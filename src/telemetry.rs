//! Request metrics, OpenTelemetry spans, and the Prometheus exporter.
//!
//! Metrics carry only configured model ids, fixed endpoint names, status codes,
//! and token counts. Prompts, headers, and credentials never reach a label.

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::OnceLock;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::http::StatusCode;
use eyre::{Result, WrapErr};
use futures_util::Stream;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use prometheus_exporter::prometheus::{
    HistogramVec, IntCounterVec, IntGaugeVec, register_histogram_vec, register_int_counter_vec,
    register_int_gauge_vec,
};
use tracing::Span;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Default scrape address; override with `INFR_PROXY_METRICS_ADDR` ("off" disables it).
const DEFAULT_METRICS_ADDR: &str = "0.0.0.0:9100";
const METRICS_ADDR_ENV: &str = "INFR_PROXY_METRICS_ADDR";
const SERVICE_NAME: &str = "infr-proxy";

/// Request durations from a short prompt to a long generation.
const DURATION_BUCKETS: &[f64] = &[
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];
/// Time a request can wait for the resident model, up to the queue timeout.
const QUEUE_BUCKETS: &[f64] = &[
    0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 240.0,
];
/// Decode rates from a slow CPU backend to a fast GPU.
const SPEED_BUCKETS: &[f64] = &[
    0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 250.0, 500.0,
];
/// An SSE line longer than this cannot be a usage chunk; drop the tail.
const MAX_SSE_PENDING: usize = 64 * 1024;
/// Below this much streamed time the decode rate is noise, not a measurement.
const MIN_DECODE_MS: f64 = 100.0;

struct Metrics {
    requests: IntCounterVec,
    duration: HistogramVec,
    queue: HistogramVec,
    tokens: IntCounterVec,
    speed: HistogramVec,
    prefill: HistogramVec,
    active: IntGaugeVec,
    http_requests: IntCounterVec,
    http_duration: HistogramVec,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

/// Token counts reported by infr's OpenAI `usage` block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt: u64,
    pub completion: u64,
    pub cached: u64,
}

impl Usage {
    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        let usage = value.get("usage")?;
        let prompt = usage.get("prompt_tokens")?.as_u64()?;
        let completion = usage
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let cached = usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        Some(Self {
            prompt,
            completion,
            cached,
        })
    }
}

/// infr's llama.cpp-style `timings`: exact prefill and decode rates for this request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    pub prompt_n: u64,
    pub prompt_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: u64,
    pub predicted_ms: f64,
    pub predicted_per_second: f64,
}

impl Timings {
    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        let timings = value.get("timings")?;
        Some(Self {
            prompt_n: timings.get("prompt_n")?.as_u64()?,
            prompt_ms: timings.get("prompt_ms")?.as_f64()?,
            prompt_per_second: timings
                .get("prompt_per_second")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
            predicted_n: timings.get("predicted_n")?.as_u64()?,
            predicted_ms: timings.get("predicted_ms")?.as_f64()?,
            predicted_per_second: timings
                .get("predicted_per_second")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
        })
    }
}

/// Keeps the OTLP provider alive and flushes it on shutdown.
pub struct Telemetry {
    provider: Option<SdkTracerProvider>,
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        let Some(provider) = &self.provider else {
            return;
        };
        if let Err(err) = provider.shutdown() {
            tracing::warn!(error = %err, "could not shut down the OTLP tracer");
        }
    }
}

/// Install logging for every command, plus metrics and OTLP traces for serve.
pub fn init(serve: bool) -> Result<Telemetry> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if !serve {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .try_init()
            .map_err(|err| eyre::eyre!("install the tracing subscriber: {err}"))?;
        return Ok(Telemetry { provider: None });
    }

    let endpoint = otlp_endpoint();
    let provider = match &endpoint {
        Some(endpoint) => Some(build_tracer(endpoint)?),
        None => None,
    };
    match &provider {
        Some(provider) => {
            let tracer = provider.tracer(SERVICE_NAME);
            tracing_subscriber::registry()
                .with(filter)
                .with(tracing_subscriber::fmt::layer())
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .try_init()
                .map_err(|err| eyre::eyre!("install the tracing subscriber: {err}"))?;
        }
        None => {
            tracing_subscriber::registry()
                .with(filter)
                .with(tracing_subscriber::fmt::layer())
                .try_init()
                .map_err(|err| eyre::eyre!("install the tracing subscriber: {err}"))?;
        }
    }
    start_metrics()?;
    if let Some(endpoint) = &endpoint {
        tracing::info!(endpoint = %endpoint, "OTLP trace export enabled");
    }
    Ok(Telemetry { provider })
}

fn start_metrics() -> Result<()> {
    let value = std::env::var(METRICS_ADDR_ENV).ok();
    let Some(addr) = metrics_addr(value)? else {
        tracing::info!("prometheus metrics disabled");
        return Ok(());
    };
    let requests = register_int_counter_vec!(
        "infr_proxy_requests_total",
        "Inference requests by model, endpoint, and HTTP status",
        &["model", "endpoint", "status"]
    )?;
    let duration = register_histogram_vec!(
        "infr_proxy_request_duration_seconds",
        "Inference request duration in seconds, including the wait for the model",
        &["model", "endpoint"],
        DURATION_BUCKETS.to_vec()
    )?;
    let queue = register_histogram_vec!(
        "infr_proxy_queue_duration_seconds",
        "Time a request waited for the resident model, before generation started",
        &["model"],
        QUEUE_BUCKETS.to_vec()
    )?;
    let tokens = register_int_counter_vec!(
        "infr_proxy_tokens_total",
        "Tokens reported by infr, by model and kind (prompt, cached, completion)",
        &["model", "kind"]
    )?;
    let speed = register_histogram_vec!(
        "infr_proxy_decode_tokens_per_second",
        "Completion tokens per second, from infr's timings when present",
        &["model"],
        SPEED_BUCKETS.to_vec()
    )?;
    let prefill = register_histogram_vec!(
        "infr_proxy_prefill_tokens_per_second",
        "Prompt tokens prefilled per second, as reported by infr",
        &["model"],
        SPEED_BUCKETS.to_vec()
    )?;
    let active = register_int_gauge_vec!(
        "infr_proxy_active_requests",
        "Requests accepted and not finished, including those waiting for the model",
        &["model"]
    )?;
    let http_requests = register_int_counter_vec!(
        "infr_proxy_http_requests_total",
        "Every HTTP request by method, route, and status",
        &["method", "route", "status"]
    )?;
    let http_duration = register_histogram_vec!(
        "infr_proxy_http_request_duration_seconds",
        "HTTP handler duration in seconds",
        &["method", "route"],
        DURATION_BUCKETS.to_vec()
    )?;
    let _ = METRICS.set(Metrics {
        requests,
        duration,
        queue,
        tokens,
        speed,
        prefill,
        active,
        http_requests,
        http_duration,
    });

    match prometheus_exporter::start(addr) {
        Ok(_) => tracing::info!(addr = %addr, "prometheus metrics listening"),
        Err(err) => {
            tracing::warn!(addr = %addr, error = %err, "could not start the metrics exporter")
        }
    }
    Ok(())
}

fn metrics_addr(value: Option<String>) -> Result<Option<SocketAddr>> {
    let value = value.unwrap_or_else(|| DEFAULT_METRICS_ADDR.to_owned());
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let addr = value
        .parse()
        .wrap_err_with(|| format!("{METRICS_ADDR_ENV} is not a socket address: {value}"))?;
    Ok(Some(addr))
}

fn build_tracer(endpoint: &str) -> Result<SdkTracerProvider> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(5))
        .build()
        .wrap_err("build the OTLP span exporter")?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            Resource::builder()
                .with_service_name(service_name())
                .build(),
        )
        .build();
    Ok(provider)
}

fn service_name() -> String {
    std::env::var("OTEL_SERVICE_NAME")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| SERVICE_NAME.to_owned())
}

fn otlp_endpoint() -> Option<String> {
    [
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ]
    .into_iter()
    .find_map(|key| {
        std::env::var(key)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}

/// Record one HTTP request, including routes that never reach a model.
pub fn record_http(method: &str, route: &str, status: StatusCode, elapsed: Duration) {
    let Some(metrics) = METRICS.get() else {
        return;
    };
    metrics
        .http_requests
        .with_label_values(&[method, route, &status.as_u16().to_string()])
        .inc();
    metrics
        .http_duration
        .with_label_values(&[method, route])
        .observe(elapsed.as_secs_f64());
}

/// Times one HTTP request and keeps its span open until the body is done.
pub struct HttpMeter {
    method: &'static str,
    route: &'static str,
    started: Instant,
    status: StatusCode,
    span: Span,
    recorded: bool,
}

impl HttpMeter {
    pub fn start(method: &'static str, route: &'static str, span: Span) -> Self {
        Self {
            method,
            route,
            started: Instant::now(),
            status: StatusCode::OK,
            span,
            recorded: false,
        }
    }

    pub fn set_status(&mut self, status: StatusCode) {
        self.status = status;
    }

    pub fn finish(&mut self) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        let elapsed = self.started.elapsed();
        self.span
            .record("http.response.status_code", i64::from(self.status.as_u16()));
        self.span.record("infr.duration_ms", millis(elapsed));
        record_http(self.method, self.route, self.status, elapsed);
    }
}

impl Drop for HttpMeter {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Passes an HTTP body through and closes its span when the body ends.
pub struct MeteredBody<S> {
    inner: Pin<Box<S>>,
    meter: HttpMeter,
}

impl<S> MeteredBody<S> {
    pub fn new(inner: S, meter: HttpMeter) -> Self {
        Self {
            inner: Box::pin(inner),
            meter,
        }
    }
}

impl<S, E> Stream for MeteredBody<S>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    type Item = Result<Bytes, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let polled = {
            let _entered = this.meter.span.enter();
            this.inner.as_mut().poll_next(cx)
        };
        match polled {
            Poll::Ready(None) => {
                this.meter.finish();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(err))) => {
                this.meter.finish();
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(Some(Ok(chunk))) => Poll::Ready(Some(Ok(chunk))),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Times one request, records its outcome exactly once, and closes its span.
pub struct RequestMeter {
    model: String,
    endpoint: &'static str,
    stream: bool,
    started: Instant,
    acquired_at: Option<Instant>,
    first_token: Option<Instant>,
    content_events: u32,
    status: StatusCode,
    usage: Option<Usage>,
    timings: Option<Timings>,
    response_model: Option<String>,
    response_id: Option<String>,
    finish_reason: Option<String>,
    response_bytes: u64,
    span: Span,
    recorded: bool,
}

impl RequestMeter {
    pub fn start(model: &str, endpoint: &'static str, stream: bool, request_bytes: u64) -> Self {
        if let Some(metrics) = METRICS.get() {
            metrics.active.with_label_values(&[model]).inc();
        }
        let span = tracing::info_span!(
            target: "infr_proxy::inference",
            "inference",
            "otel.kind" = "server",
            "gen_ai.operation.name" = operation_name(endpoint),
            "gen_ai.provider.name" = "infr",
            "http.route" = endpoint,
            "gen_ai.request.model" = model,
            "infr.stream" = stream,
            "infr.request.bytes" = as_i64(request_bytes),
            "gen_ai.response.model" = tracing::field::Empty,
            "gen_ai.response.id" = tracing::field::Empty,
            "gen_ai.response.finish_reasons" = tracing::field::Empty,
            "http.response.status_code" = tracing::field::Empty,
            "gen_ai.usage.input_tokens" = tracing::field::Empty,
            "gen_ai.usage.output_tokens" = tracing::field::Empty,
            "gen_ai.usage.cache_read.input_tokens" = tracing::field::Empty,
            "infr.queue_ms" = tracing::field::Empty,
            "infr.duration_ms" = tracing::field::Empty,
            "infr.ttft_ms" = tracing::field::Empty,
            "infr.decode_tokens_per_second" = tracing::field::Empty,
            "infr.prefill_tokens_per_second" = tracing::field::Empty,
            "infr.prefill_ms" = tracing::field::Empty,
            "infr.decode_ms" = tracing::field::Empty,
            "infr.response.bytes" = tracing::field::Empty,
        );
        Self {
            model: model.to_owned(),
            endpoint,
            stream,
            started: Instant::now(),
            acquired_at: None,
            first_token: None,
            content_events: 0,
            status: StatusCode::OK,
            usage: None,
            timings: None,
            response_model: None,
            response_id: None,
            finish_reason: None,
            response_bytes: 0,
            span,
            recorded: false,
        }
    }

    /// The resident model is ready; everything after this point is generation.
    pub fn mark_acquired(&mut self) {
        if self.acquired_at.is_none() {
            self.acquired_at = Some(Instant::now());
        }
    }

    pub fn span(&self) -> &Span {
        &self.span
    }

    pub fn set_status(&mut self, status: StatusCode) {
        self.status = status;
    }

    /// Fold one OpenAI response or stream event into the request record.
    pub fn observe_event(&mut self, value: &serde_json::Value) {
        if self.stream && event_has_content(value) {
            self.content_events = self.content_events.saturating_add(1);
            if self.first_token.is_none() {
                self.first_token = Some(Instant::now());
            }
        }
        if self.response_model.is_none()
            && let Some(model) = value.get("model").and_then(serde_json::Value::as_str)
        {
            self.response_model = Some(model.to_owned());
        }
        if self.response_id.is_none()
            && let Some(id) = value.get("id").and_then(serde_json::Value::as_str)
        {
            self.response_id = Some(id.to_owned());
        }
        if self.finish_reason.is_none()
            && let Some(reason) = value
                .pointer("/choices/0/finish_reason")
                .and_then(serde_json::Value::as_str)
        {
            self.finish_reason = Some(reason.to_owned());
        }
        if let Some(usage) = Usage::from_json(value) {
            self.usage = Some(usage);
        }
        if let Some(timings) = Timings::from_json(value) {
            self.timings = Some(timings);
        }
    }

    pub fn add_response_bytes(&mut self, bytes: u64) {
        self.response_bytes = self.response_bytes.saturating_add(bytes);
    }

    pub fn finish(&mut self) {
        if self.recorded {
            return;
        }
        self.recorded = true;

        let now = Instant::now();
        let duration = now.duration_since(self.started);
        let queue = self
            .acquired_at
            .map(|acquired| acquired.duration_since(self.started));
        let ttft = self
            .first_token
            .map(|first| first.duration_since(self.started));
        let derived_speed = match (self.first_token, self.usage) {
            (Some(first), Some(usage)) if usage.completion > 0 && self.content_events >= 2 => {
                let seconds = now.duration_since(first).as_secs_f64();
                (seconds >= MIN_DECODE_MS / 1000.0).then(|| usage.completion as f64 / seconds)
            }
            _ => None,
        };
        let decode_speed = self
            .timings
            .filter(|timings| timings.predicted_n >= 2 && timings.predicted_ms >= MIN_DECODE_MS)
            .map(|timings| timings.predicted_per_second)
            .filter(|speed| *speed > 0.0)
            .or(derived_speed);
        let prefill_speed = self
            .timings
            .map(|timings| timings.prompt_per_second)
            .filter(|speed| *speed > 0.0);
        let status = self.status.as_u16();

        if let Some(metrics) = METRICS.get() {
            metrics
                .requests
                .with_label_values(&[&self.model, self.endpoint, &status.to_string()])
                .inc();
            metrics
                .duration
                .with_label_values(&[&self.model, self.endpoint])
                .observe(duration.as_secs_f64());
            metrics.active.with_label_values(&[&self.model]).dec();
            if let Some(queue) = queue {
                metrics
                    .queue
                    .with_label_values(&[&self.model])
                    .observe(queue.as_secs_f64());
            }
            if let Some(speed) = decode_speed {
                metrics
                    .speed
                    .with_label_values(&[&self.model])
                    .observe(speed);
            }
            if let Some(speed) = prefill_speed {
                metrics
                    .prefill
                    .with_label_values(&[&self.model])
                    .observe(speed);
            }
            if let Some(usage) = self.usage {
                for (kind, value) in [
                    ("prompt", usage.prompt),
                    ("cached", usage.cached),
                    ("completion", usage.completion),
                ] {
                    if value > 0 {
                        metrics
                            .tokens
                            .with_label_values(&[&self.model, kind])
                            .inc_by(value);
                    }
                }
            }
        }

        self.span
            .record("http.response.status_code", i64::from(status));
        self.span.record("infr.duration_ms", millis(duration));
        self.span
            .record("infr.response.bytes", as_i64(self.response_bytes));
        if let Some(queue) = queue {
            self.span.record("infr.queue_ms", millis(queue));
        }
        if let Some(ttft) = ttft {
            self.span.record("infr.ttft_ms", millis(ttft));
        }
        if let Some(speed) = decode_speed {
            self.span.record("infr.decode_tokens_per_second", speed);
        }
        if let Some(speed) = prefill_speed {
            self.span.record("infr.prefill_tokens_per_second", speed);
        }
        if let Some(timings) = self.timings {
            self.span.record("infr.prefill_ms", timings.prompt_ms);
            self.span.record("infr.decode_ms", timings.predicted_ms);
        }
        if let Some(model) = &self.response_model {
            self.span.record("gen_ai.response.model", model.as_str());
        }
        if let Some(id) = &self.response_id {
            self.span.record("gen_ai.response.id", id.as_str());
        }
        if let Some(reason) = &self.finish_reason {
            self.span
                .record("gen_ai.response.finish_reasons", reason.as_str());
        }
        if let Some(usage) = self.usage {
            self.span
                .record("gen_ai.usage.input_tokens", as_i64(usage.prompt));
            self.span
                .record("gen_ai.usage.output_tokens", as_i64(usage.completion));
            self.span
                .record("gen_ai.usage.cache_read.input_tokens", as_i64(usage.cached));
        }
        if status >= 500 {
            self.span.record("otel.status_code", "ERROR");
            self.span
                .record("otel.status_description", "model or upstream failure");
        }

        let _entered = self.span.enter();
        tracing::info!(
            target: "infr_proxy::request",
            model = %self.model,
            endpoint = self.endpoint,
            status,
            stream = self.stream,
            duration_ms = millis(duration),
            queue_ms = queue.map(millis),
            ttft_ms = ttft.map(millis),
            tokens_per_second = decode_speed,
            prefill_tokens_per_second = prefill_speed,
            prefill_ms = self.timings.map(|timings| timings.prompt_ms),
            decode_ms = self.timings.map(|timings| timings.predicted_ms),
            prompt_tokens = self.usage.map_or(0, |usage| usage.prompt),
            completion_tokens = self.usage.map_or(0, |usage| usage.completion),
            cached_tokens = self.usage.map_or(0, |usage| usage.cached),
            "request finished"
        );
    }
}

impl Drop for RequestMeter {
    fn drop(&mut self) {
        self.finish();
    }
}

pub(crate) fn millis(duration: Duration) -> i64 {
    duration.as_millis().min(i64::MAX as u128) as i64
}

fn as_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn operation_name(endpoint: &str) -> &'static str {
    match endpoint {
        "/v1/chat/completions" | "/api/chat" => "chat",
        "/v1/completions" | "/api/generate" => "text_completion",
        _ => "inference",
    }
}

fn event_has_content(value: &serde_json::Value) -> bool {
    let content = value
        .pointer("/choices/0/delta/content")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|content| !content.is_empty());
    let text = value
        .pointer("/choices/0/text")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|text| !text.is_empty());
    content || text
}

/// Passes a response stream through while folding its events into the meter.
pub struct MeteredStream<S> {
    inner: Pin<Box<S>>,
    meter: RequestMeter,
    upstream: Span,
    pending: Vec<u8>,
}

impl<S> MeteredStream<S> {
    pub fn new(inner: S, meter: RequestMeter, upstream: Span) -> Self {
        Self {
            inner: Box::pin(inner),
            meter,
            upstream,
            pending: Vec::new(),
        }
    }

    fn scan(&mut self, chunk: &[u8]) {
        self.meter.add_response_bytes(chunk.len() as u64);
        self.pending.extend_from_slice(chunk);
        while let Some(index) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=index).collect();
            if let Some(event) = sse_event(&line) {
                self.meter.observe_event(&event);
            }
        }
        if self.pending.len() > MAX_SSE_PENDING {
            self.pending.clear();
        }
    }
}

impl<S, E> Stream for MeteredStream<S>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    type Item = Result<Bytes, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let polled = {
            let _entered = this.upstream.enter();
            this.inner.as_mut().poll_next(cx)
        };
        match polled {
            Poll::Ready(Some(Ok(chunk))) => {
                this.scan(&chunk);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(err))) => {
                this.meter.set_status(StatusCode::BAD_GATEWAY);
                this.meter.finish();
                this.upstream = Span::none();
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(None) => {
                this.meter.finish();
                this.upstream = Span::none();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn sse_event(line: &[u8]) -> Option<serde_json::Value> {
    let text = std::str::from_utf8(line).ok()?;
    let payload = text.trim_start().strip_prefix("data:")?.trim();
    if !payload.starts_with('{') {
        return None;
    }
    serde_json::from_str(payload).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_reads_prompt_completion_and_cached_tokens() {
        let value = serde_json::json!({
            "usage": {
                "prompt_tokens": 892,
                "completion_tokens": 295,
                "total_tokens": 1187,
                "prompt_tokens_details": { "cached_tokens": 128 }
            }
        });
        assert_eq!(
            Usage::from_json(&value),
            Some(Usage {
                prompt: 892,
                completion: 295,
                cached: 128
            })
        );
    }

    #[test]
    fn usage_is_absent_from_a_content_chunk() {
        let value = serde_json::json!({
            "choices": [{ "delta": { "content": "hello" } }]
        });
        assert_eq!(Usage::from_json(&value), None);
    }

    #[test]
    fn timings_read_prefill_and_decode_rates() {
        let value = serde_json::json!({
            "timings": {
                "prompt_n": 120,
                "prompt_ms": 600.0,
                "prompt_per_second": 200.0,
                "predicted_n": 30,
                "predicted_ms": 1000.0,
                "predicted_per_second": 30.0
            }
        });
        assert_eq!(
            Timings::from_json(&value),
            Some(Timings {
                prompt_n: 120,
                prompt_ms: 600.0,
                prompt_per_second: 200.0,
                predicted_n: 30,
                predicted_ms: 1000.0,
                predicted_per_second: 30.0,
            })
        );
    }

    #[test]
    fn an_sse_usage_event_split_across_chunks_is_recorded() {
        let meter = RequestMeter::start("test:model", "/v1/chat/completions", true, 0);
        let mut metered = MeteredStream::new(
            futures_util::stream::empty::<Result<Bytes, ()>>(),
            meter,
            Span::none(),
        );
        metered.scan(b"data: {\"choices\":[],\"usa");
        metered.scan(b"ge\":{\"prompt_tokens\":7,\"completion_tokens\":3,");
        metered.scan(b"\"prompt_tokens_details\":{\"cached_tokens\":2}}}\n\n");
        assert_eq!(
            metered.meter.usage,
            Some(Usage {
                prompt: 7,
                completion: 3,
                cached: 2
            })
        );
    }

    #[test]
    fn a_content_event_marks_the_first_token() {
        let meter = RequestMeter::start("test:model", "/v1/chat/completions", true, 0);
        let mut metered = MeteredStream::new(
            futures_util::stream::empty::<Result<Bytes, ()>>(),
            meter,
            Span::none(),
        );
        metered.scan(b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n");
        assert!(metered.meter.first_token.is_none());
        metered.scan(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n");
        assert!(metered.meter.first_token.is_some());
    }

    #[test]
    fn metrics_address_can_be_disabled() {
        assert_eq!(metrics_addr(Some("off".into())).unwrap(), None);
        assert_eq!(
            metrics_addr(None).unwrap(),
            Some(DEFAULT_METRICS_ADDR.parse().unwrap())
        );
        assert!(metrics_addr(Some("not an address".into())).is_err());
    }
}
