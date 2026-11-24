use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use axum::Json;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Query, Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tracing::Instrument;

use crate::config::Model;
use crate::gguf;
use crate::registry::Reference;
use crate::slot::{Lease, Slot, SlotError};
use crate::telemetry::{HttpMeter, MeteredBody, MeteredStream, RequestMeter, Usage};

#[derive(Clone)]
struct AppState {
    slot: Arc<Slot>,
    clients: Arc<Mutex<HashMap<PathBuf, reqwest::Client>>>,
    metadata: Arc<MetadataCache>,
}

#[derive(Default)]
struct MetadataCache {
    entries: Mutex<HashMap<PathBuf, CachedMetadata>>,
}

struct CachedMetadata {
    modified: Option<SystemTime>,
    len: u64,
    metadata: Arc<gguf::Metadata>,
}

impl AppState {
    async fn unix_client(&self, socket: &Path) -> Result<reqwest::Client, reqwest::Error> {
        let mut clients = self.clients.lock().await;
        if let Some(client) = clients.get(socket) {
            return Ok(client.clone());
        }
        let client = reqwest::Client::builder()
            .unix_socket(socket.to_path_buf())
            .build()?;
        clients.insert(socket.to_path_buf(), client.clone());
        Ok(client)
    }

    /// Parse the GGUF header once per file revision; a failed read is cached as empty.
    async fn metadata(&self, path: &str) -> Arc<gguf::Metadata> {
        let stat = tokio::fs::metadata(path).await.ok();
        let modified = stat.as_ref().and_then(|stat| stat.modified().ok());
        let len = stat.as_ref().map_or(0, std::fs::Metadata::len);
        if let Some(cached) = self.metadata.entries.lock().await.get(Path::new(path))
            && cached.modified == modified
            && cached.len == len
        {
            return Arc::clone(&cached.metadata);
        }

        let parsed = tokio::task::spawn_blocking({
            let path = path.to_owned();
            move || gguf::read(Path::new(&path))
        })
        .await;
        let metadata = match parsed {
            Ok(Ok(metadata)) => metadata,
            Ok(Err(err)) => {
                tracing::debug!(path, error = %err, "could not read GGUF metadata");
                gguf::Metadata::default()
            }
            Err(err) => {
                tracing::warn!(path, error = %err, "GGUF metadata task failed");
                gguf::Metadata::default()
            }
        };

        let metadata = Arc::new(metadata);
        self.metadata.entries.lock().await.insert(
            PathBuf::from(path),
            CachedMetadata {
                modified,
                len,
                metadata: Arc::clone(&metadata),
            },
        );
        metadata
    }

    /// An explicit override beats the live window, which beats the trained ceiling.
    async fn reported_context(
        &self,
        model: &Model,
        metadata: &gguf::Metadata,
        resident: bool,
    ) -> Option<u64> {
        if let Some(context_length) = model.context_length {
            return Some(u64::from(context_length));
        }
        if resident {
            let live = context_length(self, &model.id, &model.socket).await;
            if live > 0 {
                return Some(live);
            }
        }
        metadata.context_length.map(u64::from)
    }
}

pub fn router(slot: Arc<Slot>) -> Router {
    let state = AppState {
        slot,
        clients: Arc::new(Mutex::new(HashMap::new())),
        metadata: Arc::new(MetadataCache::default()),
    };
    Router::new()
        .route("/", get(root).head(root))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/registry/model-path", get(model_path))
        .route("/api/version", get(version))
        .route("/api/tags", get(tags))
        .route("/api/ps", get(ps))
        .route("/api/show", post(show))
        .route("/api/pull", post(pull))
        .route("/api/chat", post(ollama_chat))
        .route("/api/generate", post(ollama_generate))
        .with_state(state)
        .layer(middleware::from_fn(meter_http))
}

async fn meter_http(request: Request, next: Next) -> Response {
    let method = method_label(request.method().as_str());
    let path = request.uri().path().to_owned();
    let route = route_label(&path);
    let span = tracing::info_span!(
        "http.request",
        "otel.kind" = "server",
        "http.request.method" = method,
        "url.path" = %path,
        "http.response.status_code" = tracing::field::Empty,
        "infr.duration_ms" = tracing::field::Empty,
    );
    let mut meter = HttpMeter::start(method, route, span.clone());
    let response = next.run(request).instrument(span).await;
    meter.set_status(response.status());
    let (parts, body) = response.into_parts();
    let body = Body::from_stream(MeteredBody::new(body.into_data_stream(), meter));
    Response::from_parts(parts, body)
}

fn method_label(method: &str) -> &'static str {
    match method {
        "GET" => "GET",
        "HEAD" => "HEAD",
        "POST" => "POST",
        _ => "other",
    }
}

/// Keep route labels bounded even when a client requests an unknown path.
fn route_label(path: &str) -> &'static str {
    match path {
        "/" => "/",
        "/health" => "/health",
        "/v1/models" => "/v1/models",
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/completions" => "/v1/completions",
        "/registry/model-path" => "/registry/model-path",
        "/api/version" => "/api/version",
        "/api/tags" => "/api/tags",
        "/api/ps" => "/api/ps",
        "/api/show" => "/api/show",
        "/api/pull" => "/api/pull",
        "/api/chat" => "/api/chat",
        "/api/generate" => "/api/generate",
        _ => "unmatched",
    }
}

async fn root() -> &'static str {
    "Ollama is running"
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

async fn version() -> Json<serde_json::Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

#[derive(Deserialize)]
struct ModelPathQuery {
    id: String,
}

#[derive(Serialize)]
struct ModelPathResponse {
    id: String,
    path: String,
}

async fn model_path(
    State(state): State<AppState>,
    Query(query): Query<ModelPathQuery>,
) -> Result<Json<ModelPathResponse>, ApiError> {
    let model = state
        .slot
        .model(&query.id)
        .await
        .map_err(|err| ApiError::from_slot(err, &query.id))?;
    Ok(Json(ModelPathResponse {
        id: model.id.clone(),
        path: model.source.clone(),
    }))
}

#[derive(Serialize)]
struct TagsResponse {
    models: Vec<TagEntry>,
}

#[derive(Serialize)]
struct TagEntry {
    name: String,
    model: String,
    modified_at: String,
    size: u64,
    digest: String,
    details: ModelDetails,
}

#[derive(Serialize)]
struct ModelDetails {
    format: &'static str,
    family: String,
    families: Vec<String>,
    parameter_size: String,
    quantization_level: String,
}

fn details(metadata: &gguf::Metadata) -> ModelDetails {
    let family = metadata.architecture.clone().unwrap_or_default();
    ModelDetails {
        format: "gguf",
        families: if family.is_empty() {
            Vec::new()
        } else {
            vec![family.clone()]
        },
        family,
        parameter_size: metadata
            .parameter_count
            .map(parameter_size)
            .unwrap_or_default(),
        quantization_level: metadata
            .file_type
            .map(quantization)
            .unwrap_or_default()
            .to_owned(),
    }
}

/// The GGUF `general.parameter_count`, rendered the way Ollama renders details.
fn parameter_size(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else {
        count.to_string()
    }
}

/// `general.file_type` uses llama.cpp's `llama_ftype`; names match what the file was built as.
fn quantization(file_type: u32) -> &'static str {
    match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        39 => "NVFP4",
        40 => "Q1_0",
        41 => "Q2_0",
        42 => "F8_E4M3",
        _ => "",
    }
}

/// The `model_info` keys clients read: architecture and the context window under it.
fn model_info(metadata: &gguf::Metadata, context_length: Option<u64>) -> serde_json::Value {
    let mut info = serde_json::Map::new();
    if let Some(architecture) = &metadata.architecture {
        info.insert("general.architecture".into(), json!(architecture));
        if let Some(context_length) = context_length {
            info.insert(
                format!("{architecture}.context_length"),
                json!(context_length),
            );
        }
    }
    if let Some(parameter_count) = metadata.parameter_count {
        info.insert("general.parameter_count".into(), json!(parameter_count));
    }
    if let Some(file_type) = metadata.file_type {
        info.insert("general.file_type".into(), json!(file_type));
    }
    serde_json::Value::Object(info)
}

fn capabilities(model: &Model) -> Vec<&'static str> {
    let mut capabilities = vec!["completion"];
    if model.reasoning == Some(true) {
        capabilities.push("thinking");
    }
    if model.vision == Some(true) {
        capabilities.push("vision");
    }
    capabilities
}

async fn tags(State(state): State<AppState>) -> Json<TagsResponse> {
    let mut entries = Vec::new();
    for model in state.slot.models().await {
        let metadata = state.metadata(&model.source).await;
        let stat = tokio::fs::metadata(&model.source).await.ok();
        let modified = stat
            .as_ref()
            .and_then(|stat| stat.modified().ok())
            .map(|time| {
                OffsetDateTime::from(time)
                    .format(&Rfc3339)
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        entries.push(TagEntry {
            name: model.id.clone(),
            model: model.id.clone(),
            modified_at: modified,
            size: stat.map_or(0, |stat| stat.len()),
            digest: model.digest.clone(),
            details: details(&metadata),
        });
    }
    Json(TagsResponse { models: entries })
}

#[derive(Serialize)]
struct PsResponse {
    models: Vec<PsEntry>,
}

#[derive(Serialize)]
struct PsEntry {
    name: String,
    model: String,
    size: u64,
    digest: String,
    expires_at: String,
    context_length: u64,
    details: ModelDetails,
}

#[derive(Deserialize)]
struct ShowRequest {
    model: Option<String>,
    name: Option<String>,
}

async fn show(
    State(state): State<AppState>,
    Json(request): Json<ShowRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let id = request
        .model
        .filter(|model| !model.is_empty())
        .or(request.name)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ApiError::bad_request("model is required"))?;
    let model = state
        .slot
        .model(&id)
        .await
        .map_err(|err| ApiError::from_slot(err, &id))?;
    let metadata = state.metadata(&model.source).await;
    let resident = state.slot.resident().await.is_some_and(|id| id == model.id);
    let context_length = state.reported_context(&model, &metadata, resident).await;
    Ok(Json(json!({
        "modelfile": format!("# infr-proxy model {id}\n"),
        "parameters": "",
        "template": "",
        "details": details(&metadata),
        "model_info": model_info(&metadata, context_length),
        "capabilities": capabilities(&model),
        "modified_at": OffsetDateTime::from(
            tokio::fs::metadata(&model.source)
                .await
                .ok()
                .and_then(|meta| meta.modified().ok())
                .unwrap_or(std::time::UNIX_EPOCH)
        )
        .format(&Rfc3339)
        .unwrap_or_default(),
    })))
}

#[derive(Deserialize)]
struct PullRequest {
    name: Option<String>,
    model: Option<String>,
    #[serde(rename = "stream", default)]
    _stream: bool,
}

async fn pull(
    State(state): State<AppState>,
    Json(request): Json<PullRequest>,
) -> Result<Response, ApiError> {
    let id = request
        .name
        .filter(|name| !name.is_empty())
        .or(request.model)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| ApiError::bad_request("model is required"))?;
    let reference = Reference::parse(&id).map_err(|err| ApiError::bad_request(err.to_string()))?;
    let registry = state.slot.registry();
    let (tx, rx) = tokio::sync::mpsc::channel::<serde_json::Value>(32);

    tokio::spawn(async move {
        let _ = tx.send(json!({ "status": "pulling manifest" })).await;
        let mut reported = 0u64;
        let result = registry
            .pull_with(&reference, |completed, total, digest| {
                if completed == total || completed.saturating_sub(reported) >= 4 * 1024 * 1024 {
                    reported = completed;
                    let _ = tx.try_send(json!({
                        "status": format!("pulling {}", reference.name),
                        "digest": digest,
                        "total": total,
                        "completed": completed,
                    }));
                }
            })
            .await;
        let event = match result {
            Ok(_) => json!({ "status": "success" }),
            Err(err) => json!({ "error": err.to_string() }),
        };
        let _ = tx.send(event).await;
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|event| (Ok::<_, std::convert::Infallible>(format!("{event}\n")), rx))
    });
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    Ok(response)
}

async fn ps(State(state): State<AppState>) -> Json<PsResponse> {
    let mut entries = Vec::new();
    if let Some(id) = state.slot.resident().await
        && let Ok(model) = state.slot.model(&id).await
    {
        let size = tokio::fs::metadata(&model.source)
            .await
            .map(|meta| meta.len())
            .unwrap_or(0);
        let metadata = state.metadata(&model.source).await;
        entries.push(PsEntry {
            name: model.id.clone(),
            model: model.id.clone(),
            size,
            digest: model.digest.clone(),
            expires_at: (OffsetDateTime::now_utc() + time::Duration::minutes(5))
                .format(&Rfc3339)
                .unwrap_or_default(),
            context_length: context_length(&state, &model.id, &model.socket).await,
            details: details(&metadata),
        });
    }
    Json(PsResponse { models: entries })
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u64>,
}

/// infr reports the per-slot window a request actually gets in `/status`.
async fn context_length(state: &AppState, id: &str, socket: &Path) -> u64 {
    let status = async {
        let client = state.unix_client(socket).await?;
        client
            .get("http://localhost/status")
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await
    }
    .await;

    match status {
        Ok(body) => body
            .pointer("/models/0/ctx_per_slot")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        Err(err) => {
            tracing::debug!(model = id, error = %err, "could not read infr status");
            0
        }
    }
}

async fn models(State(state): State<AppState>) -> Json<ModelList> {
    let resident = state.slot.resident().await;
    let mut data = Vec::new();
    for model in state.slot.models().await {
        let metadata = state.metadata(&model.source).await;
        let is_resident = resident.as_deref() == Some(model.id.as_str());
        let context_length = state.reported_context(&model, &metadata, is_resident).await;
        data.push(ModelEntry {
            id: model.id,
            object: "model",
            owned_by: "infr-proxy",
            context_length,
        });
    }
    Json(ModelList {
        object: "list",
        data,
    })
}

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    #[serde(default)]
    stream: bool,
}

async fn chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    forward_openai(&state, headers, body, "/v1/chat/completions").await
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    forward_openai(&state, headers, body, "/v1/completions").await
}

async fn forward_openai(
    state: &AppState,
    headers: HeaderMap,
    body: Bytes,
    endpoint: &'static str,
) -> Result<Response, ApiError> {
    let request: ChatRequest = serde_json::from_slice(&body)
        .map_err(|err| ApiError::bad_request(format!("invalid JSON body: {err}")))?;
    let id = request
        .model
        .ok_or_else(|| ApiError::bad_request("model is required"))?;

    let mut meter = RequestMeter::start(&id, endpoint, request.stream, body.len() as u64);
    let queue_span = tracing::info_span!(parent: meter.span(), "model.queue", model = %id);
    let (model, lease) = state
        .slot
        .acquire_model(&id)
        .instrument(queue_span)
        .await
        .map_err(|err| {
            let api = ApiError::from_slot(err, &id);
            meter.set_status(api.status);
            api
        })?;
    meter.mark_acquired();

    let client = state
        .unix_client(&model.socket)
        .await
        .map_err(|err| ApiError::bad_gateway(format!("connect to {}: {err}", model.id)))?;
    let url = format!("http://localhost{endpoint}");
    let mut outgoing = client
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .body(body);
    if let Some(auth) = headers.get(AUTHORIZATION) {
        outgoing = outgoing.header(AUTHORIZATION, auth);
    }
    let upstream_span = tracing::info_span!(
        parent: meter.span(),
        "infr.upstream",
        "otel.kind" = "client",
        "http.request.method" = "POST",
        "server.address" = %model.socket.display(),
        "network.transport" = "unix",
        "http.response.status_code" = tracing::field::Empty,
    );
    let upstream = match outgoing.send().instrument(upstream_span.clone()).await {
        Ok(upstream) => upstream,
        Err(err) => {
            meter.set_status(StatusCode::BAD_GATEWAY);
            return Err(ApiError::bad_gateway(format!(
                "forward to {}: {err}",
                model.id
            )));
        }
    };

    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    upstream_span.record("http.response.status_code", i64::from(status.as_u16()));
    meter.set_status(status);
    let content_type = upstream
        .headers()
        .get(CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));

    if request.stream {
        let stream = MeteredStream::new(upstream.bytes_stream(), meter, upstream_span);
        let stream = stream.map(move |chunk| {
            let _lease: &Lease = &lease;
            chunk
        });
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = status;
        response.headers_mut().insert(CONTENT_TYPE, content_type);
        return Ok(response);
    }

    let bytes = match upstream.bytes().instrument(upstream_span).await {
        Ok(bytes) => bytes,
        Err(err) => {
            meter.set_status(StatusCode::BAD_GATEWAY);
            return Err(ApiError::bad_gateway(format!(
                "read {} reply: {err}",
                model.id
            )));
        }
    };
    drop(lease);

    if let Some(value) = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .as_ref()
    {
        meter.observe_event(value);
    }
    meter.add_response_bytes(bytes.len() as u64);
    meter.finish();

    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    response.headers_mut().insert(CONTENT_TYPE, content_type);
    Ok(response)
}

#[derive(Deserialize)]
struct OllamaChatRequest {
    model: Option<String>,
    messages: Vec<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    options: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct OllamaGenerateRequest {
    model: Option<String>,
    prompt: Option<String>,
    system: Option<String>,
    #[serde(default)]
    stream: bool,
    options: Option<serde_json::Value>,
}

async fn ollama_chat(
    State(state): State<AppState>,
    Json(request): Json<OllamaChatRequest>,
) -> Result<Response, OllamaError> {
    let id = request
        .model
        .ok_or_else(|| OllamaError::bad_request("model is required"))?;
    run_ollama(
        &state,
        &id,
        request.messages,
        request.options,
        request.stream,
        false,
    )
    .await
}

async fn ollama_generate(
    State(state): State<AppState>,
    Json(request): Json<OllamaGenerateRequest>,
) -> Result<Response, OllamaError> {
    let id = request
        .model
        .ok_or_else(|| OllamaError::bad_request("model is required"))?;
    let mut messages = Vec::new();
    if let Some(system) = request.system {
        messages.push(json!({ "role": "system", "content": system }));
    }
    messages.push(json!({
        "role": "user",
        "content": request.prompt.unwrap_or_default(),
    }));
    run_ollama(&state, &id, messages, request.options, request.stream, true).await
}

async fn run_ollama(
    state: &AppState,
    id: &str,
    messages: Vec<serde_json::Value>,
    options: Option<serde_json::Value>,
    stream: bool,
    generate: bool,
) -> Result<Response, OllamaError> {
    let model = state
        .slot
        .model(id)
        .await
        .map_err(|err| OllamaError::from_slot(err, id))?;
    let endpoint: &'static str = if generate {
        "/api/generate"
    } else {
        "/api/chat"
    };

    let mut body = json!({ "model": model.id, "messages": messages, "stream": stream });
    if let Some(options) = options {
        for (from, to) in [
            ("temperature", "temperature"),
            ("num_predict", "max_tokens"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("seed", "seed"),
            ("stop", "stop"),
        ] {
            if let Some(value) = options.get(from) {
                body[to] = value.clone();
            }
        }
    }
    if stream {
        body["stream_options"] = json!({ "include_usage": true });
    }
    let request_bytes = serde_json::to_vec(&body)
        .map(|bytes| bytes.len() as u64)
        .unwrap_or(0);

    let mut meter = RequestMeter::start(&model.id, endpoint, stream, request_bytes);
    let queue_span = tracing::info_span!(parent: meter.span(), "model.queue", model = %model.id);
    let (running, lease) = state
        .slot
        .acquire_model(&model.id)
        .instrument(queue_span)
        .await
        .map_err(|err| {
            let err = OllamaError::from_slot(err, &model.id);
            meter.set_status(err.status);
            err
        })?;
    meter.mark_acquired();

    let client = state
        .unix_client(&running.socket)
        .await
        .map_err(|err| OllamaError::backend(format!("connect to {}: {err}", running.id)))?;
    let url = "http://localhost/v1/chat/completions";
    tracing::debug!(model = %model.id, socket = %running.socket.display(), "ollama: forwarding");
    let upstream_span = tracing::info_span!(
        parent: meter.span(),
        "infr.upstream",
        "otel.kind" = "client",
        "http.request.method" = "POST",
        "server.address" = %running.socket.display(),
        "network.transport" = "unix",
        "http.response.status_code" = tracing::field::Empty,
    );
    let upstream = match client
        .post(url)
        .json(&body)
        .send()
        .instrument(upstream_span.clone())
        .await
    {
        Ok(upstream) => upstream,
        Err(err) => {
            meter.set_status(StatusCode::BAD_GATEWAY);
            return Err(OllamaError::backend(format!("forward to {id}: {err}")));
        }
    };
    let status = upstream.status();
    upstream_span.record("http.response.status_code", i64::from(status.as_u16()));
    if !status.is_success() {
        let text = upstream.text().await.unwrap_or_default();
        meter.set_status(StatusCode::BAD_GATEWAY);
        return Err(OllamaError::backend(format!(
            "infr answered {status}: {}",
            text.trim()
        )));
    }

    if !stream {
        let bytes = match upstream.bytes().instrument(upstream_span).await {
            Ok(bytes) => bytes,
            Err(err) => {
                meter.set_status(StatusCode::BAD_GATEWAY);
                return Err(OllamaError::backend(format!("read {id} reply: {err}")));
            }
        };
        let reply: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(reply) => reply,
            Err(err) => {
                meter.set_status(StatusCode::BAD_GATEWAY);
                return Err(OllamaError::backend(format!("read {id} reply: {err}")));
            }
        };
        meter.add_response_bytes(bytes.len() as u64);
        meter.observe_event(&reply);
        let usage = Usage::from_json(&reply);
        let content = reply["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default();
        drop(lease);
        meter.finish();
        return Ok(Json(ollama_final(&running.id, content, generate, usage)).into_response());
    }

    let (tx, rx) = mpsc::channel::<serde_json::Value>(32);
    let task = stream_ollama(upstream, running.id, tx, lease, generate, meter);
    tokio::spawn(task.instrument(upstream_span));
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|event| (Ok::<_, std::convert::Infallible>(format!("{event}\n")), rx))
    });
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    Ok(response)
}

async fn stream_ollama(
    upstream: reqwest::Response,
    id: String,
    tx: mpsc::Sender<serde_json::Value>,
    lease: Lease,
    generate: bool,
    mut meter: RequestMeter,
) {
    let _lease = lease;
    let mut usage = None;
    let mut buffer = Vec::new();
    let mut stream = upstream.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(_) => {
                meter.set_status(StatusCode::BAD_GATEWAY);
                break;
            }
        };
        buffer.extend_from_slice(&chunk);
        meter.add_response_bytes(chunk.len() as u64);
        while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = buffer.drain(..=position).collect();
            let line = String::from_utf8_lossy(&line);
            let Some(payload) = line.trim().strip_prefix("data:") else {
                continue;
            };
            let payload = payload.trim();
            if payload == "[DONE]" {
                let _ = tx.send(ollama_final(&id, "", generate, usage)).await;
                return;
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };
            meter.observe_event(&value);
            if let Some(found) = Usage::from_json(&value) {
                usage = Some(found);
            }
            if let Some(content) = value["choices"][0]["delta"]["content"].as_str()
                && !content.is_empty()
                && tx.send(ollama_chunk(&id, content, generate)).await.is_err()
            {
                return;
            }
        }
    }
    let _ = tx.send(ollama_final(&id, "", generate, usage)).await;
}

fn timestamp() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_default()
}

fn ollama_chunk(id: &str, content: &str, generate: bool) -> serde_json::Value {
    if generate {
        json!({ "model": id, "created_at": timestamp(), "response": content, "done": false })
    } else {
        json!({
            "model": id,
            "created_at": timestamp(),
            "message": { "role": "assistant", "content": content },
            "done": false,
        })
    }
}

fn ollama_final(
    id: &str,
    content: &str,
    generate: bool,
    usage: Option<Usage>,
) -> serde_json::Value {
    let prompt = usage.map_or(0, |usage| usage.prompt);
    let completion = usage.map_or(0, |usage| usage.completion);
    if generate {
        json!({
            "model": id,
            "created_at": timestamp(),
            "response": content,
            "done": true,
            "done_reason": "stop",
            "eval_count": completion,
            "prompt_eval_count": prompt,
        })
    } else {
        json!({
            "model": id,
            "created_at": timestamp(),
            "message": { "role": "assistant", "content": content },
            "done": true,
            "done_reason": "stop",
            "eval_count": completion,
            "prompt_eval_count": prompt,
        })
    }
}

struct OllamaError {
    status: StatusCode,
    message: String,
}

impl OllamaError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(id: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: format!("model {id} is not served by this proxy"),
        }
    }

    fn backend(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
        }
    }

    fn from_slot(err: SlotError, id: &str) -> Self {
        match err {
            SlotError::UnknownModel => Self::not_found(id),
            SlotError::TooLarge { size, max } => Self {
                status: StatusCode::BAD_REQUEST,
                message: format!(
                    "model {id} is {size} bytes, larger than the {max}-byte max_model_size"
                ),
            },
            SlotError::Busy => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: err.to_string(),
            },
            SlotError::Backend(message) => Self {
                status: StatusCode::BAD_GATEWAY,
                message,
            },
        }
    }
}

impl IntoResponse for OllamaError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

struct ApiError {
    status: StatusCode,
    kind: &'static str,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: "invalid_request",
            message: message.into(),
        }
    }

    fn model_not_found(id: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            code: "model_not_found",
            message: format!("model {id} is not served by this proxy"),
        }
    }

    fn busy(id: &str) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "server_error",
            code: "model_busy",
            message: format!("{id} is waiting for the resident model to drain"),
        }
    }

    fn from_slot(err: SlotError, id: &str) -> Self {
        match err {
            SlotError::UnknownModel => Self::model_not_found(id),
            SlotError::TooLarge { size, max } => Self {
                status: StatusCode::BAD_REQUEST,
                kind: "invalid_request_error",
                code: "model_too_large",
                message: format!(
                    "model {id} is {size} bytes, larger than the {max}-byte max_model_size"
                ),
            },
            SlotError::Busy => Self::busy(id),
            SlotError::Backend(message) => Self::bad_gateway(message),
        }
    }

    fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            kind: "server_error",
            code: "backend_error",
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let retry = self.status == StatusCode::SERVICE_UNAVAILABLE;
        let body = json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "code": self.code,
            }
        });
        let mut response = (self.status, Json(body)).into_response();
        if retry {
            response
                .headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from_static("5"));
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    use crate::config::{Config, Model, ModelEntry};
    use crate::host::ModelHost;
    use crate::slot::Slot;

    struct ReadyHost;

    #[async_trait]
    impl ModelHost for ReadyHost {
        async fn start(&self, _: &Model) -> eyre::Result<()> {
            Ok(())
        }

        async fn stop(&self, _: &Model) -> eyre::Result<()> {
            Ok(())
        }

        async fn is_active(&self, _: &Model) -> eyre::Result<bool> {
            Ok(true)
        }
    }

    struct Proxy {
        _dir: tempfile::TempDir,
        addr: std::net::SocketAddr,
    }

    impl Proxy {
        async fn start(backend: Router) -> Self {
            Self::start_with_source(backend, b"model bytes").await
        }

        async fn start_with_source(backend: Router, source_bytes: &[u8]) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let source = dir.path().join("model.gguf");
            tokio::fs::write(&source, source_bytes).await.unwrap();

            let runtime = dir.path().join("runtime");
            tokio::fs::create_dir_all(runtime.join("infr"))
                .await
                .unwrap();
            let backend_socket = runtime.join("infr/test-model.sock");
            let backend_listener = tokio::net::UnixListener::bind(&backend_socket).unwrap();
            tokio::spawn(async move {
                axum::serve(backend_listener, backend).await.unwrap();
            });

            let config = Config {
                listen: "127.0.0.1:0".parse().unwrap(),
                queue_timeout_secs: 5,
                load_timeout_secs: 1,
                poll_interval_ms: 10,
                cache_dir: Some(dir.path().join("cache")),
                extra_model_dirs: Vec::new(),
                max_model_size: None,
                registries: Vec::new(),
                models: vec![ModelEntry {
                    id: "test:model".into(),
                    instance: Some("test-model".into()),
                    source: Some(source.to_string_lossy().into_owned()),
                    context_length: None,
                    reasoning: None,
                    vision: None,
                }],
                exclude: Vec::new(),
            };
            let slot = Arc::new(
                Slot::with_host(config, Arc::new(ReadyHost), runtime)
                    .await
                    .unwrap(),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                axum::serve(listener, router(slot)).await.unwrap();
            });

            Self { _dir: dir, addr }
        }

        async fn complete(&self) {
            reqwest::Client::new()
                .post(format!("http://{}/v1/completions", self.addr))
                .json(&json!({ "model": "test:model", "prompt": "hello" }))
                .send()
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn completion_requests_reach_the_infr_completions_route() {
        let proxy = Proxy::start(
            Router::new()
                .route("/v1/completions", post(|| async { "completion backend" }))
                .route("/v1/chat/completions", post(|| async { "chat backend" })),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("http://{}/v1/completions", proxy.addr))
            .json(&json!({ "model": "test:model", "prompt": "hello" }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.text().await.unwrap(), "completion backend");
    }

    #[tokio::test]
    async fn ps_reports_the_context_window_from_infr() {
        let proxy = Proxy::start(
            Router::new()
                .route("/v1/completions", post(|| async { "ok" }))
                .route(
                    "/status",
                    get(|| async { Json(json!({ "models": [{ "ctx_per_slot": 32768 }] })) }),
                ),
        )
        .await;
        proxy.complete().await;

        let ps: serde_json::Value = reqwest::Client::new()
            .get(format!("http://{}/api/ps", proxy.addr))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(ps["models"][0]["context_length"], 32768);

        let models: serde_json::Value = reqwest::Client::new()
            .get(format!("http://{}/v1/models", proxy.addr))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(models["data"][0]["context_length"], 32768);
    }

    #[tokio::test]
    async fn show_reports_the_gguf_metadata() {
        use crate::gguf::test_support::{Value, gguf};

        let source = gguf(&[
            ("general.architecture", Value::Str("qwen3")),
            ("qwen3.context_length", Value::U32(262_144)),
            ("general.file_type", Value::U32(15)),
            ("general.parameter_count", Value::U64(7_615_000_000)),
        ]);
        let proxy = Proxy::start_with_source(Router::new(), &source).await;

        let show: serde_json::Value = reqwest::Client::new()
            .post(format!("http://{}/api/show", proxy.addr))
            .json(&json!({ "model": "test:model" }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        assert_eq!(show["model_info"]["general.architecture"], "qwen3");
        assert_eq!(show["model_info"]["qwen3.context_length"], 262_144);
        assert_eq!(show["details"]["family"], "qwen3");
        assert_eq!(show["details"]["quantization_level"], "Q4_K_M");
        assert_eq!(show["details"]["parameter_size"], "7.6B");
        assert_eq!(show["capabilities"], json!(["completion"]));
    }
}
