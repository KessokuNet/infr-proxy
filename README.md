# infr-proxy

An almost drop-in Ollama API in front of `infr serve` instances. Same model
names, same endpoints for the parts that matter, but the inference process is
infr rather than a vendored llama.cpp.

## The cache is the model list

Ollama clients address models by registry name (`qwen2.5:1.5b`),
while infr names a model after its GGUF file stem. The proxy reads the manifest
tree in its own cache and in any read-only store you point it at, and serves
everything it finds, so a pull is all it takes to make a model runnable:

```sh
ollama run hf.co/mradermacher/Qwen3.5-4B-Unfiltered-GGUF:Q4_K_M
```

`/api/tags` and `/v1/models` list the cached manifests plus any config entries.
A `[[model]]` entry is an override, not the model list: pin an id's unit name,
or add a GGUF that lives outside the cache. Each model gets a unit-safe
instance name derived from the id and a Unix socket at
`$XDG_RUNTIME_DIR/infr/<instance>.sock`.

The cache is laid out like Ollama's:

```
<cache_dir>/manifests/registry.ollama.ai/<namespace>/<name>/<tag>
<cache_dir>/blobs/sha256-<hex>
```

An existing Ollama install already has that layout. Point the proxy at its store
instead of re-pulling everything:

```toml
extra_model_dirs = ["/var/lib/ollama/.ollama/models"]
```

Pulls still land in `cache_dir`, which must be writable; extra stores are only
read. A tag that is already in an extra store is not downloaded again.
`exclude = ["embeddinggemma-2:latest"]` keeps a model out of the list when infr
cannot serve it; it is not a hardware filter like `max_model_size`.

`/api/tags` reports the SHA-256 of the matching Ollama manifest for files in an
Ollama model store. Other local GGUF files report their content SHA-256.

`source` in a `[[model]]` entry is either a path (`/models/x.gguf`,
`./x.gguf`, `thing.gguf`) or a reference (`qwen2.5:1.5b`,
`huihui_ai/...:4b`, `hf.co/Qwen/...:Q4_K_M`). A reference must already be
pulled. Without an entry, any id in the cache resolves on request:

```toml
[[model]]
id = "qwen2.5:1.5b"          # override the derived instance
instance = "qwen2.5-1.5b"

[[model]]
id = "local-model"
source = "/models/x.gguf"    # a model outside the cache
```

Pulls:

```sh
infr-proxy pull qwen2.5:1.5b      # manifest + model blob, digest-checked
infr-proxy model-path qwen2.5:1.5b # ask a running proxy where it lives
```

`model-path` talks to the proxy over HTTP (`GET /registry/model-path?id=...`),
so the path it prints is the one the systemd unit will actually load.

## Registries

References resolve the way Ollama resolves them: the host defaults to
`registry.ollama.ai`, the namespace to `library`, the tag to `latest`, a colon
is a tag only when it follows the last `/`, and part rules are Ollama's
(leading alphanumeric or `_`, `.` everywhere but the namespace, `:` only in the
host for a port). A manifest list (`application/vnd.ollama.manifest.list.v2+json`)
is followed to its first child.

Hugging Face serves the same Docker Registry v2 API at `hf.co`, so
`hf.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M` pulls without a special backend.
Override a host in the config when it needs a scheme, a token, or a mirror:

```toml
[[registry]]
host = "hf.co"
scheme = "https"          # default: the reference's scheme, then https
token_env = "HF_TOKEN"    # sent as a bearer token (gated repos)
mirrors = ["https://mirror.example"]  # tried in order when the primary fails
```

## One slot

Only one model is resident at a time. A request for a different model marks the
active one draining, waits for its in-flight requests to finish, stops the unit,
and only then starts the replacement. `INFR_SERVE_IDLE_SECS` on the unit handles
the idle exit; the proxy restarts a model that exited under it on the next
request.

A proxy restart does not evict the model. The new process lists active `infr@`
units, matches the unit's env file back to a model, and adopts it as the
resident slot; the request path then uses the unit's socket path. A unit that
no longer matches a model (deleted tag, new exclude rule) is stopped at startup
so it cannot hold VRAM invisibly. `RuntimeDirectoryPreserve=yes` on the proxy
unit keeps those env files and sockets across a restart.

A request that waits longer than `queue_timeout_secs` for the drain gets a 503
with `Retry-After`, so the caller can retry elsewhere instead of holding a
generation timeout open.

`max_model_size = "8GiB"` drops larger models from `/v1/models` and
`/api/tags`, so no client ever sees a model the card cannot hold, and a direct
request gets a 400. It is a file-size guard, not a VRAM query: a model under the
cap can still fail to load once the KV cache is added.

## API

Implemented: `HEAD /` (the CLI's connectivity probe), `GET /api/version`,
`GET /api/tags`, `GET /api/ps`, `POST /api/show`, `POST /api/pull` (NDJSON
progress), `POST /api/chat`, `POST /api/generate`, `GET /v1/models`,
`POST /v1/chat/completions` (JSON and SSE passthrough),
`POST /v1/completions` (passthrough), and `GET /registry/model-path`.

`/api/ps` reports the loaded instance's `context_length` (infr's
`ctx_per_slot`, so with `--parallel` it is the per-request window).

`/api/show` and `/api/tags` read the GGUF header, so clients get the real
architecture, trained context, parameter count and quantization instead of the
empty `model_info` an Ollama shim usually leaves behind. The header is parsed
once per file revision and cached; tensors are never touched. `/v1/models`
carries the same `context_length`. For the resident model the reported window is
the live `ctx_per_slot`, not the trained ceiling, because infr clamps context to
VRAM at load; an idle model reports its trained context until it has run once.

A `[[model]]` entry can correct what the header cannot answer:

```toml
[[model]]
id = "qwen2.5:1.5b"
context_length = 32768  # the window the card actually fits
reasoning = true        # template has a thinking switch; Pi budgets for it
vision = false          # only when the backend accepts images
```

`reasoning` and `vision` map to Ollama's `thinking` and `vision` capabilities;
both default to unset, so no capability is claimed without evidence.

The Ollama chat paths translate infr's OpenAI SSE into Ollama's NDJSON, so
`ollama run` streams through the proxy exactly as it would against Ollama.
`/api/chat` also carries tool calling: Ollama's `tool_calls` (object arguments,
no id) and `tool_name` tool results are converted to OpenAI's id/string form on
the way in, and infr's streamed `tool_calls` deltas are reassembled into one
Ollama chunk with parsed arguments on the way out. `/api/generate` takes no
tools, matching Ollama.
`ollama list`, `ps`, `show`, `pull` and `run` were all exercised against a live
infr instance; `ollama rm` (`DELETE /api/delete`) is not wired yet.

## Install

```sh
cargo build --release
install -Dm755 target/release/infr-proxy ~/.local/bin/infr-proxy
install -Dm644 contrib/infr@.service ~/.config/systemd/user/infr@.service
install -Dm644 contrib/infr@.socket ~/.config/systemd/user/infr@.socket
install -Dm644 contrib/infr-proxy.service ~/.config/systemd/user/infr-proxy.service
install -Dm644 infr-proxy.example.toml ~/.config/infr-proxy/config.toml
systemctl --user daemon-reload
systemctl --user enable --now infr-proxy.service
```

The config file is optional: with none, the proxy listens on `127.0.0.1:11435`
and serves the default cache. Keep the example only if you need overrides.

The `infr@.service` template is started on demand; do not enable it. It runs
`~/.local/bin/infr`; edit `ExecStart` if infr lives elsewhere. `infr@.socket`
listens on `$XDG_RUNTIME_DIR/infr/<instance>.sock`, and the service requires it,
so systemd passes the Unix listener as fd 3 and infr serves the HTTP API there.
The proxy writes `$XDG_RUNTIME_DIR/infr/<instance>.env` with `INFR_MODEL` and
`INFR_SOCKET` before starting the service, and the template reads it through
`EnvironmentFile=`; the socket path in that file is what restart adoption uses.

## Metrics and traces

`infr serve` metrics are exported on `0.0.0.0:9100/metrics`. Override the
address with `INFR_PROXY_METRICS_ADDR`; `off` or an empty value disables the
exporter.

| metric | labels | meaning |
| --- | --- | --- |
| `infr_proxy_http_requests_total` | `method`, `route`, `status` | every request, including model listing and health checks |
| `infr_proxy_http_request_duration_seconds` | `method`, `route` | handler duration for every route |
| `infr_proxy_requests_total` | `model`, `endpoint`, `status` | finished inference requests |
| `infr_proxy_request_duration_seconds` | `model`, `endpoint` | arrival to last streamed chunk, including the wait for the model |
| `infr_proxy_queue_duration_seconds` | `model` | time spent waiting for the resident model |
| `infr_proxy_tokens_total` | `model`, `kind` | `prompt`, `cached`, and `completion` tokens |
| `infr_proxy_decode_tokens_per_second` | `model` | completion rate from infr's timings, else between the first and last streamed chunk |
| `infr_proxy_prefill_tokens_per_second` | `model` | prompt tokens prefilled per second, from infr's timings |
| `infr_proxy_active_requests` | `model` | accepted and unfinished, including queued requests |

Token counts come from infr's OpenAI `usage` block. Non-streaming replies
always carry it; a streamed reply carries it only with
`stream_options.include_usage`. The proxy sets that option on its own Ollama
routes and leaves OpenAI passthrough requests untouched, so a streaming OpenAI
client only contributes token counts when it asks for them. Queue time, token
rates, and cache reads only appear where the measurement exists.

Set `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) to
export traces over OTLP/gRPC. Every route gets an `http.request` span with the
method, path, status, and duration. An inference route adds an `inference` child
with `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`,
`infr.stream`, `gen_ai.response.model` (infr's SHA-256 model id),
`gen_ai.response.id`, `gen_ai.response.finish_reasons`, the `gen_ai.usage.*`
token counts, and the `infr.*` timing and payload-size attributes
(`infr.prefill_ms`, `infr.prefill_tokens_per_second`, `infr.decode_ms`,
`infr.ttft_ms`). The queue
wait and the upstream call are separate children named `model.queue` and
`infr.upstream`. Without the variable no trace exporter runs.

Neither endpoint identifies the caller: requests arrive from localhost. The
metrics port has no authentication either, so keep it on a trusted network.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
