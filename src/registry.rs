use std::path::{Path, PathBuf};
use std::time::Duration;

use eyre::{Context, Result, bail, ensure};
use futures_util::StreamExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::config::RegistryEntry;

const DEFAULT_HOST: &str = "registry.ollama.ai";
const DEFAULT_NAMESPACE: &str = "library";
const DEFAULT_TAG: &str = "latest";
const MANIFEST_ACCEPT: &str = "application/vnd.docker.distribution.manifest.v2+json";
const MANIFEST_LIST: &str = "application/vnd.ollama.manifest.list.v2+json";
const MODEL_LAYER: &str = "application/vnd.ollama.image.model";
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(300);

/// An Ollama registry reference: `[scheme://][host/][namespace/]name[:tag]`.
///
/// Part rules follow Ollama's `ParseName`: a leading alphanumeric or `_`, then
/// alphanumerics plus `_` and `-`; `.` everywhere except the namespace; `:`
/// only in the host (a port).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    pub scheme: Option<String>,
    pub host: String,
    pub namespace: String,
    pub name: String,
    pub tag: String,
}

impl Reference {
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();
        ensure!(!input.is_empty(), "empty model reference");
        ensure!(
            !input.contains('@'),
            "digest references are not supported: {input}"
        );

        let (scheme, rest) = if let Some(rest) = input.strip_prefix("http://") {
            (Some("http".to_string()), rest)
        } else if let Some(rest) = input.strip_prefix("https://") {
            (Some("https".to_string()), rest)
        } else {
            (None, input)
        };

        let last_slash = rest.rfind('/');
        let last_colon = rest.rfind(':');
        let (path, tag) = match last_colon {
            Some(colon) if last_slash.is_none_or(|slash| colon > slash) => {
                (&rest[..colon], &rest[colon + 1..])
            }
            _ => (rest, DEFAULT_TAG),
        };
        ensure!(!tag.is_empty(), "empty tag in {input}");

        let parts: Vec<&str> = path.split('/').collect();
        let (host, namespace, name) = match parts.as_slice() {
            [name] => (DEFAULT_HOST, DEFAULT_NAMESPACE, *name),
            [namespace, name] => (DEFAULT_HOST, *namespace, *name),
            [host, namespace, name] => (*host, *namespace, *name),
            _ => bail!("too many path segments in {input}"),
        };

        ensure!(valid_part(Part::Host, host), "invalid host in {input}");
        ensure!(
            valid_part(Part::Namespace, namespace),
            "invalid namespace in {input}"
        );
        ensure!(valid_part(Part::Model, name), "invalid model in {input}");
        ensure!(valid_part(Part::Tag, tag), "invalid tag in {input}");

        Ok(Self {
            scheme,
            host: host.to_string(),
            namespace: namespace.to_string(),
            name: name.to_string(),
            tag: tag.to_string(),
        })
    }

    /// The short id Ollama shows: the default host and `library` namespace are omitted.
    pub fn display_id(&self) -> String {
        match (self.host.as_str(), self.namespace.as_str()) {
            (DEFAULT_HOST, DEFAULT_NAMESPACE) => format!("{}:{}", self.name, self.tag),
            (DEFAULT_HOST, _) => format!("{}/{}:{}", self.namespace, self.name, self.tag),
            _ => format!(
                "{}/{}/{}:{}",
                self.host, self.namespace, self.name, self.tag
            ),
        }
    }
}

impl std::fmt::Display for Reference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}/{}:{}",
            self.host, self.namespace, self.name, self.tag
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    Host,
    Namespace,
    Model,
    Tag,
}

fn valid_part(part: Part, value: &str) -> bool {
    let max = match part {
        Part::Host => 350,
        _ => 80,
    };
    if value.is_empty() || value.len() > max {
        return false;
    }
    for (index, byte) in value.bytes().enumerate() {
        if index == 0 {
            if !byte.is_ascii_alphanumeric() && byte != b'_' {
                return false;
            }
            continue;
        }
        match byte {
            b'_' | b'-' => {}
            b'.' if part != Part::Namespace => {}
            b':' if part == Part::Host => {}
            _ if byte.is_ascii_alphanumeric() => {}
            _ => return false,
        }
    }
    true
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    #[serde(default)]
    pub layers: Vec<Layer>,
    #[serde(default)]
    pub manifests: Vec<ManifestRef>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Layer {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
}

#[derive(Debug, Deserialize)]
pub struct ManifestRef {
    pub digest: String,
}

impl Manifest {
    fn model_layer(&self) -> Result<&Layer> {
        ensure!(
            self.media_type == MANIFEST_ACCEPT,
            "unsupported manifest type {}",
            self.media_type
        );
        self.layers
            .iter()
            .find(|layer| layer.media_type == MODEL_LAYER)
            .ok_or_else(|| eyre::eyre!("manifest has no {} layer", MODEL_LAYER))
    }
}

/// Content-addressed model cache, laid out like Ollama's:
/// `manifests/<host>/<namespace>/<name>/<tag>` and `blobs/sha256-<hex>`.
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn manifest_path(&self, reference: &Reference) -> PathBuf {
        self.root
            .join("manifests")
            .join(&reference.host)
            .join(&reference.namespace)
            .join(&reference.name)
            .join(&reference.tag)
    }

    fn blob_path(&self, digest: &str) -> Result<PathBuf> {
        let hex = digest
            .strip_prefix("sha256:")
            .ok_or_else(|| eyre::eyre!("unsupported digest {digest}"))?;
        ensure!(
            hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()),
            "malformed sha256 digest {digest}"
        );
        Ok(self.root.join("blobs").join(format!("sha256-{hex}")))
    }

    fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    pub async fn read_manifest(
        &self,
        reference: &Reference,
    ) -> Result<Option<(Manifest, Vec<u8>)>> {
        let path = self.manifest_path(reference);
        let raw = match tokio::fs::read(&path).await {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err).wrap_err_with(|| format!("read {}", path.display())),
        };
        let manifest =
            serde_json::from_slice(&raw).wrap_err_with(|| format!("parse {}", path.display()))?;
        Ok(Some((manifest, raw)))
    }

    pub async fn model_path(&self, reference: &Reference) -> Result<(PathBuf, String, u64)> {
        let (manifest, raw) = self
            .read_manifest(reference)
            .await?
            .ok_or_else(|| eyre::eyre!("{reference} is not cached"))?;
        let layer = manifest.model_layer()?;
        let path = self.blob_path(&layer.digest)?;
        let meta = tokio::fs::metadata(&path)
            .await
            .wrap_err_with(|| format!("stat {}", path.display()))?;
        ensure!(
            meta.len() == layer.size,
            "cached blob for {reference} has the wrong size; pull it again"
        );
        Ok((path, format!("sha256:{}", sha256_hex(&raw)), meta.len()))
    }

    /// Every manifest in the cache, in display-id order.
    pub async fn manifests(&self) -> Result<Vec<Reference>> {
        let root = self.root.join("manifests");
        let mut references = Vec::new();
        for host in child_names(&root, true).await? {
            let host_dir = root.join(&host);
            for namespace in child_names(&host_dir, true).await? {
                let namespace_dir = host_dir.join(&namespace);
                for name in child_names(&namespace_dir, true).await? {
                    let name_dir = namespace_dir.join(&name);
                    for tag in child_names(&name_dir, false).await? {
                        let input = format!("{host}/{namespace}/{name}:{tag}");
                        if let Ok(reference) = Reference::parse(&input) {
                            references.push(reference);
                        }
                    }
                }
            }
        }
        references.sort_by_key(Reference::display_id);
        Ok(references)
    }

    pub async fn write_manifest(&self, reference: &Reference, raw: &[u8]) -> Result<()> {
        let path = self.manifest_path(reference);
        let dir = path
            .parent()
            .ok_or_else(|| eyre::eyre!("manifest path has no parent"))?;
        tokio::fs::create_dir_all(dir)
            .await
            .wrap_err_with(|| format!("create {}", dir.display()))?;
        let tmp = path.with_extension("part");
        tokio::fs::write(&tmp, raw)
            .await
            .wrap_err_with(|| format!("write {}", tmp.display()))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .wrap_err_with(|| format!("rename {}", path.display()))?;
        Ok(())
    }
}

pub struct Resolved {
    pub path: String,
    pub digest: String,
    pub size: u64,
}

async fn child_names(path: &Path, dirs: bool) -> Result<Vec<String>> {
    let mut names = Vec::new();
    let mut entries = match tokio::fs::read_dir(path).await {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(names),
        Err(err) => return Err(err).wrap_err_with(|| format!("read {}", path.display())),
    };
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".part") || entry.file_type().await?.is_dir() != dirs {
            continue;
        }
        names.push(name);
    }
    Ok(names)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    digest_hex(hasher)
}

fn digest_hex(hasher: Sha256) -> String {
    let mut hex = String::with_capacity(64);
    for byte in hasher.finalize() {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

async fn sha256_file(path: &Path) -> Result<String> {
    let canonical = tokio::fs::canonicalize(path)
        .await
        .wrap_err_with(|| format!("resolve {} for hashing", path.display()))?;
    if canonical
        .parent()
        .is_some_and(|dir| dir.file_name().is_some_and(|name| name == "blobs"))
        && let Some(hash) = canonical
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Ok(format!("sha256:{hash}"));
    }
    let mut file = tokio::fs::File::open(&canonical)
        .await
        .wrap_err_with(|| format!("open {} for hashing", path.display()))?;
    let mut buffer = [0; 64 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let count = file
            .read(&mut buffer)
            .await
            .wrap_err_with(|| format!("hash {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("sha256:{}", digest_hex(hasher)))
}

async fn ollama_manifest_digest(model_id: &str, source: &Path) -> Result<Option<String>> {
    let Some(blob_dir) = source
        .parent()
        .filter(|dir| dir.file_name().is_some_and(|name| name == "blobs"))
    else {
        return Ok(None);
    };
    let Some(models_dir) = blob_dir.parent() else {
        return Ok(None);
    };
    let Some(blob_hash) = source
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("sha256-"))
        .filter(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
    else {
        return Ok(None);
    };
    let Ok(reference) = Reference::parse(model_id) else {
        return Ok(None);
    };
    let manifest_path = models_dir
        .join("manifests")
        .join(reference.host)
        .join(reference.namespace)
        .join(reference.name)
        .join(reference.tag);
    let raw = match tokio::fs::read(&manifest_path).await {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).wrap_err_with(|| format!("read {}", manifest_path.display())),
    };
    let manifest: Manifest = serde_json::from_slice(&raw)
        .wrap_err_with(|| format!("parse {}", manifest_path.display()))?;
    let layer = manifest.model_layer()?;
    let metadata = tokio::fs::metadata(source)
        .await
        .wrap_err_with(|| format!("stat {}", source.display()))?;
    ensure!(
        layer.digest == format!("sha256:{blob_hash}") && layer.size == metadata.len(),
        "Ollama manifest {} does not describe {}",
        manifest_path.display(),
        source.display()
    );
    Ok(Some(format!("sha256:{}", sha256_hex(&raw))))
}

pub fn is_path(source: &str) -> bool {
    source.starts_with('/')
        || source.starts_with("./")
        || source.starts_with("../")
        || source.starts_with('~')
        || source.ends_with(".gguf")
}

#[derive(Clone)]
struct Endpoint {
    base: String,
    token: Option<String>,
}

impl Endpoint {
    fn manifest_url(&self, reference: &Reference, digest: Option<&str>) -> String {
        let reference_part = digest.unwrap_or(&reference.tag);
        format!(
            "{}/v2/{}/{}/manifests/{}",
            self.base, reference.namespace, reference.name, reference_part
        )
    }

    fn blob_url(&self, reference: &Reference, digest: &str) -> String {
        format!(
            "{}/v2/{}/{}/blobs/{}",
            self.base, reference.namespace, reference.name, digest
        )
    }
}

pub struct Registry {
    http: reqwest::Client,
    cache: Cache,
    stores: Vec<Cache>,
    entries: Vec<RegistryEntry>,
}

impl Registry {
    pub fn new(
        cache_root: PathBuf,
        store_roots: Vec<PathBuf>,
        entries: Vec<RegistryEntry>,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("infr-proxy/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            http,
            cache: Cache::new(cache_root),
            stores: store_roots.into_iter().map(Cache::new).collect(),
            entries,
        })
    }

    fn endpoints(&self, reference: &Reference) -> Vec<Endpoint> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.host == reference.host);
        let scheme = entry
            .and_then(|entry| entry.scheme.as_deref())
            .or(reference.scheme.as_deref())
            .unwrap_or("https");
        let token = entry
            .and_then(|entry| entry.token_env.as_deref())
            .and_then(|name| std::env::var(name).ok());
        let mut bases = vec![format!("{scheme}://{}", reference.host)];
        if let Some(entry) = entry {
            bases.extend(entry.mirrors.iter().cloned());
        }
        bases
            .into_iter()
            .map(|base| Endpoint {
                base,
                token: token.clone(),
            })
            .collect()
    }

    /// Resolve an id from the cache or a read-only store, with its Ollama digest.
    pub async fn cached(&self, reference: &Reference) -> Result<Resolved> {
        let mut last = None;
        for cache in std::iter::once(&self.cache).chain(&self.stores) {
            match cache.model_path(reference).await {
                Ok((path, digest, size)) => {
                    return Ok(Resolved {
                        path: path.to_string_lossy().into_owned(),
                        digest,
                        size,
                    });
                }
                Err(err) => last = Some(err),
            }
        }
        Err(last.unwrap_or_else(|| eyre::eyre!("{reference} is not cached")))
    }

    /// Every manifest in the writable cache and the read-only stores.
    pub async fn manifests(&self) -> Result<Vec<Reference>> {
        let mut references = self.cache.manifests().await?;
        for store in &self.stores {
            match store.manifests().await {
                Ok(more) => references.extend(more),
                Err(err) => tracing::warn!(
                    store = %store.root().display(),
                    error = %err,
                    "could not scan a model store"
                ),
            }
        }
        references.sort_by_key(Reference::display_id);
        references.dedup_by_key(|reference| reference.display_id());
        Ok(references)
    }

    /// Resolve a config source and its Ollama-compatible model digest.
    pub async fn resolve_source(&self, model_id: &str, source: &str) -> Result<Resolved> {
        if is_path(source) {
            let path = PathBuf::from(source);
            let meta = tokio::fs::metadata(&path)
                .await
                .wrap_err_with(|| format!("stat {}", path.display()))?;
            let digest = match ollama_manifest_digest(model_id, &path).await? {
                Some(digest) => digest,
                None => sha256_file(&path).await?,
            };
            return Ok(Resolved {
                path: source.to_string(),
                digest,
                size: meta.len(),
            });
        }
        let reference = Reference::parse(source)?;
        self.cached(&reference).await.map_err(|err| {
            eyre::eyre!("{source} is not usable: {err}; run `infr-proxy pull {source}`")
        })
    }

    pub async fn pull(&self, reference: &Reference) -> Result<PathBuf> {
        self.pull_with(reference, |_, _, _| {}).await
    }

    /// Pull with a progress callback: `(completed, total, digest)`.
    pub async fn pull_with<F: FnMut(u64, u64, &str)>(
        &self,
        reference: &Reference,
        mut progress: F,
    ) -> Result<PathBuf> {
        tracing::info!(reference = %reference, "pulling manifest");
        let (manifest, raw, endpoint) = self.fetch_manifest(reference, None).await?;
        let (manifest, raw) = if manifest.media_type == MANIFEST_LIST {
            let child = manifest
                .manifests
                .first()
                .ok_or_else(|| eyre::eyre!("{reference} has an empty manifest list"))?;
            let url = endpoint.manifest_url(reference, Some(&child.digest));
            self.get_manifest(&url, endpoint.token.as_deref())
                .await
                .wrap_err_with(|| format!("fetch child manifest for {reference}"))?
        } else {
            (manifest, raw)
        };

        let layer = manifest.model_layer()?.clone();
        let blob = self.cache.blob_path(&layer.digest)?;
        if let Ok(meta) = tokio::fs::metadata(&blob).await
            && meta.len() == layer.size
        {
            tracing::info!(reference = %reference, path = %blob.display(), "already cached");
            self.cache.write_manifest(reference, &raw).await?;
            return Ok(blob);
        }
        for store in &self.stores {
            if let Ok((path, _, size)) = store.model_path(reference).await
                && size == layer.size
                && store
                    .blob_path(&layer.digest)
                    .is_ok_and(|blob| blob == path)
            {
                tracing::info!(
                    reference = %reference,
                    path = %path.display(),
                    "already in a read-only store"
                );
                return Ok(path);
            }
        }

        self.download(reference, &layer, &blob, &endpoint, &mut progress)
            .await?;
        self.cache.write_manifest(reference, &raw).await?;
        tracing::info!(reference = %reference, path = %blob.display(), "cached");
        Ok(blob)
    }

    async fn fetch_manifest(
        &self,
        reference: &Reference,
        digest: Option<&str>,
    ) -> Result<(Manifest, Vec<u8>, Endpoint)> {
        let mut last = None;
        for endpoint in self.endpoints(reference) {
            let url = endpoint.manifest_url(reference, digest);
            match self.get_manifest(&url, endpoint.token.as_deref()).await {
                Ok((manifest, raw)) => return Ok((manifest, raw, endpoint)),
                Err(err) => last = Some(err),
            }
        }
        Err(last.unwrap_or_else(|| eyre::eyre!("no registry endpoint for {reference}")))
    }

    async fn get_manifest(&self, url: &str, token: Option<&str>) -> Result<(Manifest, Vec<u8>)> {
        let mut request = self
            .http
            .get(url)
            .header(
                reqwest::header::ACCEPT,
                format!("{MANIFEST_ACCEPT}, {MANIFEST_LIST}"),
            )
            .timeout(Duration::from_secs(60));
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .wrap_err_with(|| format!("fetch {url}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!("{url} answered {status}: {}", body.trim());
        }
        let raw = response
            .bytes()
            .await
            .wrap_err_with(|| format!("read {url}"))?;
        let manifest: Manifest =
            serde_json::from_slice(&raw).wrap_err_with(|| format!("parse manifest {url}"))?;
        Ok((manifest, raw.to_vec()))
    }

    async fn download<F: FnMut(u64, u64, &str)>(
        &self,
        reference: &Reference,
        layer: &Layer,
        blob: &Path,
        endpoint: &Endpoint,
        progress: &mut F,
    ) -> Result<()> {
        tokio::fs::create_dir_all(self.cache.blobs_dir())
            .await
            .wrap_err("create blob directory")?;
        let tmp = self.cache.blobs_dir().join(format!(
            ".{}.part",
            blob.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("download")
        ));

        let mut request = self
            .http
            .get(endpoint.blob_url(reference, &layer.digest))
            .timeout(DOWNLOAD_TIMEOUT);
        if let Some(token) = &endpoint.token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .wrap_err_with(|| format!("download {} for {reference}", layer.digest))?;
        let status = response.status();
        if !status.is_success() {
            bail!(
                "registry answered {status} for {} of {reference}",
                layer.digest
            );
        }

        let mut file = tokio::fs::File::create(&tmp)
            .await
            .wrap_err_with(|| format!("create {}", tmp.display()))?;
        let mut hasher = Sha256::new();
        let mut completed = 0u64;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.wrap_err("read blob chunk")?;
            hasher.update(&chunk);
            completed += chunk.len() as u64;
            ensure!(
                completed <= layer.size,
                "blob for {reference} is larger than the manifest says"
            );
            file.write_all(&chunk).await.wrap_err("write blob chunk")?;
            progress(completed, layer.size, &layer.digest);
        }
        file.flush().await.wrap_err("flush blob")?;
        drop(file);

        ensure!(
            completed == layer.size,
            "blob for {reference} is {completed} bytes, manifest says {}",
            layer.size
        );
        let mut digest = String::with_capacity(64);
        for byte in hasher.finalize() {
            use std::fmt::Write as _;
            let _ = write!(digest, "{byte:02x}");
        }
        let expected = layer
            .digest
            .strip_prefix("sha256:")
            .ok_or_else(|| eyre::eyre!("unsupported digest {}", layer.digest))?;
        ensure!(digest == expected, "digest mismatch for {reference}");

        tokio::fs::rename(&tmp, blob)
            .await
            .wrap_err_with(|| format!("rename {}", blob.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn write_model(cache: &Cache, id: &str, bytes: &[u8]) {
        let reference = Reference::parse(id).unwrap();
        let digest = sha256_hex(bytes);
        let blob = cache.blob_path(&format!("sha256:{digest}")).unwrap();
        tokio::fs::create_dir_all(blob.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&blob, bytes).await.unwrap();
        let raw = format!(
            "{{\"mediaType\":\"{MANIFEST_ACCEPT}\",\"layers\":[{{\"mediaType\":\"{MODEL_LAYER}\",\"digest\":\"sha256:{digest}\",\"size\":{}}}]}}",
            bytes.len()
        );
        cache
            .write_manifest(&reference, raw.as_bytes())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn local_ollama_blob_uses_the_sha256_of_its_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path().join("models");
        let blob = b"test model bytes";
        let blob_hash = sha256_hex(blob);
        let source = models_dir.join("blobs").join(format!("sha256-{blob_hash}"));
        tokio::fs::create_dir_all(source.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&source, blob).await.unwrap();

        let raw = format!(
            "{{\"mediaType\":\"{MANIFEST_ACCEPT}\",\"layers\":[{{\"mediaType\":\"{MODEL_LAYER}\",\"digest\":\"sha256:{blob_hash}\",\"size\":{}}}]}}",
            blob.len()
        );
        let reference = Reference::parse("qwen2.5:0.5b").unwrap();
        let manifest_path = models_dir
            .join("manifests")
            .join(reference.host)
            .join(reference.namespace)
            .join(reference.name)
            .join(reference.tag);
        tokio::fs::create_dir_all(manifest_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&manifest_path, &raw).await.unwrap();

        let digest = ollama_manifest_digest("qwen2.5:0.5b", &source)
            .await
            .unwrap();
        assert_eq!(
            digest,
            Some(format!("sha256:{}", sha256_hex(raw.as_bytes())))
        );
    }

    #[tokio::test]
    async fn standalone_path_uses_the_content_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model.gguf");
        tokio::fs::write(&source, b"standalone model bytes")
            .await
            .unwrap();
        let registry = Registry::new(dir.path().join("cache"), Vec::new(), Vec::new()).unwrap();

        let resolved = registry
            .resolve_source("local-model", source.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            resolved.digest,
            format!("sha256:{}", sha256_hex(b"standalone model bytes"))
        );
    }

    #[test]
    fn parses_ollama_references_with_defaults() {
        let short = Reference::parse("qwen2.5:1.5b").unwrap();
        assert_eq!(short.host, DEFAULT_HOST);
        assert_eq!(short.namespace, DEFAULT_NAMESPACE);
        assert_eq!(short.name, "qwen2.5");
        assert_eq!(short.tag, "1.5b");
        assert_eq!(short.display_id(), "qwen2.5:1.5b");

        let namespaced = Reference::parse("huihui_ai/qwen3.5-abliterated:4b").unwrap();
        assert_eq!(namespaced.namespace, "huihui_ai");
        assert_eq!(namespaced.name, "qwen3.5-abliterated");
        assert_eq!(namespaced.display_id(), "huihui_ai/qwen3.5-abliterated:4b");

        let hosted = Reference::parse("hf.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M").unwrap();
        assert_eq!(hosted.host, "hf.co");
        assert_eq!(hosted.namespace, "Qwen");
        assert_eq!(hosted.tag, "Q4_K_M");
        assert_eq!(
            hosted.display_id(),
            "hf.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"
        );
        assert_eq!(
            Reference::parse("registry.ollama.ai/library/qwen2.5:1.5b")
                .unwrap()
                .display_id(),
            "qwen2.5:1.5b"
        );
    }

    #[tokio::test]
    async fn scans_the_manifest_tree_into_display_ids() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path().to_path_buf());
        write_model(&cache, "qwen2.5:1.5b", b"model").await;
        write_model(&cache, "hf.co/ns/model:Q4_K_M", b"model").await;

        let ids: Vec<String> = cache
            .manifests()
            .await
            .unwrap()
            .iter()
            .map(Reference::display_id)
            .collect();
        assert_eq!(ids, ["hf.co/ns/model:Q4_K_M", "qwen2.5:1.5b"]);
    }

    #[tokio::test]
    async fn extra_stores_resolve_models_and_the_cache_wins() {
        let dir = tempfile::tempdir().unwrap();
        let writable = Cache::new(dir.path().join("cache"));
        let store = Cache::new(dir.path().join("store"));
        write_model(&store, "qwen2.5:1.5b", b"from the store").await;

        let registry = Registry::new(
            dir.path().join("cache"),
            vec![dir.path().join("store")],
            Vec::new(),
        )
        .unwrap();
        let reference = Reference::parse("qwen2.5:1.5b").unwrap();
        let resolved = registry.cached(&reference).await.unwrap();
        assert_eq!(resolved.size, b"from the store".len() as u64);
        assert_eq!(registry.manifests().await.unwrap().len(), 1);

        write_model(&writable, "qwen2.5:1.5b", b"from the cache").await;
        let resolved = registry.cached(&reference).await.unwrap();
        assert_eq!(resolved.size, b"from the cache".len() as u64);
        assert_eq!(registry.manifests().await.unwrap().len(), 1);
    }

    #[test]
    fn keeps_a_registry_port_and_an_explicit_scheme() {
        let port = Reference::parse("localhost:5000/ns/model:tag").unwrap();
        assert_eq!(port.host, "localhost:5000");
        assert_eq!(port.tag, "tag");
        assert_eq!(port.scheme, None);

        let http = Reference::parse("http://localhost:5000/ns/model").unwrap();
        assert_eq!(http.scheme.as_deref(), Some("http"));
        assert_eq!(http.host, "localhost:5000");
        assert_eq!(http.tag, DEFAULT_TAG);
    }

    #[test]
    fn matches_ollama_part_rules() {
        assert!(Reference::parse("Qwen/Model:Q4_K_M").is_ok());
        assert!(Reference::parse("ns.with.dot/model").is_err());
        assert!(Reference::parse("-bad/model").is_err());
        assert!(Reference::parse("qwen2.5@sha256:abc").is_err());
        assert!(Reference::parse("qwen2.5").unwrap().tag == "latest");
    }

    #[test]
    fn picks_the_model_layer_from_a_manifest() {
        let raw = r#"{
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [
                {"mediaType": "application/vnd.ollama.image.system", "digest": "sha256:aa", "size": 1},
                {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:bb", "size": 2}
            ]
        }"#;
        let manifest: Manifest = serde_json::from_str(raw).unwrap();
        assert_eq!(manifest.model_layer().unwrap().digest, "sha256:bb");
    }

    #[test]
    fn path_sources_are_not_references() {
        assert!(is_path("/models/x.gguf"));
        assert!(is_path("./x.gguf"));
        assert!(is_path("qwen2.5-1.5b.gguf"));
        assert!(!is_path("qwen2.5:1.5b"));
    }
}
