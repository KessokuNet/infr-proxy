use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use eyre::{Context, Result, bail};
use serde::Deserialize;

use bytesize::ByteSize;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    /// Seconds a request waits for the resident model to drain before 503.
    #[serde(default = "default_queue_timeout")]
    pub queue_timeout_secs: u64,
    /// Seconds a started unit has to answer before the request fails.
    #[serde(default = "default_load_timeout")]
    pub load_timeout_secs: u64,
    /// Milliseconds between readiness probes.
    #[serde(default = "default_poll_interval")]
    pub poll_interval_ms: u64,
    /// Model cache root; defaults to `$XDG_DATA_HOME/infr-proxy/models`.
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    /// Read-only model stores scanned in addition to the cache, e.g. an Ollama store.
    #[serde(default)]
    pub extra_model_dirs: Vec<PathBuf>,
    /// Models larger than this are not advertised (a VRAM fit check).
    #[serde(default)]
    pub max_model_size: Option<ByteSize>,
    /// Per-host registry overrides: scheme, mirrors, auth token env.
    #[serde(default, rename = "registry")]
    pub registries: Vec<RegistryEntry>,
    /// Overrides and local files. Cached models are served without an entry.
    #[serde(default, rename = "model")]
    pub models: Vec<ModelEntry>,
    /// Ids never served, even when a store or an entry has them.
    #[serde(default)]
    pub exclude: Vec<String>,
}

/// An entry overrides a cached id or adds a GGUF that is not in the cache.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    /// Network-facing id, e.g. the Ollama registry name a client asks for.
    pub id: String,
    /// systemd instance for `infr@<instance>.service`; derived from the id when absent.
    #[serde(default)]
    pub instance: Option<String>,
    /// GGUF path or registry reference; absent means resolve the id from the cache.
    #[serde(default)]
    pub source: Option<String>,
    /// Runtime window override; the GGUF's trained context is a ceiling, not what the card fits.
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Advertise a thinking model, so Pi budgets the reply for reasoning tokens.
    #[serde(default)]
    pub reasoning: Option<bool>,
    /// Advertise image input; only when the backend actually accepts images.
    #[serde(default)]
    pub vision: Option<bool>,
}

/// A model that can be started: the cache or a config source resolved to a file.
#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    pub instance: String,
    pub source: String,
    pub socket: PathBuf,
    pub digest: String,
    pub context_length: Option<u32>,
    pub reasoning: Option<bool>,
    pub vision: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegistryEntry {
    pub host: String,
    /// `http` for a local registry; defaults to the reference's scheme, then https.
    #[serde(default)]
    pub scheme: Option<String>,
    /// Additional bases tried in order when the primary does not answer.
    #[serde(default)]
    pub mirrors: Vec<String>,
    /// Environment variable holding a bearer token for this host.
    #[serde(default)]
    pub token_env: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            queue_timeout_secs: default_queue_timeout(),
            load_timeout_secs: default_load_timeout(),
            poll_interval_ms: default_poll_interval(),
            cache_dir: None,
            extra_model_dirs: Vec::new(),
            max_model_size: None,
            registries: Vec::new(),
            models: Vec::new(),
            exclude: Vec::new(),
        }
    }
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = match path {
            Some(path) => path.to_path_buf(),
            None => default_path()?,
        };
        Self::read(&path)
    }

    /// Load a config when one exists; used by subcommands that can run without one.
    pub fn load_optional(path: Option<&Path>) -> Result<Option<Self>> {
        let path = match path {
            Some(path) => Some(path.to_path_buf()),
            None => default_path_opt(),
        };
        match path {
            Some(path) => Self::read(&path).map(Some),
            None => Ok(None),
        }
    }

    fn read(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).wrap_err_with(|| format!("read {}", path.display()))?;
        let config: Self =
            toml::from_str(&raw).wrap_err_with(|| format!("parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn cache_root(&self) -> Result<PathBuf> {
        match &self.cache_dir {
            Some(dir) => Ok(dir.clone()),
            None => default_cache_dir(),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.max_model_size.is_some_and(|size| size.as_u64() == 0) {
            bail!("max_model_size must be greater than zero");
        }
        let mut ids = HashSet::new();
        let mut instances = HashSet::new();
        let mut hosts = HashSet::new();
        for registry in &self.registries {
            if registry.host.is_empty() {
                bail!("a [[registry]] entry has an empty host");
            }
            if let Some(scheme) = &registry.scheme
                && scheme != "http"
                && scheme != "https"
            {
                bail!("registry {} has an unknown scheme {scheme}", registry.host);
            }
            for mirror in &registry.mirrors {
                if !(mirror.starts_with("http://") || mirror.starts_with("https://")) {
                    bail!("registry mirror {mirror} must be an http(s) URL");
                }
            }
            if !hosts.insert(registry.host.as_str()) {
                bail!("duplicate [[registry]] host {}", registry.host);
            }
        }
        for model in &self.models {
            if model.id.is_empty() || model.id.len() > 128 {
                bail!("model id {:?} must be 1-128 characters", model.id);
            }
            if !model.id.chars().all(|c| {
                c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/' | '@')
            }) {
                bail!("model id {:?} has characters the network rejects", model.id);
            }
            if let Some(instance) = &model.instance
                && (instance.is_empty()
                    || instance.len() > 64
                    || !instance
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':')))
            {
                bail!("instance {instance:?} must be a unit-safe name of at most 64 characters");
            }
            if model
                .source
                .as_ref()
                .is_some_and(|source| source.is_empty())
            {
                bail!("model {:?} has an empty source", model.id);
            }
            if model.context_length == Some(0) {
                bail!("model {:?} has a zero context_length", model.id);
            }
            if !ids.insert(model.id.as_str()) {
                bail!("duplicate model id {:?}", model.id);
            }
            if let Some(instance) = &model.instance
                && !instances.insert(instance.as_str())
            {
                bail!("duplicate instance {instance:?}");
            }
        }
        Ok(())
    }
}

fn default_path() -> Result<PathBuf> {
    default_path_opt().ok_or_else(|| {
        eyre::eyre!(
            "no config found (tried ./infr-proxy.toml and $XDG_CONFIG_HOME/infr-proxy/config.toml)"
        )
    })
}

fn default_path_opt() -> Option<PathBuf> {
    let local = PathBuf::from("infr-proxy.toml");
    if local.exists() {
        return Some(local);
    }
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    let path = base.join("infr-proxy/config.toml");
    path.exists().then_some(path)
}

pub fn default_cache_dir() -> Result<PathBuf> {
    if let Some(data) = std::env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(data).join("infr-proxy/models"));
    }
    let home =
        std::env::var_os("HOME").ok_or_else(|| eyre::eyre!("HOME is not set; pass --cache-dir"))?;
    Ok(PathBuf::from(home).join(".local/share/infr-proxy/models"))
}

fn default_listen() -> SocketAddr {
    "127.0.0.1:11435".parse().expect("static listen address")
}

fn default_queue_timeout() -> u64 {
    240
}

fn default_load_timeout() -> u64 {
    120
}

fn default_poll_interval() -> u64 {
    250
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(toml: &str) -> Result<Config> {
        let config: Config = toml::from_str(toml)?;
        config.validate()?;
        Ok(config)
    }

    #[test]
    fn defaults_fill_in_and_an_override_parses() {
        let config = config(
            r#"
            max_model_size = "8GiB"

            [[model]]
            id = "qwen2.5:1.5b"
            "#,
        )
        .unwrap();
        assert_eq!(config.listen, "127.0.0.1:11435".parse().unwrap());
        assert_eq!(config.queue_timeout_secs, 240);
        assert_eq!(config.models[0].id, "qwen2.5:1.5b");
        assert_eq!(config.models[0].instance, None);
        assert_eq!(config.models[0].source, None);
        assert_eq!(
            config.max_model_size.map(|size| size.as_u64()),
            Some(8 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn a_config_without_models_is_valid() {
        let config = config("max_model_size = \"8GiB\"\n").unwrap();
        assert!(config.models.is_empty());
    }

    #[test]
    fn rejects_network_unsafe_ids_and_duplicates() {
        let unsafe_id = r#"
            [[model]]
            id = "has space"
            "#;
        assert!(config(unsafe_id).is_err());

        let duplicate_id = r#"
            [[model]]
            id = "a"
            [[model]]
            id = "a"
        "#;
        assert!(config(duplicate_id).is_err());
    }
}
