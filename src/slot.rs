use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use eyre::Result;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, Notify};

use crate::config::{Config, Model};
use crate::host::ModelHost;
use crate::registry::{Reference, Registry};
use crate::systemd::Systemd;

#[derive(Debug)]
pub enum SlotError {
    UnknownModel,
    TooLarge { size: u64, max: u64 },
    Busy,
    Backend(String),
}

impl std::fmt::Display for SlotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownModel => f.write_str("unknown model"),
            Self::TooLarge { size, max } => {
                write!(
                    f,
                    "model is {size} bytes, larger than the {max}-byte max_model_size"
                )
            }
            Self::Busy => f.write_str("the resident model is still draining"),
            Self::Backend(err) => f.write_str(err),
        }
    }
}

impl std::error::Error for SlotError {}

#[derive(Clone, Debug)]
enum Phase {
    Idle,
    Running(Model),
    Draining(Model),
    Starting,
}

enum Action {
    Leased(Model),
    Switch(Option<Model>),
    Wait,
}

#[derive(Debug, Clone, Default)]
struct ModelOverride {
    instance: Option<String>,
}

/// One resident model at a time; ids resolve through config, then the cache.
/// A switch drains the active model and stops it before starting the next.
pub struct Slot {
    models: HashMap<String, Model>,
    aliases: HashMap<String, String>,
    overrides: HashMap<String, ModelOverride>,
    excluded: HashSet<String>,
    host: Arc<dyn ModelHost>,
    registry: Arc<Registry>,
    socket_dir: PathBuf,
    max_model_size: Option<u64>,
    phase: Mutex<Phase>,
    inflight: AtomicUsize,
    notify: Notify,
    switch: Mutex<()>,
    queue_timeout: Duration,
}

impl Slot {
    pub async fn new(config: Config) -> Result<Self> {
        let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .ok_or_else(|| eyre::eyre!("XDG_RUNTIME_DIR is not set"))?;
        let host = Arc::new(
            Systemd::connect(
                runtime_dir.clone(),
                Duration::from_secs(config.load_timeout_secs),
                Duration::from_millis(config.poll_interval_ms),
            )
            .await?,
        );
        Self::with_host(config, host, runtime_dir).await
    }

    pub async fn with_host(
        config: Config,
        host: Arc<dyn ModelHost>,
        runtime_dir: PathBuf,
    ) -> Result<Self> {
        let registry = Arc::new(Registry::new(
            config.cache_root()?,
            config.extra_model_dirs.clone(),
            config.registries.clone(),
        )?);
        let socket_dir = runtime_dir.join("infr");
        let mut models = HashMap::new();
        let mut aliases = HashMap::new();
        let mut overrides = HashMap::new();
        let excluded = config.exclude.iter().map(|id| canonical_key(id)).collect();

        for entry in &config.models {
            overrides.insert(
                canonical_key(&entry.id),
                ModelOverride {
                    instance: entry.instance.clone(),
                },
            );
            let Some(source) = &entry.source else {
                continue;
            };
            let resolved = registry.resolve_source(&entry.id, source).await?;
            let instance = entry
                .instance
                .clone()
                .unwrap_or_else(|| instance_for(&entry.id));
            let model = Model {
                id: entry.id.clone(),
                socket: socket_dir.join(format!("{instance}.sock")),
                instance,
                source: resolved.path,
                digest: resolved.digest,
            };
            aliases.insert(canonical_key(&model.id), model.id.clone());
            models.insert(model.id.clone(), model);
        }

        let slot = Self {
            models,
            aliases,
            overrides,
            excluded,
            host,
            registry,
            socket_dir,
            max_model_size: config.max_model_size.map(|size| size.as_u64()),
            phase: Mutex::new(Phase::Idle),
            inflight: AtomicUsize::new(0),
            notify: Notify::new(),
            switch: Mutex::new(()),
            queue_timeout: Duration::from_secs(config.queue_timeout_secs),
        };
        slot.adopt_running().await;
        Ok(slot)
    }

    /// Take over the unit an earlier proxy left running, and stop any others.
    async fn adopt_running(&self) {
        let running = match self.host.running().await {
            Ok(running) => running,
            Err(err) => {
                tracing::warn!(error = %err, "could not list running model units");
                return;
            }
        };
        if running.is_empty() {
            return;
        }

        let mut matched = Vec::new();
        for unit in &running {
            if let Some(source) = &unit.source
                && unit.socket.is_some()
                && let Some(model) = self.model_for_source(source).await
            {
                matched.push((unit, model));
            }
        }

        let adopted = if matched.len() == 1 {
            matched.pop()
        } else {
            if matched.len() > 1 {
                tracing::warn!(
                    units = matched.len(),
                    "several model units are running; stopping them and starting fresh"
                );
            }
            None
        };
        let adopted_instance = adopted.as_ref().map(|(unit, _)| unit.instance.clone());
        if let Some((unit, mut model)) = adopted {
            model.instance = unit.instance.clone();
            if let Some(socket) = &unit.socket {
                model.socket = socket.clone();
            }
            tracing::info!(
                model = %model.id,
                instance = %model.instance,
                socket = %model.socket.display(),
                "adopted the running model"
            );
            *self.phase.lock().await = Phase::Running(model);
        }

        for unit in &running {
            if adopted_instance.as_deref() == Some(unit.instance.as_str()) {
                continue;
            }
            tracing::info!(instance = %unit.instance, "stopping a unit from a previous run");
            if let Err(err) = self.host.stop_instance(&unit.instance).await {
                tracing::warn!(instance = %unit.instance, error = %err, "could not stop a stale unit");
            }
        }
    }

    /// Match a running unit's GGUF path back to a servable model.
    async fn model_for_source(&self, source: &str) -> Option<Model> {
        for model in self.models.values() {
            if model.source == source {
                return self.model(&model.id).await.ok();
            }
        }
        let references = self.registry.manifests().await.ok()?;
        for reference in references {
            if let Ok(model) = self.model(&reference.display_id()).await
                && model.source == source
            {
                return Some(model);
            }
        }
        None
    }

    /// Resolve an id to a servable model: a config entry first, then the cache.
    pub async fn model(&self, id: &str) -> Result<Model, SlotError> {
        if self.excluded.contains(id) || self.excluded.contains(&canonical_key(id)) {
            return Err(SlotError::UnknownModel);
        }
        if let Some(model) = self.config_model(id) {
            return self.checked(model).await;
        }

        let Ok(reference) = Reference::parse(id) else {
            return Err(SlotError::UnknownModel);
        };
        let canonical = reference.display_id();
        if let Some(model) = self.config_model(&canonical) {
            return self.checked(model).await;
        }

        let resolved = match self.registry.cached(&reference).await {
            Ok(resolved) => resolved,
            Err(err) => {
                tracing::debug!(model = %id, error = %err, "model is not in the cache");
                return Err(SlotError::UnknownModel);
            }
        };
        if let Some(max) = self.max_model_size
            && resolved.size > max
        {
            return Err(SlotError::TooLarge {
                size: resolved.size,
                max,
            });
        }

        let over = self.overrides.get(&canonical);
        let instance = over
            .and_then(|over| over.instance.clone())
            .unwrap_or_else(|| instance_for(&canonical));
        Ok(Model {
            id: canonical,
            socket: self.socket_dir.join(format!("{instance}.sock")),
            instance,
            source: resolved.path,
            digest: resolved.digest,
        })
    }

    /// Every model the proxy can start: config entries plus cached manifests.
    pub async fn models(&self) -> Vec<Model> {
        let mut listed = Vec::new();
        let mut covered = HashSet::new();

        for model in self.models.values() {
            covered.insert(model.id.clone());
            covered.insert(canonical_key(&model.id));
            match self.model(&model.id).await {
                Ok(model) => listed.push(model),
                Err(SlotError::TooLarge { size, .. }) => {
                    tracing::debug!(
                        model = %model.id,
                        size,
                        "not advertising: model exceeds max_model_size"
                    );
                }
                Err(_) => {}
            }
        }

        let references = match self.registry.manifests().await {
            Ok(references) => references,
            Err(err) => {
                tracing::warn!(error = %err, "could not read the model cache");
                Vec::new()
            }
        };
        for reference in references {
            let id = reference.display_id();
            if covered.contains(&id) || self.models.contains_key(&id) {
                continue;
            }
            match self.model(&id).await {
                Ok(model) => listed.push(model),
                Err(SlotError::TooLarge { size, .. }) => {
                    tracing::debug!(
                        model = %id,
                        size,
                        "not advertising: model exceeds max_model_size"
                    );
                }
                Err(_) => {}
            }
        }

        listed.sort_by(|a, b| a.id.cmp(&b.id));
        listed
    }

    pub fn registry(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }

    pub async fn resident(&self) -> Option<String> {
        match &*self.phase.lock().await {
            Phase::Running(model) | Phase::Draining(model) => Some(model.id.clone()),
            _ => None,
        }
    }

    #[cfg(test)]
    pub async fn acquire(self: &Arc<Self>, id: &str) -> Result<Lease, SlotError> {
        self.acquire_model(id).await.map(|(_, lease)| lease)
    }

    /// Acquire the slot and return the model that was resolved for the request.
    pub async fn acquire_model(self: &Arc<Self>, id: &str) -> Result<(Model, Lease), SlotError> {
        let model = self.model(id).await?;
        tracing::debug!(model = %model.id, socket = %model.socket.display(), "slot: resolved");
        if let Some(running) = self.try_lease(&model).await {
            tracing::debug!(model = %running.id, socket = %running.socket.display(), "slot: leased the resident model");
            return Ok((running, self.lease()));
        }

        let deadline = Instant::now() + self.queue_timeout;
        let _switch = tokio::time::timeout(self.queue_timeout, self.switch.lock())
            .await
            .map_err(|_| SlotError::Busy)?;
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            match self.next_action(&model).await {
                Action::Leased(running) => return Ok((running, self.lease())),
                Action::Switch(previous) => {
                    tracing::debug!(
                        model = %model.id,
                        previous = previous.as_ref().map(|model| model.id.as_str()),
                        "slot: switching"
                    );
                    self.switch_to(model.clone(), previous).await?;
                    self.inflight.fetch_add(1, Ordering::SeqCst);
                    self.notify.notify_waiters();
                    return Ok((model, self.lease()));
                }
                Action::Wait => {}
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(SlotError::Busy);
            }
            if tokio::time::timeout(remaining, notified).await.is_err() {
                return Err(SlotError::Busy);
            }
        }
    }

    fn lease(self: &Arc<Self>) -> Lease {
        Lease {
            slot: Arc::clone(self),
        }
    }

    fn config_model(&self, id: &str) -> Option<Model> {
        self.models
            .get(id)
            .or_else(|| self.aliases.get(id).and_then(|key| self.models.get(key)))
            .cloned()
    }

    async fn checked(&self, model: Model) -> Result<Model, SlotError> {
        let size = match tokio::fs::metadata(&model.source).await {
            Ok(meta) => meta.len(),
            Err(err) => {
                tracing::warn!(model = %model.id, error = %err, "configured model source is missing");
                return Err(SlotError::UnknownModel);
            }
        };
        if let Some(max) = self.max_model_size
            && size > max
        {
            return Err(SlotError::TooLarge { size, max });
        }
        Ok(model)
    }

    async fn try_lease(&self, model: &Model) -> Option<Model> {
        let running = match &*self.phase.lock().await {
            Phase::Running(current) if current.id == model.id => current.clone(),
            _ => return None,
        };
        if !self.host.is_active(&running).await.unwrap_or(false) {
            if let Err(err) = self.host.stop(&running).await {
                tracing::warn!(model = %running.id, error = %err, "could not stop the exited model");
            }
            let mut phase = self.phase.lock().await;
            if matches!(&*phase, Phase::Running(current) if current.id == model.id) {
                *phase = Phase::Idle;
            }
            drop(phase);
            self.notify.notify_waiters();
            return None;
        }
        let phase = self.phase.lock().await;
        if matches!(&*phase, Phase::Running(current) if current.id == model.id) {
            self.inflight.fetch_add(1, Ordering::SeqCst);
            Some(running)
        } else {
            None
        }
    }

    async fn next_action(&self, model: &Model) -> Action {
        let mut phase = self.phase.lock().await;
        let empty = self.inflight.load(Ordering::SeqCst) == 0;
        match phase.clone() {
            Phase::Running(current) if current.id == model.id => {
                self.inflight.fetch_add(1, Ordering::SeqCst);
                Action::Leased(current)
            }
            Phase::Running(current) if empty => {
                *phase = Phase::Starting;
                Action::Switch(Some(current))
            }
            Phase::Running(current) => {
                *phase = Phase::Draining(current);
                Action::Wait
            }
            Phase::Draining(current) if empty => {
                *phase = Phase::Starting;
                Action::Switch(Some(current))
            }
            Phase::Draining(_) | Phase::Starting => Action::Wait,
            Phase::Idle => {
                *phase = Phase::Starting;
                Action::Switch(None)
            }
        }
    }

    /// The load outlives the request that asked for it, so a client hanging up
    /// mid-switch cannot leave the slot in `Starting`.
    async fn switch_to(
        self: &Arc<Self>,
        model: Model,
        previous: Option<Model>,
    ) -> Result<(), SlotError> {
        let slot = Arc::clone(self);
        let switch = tokio::spawn(async move {
            let result = slot.hand_over(&model, previous.as_ref()).await;
            let mut phase = slot.phase.lock().await;
            match &result {
                Ok(()) => *phase = Phase::Running(model.clone()),
                Err(_) => *phase = Phase::Idle,
            }
            drop(phase);
            slot.notify.notify_waiters();
            result
        });
        match switch.await {
            Ok(result) => result,
            Err(err) => {
                *self.phase.lock().await = Phase::Idle;
                self.notify.notify_waiters();
                Err(SlotError::Backend(format!("switch task failed: {err}")))
            }
        }
    }

    async fn hand_over(&self, model: &Model, previous: Option<&Model>) -> Result<(), SlotError> {
        if let Some(previous) = previous {
            self.host
                .stop(previous)
                .await
                .map_err(|err| SlotError::Backend(format!("stop {}: {err}", previous.instance)))?;
        }
        self.host
            .start(model)
            .await
            .map_err(|err| SlotError::Backend(format!("start {}: {err}", model.instance)))?;
        Ok(())
    }
}

/// Holds the model resident: a switch cannot stop the unit until the last
/// lease is dropped, which for a streaming reply is after the last chunk.
pub struct Lease {
    slot: Arc<Slot>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.slot.inflight.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.slot.notify.notify_waiters();
        }
    }
}

/// The lookup key shared by a config id and the cached manifest it describes.
fn canonical_key(id: &str) -> String {
    Reference::parse(id)
        .map(|reference| reference.display_id())
        .unwrap_or_else(|_| id.to_string())
}

/// Unit-safe instance name that stays stable for a given model id.
fn instance_for(id: &str) -> String {
    let base: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '-'
            }
        })
        .take(48)
        .collect();
    let digest = Sha256::digest(id.as_bytes());
    let mut hash = String::with_capacity(8);
    for byte in &digest[..4] {
        let _ = write!(hash, "{byte:02x}");
    }
    format!("{base}-{hash}")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use bytesize::ByteSize;

    use super::*;
    use crate::config::ModelEntry;
    use crate::host::RunningModel;

    #[derive(Default)]
    struct FakeHost {
        active: StdMutex<HashSet<String>>,
        calls: StdMutex<Vec<String>>,
        running: StdMutex<Vec<RunningModel>>,
        start_delay: StdMutex<Duration>,
    }

    impl FakeHost {
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }

        fn exit(&self, id: &str) {
            self.active.lock().unwrap().remove(id);
        }

        fn run(&self, unit: RunningModel) {
            self.running.lock().unwrap().push(unit);
        }
    }

    #[async_trait]
    impl ModelHost for FakeHost {
        async fn start(&self, model: &Model) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("start {}", model.id));
            let delay = *self.start_delay.lock().unwrap();
            tokio::time::sleep(delay).await;
            self.active.lock().unwrap().insert(model.id.clone());
            Ok(())
        }

        async fn stop(&self, model: &Model) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("stop {}", model.id));
            self.active.lock().unwrap().remove(&model.id);
            Ok(())
        }

        async fn is_active(&self, model: &Model) -> Result<bool> {
            Ok(self.active.lock().unwrap().contains(&model.id))
        }

        async fn running(&self) -> Result<Vec<RunningModel>> {
            Ok(self.running.lock().unwrap().clone())
        }

        async fn stop_instance(&self, instance: &str) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("stop-instance {instance}"));
            Ok(())
        }
    }

    fn entry(id: &str, source: &Path) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            instance: Some(id.into()),
            source: Some(source.to_string_lossy().into_owned()),
        }
    }

    fn config(dir: &Path) -> Config {
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        std::fs::write(&a, b"model a").unwrap();
        std::fs::write(&b, b"model b").unwrap();
        Config {
            listen: "127.0.0.1:0".parse().unwrap(),
            queue_timeout_secs: 5,
            load_timeout_secs: 1,
            poll_interval_ms: 10,
            cache_dir: Some(dir.to_path_buf()),
            extra_model_dirs: Vec::new(),
            max_model_size: None,
            registries: Vec::new(),
            models: vec![entry("a", &a), entry("b", &b)],
            exclude: Vec::new(),
        }
    }

    fn empty_config(dir: &Path) -> Config {
        Config {
            cache_dir: Some(dir.to_path_buf()),
            ..Config::default()
        }
    }

    /// Write a manifest and blob the way a pull does, without a registry.
    async fn cache_model(dir: &Path, id: &str, bytes: &[u8]) {
        let reference = Reference::parse(id).unwrap();
        let digest = crate::registry::sha256_hex(bytes);
        let blob = dir.join("blobs").join(format!("sha256-{digest}"));
        tokio::fs::create_dir_all(blob.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&blob, bytes).await.unwrap();
        let raw = format!(
            "{{\"mediaType\":\"application/vnd.docker.distribution.manifest.v2+json\",\"layers\":[{{\"mediaType\":\"application/vnd.ollama.image.model\",\"digest\":\"sha256:{digest}\",\"size\":{}}}]}}",
            bytes.len()
        );
        let path = dir
            .join("manifests")
            .join(&reference.host)
            .join(&reference.namespace)
            .join(&reference.name)
            .join(&reference.tag);
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(path, raw).await.unwrap();
    }

    #[tokio::test]
    async fn adopts_a_running_config_model() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        let source = dir.path().join("a.gguf").to_string_lossy().into_owned();
        host.active.lock().unwrap().insert("a".into());
        host.run(RunningModel {
            instance: "a".into(),
            source: Some(source),
            socket: Some(dir.path().join("infr/old.sock")),
        });
        let slot = Arc::new(
            Slot::with_host(config(dir.path()), host.clone(), dir.path().to_path_buf())
                .await
                .unwrap(),
        );

        assert_eq!(slot.resident().await.as_deref(), Some("a"));
        let (model, lease) = slot.acquire_model("a").await.unwrap();
        assert_eq!(model.socket, dir.path().join("infr/old.sock"));
        assert!(host.calls().is_empty());
        drop(lease);
    }

    #[tokio::test]
    async fn adopts_a_running_cached_model() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        cache_model(dir.path(), "qwen2.5:1.5b", b"cached model").await;
        let blob = format!("sha256-{}", crate::registry::sha256_hex(b"cached model"));
        let source = dir
            .path()
            .join("blobs")
            .join(blob)
            .to_string_lossy()
            .into_owned();
        host.active.lock().unwrap().insert("qwen2.5:1.5b".into());
        host.run(RunningModel {
            instance: instance_for("qwen2.5:1.5b"),
            source: Some(source),
            socket: Some(
                dir.path()
                    .join("infr")
                    .join(format!("{}.sock", instance_for("qwen2.5:1.5b"))),
            ),
        });
        let slot = Arc::new(
            Slot::with_host(
                empty_config(dir.path()),
                host.clone(),
                dir.path().to_path_buf(),
            )
            .await
            .unwrap(),
        );

        let lease = slot.acquire("qwen2.5:1.5b").await.unwrap();
        assert!(host.calls().is_empty());
        drop(lease);
    }

    #[tokio::test]
    async fn stops_running_units_that_do_not_match() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        host.run(RunningModel {
            instance: "gone".into(),
            source: Some("/gone.gguf".into()),
            socket: None,
        });
        host.run(RunningModel {
            instance: "unreadable".into(),
            source: None,
            socket: None,
        });
        let slot = Slot::with_host(config(dir.path()), host.clone(), dir.path().to_path_buf())
            .await
            .unwrap();

        assert_eq!(slot.resident().await, None);
        assert_eq!(
            host.calls(),
            vec!["stop-instance gone", "stop-instance unreadable"]
        );
    }

    #[tokio::test]
    async fn switching_returns_the_new_model() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        let slot = Arc::new(
            Slot::with_host(config(dir.path()), host.clone(), dir.path().to_path_buf())
                .await
                .unwrap(),
        );

        let (a, lease) = slot.acquire_model("a").await.unwrap();
        assert_eq!(a.id, "a");
        assert_eq!(a.socket, dir.path().join("infr/a.sock"));
        drop(lease);

        let (b, lease) = slot.acquire_model("b").await.unwrap();
        assert_eq!(b.id, "b");
        assert_eq!(b.socket, dir.path().join("infr/b.sock"));
        assert_eq!(host.calls(), vec!["start a", "stop a", "start b"]);
        drop(lease);
    }

    #[tokio::test]
    async fn switch_waits_for_the_lease_then_stops_before_starting() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        let slot = Arc::new(
            Slot::with_host(config(dir.path()), host.clone(), dir.path().to_path_buf())
                .await
                .unwrap(),
        );

        let lease = slot.acquire("a").await.unwrap();
        let waiting = {
            let slot = Arc::clone(&slot);
            tokio::spawn(async move { slot.acquire("b").await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(host.calls(), vec!["start a"]);

        drop(lease);
        let lease = waiting.await.unwrap().unwrap();
        assert_eq!(host.calls(), vec!["start a", "stop a", "start b"]);
        drop(lease);
    }

    #[tokio::test]
    async fn a_cancelled_switch_does_not_wedge_the_slot() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        *host.start_delay.lock().unwrap() = Duration::from_millis(100);
        let slot = Arc::new(
            Slot::with_host(config(dir.path()), host.clone(), dir.path().to_path_buf())
                .await
                .unwrap(),
        );

        let request = {
            let slot = Arc::clone(&slot);
            tokio::spawn(async move { slot.acquire("a").await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        request.abort();

        let lease = slot.acquire("b").await.unwrap();
        assert_eq!(host.calls(), vec!["start a", "stop a", "start b"]);
        drop(lease);
    }

    #[tokio::test]
    async fn an_idle_exited_model_is_started_again() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        let slot = Arc::new(
            Slot::with_host(config(dir.path()), host.clone(), dir.path().to_path_buf())
                .await
                .unwrap(),
        );

        let lease = slot.acquire("a").await.unwrap();
        drop(lease);
        host.exit("a");

        let lease = slot.acquire("a").await.unwrap();
        assert_eq!(host.calls(), vec!["start a", "stop a", "start a"]);
        drop(lease);
    }

    #[tokio::test]
    async fn serves_a_cached_model_without_a_config_entry() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        cache_model(dir.path(), "qwen2.5:1.5b", b"cached model").await;
        let slot = Arc::new(
            Slot::with_host(
                empty_config(dir.path()),
                host.clone(),
                dir.path().to_path_buf(),
            )
            .await
            .unwrap(),
        );

        let lease = slot
            .acquire("registry.ollama.ai/library/qwen2.5:1.5b")
            .await
            .unwrap();
        assert_eq!(host.calls(), vec!["start qwen2.5:1.5b"]);
        drop(lease);

        let model = slot.model("qwen2.5:1.5b").await.unwrap();
        assert_eq!(model.id, "qwen2.5:1.5b");
        let instance = instance_for("qwen2.5:1.5b");
        assert_eq!(model.instance, instance);
        assert_eq!(
            model.socket,
            dir.path().join("infr").join(format!("{instance}.sock"))
        );
        assert_eq!(slot.models().await.len(), 1);

        let lease = slot.acquire("qwen2.5:1.5b").await.unwrap();
        assert_eq!(host.calls(), vec!["start qwen2.5:1.5b"]);
        drop(lease);
    }

    #[tokio::test]
    async fn a_config_entry_overrides_a_cached_model() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        cache_model(dir.path(), "qwen2.5:1.5b", b"cached model").await;
        let mut config = empty_config(dir.path());
        config.models.push(ModelEntry {
            id: "qwen2.5:1.5b".into(),
            instance: Some("tiny".into()),
            source: None,
        });
        let slot = Arc::new(
            Slot::with_host(config, host, dir.path().to_path_buf())
                .await
                .unwrap(),
        );

        let model = slot.model("qwen2.5:1.5b").await.unwrap();
        assert_eq!(model.instance, "tiny");
        assert_eq!(model.socket, dir.path().join("infr/tiny.sock"));
    }

    #[tokio::test]
    async fn excluded_models_are_not_served_or_listed() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        cache_model(dir.path(), "big:1", b"model").await;
        let mut config = empty_config(dir.path());
        config.exclude = vec!["big:1".into()];
        let slot = Arc::new(
            Slot::with_host(config, host, dir.path().to_path_buf())
                .await
                .unwrap(),
        );

        assert!(slot.model("big:1").await.is_err());
        assert!(slot.models().await.is_empty());
    }

    #[tokio::test]
    async fn oversized_cached_models_are_not_offered() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(FakeHost::default());
        cache_model(dir.path(), "big:1", b"0123456789").await;
        let mut config = empty_config(dir.path());
        config.max_model_size = Some(ByteSize::b(4));
        let slot = Arc::new(
            Slot::with_host(config, host, dir.path().to_path_buf())
                .await
                .unwrap(),
        );

        assert!(matches!(
            slot.model("big:1").await,
            Err(SlotError::TooLarge { .. })
        ));
        assert!(matches!(
            slot.acquire("big:1").await,
            Err(SlotError::TooLarge { .. })
        ));
        assert!(slot.models().await.is_empty());
    }
}
