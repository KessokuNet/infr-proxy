use std::path::PathBuf;

use async_trait::async_trait;
use eyre::Result;

use crate::config::Model;

/// A unit that was already active when the proxy started.
#[derive(Debug, Clone)]
pub struct RunningModel {
    pub instance: String,
    pub source: Option<String>,
    pub socket: Option<PathBuf>,
}

/// Where a model's process comes from. The scheduler only needs these three
/// operations; the systemd implementation turns them into unit jobs, a test
/// can answer them in memory.
#[async_trait]
pub trait ModelHost: Send + Sync {
    /// Bring the model up and do not return until it answers.
    async fn start(&self, model: &Model) -> Result<()>;
    /// Stop the model and do not return until its memory is released.
    async fn stop(&self, model: &Model) -> Result<()>;
    /// Whether the model's process is currently loaded.
    async fn is_active(&self, model: &Model) -> Result<bool>;
    /// Units left active by a previous proxy process.
    async fn running(&self) -> Result<Vec<RunningModel>> {
        Ok(Vec::new())
    }
    /// Stop a unit by instance name, for one that no longer matches a model.
    async fn stop_instance(&self, instance: &str) -> Result<()> {
        let _ = instance;
        Ok(())
    }
}
