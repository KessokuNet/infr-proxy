mod api;
mod config;
mod gguf;
mod host;
mod registry;
mod slot;
mod systemd;
mod telemetry;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use eyre::{Result, bail};

use crate::config::{Config, default_cache_dir};
use crate::registry::{Reference, Registry};
use crate::slot::Slot;

#[derive(Parser)]
#[command(version, about = "Ollama-compatible router for infr model units")]
struct Cli {
    /// Config file; defaults to ./infr-proxy.toml, then $XDG_CONFIG_HOME/infr-proxy/config.toml
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Model cache root override
    #[arg(long, global = true)]
    cache_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the HTTP server (default)
    Serve {
        /// Listen address override
        #[arg(long)]
        listen: Option<SocketAddr>,
    },
    /// Download a model into the cache
    Pull { reference: String },
    /// Print the cached path of a model, asking a running proxy
    ModelPath {
        id: String,
        #[arg(long, default_value = "http://127.0.0.1:11435")]
        proxy: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let serve = matches!(cli.command, None | Some(Command::Serve { .. }));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let _telemetry = telemetry::init(serve)?;
        run(cli).await
    })
}

async fn run(cli: Cli) -> Result<()> {
    match cli.command.unwrap_or(Command::Serve { listen: None }) {
        Command::Serve { listen } => serve(cli.config, cli.cache_dir, listen).await,
        Command::Pull { reference } => pull(cli.config, cli.cache_dir, &reference).await,
        Command::ModelPath { id, proxy } => model_path(&proxy, &id).await,
    }
}

async fn serve(
    config_path: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    listen: Option<SocketAddr>,
) -> Result<()> {
    let mut config = match config_path {
        Some(path) => load_config(Some(path)).await?,
        None => load_optional_config(None).await?.unwrap_or_default(),
    };
    if let Some(cache_dir) = cache_dir {
        config.cache_dir = Some(cache_dir);
    }
    let listen = listen.unwrap_or(config.listen);
    let slot = Arc::new(Slot::new(config).await?);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!(addr = %listener.local_addr()?, "infr-proxy listening");
    axum::serve(listener, api::router(Arc::clone(&slot)))
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn pull(
    config_path: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    reference: &str,
) -> Result<()> {
    let reference = Reference::parse(reference)?;
    let config = load_optional_config(config_path).await?;
    let root = match cache_dir {
        Some(dir) => dir,
        None => match &config {
            Some(config) => config.cache_root()?,
            None => default_cache_dir()?,
        },
    };
    let (registries, store_dirs) = match config {
        Some(config) => (config.registries, config.extra_model_dirs),
        None => (Vec::new(), Vec::new()),
    };
    let registry = Registry::new(root, store_dirs, registries)?;
    let path = registry.pull(&reference).await?;
    println!("{}", path.display());
    Ok(())
}

async fn model_path(proxy: &str, id: &str) -> Result<()> {
    let url = format!("{}/registry/model-path", proxy.trim_end_matches('/'));
    let response = reqwest::Client::new()
        .get(&url)
        .query(&[("id", id)])
        .send()
        .await?;
    let status = response.status();
    let body: serde_json::Value = response.json().await?;
    if !status.is_success() {
        let message = body["error"]["message"]
            .as_str()
            .unwrap_or("request failed");
        bail!("{message}");
    }
    let path = body["path"]
        .as_str()
        .ok_or_else(|| eyre::eyre!("no path in response"))?;
    println!("{path}");
    Ok(())
}

async fn load_config(path: Option<PathBuf>) -> Result<Config> {
    tokio::task::spawn_blocking(move || Config::load(path.as_deref())).await?
}

async fn load_optional_config(path: Option<PathBuf>) -> Result<Option<Config>> {
    tokio::task::spawn_blocking(move || Config::load_optional(path.as_deref())).await?
}

async fn shutdown() {
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("install SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = sigterm.recv() => {}
    }
}
