use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use eyre::{Context, Result, bail};
use zbus::Connection;
use zbus::OwnedObjectPath;

use crate::config::Model;
use crate::host::{ModelHost, RunningModel};

/// The user-manager implementation of [`ModelHost`]: one `infr@<instance>.service`
/// per model, fed a per-instance env file with its GGUF path and socket.
pub struct Systemd {
    conn: Connection,
    runtime_dir: PathBuf,
    load_timeout: Duration,
    poll: Duration,
}

#[zbus::proxy(
    interface = "org.freedesktop.systemd1.Manager",
    default_service = "org.freedesktop.systemd1",
    default_path = "/org/freedesktop/systemd1"
)]
trait Manager {
    fn start_unit(&self, name: &str, mode: &str) -> zbus::Result<OwnedObjectPath>;
    fn stop_unit(&self, name: &str, mode: &str) -> zbus::Result<OwnedObjectPath>;
    fn get_unit(&self, name: &str) -> zbus::Result<OwnedObjectPath>;
    fn list_units(&self) -> zbus::Result<Vec<UnitStatus>>;
}

/// `a(ssssssouso)`: everything `ListUnits` returns per unit.
type UnitStatus = (
    String,
    String,
    String,
    String,
    String,
    String,
    OwnedObjectPath,
    u32,
    String,
    OwnedObjectPath,
);

#[zbus::proxy(
    interface = "org.freedesktop.systemd1.Unit",
    default_service = "org.freedesktop.systemd1",
    default_path = "/org/freedesktop/systemd1/unit/_"
)]
trait Unit {
    #[zbus(property)]
    fn active_state(&self) -> zbus::Result<String>;
}

const POLL: Duration = Duration::from_millis(200);

impl Systemd {
    pub async fn connect(
        runtime_dir: PathBuf,
        load_timeout: Duration,
        poll: Duration,
    ) -> Result<Self> {
        let conn = Connection::session()
            .await
            .wrap_err("connect to the user bus (is DBUS_SESSION_BUS_ADDRESS set?)")?;
        Ok(Self {
            conn,
            runtime_dir,
            load_timeout,
            poll,
        })
    }

    /// Active `infr@` units and the env they were started with.
    async fn active_units(&self) -> Result<Vec<RunningModel>> {
        let units = ManagerProxy::new(&self.conn).await?.list_units().await?;
        let mut running = Vec::new();
        for (name, _, _, active_state, _, _, _, _, _, _) in units {
            if !name.starts_with("infr@") || !name.ends_with(".service") {
                continue;
            }
            if !matches!(active_state.as_str(), "active" | "activating" | "reloading") {
                continue;
            }
            let Some(instance) = name
                .strip_prefix("infr@")
                .and_then(|name| name.strip_suffix(".service"))
            else {
                continue;
            };
            let path = self
                .runtime_dir
                .join("infr")
                .join(format!("{instance}.env"));
            let body = tokio::fs::read_to_string(&path).await.unwrap_or_default();
            let source = env_value(&body, "INFR_MODEL").map(str::to_string);
            let socket = env_value(&body, "INFR_SOCKET").map(PathBuf::from);
            running.push(RunningModel {
                instance: instance.to_string(),
                source,
                socket,
            });
        }
        Ok(running)
    }

    async fn start_unit(&self, unit: &str) -> Result<()> {
        ManagerProxy::new(&self.conn)
            .await?
            .start_unit(unit, "replace")
            .await
            .wrap_err_with(|| format!("start {unit}"))?;
        Ok(())
    }

    async fn stop_unit(&self, unit: &str) -> Result<()> {
        if self.unit_active(unit).await? {
            ManagerProxy::new(&self.conn)
                .await?
                .stop_unit(unit, "replace")
                .await
                .wrap_err_with(|| format!("stop {unit}"))?;
        }
        self.wait_inactive(unit).await
    }

    async fn active_state(&self, unit: &str) -> Result<Option<String>> {
        let Some(path) = self.unit_path(unit).await? else {
            return Ok(None);
        };
        let state = UnitProxy::builder(&self.conn)
            .path(path)
            .build()
            .await?
            .active_state()
            .await
            .wrap_err_with(|| format!("read ActiveState of {unit}"))?;
        Ok(Some(state))
    }

    async fn unit_active(&self, unit: &str) -> Result<bool> {
        Ok(matches!(
            self.active_state(unit).await?.as_deref(),
            Some("active" | "activating" | "reloading")
        ))
    }

    async fn wait_inactive(&self, unit: &str) -> Result<()> {
        let deadline = Instant::now() + self.load_timeout;
        loop {
            let state = self.active_state(unit).await?;
            if !matches!(
                state.as_deref(),
                Some("active" | "activating" | "reloading" | "deactivating")
            ) {
                return Ok(());
            }
            if Instant::now() >= deadline {
                bail!("{unit} is still active after {:?}", self.load_timeout);
            }
            tokio::time::sleep(POLL).await;
        }
    }

    async fn wait_ready(&self, model: &Model) -> Result<()> {
        let unit = unit_name(&model.instance);
        let deadline = Instant::now() + self.load_timeout;
        let client = reqwest::Client::builder()
            .unix_socket(model.socket.clone())
            .timeout(Duration::from_secs(2))
            .build()
            .wrap_err("build unix-socket client")?;
        loop {
            let ready = client
                .get("http://localhost/v1/models")
                .send()
                .await
                .is_ok_and(|response| response.status().is_success());
            if ready {
                if self.unit_active(&unit).await? {
                    return Ok(());
                }
                bail!("{unit} exited before it answered");
            }
            if !self.unit_active(&unit).await? {
                bail!("{unit} exited before it answered");
            }
            if Instant::now() >= deadline {
                bail!("{} did not answer within {:?}", model.id, self.load_timeout);
            }
            tokio::time::sleep(self.poll).await;
        }
    }

    async fn write_env(&self, model: &Model) -> Result<()> {
        let dir = self.runtime_dir.join("infr");
        let path = dir.join(format!("{}.env", model.instance));
        let body = format!(
            "INFR_MODEL={}\nINFR_SOCKET={}\n",
            model.source,
            model.socket.display()
        );
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            std::fs::create_dir_all(&dir)?;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(path)?;
            file.write_all(body.as_bytes())
        })
        .await
        .wrap_err("write model env")??;
        Ok(())
    }

    async fn unit_path(&self, unit: &str) -> Result<Option<OwnedObjectPath>> {
        match ManagerProxy::new(&self.conn).await?.get_unit(unit).await {
            Ok(path) => Ok(Some(path)),
            Err(zbus::Error::MethodError(name, _, _))
                if name.as_str() == "org.freedesktop.systemd1.NoSuchUnit" =>
            {
                Ok(None)
            }
            Err(err) => Err(err).wrap_err_with(|| format!("look up {unit}")),
        }
    }
}

#[async_trait]
impl ModelHost for Systemd {
    async fn start(&self, model: &Model) -> Result<()> {
        self.write_env(model).await?;
        let unit = unit_name(&model.instance);
        self.start_unit(&unit).await?;
        self.wait_ready(model).await
    }

    async fn stop(&self, model: &Model) -> Result<()> {
        self.stop_instance(&model.instance).await
    }

    async fn is_active(&self, model: &Model) -> Result<bool> {
        self.unit_active(&unit_name(&model.instance)).await
    }

    async fn running(&self) -> Result<Vec<RunningModel>> {
        self.active_units().await
    }

    async fn stop_instance(&self, instance: &str) -> Result<()> {
        self.stop_unit(&unit_name(instance)).await
    }
}

fn env_value<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    body.lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
        .map(str::trim)
}

fn unit_name(instance: &str) -> String {
    format!("infr@{instance}.service")
}
