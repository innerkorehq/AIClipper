//! Resumability & failure handling (PRD §9).
//!
//! Every stage writes its output to `<workdir>/stages/<id>.json`. A re-run
//! detects an existing stage artifact and skips re-computation unless `--force`.

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// Artifact existed already — stage output reused.
    Reused,
    /// Stage ran and produced a new artifact.
    Computed,
}

const STAGE_EXT: &str = "json";

/// The on-disk artifact store. Not a database — plain JSON files keyed by stage.
#[derive(Clone)]
pub struct ArtifactStore {
    dir: PathBuf,
}

#[allow(dead_code)]
impl ArtifactStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    pub fn stage_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.{STAGE_EXT}"))
    }

    #[allow(dead_code)]
    pub fn exists(&self, id: &str) -> bool {
        self.stage_path(id).exists()
    }

    pub fn write<T: Serialize>(&self, id: &str, value: &T) -> Result<()> {
        let path = self.stage_path(id);
        let bytes = serde_json::to_vec_pretty(value)
            .with_context(|| format!("serializing stage `{id}`"))?;
        atomic_write(&path, &bytes)?;
        tracing::info!(stage = id, path = %path.display(), "wrote stage artifact");
        Ok(())
    }

    pub fn write_raw(&self, id: &str, bytes: &[u8]) -> Result<()> {
        let path = self.stage_path(id);
        atomic_write(&path, bytes)?;
        Ok(())
    }

    pub fn read_raw(&self, id: &str) -> Result<Option<Vec<u8>>> {
        let path = self.stage_path(id);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?))
    }

    pub fn read<T: DeserializeOwned>(&self, id: &str) -> Result<Option<T>> {
        let path = self.stage_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing stage artifact {}", path.display()))?;
        Ok(Some(value))
    }

    /// Run `compute` and persist, or reuse an existing artifact unless `force`.
    pub fn stage<T: Serialize + DeserializeOwned>(
        &self,
        id: &str,
        force: bool,
        compute: impl FnOnce() -> Result<T>,
    ) -> Result<(T, Freshness)> {
        if !force {
            if let Some(existing) = self.read::<T>(id)? {
                tracing::info!(stage = id, "reusing existing artifact");
                return Ok((existing, Freshness::Reused));
            }
        }
        let value = compute().with_context(|| format!("stage `{id}` failed"))?;
        self.write(id, &value)?;
        Ok((value, Freshness::Computed))
    }

    /// Drop an artifact (used when an upstream stage is re-forced).
    pub fn remove(&self, id: &str) {
        let _ = std::fs::remove_file(self.stage_path(id));
    }
}

/// Write-then-rename so a partially-written file can never be read as a
/// completed stage artifact.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let tmp = path.with_extension(format!("{STAGE_EXT}.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path).with_context(|| format!("finalizing {}", path.display()))?;
    Ok(())
}