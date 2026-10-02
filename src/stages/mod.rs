//! Pipeline stages (PRD §5). Each stage is an async fn over shared structs,
//! writing its output through the `ArtifactStore` so runs resume (PRD §9).

pub mod audio_events;
pub mod content_edit;
pub mod export;
pub mod ingest;
pub mod render;
pub mod rough_cut;
pub mod score;
pub mod shot_classify;
pub mod speaker;
pub mod time_remap;
pub mod trajectory;
pub mod transcribe;

use crate::artifact::ArtifactStore;
use crate::config::Config;
use crate::sidecar::SidecarClient;

/// Shared context handed to every stage.
#[derive(Clone)]
#[allow(dead_code)]
pub struct PipelineContext {
    pub cfg: Config,
    pub store: ArtifactStore,
    pub sidecar: SidecarClient,
}

impl PipelineContext {
    #[allow(dead_code)]
    pub fn new(cfg: Config, store: ArtifactStore, sidecar: SidecarClient) -> Self {
        Self { cfg, store, sidecar }
    }
}