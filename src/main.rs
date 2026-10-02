//! AIClipper orchestrator — runs the §5 pipeline stages in order with
//! artifact-based resumability (§9), a persistent Python ML sidecar (§4),
//! and the warm-process Remotion renderer.
//!
//! Not-a-lib: this binary is the crate root, so all `crate::` module paths
//! resolve from here.

mod artifact;
mod cli;
mod config;
mod sidecar;
mod stages;
mod types;
mod util;
mod vision;

use crate::artifact::{ArtifactStore, Freshness};
use crate::cli::Cli;
use crate::config::Config;
use crate::sidecar::SidecarClient;
use crate::stages::ingest::{apply_source_to_state, ingest, SourceInfo};
use crate::stages::time_remap::build_manifest;
use crate::stages::trajectory::build_segments_for_clip;
use crate::types::{ContentEdit, RunState, ShotSegment, SpeechTurn};
use anyhow::{Context, Result};
use clap::Parser;
use std::future::Future;
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Child;

/// Target vertical resolution for composition (matches default export profiles).
const TARGET_WIDTH: u32 = 1080;
const TARGET_HEIGHT: u32 = 1920;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aiclipper=info,info".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let exit = run(cli).await;
    match exit {
        Ok(()) => Ok(()),
        Err(e) => {
            tracing::error!("{e:#}");
            Err(e)
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    let input = cli.input();
    let cfg = Config::default_with_workdir(cli.workdir.clone());
    cfg.ensure_dirs()?;
    let store = ArtifactStore::new(cfg.stage_dir());
    let sidecar = SidecarClient::new(&cfg, &cfg.sidecar_url);

    // ---- sidecar lifecycle ------------------------------------------------
    let mut guard = SidecarGuard::None;
    if cfg.sidecar_start && !cli.no_sidecar {
        guard = spawn_sidecar(&cfg).await?;
        sidecar
            .wait_ready(Duration::from_secs(180), "python sidecar")
            .await
            .with_context(|| {
                "install ML deps first:\n  pip3 install -r python_sidecar/requirements.txt".to_string()
            })?;
        tracing::info!("python sidecar ready at {}", cfg.sidecar_url);
    }

    let result = pipeline(cfg, store, sidecar.clone(), cli, &input).await;
    if guard.is_live() {
        let _ = sidecar.shutdown().await;
        guard.kill().await;
    }
    result
}

enum SidecarGuard {
    None,
    Child { child: Child, _cwd: PathBuf },
}

impl SidecarGuard {
    fn is_live(&self) -> bool {
        matches!(self, SidecarGuard::Child { .. })
    }
    async fn kill(&mut self) {
        if let SidecarGuard::Child { child, .. } = self {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
    }
}

/// Spawn `python3 -m sidecar` inside `python_sidecar/`.
async fn spawn_sidecar(cfg: &Config) -> Result<SidecarGuard> {
    let cwd = cfg
        .remotion_dir
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("python_sidecar");
    let cwd = std::fs::canonicalize(&cwd).unwrap_or(cwd);
    if !cwd.join("sidecar").join("__main__.py").exists() {
        anyhow::bail!("python_sidecar/ not found next to remotion/ — cloned repo incomplete?");
    }
    // Prefer a project virtualenv, then $AICLIPPER_PYTHON, then system python3.
    let venv_py = cwd.join(".venv").join("bin").join("python");
    let python = if let Ok(p) = std::env::var("AICLIPPER_PYTHON") {
        PathBuf::from(p)
    } else if venv_py.exists() {
        venv_py
    } else {
        PathBuf::from("python3")
    };
    tracing::info!(python = %python.display(), "starting python sidecar");
    let child = tokio::process::Command::new(&python)
        .current_dir(&cwd)
        .arg("-m")
        .arg("sidecar")
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning python sidecar with {}", python.display()))?;
    Ok(SidecarGuard::Child { child, _cwd: cwd })
}

/// Run every artifact-based stage in order.
async fn pipeline(cfg: Config, store: ArtifactStore, sidecar: SidecarClient, cli: Cli, input: &str) -> Result<()> {
    // ---- 5.1 ingest --------------------------------------------------------
    let (source, _): (SourceInfo, Freshness) =
        run_stage(&store, "ingest", cli.force, ingest(&cfg, input, cli.force)).await?;

    // ---- 5.2 transcribe -----------------------------------------------------
    let (transcript, _) =
        run_stage(&store, "transcript", cli.force, stages::transcribe::transcribe(&cfg, &source.path, &sidecar)).await?;

    // ---- 5.3 clip scoring (Gemma) ------------------------------------------
    let (candidates, _) =
        run_stage(&store, "candidates", cli.force, stages::score::score(&cfg, &sidecar, &transcript)).await?;
    tracing::info!("{} candidates to process", candidates.len());

    if candidates.is_empty() {
        tracing::warn!("no candidates — check that the sidecar returned clips");
    }

    let fps_u32 = (source.fps.round() as u32).max(1);
    let fps_f = source.fps.max(1.0);

    // ---- diarization (once) — needed by speaker resolution ------------------
    let wav = cfg.media_dir().join("source.wav");
    stages::transcribe::extract_wav(&cfg, &source.path, &wav).await?;
    let (turns, _): (Vec<SpeechTurn>, Freshness) = if cfg.assume_single_speaker || cli.single_speaker {
        (vec![], Freshness::Computed)
    } else {
        run_stage(&store, "diarization", cli.force, stages::speaker::diarize_global(&sidecar, &wav)).await?
    };

    let (vision, vision_ok) = vision::VisionService::new(&cfg);
    if !vision_ok {
        anyhow::bail!("Apple Vision shim not available — run `cargo build` so build.rs compiles it");
    }

    // ---- per-candidate pipeline: 5.4 → 5.8 ----------------------------------
    let mut edits: Vec<ContentEdit> = Vec::with_capacity(candidates.len());
    let mut roughs: Vec<stages::rough_cut::RoughCut> = Vec::with_capacity(candidates.len());
    for (idx, candidate) in candidates.iter().enumerate() {
        let (rough, _) = run_stage(
            &store,
            &rough_id(idx),
            cli.force,
            stages::rough_cut::rough_cut(&cfg, &source.path, candidate, idx),
        )
        .await?;
        let rough_abs = rough.path.canonicalize().unwrap_or_else(|_| rough.path.clone());
        roughs.push(rough);

        let (edit, _) = run_stage(
            &store,
            &edit_id(idx),
            cli.force,
            stages::content_edit::content_edit(
                &cfg,
                &sidecar,
                &stages::content_edit::ContentEditInput {
                    candidate_idx: idx,
                    candidate,
                    transcript: &transcript,
                    scene_cuts: roughs[idx].scene_cuts.clone(),
                    source_offset: candidate.start,
                },
            ),
        )
        .await?;
        edits.push(edit.clone());

        // 5.6 classify → 5.7 speaker resolve → 5.8 trajectory, persisted as
        // one segments artifact per candidate (all deterministic stages).
        let (segments, _): (Vec<ShotSegment>, Freshness) = run_stage(
            &store,
            &segments_id(idx),
            cli.force,
            async {
                let classified = stages::shot_classify::classify_shots(
                    &cfg,
                    &vision,
                    &rough_abs,
                    candidate.start,
                    fps_f,
                    &edit.shot_list,
                    &roughs[idx].scene_cuts.iter().map(|t| candidate.start + t).collect::<Vec<_>>(),
                )
                .await?;
                let speakers = stages::speaker::resolve(
                    &cfg,
                    &wav,
                    &turns,
                    &rough_abs,
                    candidate.start,
                    idx,
                    &classified,
                    &vision,
                )
                .await?;
                Ok(build_segments_for_clip(&cfg, fps_f, candidate.start, &classified, &speakers))
            },
        )
        .await?;

        tracing::info!("candidate #{idx}: {} shot segments", segments.len());
    }

    // ---- 5.9 time remap → render manifests (the single source→output pass) --
    let mut manifests = Vec::with_capacity(candidates.len());
    for (idx, candidate) in candidates.iter().enumerate() {
        let rough = &roughs[idx];
        let rough_abs = rough.path.canonicalize().unwrap_or_else(|_| rough.path.clone());
        let info = util::probe_media(&cfg.ffmpeg, &rough_abs)
            .await
            .with_context(|| format!("probing rough cut #{}", rough.path.display()))?;
        let segments: Vec<ShotSegment> = store
            .read::<Vec<ShotSegment>>(&segments_id(idx))?
            .context("missing segments artifact")?;
        let (m, _) = run_stage(
            &store,
            &manifests_id(idx),
            cli.force,
            async {
                build_manifest(
                    &format!("clip_{idx:03}"),
                    candidate.start,
                    candidate.end,
                    fps_u32,
                    &rough.path,
                    &rough_abs,
                    info.width,
                    info.height,
                    &transcript,
                    &edits[idx],
                    &segments,
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                )
            },
        )
        .await?;
        manifests.push(m);
    }

    // ---- persist run.json summary --------------------------------------------
    let mut run = RunState::default();
    apply_source_to_state(&mut run, &source, Some(&transcript), candidates.clone(), edits);
    run.manifests = manifests.clone();
    std::fs::create_dir_all(&cfg.workdir)?;
    std::fs::write(cfg.workdir.join("run.json"), serde_json::to_vec_pretty(&run)?)?;

    // ---- 5.10 render + 5.11 export --------------------------------------------
    if cli.no_render {
        tracing::info!("--no-render: stopping after {} manifests", manifests.len());
        return Ok(());
    }
    let summary = stages::render::render_all(&cfg, &manifests).await?;
    let mut ok_render = manifests.clone();
    ok_render.retain(|m| summary.rendered.contains(&m.clip_id));

    let export_summary = stages::export::export_all(&cfg, &cli.profile, cli.profiles.clone()).await?;
    tracing::info!(
        "export summary: {} exported, {} skipped",
        export_summary.exported.len(),
        export_summary.skipped.len()
    );
    let _ = ok_render;
    Ok(())
}

// ---- stage artifact ids ----------------------------------------------------

fn rough_id(i: usize) -> String {
    format!("rough_{i}")
}
fn edit_id(i: usize) -> String {
    format!("edit_{i}")
}
fn segments_id(i: usize) -> String {
    format!("segments_{i}")
}
fn manifests_id(i: usize) -> String {
    format!("manifest_{i}")
}

/// Async variant of `ArtifactStore::stage` for stages that are async.
async fn run_stage<T, F>(store: &ArtifactStore, id: &str, force: bool, fut: F) -> Result<(T, Freshness)>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
    F: Future<Output = Result<T>>,
{
    if !force {
        if let Some(existing) = store.read::<T>(id)? {
            tracing::info!(stage = id, "reusing existing artifact");
            return Ok((existing, Freshness::Reused));
        }
    }
    let value = fut.await.with_context(|| format!("stage `{id}` failed"))?;
    store.write(id, &value)?;
    Ok((value, Freshness::Computed))
}