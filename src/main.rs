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
use crate::types::{ClipCandidate, ContentEdit, RunState, ShotSegment, SpeechTurn, Transcript};
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
    // Cached stage artifacts belong to one specific input; never mix inputs.
    reset_if_new_input(&cfg, input)?;

    // ---- 5.1 ingest --------------------------------------------------------
    let (source, _): (SourceInfo, Freshness) =
        run_stage(&store, "ingest", cli.force, ingest(&cfg, input, cli.force)).await?;

    let manual = cli.range_from.is_some() || cli.range_to.is_some();

    // ---- 5.2 transcribe -----------------------------------------------------
    // For an explicit range the clip may have no speech at all (action
    // footage), so a failed/empty transcription is not fatal there.
    let (transcript, _) = run_stage(&store, "transcript", cli.force, async {
        match stages::transcribe::transcribe(&cfg, &source.path, &sidecar).await {
            Err(e) if manual => {
                tracing::warn!("no usable transcript ({e:#}); continuing without speech");
                Ok(Transcript { words: Vec::new() })
            }
            other => other,
        }
    })
    .await?;

    // A changed range / cut mode invalidates every clip-level artifact (they are
    // keyed by candidate index only) and the per-frame detection cache.
    let force_clip = cli.force || request_changed(&cfg, &cli)?;

    // ---- 5.2b audio events (laughter / applause / loudness peaks) -------------
    // Optional signal: any failure just means text-only clip selection.
    let audio: Option<stages::audio_events::AudioEvents> = if manual || !cfg.audio_events {
        None
    } else {
        let wav = cfg.media_dir().join("source.wav");
        match run_stage(&store, "audio_events", cli.force, stages::audio_events::analyze(&cfg, &sidecar, &wav)).await {
            Ok((a, _)) => Some(a),
            Err(e) => {
                tracing::warn!("audio analysis unavailable ({e:#}); selecting clips from the transcript only");
                None
            }
        }
    };

    // ---- 5.3 clip selection --------------------------------------------------
    let candidates: Vec<ClipCandidate> = if manual {
        vec![manual_candidate(&cli, &transcript, source.duration)?]
    } else {
        let (found, fresh) =
            run_stage(&store, "candidates", force_clip, stages::score::score(&cfg, &sidecar, &transcript, audio.as_ref())).await?;
        if found.is_empty() && matches!(fresh, Freshness::Reused) {
            // An empty cached result is a past failure, not an answer — rescore.
            tracing::info!("cached candidates were empty — rescoring");
            run_stage(&store, "candidates", true, stages::score::score(&cfg, &sidecar, &transcript, audio.as_ref())).await?.0
        } else {
            found
        }
    };
    tracing::info!("{} candidates to process", candidates.len());

    if candidates.is_empty() {
        tracing::warn!("no clip candidates found — nothing to render (try --from/--to for an exact range)");
        return Ok(());
    }

    let fps_u32 = (source.fps.round() as u32).max(1);
    let fps_f = source.fps.max(1.0);

    // ---- diarization (once) — needed by speaker resolution ------------------
    let wav = cfg.media_dir().join("source.wav");
    stages::transcribe::extract_wav(&cfg, &source.path, &wav).await?;
    let (turns, _): (Vec<SpeechTurn>, Freshness) = if cfg.assume_single_speaker || cli.single_speaker {
        (vec![], Freshness::Computed)
    } else {
        // Diarization needs a gated model + HF token. Without it, fall back to
        // face-based speaker resolution instead of failing the whole run.
        match run_stage(&store, "diarization", cli.force, stages::speaker::diarize_global(&sidecar, &wav)).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "diarization unavailable ({e:#}); continuing without it. \
                     To enable: accept https://hf.co/pyannote/speaker-diarization-3.1 and set HF_TOKEN \
                     (or pass --single-speaker to silence this)"
                );
                (vec![], Freshness::Computed)
            }
        }
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
            force_clip,
            stages::rough_cut::rough_cut(&cfg, &source.path, candidate, idx),
        )
        .await?;
        let rough_abs = rough.path.canonicalize().unwrap_or_else(|_| rough.path.clone());
        roughs.push(rough);

        let (edit, _) = run_stage(
            &store,
            &edit_id(idx),
            force_clip,
            stages::content_edit::content_edit(
                &cfg,
                &sidecar,
                &stages::content_edit::ContentEditInput {
                    candidate_idx: idx,
                    candidate,
                    transcript: &transcript,
                    scene_cuts: roughs[idx].scene_cuts.clone(),
                    source_offset: candidate.start,
                    allow_cuts: !cli.no_cuts,
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
            force_clip,
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
            force_clip,
            async {
                let mut manifest = build_manifest(
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
                )?;
                // Show captions slightly ahead of the ASR word times (clamped at 0).
                let lead = cfg.caption_lead_s;
                for w in &mut manifest.captions {
                    w.start = (w.start - lead).max(0.0);
                    w.end = (w.end - lead).max(w.start + 0.05);
                }
                Ok(manifest)
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

    // `--profiles` is either a JSON profile table or a comma-separated list of
    // platform names (e.g. instagram_reels,tiktok,shorts).
    let raw_profiles = cli.profiles.as_ref().map(|p| p.to_string_lossy().to_string());
    let (table, names): (Option<PathBuf>, Vec<String>) = match raw_profiles {
        Some(r) if std::path::Path::new(&r).is_file() => (Some(PathBuf::from(r)), vec![cli.profile.clone()]),
        Some(r) => (None, r.split(',').map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).collect()),
        None => (None, vec![cli.profile.clone()]),
    };
    let multi = names.len() > 1;
    let mut export_summary = stages::export::ExportSummary::default();
    for name in &names {
        let one = stages::export::export_all(&cfg, name, table.clone(), multi.then_some(name.as_str())).await?;
        export_summary.exported.extend(one.exported);
        export_summary.skipped.extend(one.skipped);
    }
    tracing::info!(
        "export summary: {} exported, {} skipped",
        export_summary.exported.len(),
        export_summary.skipped.len()
    );
    let _ = ok_render;
    Ok(())
}

/// Identity of the input: the URL, or a local file's path + size + mtime.
fn input_key(input: &str) -> String {
    let p = std::path::Path::new(input);
    match std::fs::metadata(p) {
        Ok(m) if p.exists() => {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let abs = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
            format!("file:{}|{}|{}", abs.display(), m.len(), mtime)
        }
        _ => format!("url:{input}"),
    }
}

/// If this workdir was last used for a different input, clear everything
/// derived from the old one (stage artifacts, media, frame cache, renders).
/// Finished exports are left alone. Records the current input either way.
fn reset_if_new_input(cfg: &Config, input: &str) -> Result<()> {
    let key = input_key(input);
    let path = cfg.workdir.join("input.txt");
    let previous = std::fs::read_to_string(&path).ok();
    let has_old_work = cfg.stage_dir().join("ingest.json").exists();
    let changed = match &previous {
        Some(p) => p.trim() != key,
        // legacy workdir with no record: its source is unknown, so don't trust it
        None => has_old_work,
    };
    if changed {
        tracing::warn!("input differs from the one this workdir was built from — clearing cached stages");
        for dir in [cfg.stage_dir(), cfg.media_dir(), cfg.frames_dir(), cfg.renders_dir(), cfg.clips_dir()] {
            let _ = std::fs::remove_dir_all(&dir);
        }
        cfg.ensure_dirs()?;
    }
    std::fs::write(&path, &key)?;
    Ok(())
}

/// Candidate for an explicit `--from/--to` range (skips the AI clip picker).
fn manual_candidate(cli: &cli::Cli, transcript: &Transcript, duration: f64) -> Result<ClipCandidate> {
    let start = cli.range_from.unwrap_or(0.0);
    let end = cli.range_to.unwrap_or(duration).min(duration);
    if start >= duration {
        anyhow::bail!("--from {start:.2}s is past the end of the source ({duration:.2}s)");
    }
    if end <= start {
        anyhow::bail!("--to ({end:.2}s) must be after --from ({start:.2}s)");
    }
    let (ws, we) = stages::content_edit::word_span(transcript, start, end);
    tracing::info!("manual range {start:.2}s–{end:.2}s ({:.1}s)", end - start);
    Ok(ClipCandidate {
        start,
        end,
        score: 1.0,
        rationale: "manual range".into(),
        word_start: Some(ws),
        word_end: Some(we),
    })
}

/// True when the clip request (range / cut mode) differs from the last run in
/// this workdir. In that case stale per-clip artifacts and cached frame
/// detections are cleared. Records the current request either way.
fn request_changed(cfg: &Config, cli: &cli::Cli) -> Result<bool> {
    // `rev` bumps when clip-stage semantics change (rev 2: frame-accurate rough cut).
    let current = format!("from={:?} to={:?} cuts={} rev=2", cli.range_from, cli.range_to, !cli.no_cuts);
    let default = "from=None to=None cuts=true rev=2";
    let path = cfg.stage_dir().join("request.txt");
    let previous = std::fs::read_to_string(&path).ok();
    let changed = match &previous {
        Some(p) => p.trim() != current,
        None => current != default,
    };
    if changed {
        tracing::info!("clip request changed ({current}) — recomputing clip stages");
        let _ = std::fs::remove_dir_all(cfg.frames_dir());
        std::fs::create_dir_all(cfg.frames_dir())?;
    }
    std::fs::write(&path, &current)?;
    Ok(changed)
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