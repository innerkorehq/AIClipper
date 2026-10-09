//! Stage 5.1 — Ingest via `yt-dlp` / local file.
//!
//! Cap source resolution to what's needed for the export (no 4K for a 1080-wide
//! vertical profile). Output: source video file + metadata.

use crate::types::{ClipCandidate, ContentEdit, RunState, Transcript};
use crate::{config::Config, util};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Absolute path to the source media file.
    pub path: PathBuf,
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub audio: bool,
}

/// Resolve `--url` or `--file` into a local source file.
pub async fn ingest(cfg: &Config, input: &str, force: bool) -> Result<SourceInfo> {
    let media = cfg.media_dir();
    if is_local(input) {
        let src = Path::new(input);
        if !src.exists() {
            anyhow::bail!("local source not found: {input}");
        }
        let dst = media.join(format!("source{}", ext_of(src)));
        if force || !dst.exists() {
            if src.canonicalize()?.as_path() == dst {
                // already inside workdir — nothing to do
            } else {
                std::fs::copy(src, &dst)
                    .with_context(|| format!("copying {}", src.display()))?;
            }
        }
        return finish_source(cfg, &dst).await;
    }

    // URL path: resolve & download with yt-dlp, resolution-capped.
    let dst = media.join("source.mp4");
    let rules = format!(
        "bv*[height<={}]+ba/b[height<={}]",
        cfg.source_height_cap, cfg.source_height_cap
    );
    let mut cmd = Command::new(&cfg.yt_dlp);
    cmd.args([
        "-f", &rules,
        "--merge-output-format", "mp4",
        "-o", dst.to_str().unwrap_or("source.mp4"),
        "-N", "4",
        "--no-playlist",
    ])
    .arg(input);
    if cfg.use_subtitles {
        // The uploader's own subtitles only (never auto-generated captions, which are
        // just someone else's ASR). A video without any simply downloads without them.
        let langs = cfg
            .sub_langs
            .split(',')
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .map(|l| format!("{l}.*"))
            .collect::<Vec<_>>()
            .join(",");
        cmd.args(["--write-subs", "--sub-langs", &langs, "--convert-subs", "srt"]);
    }
    if force || !dst.exists() {
        util::run_ok(&mut cmd, "yt-dlp").await
            .with_context(|| format!("downloading {input}. Is yt-dlp installed?"))?;
    } else {
        tracing::info!("reusing existing download {}", dst.display());
    }
    finish_source(cfg, &dst).await
}

fn is_local(input: &str) -> bool {
    Path::new(input).is_file()
}

fn ext_of(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{e}"))
        .unwrap_or_else(|| ".mp4".into())
}

async fn finish_source(cfg: &Config, dst: &Path) -> Result<SourceInfo> {
    let info = util::probe_media(&cfg.ffmpeg, dst).await?;
    if info.width == 0 || info.height == 0 {
        anyhow::bail!("could not determine resolution of {}", dst.display());
    }
    let abs = dst.canonicalize()?;
    tracing::info!(
        "source ready: {} ({}x{}, {:.1}s, {:.2}fps, audio={})",
        abs.display(),
        info.width,
        info.height,
        info.duration,
        info.fps,
        info.audio
    );
    Ok(SourceInfo {
        path: abs,
        duration: info.duration,
        width: info.width,
        height: info.height,
        fps: info.fps,
        audio: info.audio,
    })
}

/// Convenience: hydrate the on-disk `RunState` with ingest + transcript facts.
pub fn apply_source_to_state(state: &mut RunState, src: &SourceInfo, transcript: Option<&Transcript>, candidates: Vec<ClipCandidate>, edits: Vec<ContentEdit>) {
    state.source = Some(src.path.display().to_string());
    state.duration = Some(src.duration);
    state.fps = Some(src.fps as u32);
    state.width = Some(src.width);
    state.height = Some(src.height);
    if let Some(t) = transcript {
        state.transcript = Some(t.clone());
    }
    state.candidates = candidates;
    state.content_edits = edits;
}