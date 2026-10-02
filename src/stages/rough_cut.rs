//! Stage 5.4 — Rough cut via ffmpeg stream-copy.
//!
//! Isolates each candidate's time range from the source into its own file, no
//! re-encode (PRD §5.4). Keeps later stages (Remotion, Vision) operating on
//! tight per-clip files instead of seeking into the full-length source repeatedly.

use crate::config::Config;
use crate::types::ClipCandidate;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::process::Command;

/// Result of a rough cut: the segment file and scene-cut timestamps within it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoughCut {
    pub path: PathBuf,
    pub start: f64,
    pub end: f64,
    /// Absolute in-source timestamps of detected scene boundaries.
    pub scene_cuts: Vec<f64>,
}

/// Stream-copy a candidate's segment from the source.
pub async fn rough_cut(
    cfg: &Config,
    source: &Path,
    candidate: &ClipCandidate,
    index: usize,
) -> Result<RoughCut> {
    let out = cfg.media_dir().join(format!("rough_{index:03}.mp4"));
    let start = candidate.start;
    let dur = candidate.end - start;

    let mut cmd = Command::new(&cfg.ffmpeg);
    cmd.arg("-y")
        .arg("-ss")
        .arg(start.to_string())
        .arg("-i")
        .arg(source.as_os_str())
        .arg("-t")
        .arg(dur.to_string())
        .arg("-c")
        .arg("copy")
        .arg("-avoid_negative_ts")
        .arg("make_zero")
        .arg(out.as_os_str());

    crate::util::run_ok(&mut cmd, &format!("ffmpeg rough cut #{index}"))
        .await
        .with_context(|| {
            format!(
                "failed to rough cut candidate #{index} ({:.1}s–{:.1}s)",
                start, candidate.end
            )
        })?;

    // Scene detection pre-pass (PRD §5.5).  This finds potential multi-shot
    // boundaries inside the rough-cut segment.
    let scene_cuts = detect_scenes(cfg, &out, cfg.scene_threshold).await?;
    tracing::info!(
        "rough cut #{index}: {} ({:.1}s–{:.1}s, {} scene cuts)",
        out.display(),
        start,
        candidate.end,
        scene_cuts.len(),
    );
    Ok(RoughCut { path: out, start, end: candidate.end, scene_cuts })
}

/// Run ffmpeg scene detection (`select='gt(scene,THRESH)'` + showinfo) and
/// extract the `pts_time` of each detected boundary inside `input`.
async fn detect_scenes(cfg: &Config, input: &Path, threshold: f64) -> Result<Vec<f64>> {
    let vf = format!("select='gt(scene,{threshold})',showinfo");
    let out = Command::new(&cfg.ffmpeg)
        .args([
            "-i",
            input.to_str().unwrap_or(""),
            "-vf",
            &vf,
            "-vsync",
            "vfr",
            "-f",
            "null",
            "-",
        ])
        .output()
        .await
        .context("scene detection ffmpeg")?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let mut cuts = Vec::new();
    for line in stderr.lines() {
        if let Some(seg) = line.split("pts_time:").nth(1) {
            let ts: f64 = seg.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(-1.0);
            if ts >= 0.0 {
                cuts.push(ts);
            }
        }
    }
    Ok(cuts)
}