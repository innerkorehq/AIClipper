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

    // Frame-accurate cut. A stream copy (`-c copy`) can only start on a keyframe,
    // which in long-GOP sources (YouTube VP9/H.264) is seconds before the clip
    // start — the rough cut then no longer lines up with the transcript, the
    // captions or the face/shot analysis, all of which assume t=0 is the clip
    // start. Re-encoding decodes up to the exact frame instead.
    let mut cmd = Command::new(&cfg.ffmpeg);
    cmd.arg("-y")
        .arg("-ss")
        .arg(format!("{start:.3}"))
        .arg("-i")
        .arg(source.as_os_str())
        .arg("-t")
        .arg(format!("{dur:.3}"))
        // Short GOP (-g 12): the renderer seeks around this file constantly, and with a
        // long GOP every seek in a 4K stream decodes seconds of video first.
        .args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"])
        .args(["-g", "12", "-keyint_min", "12"])
        .args(["-c:a", "aac", "-b:a", "192k"])
        .args(["-avoid_negative_ts", "make_zero", "-movflags", "+faststart"])
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
            // NB: no `-vsync vfr` here — with current ffmpeg builds it makes the
            // select/showinfo chain emit nothing, silently disabling cut detection.
            "-an",
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