//! Stage 5.11 — platform export.
//!
//! Encodes each Remotion-composed clip to the target platform profile in
//! `config/` (defaults == standard 1080×1920@30 profiles) with compliance
//! args: Main profile, 4:2:0, +cgop, loudness-normalized audio (-18 LUFS).

use crate::config::{Config, ExportProfile, ProfileTable};
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::PathBuf;
use tokio::process::Command;

#[derive(Debug, Serialize, Default)]
pub struct ExportSummary {
    pub exported: Vec<String>,
    pub skipped: Vec<String>,
}

/// Fallback profile for an unknown platform name (additive platforms, §5.11).
fn fallback_profile(name: &str) -> ExportProfile {
    ExportProfile {
        name: name.to_string(),
        width: 1080,
        height: 1920,
        fps: 30,
        video_bitrate_mbps: 25,
        audio_bitrate_kbps: 128,
        extra_args: vec![
            "-flags".into(),
            "+cgop".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-profile:v".into(),
            "main".into(),
        ],
    }
}

/// Encode every rendered clip in `renders/out` into `exports/` under the
/// given profile. Clips whose raw render is missing are listed as skipped.
pub async fn export_all(
    cfg: &Config,
    profile_name: &str,
    profiles_opt: Option<PathBuf>,
    out_subdir: Option<&str>,
) -> Result<ExportSummary> {
    let profiles = ProfileTable::load(profiles_opt.as_deref())?;
    let profile = profiles
        .iter()
        .find(|p| p.name == profile_name)
        .cloned()
        .unwrap_or_else(|| fallback_profile(profile_name));

    let rendered_dir = cfg.renders_dir().join("out");
    // Several platforms → one folder per platform so outputs don't collide.
    let out_dir = match out_subdir {
        Some(d) => cfg.exports_dir().join(d),
        None => cfg.exports_dir(),
    };
    std::fs::create_dir_all(&out_dir)?;

    let mut summary = ExportSummary::default();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&rendered_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "mp4").unwrap_or(false))
        .collect();
    entries.sort();

    let width = profile.width;
    let height = profile.height;
    let fps = profile.fps;
    let vbr = format!("{}k", profile.video_bitrate_mbps * 1000);
    let abr = format!("{}k", profile.audio_bitrate_kbps);

    for entry in entries {
        let clip_id = entry
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("clip")
            .to_string();
        let out_path = out_dir.join(format!("{clip_id}.mp4"));

        let mut cmd = Command::new(&cfg.ffmpeg);
        cmd.arg("-y")
            .arg("-i")
            .arg(&entry)
            .arg("-map")
            .arg("0:v:0")
            // `?` tolerates a video-only render instead of hard-failing.
            .arg("-map")
            .arg("0:a:0?")
            .args(&profile.extra_args)
            .arg("-vf")
            .arg(format!(
                "scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
            ))
            .arg("-r")
            .arg(fps.to_string())
            .arg("-b:v")
            .arg(&vbr)
            .arg("-b:a")
            .arg(&abr)
            .arg("-af")
            .arg("loudnorm=I=-18:TP=-1.5:LRA=7")
            .arg("-movflags")
            .arg("+faststart")
            .arg("-c:v")
            .arg("libx264")
            .arg("-preset")
            .arg("medium")
            .arg("-c:a")
            .arg("aac")
            .arg(&out_path);
        crate::util::run_ok(&mut cmd, &format!("export {clip_id}"))
            .await
            .with_context(|| format!("export of {clip_id} failed"))?;
        summary.exported.push(clip_id);
    }

    tracing::info!("exported {} clips to {}", summary.exported.len(), out_dir.display());
    Ok(summary)
}