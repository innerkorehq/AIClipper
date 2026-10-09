//! Stage 5.10 — Remotion render.
//!
//! Writes each clip's RenderManifest to JSON, then drives a single warm Node
//! process (`remotion/src/render-worker.ts`) which bundles once, opens
//! headless Chromium once, and renders all clips reusing the browser instance
//! (PRD §4). One failed clip never aborts the run (PRD §9).

use crate::config::Config;
use crate::types::RenderManifest;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;
use tokio::process::Command;

#[derive(Debug, Default, Deserialize)]
pub struct RenderSummary {
    pub rendered: Vec<String>,
    pub failed: Vec<RenderFailure>,
}

#[derive(Debug, Deserialize)]
pub struct RenderFailure {
    pub clip_id: String,
    pub error: String,
}

/// Render every manifest in `renders` to `out`. Returns the summary.
pub async fn render_all(
    cfg: &Config,
    manifests: &[RenderManifest],
) -> Result<RenderSummary> {
    let renders = cfg.renders_dir();
    let out = cfg.renders_dir().join("out");
    std::fs::create_dir_all(&out)?;
    // The worker runs with cwd = remotion/, so every path we hand it must be
    // absolute or it will resolve against the wrong directory.
    let renders_abs = std::fs::canonicalize(&renders).unwrap_or_else(|_| renders.clone());
    let out_abs = std::fs::canonicalize(&out).unwrap_or_else(|_| out.clone());
    let remotion_dir = std::fs::canonicalize(&cfg.remotion_dir).unwrap_or_else(|_| cfg.remotion_dir.clone());

    // Remotion (running in Chromium) can only load assets served from its
    // `public/` folder — absolute filesystem paths render black. Stage each
    // clip's rough-cut source there and reference it via `staticFile()`.
    let public_src = remotion_dir.join("public").join("_sources");
    std::fs::create_dir_all(&public_src)
        .with_context(|| format!("creating {}", public_src.display()))?;

    for m in manifests {
        let mut m = m.clone();
        let ext = Path::new(&m.source)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("mp4");
        let staged_name = format!("{}.{}", m.clip_id, ext);
        let src_path: String = if m.source_abs.is_empty() { m.source.clone() } else { m.source_abs.clone() };
        std::fs::copy(&src_path, public_src.join(&staged_name))
            .with_context(|| format!("staging source {src_path} for Remotion"))?;
        m.source_abs = format!("_sources/{staged_name}");

        // Tiny silent proxy for the blurred background layer: decoding the full
        // (often 4K) source a second/third time per frame only to blur it away is
        // what stalls the renderer.
        let blur_name = format!("{}_blur.mp4", m.clip_id);
        let mut blur = Command::new(&cfg.ffmpeg);
        blur.arg("-y")
            .arg("-i")
            .arg(&src_path)
            .args(["-vf", "scale=-2:360", "-an"])
            .args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p"])
            .args(["-g", "12", "-keyint_min", "12", "-movflags", "+faststart"])
            .arg(public_src.join(&blur_name));
        crate::util::run_ok(&mut blur, &format!("blur proxy for {}", m.clip_id)).await?;
        m.blur_src = format!("_sources/{blur_name}");

        let path = renders.join(format!("manifest_{}.json", m.clip_id));
        let bytes = serde_json::to_vec_pretty(&m).context("serializing render manifest")?;
        std::fs::write(&path, bytes)
            .with_context(|| format!("writing {}", path.display()))?;
    }

    let entry = remotion_dir.join("src").join("render-worker.ts");
    let mut cmd = Command::new("npx");
    cmd.current_dir(&remotion_dir)
        .args(["tsx", entry.to_str().expect("utf8 remotion entry")])
        .arg("--manifests-dir")
        .arg(&renders_abs)
        .arg("--out-dir")
        .arg(&out_abs);
    crate::util::run_ok(&mut cmd, "Remotion render worker")
        .await
        .with_context(|| {
            format!(
                "Remotion render failed. Ensure `cd {} && npm install` ran.",
                cfg.remotion_dir.display()
            )
        })?;
    let summary_path = out_abs.join("render-summary.json");
    let summary_bytes = std::fs::read(&summary_path)
        .with_context(|| format!("reading render summary {}", summary_path.display()))?;
    let summary: RenderSummary =
        serde_json::from_slice(&summary_bytes).context("parsing render summary")?;
    tracing::info!(
        "rendered {} clips ({} failed)",
        summary.rendered.len(),
        summary.failed.len()
    );
    for f in &summary.failed {
        tracing::warn!("clip {} failed to render: {}", f.clip_id, f.error);
    }
    Ok(summary)
}

/// Locate the composed clip file produced by Remotion.
#[allow(dead_code)]
pub fn clip_render_path(cfg: &Config, clip_id: &str, existing: Option<&Path>) -> Option<std::path::PathBuf> {
    if let Some(p) = existing {
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    let p = cfg.renders_dir().join("out").join(format!("{clip_id}.mp4"));
    p.exists().then_some(p)
}