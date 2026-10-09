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
    // Title / description / chapters: context the clip picker can use (who is in
    // the video, what each stretch is about).
    cmd.arg("--write-info-json");
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
    ensure_info_json(cfg, input).await;
    finish_source(cfg, &dst).await
}

/// Make sure `media/source.info.json` exists for a URL source (a reused download
/// may predate it). Metadata only — nothing is downloaded. Failure is not fatal:
/// the context is a nice-to-have.
pub async fn ensure_info_json(cfg: &Config, url: &str) {
    if Path::new(url).is_file() {
        return; // a local file has no URL metadata
    }
    let info = cfg.media_dir().join("source.info.json");
    if info.exists() {
        return;
    }
    let mut cmd = Command::new(&cfg.yt_dlp);
    cmd.args(["--skip-download", "--write-info-json", "--no-playlist", "-o"])
        .arg(cfg.media_dir().join("source.%(ext)s"))
        .arg(url);
    if let Err(e) = util::run_ok(&mut cmd, "yt-dlp metadata").await {
        tracing::warn!("could not fetch video metadata ({e:#}); continuing without it");
    }
}

/// What the video says about itself — title, channel, a trimmed description and
/// chapter list — as plain text for the model (empty when unavailable). It tells
/// the model who is in the video and what each stretch is about, which the
/// transcript alone (no speaker labels, often garbled) cannot.
pub fn video_context(cfg: &Config) -> String {
    let path = cfg.media_dir().join("source.info.json");
    let Ok(raw) = std::fs::read(&path) else { return String::new() };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&raw) else { return String::new() };
    build_context(&v)
}

fn build_context(v: &serde_json::Value) -> String {
    let get = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
    let mut out = String::new();
    let title = get("title");
    if !title.is_empty() {
        out.push_str(&format!("Title: {title}\n"));
    }
    let who = if !get("channel").is_empty() { get("channel") } else { get("uploader") };
    if !who.is_empty() {
        out.push_str(&format!("Channel: {who}\n"));
    }
    // description: keep prose, drop links, hashtags and separator lines
    let desc: Vec<String> = get("description")
        .lines()
        .map(|l| l.trim())
        .filter(|l| {
            !l.is_empty()
                && !l.contains("http")
                && !l.starts_with('#')
                && l.chars().filter(|c| c.is_alphabetic()).count() >= 8
        })
        .map(|l| l.to_string())
        .collect();
    let mut desc_text = desc.join(" ");
    if desc_text.chars().count() > 600 {
        desc_text = desc_text.chars().take(600).collect::<String>() + "…";
    }
    if !desc_text.is_empty() {
        out.push_str(&format!("Description: {desc_text}\n"));
    }
    if let Some(ch) = v.get("chapters").and_then(|c| c.as_array()) {
        let lines: Vec<String> = ch
            .iter()
            .take(30)
            .filter_map(|c| {
                let t = c.get("start_time")?.as_f64()?;
                let title = c.get("title")?.as_str()?.trim();
                Some(format!("  {} {title}", crate::stages::requests::fmt_clock(t)))
            })
            .collect();
        if !lines.is_empty() {
            out.push_str("Chapters (start time, title):\n");
            out.push_str(&lines.join("\n"));
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_keeps_prose_and_chapters_and_drops_links() {
        let v = serde_json::json!({
            "title": "I Regret Inviting Ravi Gupta on My Podcast!!",
            "uploader": "Prakhar Gupta",
            "description": "Episode 88 features Ravi Gupta. Ravi is a stand-up comedian.\n\n#Comedy #RaviGupta\n------------------\nListen on Spotify: https://open.spotify.com/x\nshort",
            "chapters": [{"start_time": 0.0, "title": "The Legend is back"}, {"start_time": 797.0, "title": "Hilarious bit on the singer - Kaka"}]
        });
        let c = build_context(&v);
        assert!(c.contains("Title: I Regret Inviting Ravi Gupta"));
        assert!(c.contains("Channel: Prakhar Gupta"));
        assert!(c.contains("Ravi is a stand-up comedian"));
        assert!(!c.contains("http") && !c.contains("#Comedy") && !c.contains("-----"));
        assert!(c.contains("0:00 The Legend is back") && c.contains("13:17 Hilarious bit on the singer - Kaka"));
        assert_eq!(build_context(&serde_json::json!({})), "");
    }
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