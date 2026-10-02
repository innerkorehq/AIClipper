//! Stage 5.2 — Transcription via whisper.cpp (`whisper-cli -ojf`).
//!
//! Word-level timestamps required (not just segments) — this is the timing
//! backbone for clip boundaries, caption word-reveal, silence-gap detection,
//! and content-cut snapping.

use crate::config::Config;
use crate::sidecar::SidecarClient;
use crate::types::{Transcript, Word};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::process::Command;

const MODEL_CANDIDATES: &[&str] = &[
    "models/ggml-large-v3-turbo.bin",
    "models/ggml-medium.bin",
    "models/ggml-small.en.bin",
    "models/ggml-base.en.bin",
];

/// Extract mono 16 kHz WAV (needed both by whisper and pyannote diarization).
pub async fn extract_wav(cfg: &Config, source: &Path, out: &Path) -> Result<()> {
    if out.exists() {
        return Ok(());
    }
    let mut cmd = Command::new(&cfg.ffmpeg);
    cmd.args(["-y", "-i"])
        .arg(source.as_os_str())
        .args([
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        ])
        .arg(out.as_os_str());
    crate::util::run_ok(&mut cmd, "ffmpeg wav extraction")
        .await
        .with_context(|| format!("extracting wav from {}", source.display()))?;
    Ok(())
}

/// Resolve a whisper model path: try configured name in a few well-known roots,
/// including models downloaded through the HuggingFace hub cache.
fn resolve_model(cfg: &Config) -> Result<PathBuf> {
    // Absolute paths (or `AICLIPPER_WHISPER_MODEL=/abs/ggml-*.bin`) win outright.
    let direct = PathBuf::from(&cfg.whisper_model);
    if direct.is_absolute() && direct.exists() {
        return Ok(direct);
    }
    let roots: Vec<PathBuf> = vec![
        PathBuf::from("."),
        home_dir().join(".cache/whisper"),
        home_dir().join(".local/share/whisper.cpp"),
        home_dir().join(".whisper"),
    ];
    for r in &roots {
        let c = r.join(&cfg.whisper_model);
        if c.exists() {
            return Ok(c);
        }
    }
    // HuggingFace hub cache: ~/.cache/huggingface/hub/models--*/snapshots/*/<name>
    if let Some(p) = find_in_hf_cache(&cfg.whisper_model) {
        return Ok(p);
    }
    // Also allow whisper-cli's convention of downloading to ./models.
    for name in MODEL_CANDIDATES {
        let p = PathBuf::from(name);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!(
        "whisper model `{}` not found (looked in {:?}). Download one, e.g.\n  \
         curl -L -o ~/.cache/whisper/{} https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        cfg.whisper_model,
        roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
        cfg.whisper_model,
        cfg.whisper_model,
    )
}

pub async fn transcribe(cfg: &Config, source: &Path, client: &SidecarClient) -> Result<Transcript> {
    let wav = cfg.media_dir().join("source.wav");
    extract_wav(cfg, source, &wav).await?;

    if cfg.whisper_backend == "hf" {
        tracing::info!(
            "transcribing with transformers model `{}` (Hinglish/word-level)",
            cfg.whisper_hf_model
        );
        return client.transcribe_hf(&wav, &cfg.whisper_hf_model).await;
    }

    let model = resolve_model(cfg)?;
    let out_base = cfg.media_dir().join("transcript");
    let json_path = out_base.with_extension("json");
    if json_path.exists() {
        return parse_whisper_json(&json_path);
    }

    tracing::info!("transcribing with {} (word-level)", model.display());
    let model = model.to_string_lossy();
    let wav = wav.to_string_lossy();
    let out_base = out_base.to_string_lossy();
    let mut cmd = Command::new(&cfg.whisper);
    cmd.arg("-m").arg(model.as_ref());
    cmd.arg("-f").arg(wav.as_ref());
    cmd.arg("-ojf");
    cmd.arg("-sow"); // split on words → one word per segment (word-level timing)
    cmd.arg("-of").arg(out_base.as_ref());
    cmd.arg("-t").arg(cfg.whisper_threads.to_string());
    cmd.arg("-l").arg("auto");
    cmd.arg("-np");
    crate::util::run_checked(&mut cmd, "whisper.cpp").await?;
    parse_whisper_json(&json_path)
}

/// whisper.cpp special tokens look like `[_BEG_]`, `[_TT_103]`, or `<|...|>`.
fn is_special_token(text: &str) -> bool {
    let t = text.trim();
    (t.starts_with('[') && t.ends_with(']')) || (t.starts_with("<|") && t.ends_with("|>"))
}

pub fn parse_whisper_json(path: &Path) -> Result<Transcript> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading whisper output {}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    let mut words = Vec::new();
    if let Some(segs) = v["transcription"].as_array() {
        for seg in segs {
            let start = seg["offsets"]["from"].as_f64().unwrap_or(0.0) / 1000.0;
            let end = seg["offsets"]["to"].as_f64().unwrap_or(start * 1000.0) / 1000.0;
            // whisper.cpp `-sow` yields one word per segment (leading space padded).
            let text = seg["text"].as_str().unwrap_or("").trim().to_string();
            if !text.is_empty() && !is_special_token(&text) {
                words.push(Word { text, start, end });
            }
        }
    }
    if words.is_empty() {
        anyhow::bail!("whisper produced no words — check source duration and model");
    }
    tracing::info!("transcript: {} words over {:.1}s", words.len(), words.last().unwrap().end);
    Ok(Transcript { words })
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

/// Look for `name` inside the HuggingFace hub cache, across all model repos:
/// `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>/<name>`.
fn find_in_hf_cache(name: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".cache/huggingface"));
    let hub = if hub.ends_with("hub") { hub } else { hub.join("hub") };
    let entries = std::fs::read_dir(&hub).ok()?;
    for entry in entries.flatten() {
        let snapshots = entry.path().join("snapshots");
        if !snapshots.is_dir() {
            continue;
        }
        let Ok(revisions) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        for rev in revisions.flatten() {
            let candidate = rev.path().join(name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}