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
    // `transcript_words` (not `transcript`): older runs cached 30s-segment output
    // under the previous name, which must not be reused as word-level data.
    let out_base = cfg.media_dir().join("transcript_words");
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
    // `-ml 1 -sow`: one word per segment (word-level timing). `-sow` alone does
    // nothing — whisper keeps its ~30s segments.
    cmd.arg("-ml").arg("1");
    cmd.arg("-sow");
    // No context carry-over: stops the repetition loops small models fall into.
    cmd.arg("-mc").arg("0");
    cmd.arg("-of").arg(out_base.as_ref());
    cmd.arg("-t").arg(cfg.whisper_threads.to_string());
    cmd.arg("-l").arg(&cfg.whisper_lang);
    cmd.arg("-np");
    crate::util::run_checked(&mut cmd, "whisper.cpp").await?;
    parse_whisper_json(&json_path)
}

/// whisper.cpp special tokens look like `[_BEG_]`, `[_TT_103]`, or `<|...|>`.
fn is_special_token(text: &str) -> bool {
    let t = text.trim();
    (t.starts_with('[') && t.ends_with(']')) || (t.starts_with("<|") && t.ends_with("|>"))
}

/// Words longer than this are whisper hallucinations (e.g. one "word" spanning
/// 30s of silence), not speech.
const MAX_WORD_SECONDS: f64 = 8.0;
/// A character repeated this many times in a row is a decoding loop.
const MAX_CHAR_RUN: usize = 6;

/// whisper.cpp splits tokens inside multi-byte characters (Devanagari, emoji…),
/// so its JSON can contain invalid UTF-8 until neighbouring tokens are joined.
/// Map each invalid byte to a private-use char so the JSON parses, then undo it.
const PUA_BASE: u32 = 0xE000;

fn sanitize_utf8(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut rest = bytes;
    loop {
        match std::str::from_utf8(rest) {
            Ok(ok) => {
                out.push_str(ok);
                return out;
            }
            Err(e) => {
                let (good, bad) = rest.split_at(e.valid_up_to());
                out.push_str(std::str::from_utf8(good).unwrap_or_default());
                let n = e.error_len().unwrap_or(bad.len()).max(1);
                for b in &bad[..n] {
                    out.push(char::from_u32(PUA_BASE + *b as u32).unwrap_or('\u{FFFD}'));
                }
                rest = &bad[n..];
            }
        }
    }
}

/// Inverse of `sanitize_utf8`: private-use chars back to their raw bytes.
fn to_raw_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        let c = ch as u32;
        if (PUA_BASE..PUA_BASE + 256).contains(&c) {
            out.push((c - PUA_BASE) as u8);
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}

fn has_char_run(text: &str, n: usize) -> bool {
    let mut prev = None;
    let mut run = 0usize;
    for ch in text.chars() {
        if Some(ch) == prev {
            run += 1;
            if run >= n {
                return true;
            }
        } else {
            prev = Some(ch);
            run = 1;
        }
    }
    false
}

/// A phrase (1–8 words) repeated back-to-back this many times is a decoding
/// loop — whisper's habit on music/silence — not speech.
const MIN_LOOP_REPEATS: usize = 3;
const MAX_LOOP_PHRASE: usize = 8;

/// Remove every word that sits inside a phrase loop (all repetitions, since the
/// "first" occurrence of a hallucination is just as fake).
fn drop_phrase_loops(words: &mut Vec<Word>) {
    let key = |w: &Word| w.text.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    let toks: Vec<String> = words.iter().map(key).collect();
    let n = toks.len();
    let mut in_loop = vec![false; n];
    for len in 1..=MAX_LOOP_PHRASE {
        let mut i = 0;
        while i + len * MIN_LOOP_REPEATS <= n {
            let mut reps = 1;
            while i + (reps + 1) * len <= n && toks[i + reps * len..i + (reps + 1) * len] == toks[i..i + len] {
                reps += 1;
            }
            if reps >= MIN_LOOP_REPEATS && toks[i..i + len].iter().any(|t| !t.is_empty()) {
                in_loop[i..i + reps * len].iter_mut().for_each(|f| *f = true);
                i += reps * len;
            } else {
                i += 1;
            }
        }
    }
    let mut idx = 0;
    words.retain(|_| {
        let keep = !in_loop[idx];
        idx += 1;
        keep
    });
}

pub fn parse_whisper_json(path: &Path) -> Result<Transcript> {
    let raw = std::fs::read(path).with_context(|| format!("reading whisper output {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&sanitize_utf8(&raw))
        .with_context(|| format!("parsing {}", path.display()))?;

    // (start, end, raw text bytes) per whisper segment.
    let mut segs: Vec<(f64, f64, Vec<u8>)> = Vec::new();
    if let Some(arr) = v["transcription"].as_array() {
        for seg in arr {
            let start = seg["offsets"]["from"].as_f64().unwrap_or(0.0) / 1000.0;
            let end = seg["offsets"]["to"].as_f64().unwrap_or(start * 1000.0) / 1000.0;
            segs.push((start, end, to_raw_bytes(seg["text"].as_str().unwrap_or(""))));
        }
    }

    let mut words = Vec::new();
    let mut dropped = 0usize;
    let mut i = 0;
    while i < segs.len() {
        let (start, mut end, mut bytes) = segs[i].clone();
        // Rejoin a character that whisper split across consecutive segments.
        let mut joined = 0;
        while std::str::from_utf8(&bytes).is_err() && i + 1 < segs.len() && joined < 3 {
            i += 1;
            joined += 1;
            bytes.extend_from_slice(&segs[i].2);
            end = segs[i].1;
        }
        i += 1;

        let text = String::from_utf8_lossy(&bytes).trim().to_string();
        if text.is_empty() || is_special_token(&text) {
            continue;
        }
        if end - start > MAX_WORD_SECONDS || has_char_run(&text, MAX_CHAR_RUN) {
            dropped += 1;
            continue;
        }
        // Zero-length words would be dropped downstream; give them a sliver.
        let end = if end > start { end } else { start + 0.1 };
        words.push(Word { text, start, end });
    }
    let before = words.len();
    drop_phrase_loops(&mut words);
    dropped += before - words.len();
    if dropped > 0 {
        tracing::warn!("dropped {dropped} hallucinated words (over {MAX_WORD_SECONDS}s, repeated characters or phrase loops)");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejoins_characters_split_across_segments() {
        // "है" = E0 A4 B9 E0 A5 88; split mid-character across two segments.
        let full = "है".as_bytes();
        let (a, b) = (&full[..2], &full[2..]);
        let mut json = br#"{"transcription":[{"offsets":{"from":0,"to":100},"text":" "#.to_vec();
        json.extend_from_slice(a);
        json.extend_from_slice(br#""},{"offsets":{"from":100,"to":300},"text":""#);
        json.extend_from_slice(b);
        json.extend_from_slice(br#""},{"offsets":{"from":300,"to":500},"text":" ok"}]}"#);
        let dir = std::env::temp_dir().join("aiclipper_whisper_test.json");
        std::fs::write(&dir, &json).unwrap();
        let t = parse_whisper_json(&dir).unwrap();
        assert_eq!(t.words.len(), 2);
        assert_eq!(t.words[0].text, "है");
        assert_eq!((t.words[0].start, t.words[0].end), (0.0, 0.3));
        assert_eq!(t.words[1].text, "ok");
    }

    #[test]
    fn drops_hallucinations_and_fixes_zero_length() {
        let json = br#"{"transcription":[
            {"offsets":{"from":0,"to":400},"text":" hi"},
            {"offsets":{"from":400,"to":30000},"text":" long"},
            {"offsets":{"from":30000,"to":30100},"text":" aaaaaaaa"},
            {"offsets":{"from":31000,"to":31000},"text":" end"},
            {"offsets":{"from":0,"to":0},"text":"[_BEG_]"}]}"#;
        let p = std::env::temp_dir().join("aiclipper_whisper_test2.json");
        std::fs::write(&p, json).unwrap();
        let t = parse_whisper_json(&p).unwrap();
        let texts: Vec<_> = t.words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["hi", "end"]);
        assert!(t.words[1].end > t.words[1].start);
    }

    #[test]
    fn drops_phrase_loops_but_keeps_normal_repeats() {
        let mk = |t: &str, i: usize| Word { text: t.into(), start: i as f64, end: i as f64 + 0.5 };
        let mut v: Vec<Word> = ["so", "yeh", "capsule", "lita", "yeh", "capsule", "lita", "yeh", "capsule", "lita", "bye"]
            .iter()
            .enumerate()
            .map(|(i, t)| mk(t, i))
            .collect();
        drop_phrase_loops(&mut v);
        let t: Vec<_> = v.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(t, ["so", "bye"]);
        // natural emphasis ("no no") and a twice-repeated phrase survive
        let mut ok: Vec<Word> = ["no", "no", "yes", "go", "yes", "go"].iter().enumerate().map(|(i, t)| mk(t, i)).collect();
        drop_phrase_loops(&mut ok);
        assert_eq!(ok.len(), 6);
    }
}
