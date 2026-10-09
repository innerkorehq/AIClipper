//! HTTP client for the persistent Python ML sidecar (Gemma MLX + pyannote).
//!
//! The sidecar is a long-running process hit over localhost HTTP — both Gemma
//! calls (clip scoring; content-edit + shot-list) and diarization go through it
//! to avoid Python/model cold-start cost per invocation (PRD §4).

use crate::config::Config;
use crate::types::{ClipCandidate, SpeechTurn, Transcript, Word};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::Path;
use std::time::Duration;

#[derive(Clone)]
pub struct SidecarClient {
    pub base_url: String,
    http: reqwest::Client,
    llm_log: std::path::PathBuf,
}

// ---- request/response shapes -------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct ScoreClipsResp {
    pub candidates: Vec<ClipCandidate>,
    /// Full raw LLM output (audit trail).
    pub llm_output: String,
    pub prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SelectClipsResp {
    pub candidates: Vec<ClipCandidate>,
    /// Best relevance (0..1) seen per request, kept or not — explains "no match".
    pub best: Vec<f64>,
    pub llm_output: String,
    pub prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContentEditResp {
    pub cut_ranges: Vec<RawCutRange>,
    pub shot_list: Vec<RawShotEntry>,
    pub rationale: String,
    pub llm_output: String,
    pub prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RawCutRange {
    pub start_word_idx: u32,
    pub end_word_idx: u32,
    /// "filler" | "redundant" | "tangent" | "false_start"
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RawShotEntry {
    pub start: f64,
    pub end: f64,
    pub frame: String,
    #[serde(default)]
    pub target_track_id: Option<String>,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiarizeResp {
    pub turns: Vec<RawTurn>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RawTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct HealthResp {
    pub ok: bool,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub gemma_loaded: bool,
    #[serde(default)]
    pub diarization_available: bool,
    #[serde(default)]
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TranscribeResp {
    pub ok: bool,
    pub words: Vec<Word>,
}

impl SidecarClient {
    pub fn new(cfg: &Config, base_url: impl Into<String>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(3600)) // chunked clip scoring makes many LLM calls
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("build sidecar http client");
        Self { base_url: base_url.into(), http, llm_log: cfg.llm_log_path.clone() }
    }

    // ---- lifecycle helpers -------------------------------------------------

    pub async fn ping(&self) -> bool {
        self.http_get("/v1/health")
            .await
            .ok()
            .and_then(|v| serde_json::from_value::<HealthResp>(v).ok())
            .map(|h| h.ok)
            .unwrap_or(false)
    }

    /// Wait up to `timeout` for the sidecar to answer health.
    pub async fn wait_ready(&self, timeout: Duration, what: &str) -> Result<()> {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if self.ping().await {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        anyhow::bail!(
            "sidecar at {} not ready ({what}); start it with `python3 -m sidecar` in python_sidecar/",
            self.base_url
        )
    }

    /// Perform a Gemma clip-scoring call (stage 5.3).
    pub async fn score_clips(
        &self,
        words: &[Word],
        cfg: &Config,
        extra: &[(f64, f64, String)],
    ) -> Result<ScoreClipsResp> {
        let body = json!({
            "transcript": { "words": words },
            "model": cfg.llm_model,
            "max_clips": cfg.top_n,
            "min_duration": cfg.min_clip_duration,
            "max_duration": cfg.max_clip_duration,
            "max_tokens": cfg.llm_max_tokens,
            "extra_candidates": extra.iter().map(|(s, e, src)| json!({"start": s, "end": e, "source": src})).collect::<Vec<_>>(),
        });
        let out: ScoreClipsResp = self.call("/v1/score_clips", body.clone(), "score_clips").await?;
        crate::util::log_llm_call(&self.llm_log, "score_clips", &body, &serde_json::to_value(&out).unwrap_or_default());
        Ok(out)
    }

    /// Perform the combined content-edit + shot-list call (stage 5.5).
    pub async fn content_edit(
        &self,
        candidate: &ClipCandidate,
        words: &[Word],
        scene_cuts: &[f64],
        cfg: &Config,
    ) -> Result<ContentEditResp> {
        let body = json!({
            "candidate": candidate,
            "words": words,
            "candidate_rationale": candidate.rationale,
            "scene_cuts": scene_cuts,
            "model": cfg.llm_model,
            "max_cut_fraction": cfg.max_cut_fraction,
            "sample_interval": cfg.sample_interval,
            "max_tokens": cfg.llm_max_tokens,
        });
        let out: ContentEditResp = self.call("/v1/content_edit", body.clone(), "content_edit").await?;
        crate::util::log_llm_call(&self.llm_log, "content_edit", &body, &serde_json::to_value(&out).unwrap_or_default());
        Ok(out)
    }

    /// Diarize a WAV file (pyannote.audio).
    pub async fn diarize(&self, audio_wav: &Path) -> Result<Vec<SpeechTurn>> {
        let wav_abs = audio_wav
            .canonicalize()
            .unwrap_or_else(|_| audio_wav.to_path_buf());
        let body = json!({ "audio": wav_abs.display().to_string() });
        let resp = self.call("/v1/diarize", body.clone(), "diarize").await?;
        let out: DiarizeResp = resp;
        crate::util::log_llm_call(&self.llm_log, "diarize", &body, &serde_json::to_value(&out).unwrap_or_default());
        Ok(out
            .turns
            .into_iter()
            .map(|t| SpeechTurn { start: t.start, end: t.end, speaker: t.speaker })
            .collect())
    }

    /// Transcribe a WAV via the HuggingFace (transformers) backend — used for
    /// Hinglish ASR models that have no ggml/whisper.cpp build.
    pub async fn transcribe_hf(&self, audio_wav: &Path, model: &str) -> Result<Transcript> {
        let wav_abs = audio_wav
            .canonicalize()
            .unwrap_or_else(|_| audio_wav.to_path_buf());
        let body = json!({ "audio": wav_abs.display().to_string(), "model": model });
        let out: TranscribeResp = self.call("/v1/transcribe", body.clone(), "transcribe_hf").await?;
        crate::util::log_llm_call(
            &self.llm_log,
            "transcribe_hf",
            &body,
            &serde_json::to_value(&out).unwrap_or_default(),
        );
        if !out.ok {
            anyhow::bail!("sidecar transcribe_hf returned ok=false");
        }
        if out.words.is_empty() {
            anyhow::bail!("sidecar transcribe_hf produced no words");
        }
        let last_end = out.words.last().map(|w| w.end).unwrap_or(0.0);
        tracing::info!(
            "transcript (hf): {} words over {:.1}s",
            out.words.len(),
            last_end
        );
        Ok(Transcript { words: out.words })
    }

    /// Find clips for client requests (one proposal pass covers all requests).
    /// `extra` = (request id, start, end, source) spans found by audio analysis.
    pub async fn select_clips(
        &self,
        words: &[Word],
        requests: &[crate::stages::requests::ClipRequest],
        extra: &[(usize, f64, f64, String)],
        context: &str,
        cfg: &Config,
    ) -> Result<SelectClipsResp> {
        // each request carries its effective length limits (own or default)
        let reqs: Vec<serde_json::Value> = requests
            .iter()
            .map(|r| {
                let (min_d, max_d) = crate::stages::score::bounds(cfg, r);
                let mut v = serde_json::to_value(r).unwrap_or_default();
                v["min_eff"] = json!(min_d);
                v["max_eff"] = json!(max_d);
                v
            })
            .collect();
        let body = json!({
            "transcript": { "words": words },
            "model": cfg.llm_model,
            "context": context,
            "requests": reqs,
            "max_tokens": cfg.llm_max_tokens,
            "extra_candidates": extra.iter().map(|(r, s, e, src)| json!({"request": r, "start": s, "end": e, "source": src})).collect::<Vec<_>>(),
        });
        let out: SelectClipsResp = self.call("/v1/select_clips", body.clone(), "select_clips").await?;
        crate::util::log_llm_call(&self.llm_log, "select_clips", &body, &serde_json::to_value(&out).unwrap_or_default());
        Ok(out)
    }

    /// Laughter / applause / loudness analysis of a 16 kHz mono WAV.
    pub async fn audio_events(
        &self,
        wav: &Path,
        model: &str,
    ) -> Result<crate::stages::audio_events::AudioEvents> {
        let wav_abs = wav.canonicalize().unwrap_or_else(|_| wav.to_path_buf());
        let body = json!({ "audio": wav_abs.display().to_string(), "model": model });
        self.call("/v1/audio_events", body, "audio_events").await
    }

    pub async fn shutdown(&self) {
        let _ = self.http_call("/v1/shutdown", json!({})).await;
    }

    // ---- plumbing ----------------------------------------------------------

    async fn http_call(&self, path: &str, body: serde_json::Value) -> Result<serde_json::Value> {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {path}"))?;
        let status = resp.status();
        let text = resp.text().await.with_context(|| format!("reading {path} response"))?;
        if !status.is_success() {
            anyhow::bail!("sidecar {path} returned {status}: {}", truncate(&text, 2000));
        }
        serde_json::from_str(&text).with_context(|| format!("parsing sidecar JSON for {path}"))
    }

    async fn http_get(&self, path: &str) -> Result<serde_json::Value> {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("GET {path}"))?;
        let status = resp.status();
        let text = resp.text().await.with_context(|| format!("reading {path} response"))?;
        if !status.is_success() {
            anyhow::bail!("sidecar {path} returned {status}: {}", truncate(&text, 2000));
        }
        serde_json::from_str(&text).with_context(|| format!("parsing sidecar JSON for {path}"))
    }

    async fn call<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: serde_json::Value,
        what: &str,
    ) -> Result<T> {
        let v = self.http_call(path, body).await.with_context(|| format!("sidecar {what}"))?;
        serde_json::from_value(v).with_context(|| format!("parsing sidecar {what} response"))
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}