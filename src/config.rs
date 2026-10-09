//! Runtime configuration: tool paths, thresholds, and platform export profiles.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Per-platform ffmpeg arg sets. Mapping table per PRD §5.11 — adding a platform
/// later is additive, not a redesign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportProfile {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Target fps (a single value; ffmpeg is told via `-r`).
    pub fps: u32,
    pub video_bitrate_mbps: u32,
    pub audio_bitrate_kbps: u32,
    /// Extra ffmpeg output args appended after the base compliance set.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

impl ExportProfile {
    /// The instagram_reels profile encodes the tightest real constraint
    /// (Instagram Graph API); it is the safe default everywhere.
    pub fn defaults() -> Vec<ExportProfile> {
        let base: Vec<String> = vec![
            "-flags".into(),
            "+cgop".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-profile:v".into(),
            "main".into(),
        ];
        vec![
            ExportProfile {
                name: "instagram_reels".into(),
                width: 1080,
                height: 1920,
                fps: 30,
                video_bitrate_mbps: 25,
                audio_bitrate_kbps: 128,
                extra_args: base.clone(),
            },
            ExportProfile {
                name: "tiktok".into(),
                width: 1080,
                height: 1920,
                fps: 30,
                video_bitrate_mbps: 25,
                audio_bitrate_kbps: 128,
                extra_args: base.clone(),
            },
            ExportProfile {
                name: "shorts".into(),
                width: 1080,
                height: 1920,
                fps: 30,
                video_bitrate_mbps: 25,
                audio_bitrate_kbps: 128,
                extra_args: base,
            },
        ]
    }
}

/// All tunable pipeline parameters. Defaults encode the PRD's stated starting values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub workdir: PathBuf,
    pub yt_dlp: String,
    pub ffmpeg: String,
    pub whisper: String,
    pub whisper_model: String,
    pub whisper_threads: usize,
    /// Spoken language for whisper.cpp (`auto`, `hi`, `en`, …).
    pub whisper_lang: String,
    /// Transcription backend: `"cpp"` (whisper-cli/ggml) or `"hf"`
    /// (transformers via the sidecar — for Hinglish custom models).
    pub whisper_backend: String,
    /// HF model id used when `whisper_backend == "hf"`.
    pub whisper_hf_model: String,
    pub sidecar_url: String,
    pub sidecar_start: bool,
    pub remotion_dir: PathBuf,
    pub vision_shim: PathBuf,
    /// MLX model id used by the sidecar for Gemma (clip scoring / content edit).
    pub llm_model: String,
    /// LLM context window in tokens.
    pub llm_max_tokens: u32,
    /// Source height cap for ingest (PRD §5.1: no 4K for a 1080-wide vertical).
    pub source_height_cap: u32,
    /// Seconds between sampled frames in Vision passes.
    pub sample_interval: f64,
    /// Word-gap (ms) above which a silence/dead-air cut is created (§5.5 guardrail).
    pub silence_threshold_ms: u64,
    /// Max fraction of clip duration allowed to be cut (§5.5).
    pub max_cut_fraction: f64,
    /// Cuts closer than this (s) are coalesced into one contiguous range (§5.5).
    pub coalesce_gap_s: f64,
    /// Scene detection threshold passed to ffmpeg `select='gt(scene,THRESH)'`.
    pub scene_threshold: f64,
    /// Vision face confidence threshold for `talking_head` classification.
    pub face_confidence: f32,
    /// Min fraction of a shot's sampled frames a face must appear in to classify talking_head.
    pub face_coverage: f32,
    /// Framing slack around a tracked bbox (panels etc.).
    pub framing_padding: f32,
    /// Min dwell time (s) before an active-speaker switch is honored (§5.8.4).
    pub speaker_switch_dwell_s: f64,
    /// Audio crossfade at cut joins (ms), §5.10.
    pub join_crossfade_ms: u32,
    /// Max concurrent clip renders.
    pub render_concurrency: usize,
    /// Per-candidate max duration (s) — clips longer than this are trimmed.
    pub max_clip_duration: f64,
    /// Min clip duration (s) — candidates shorter than this are dropped.
    pub min_clip_duration: f64,
    /// Number of top candidates to carry into rendering.
    pub top_n: usize,
    /// Candidates the rubric rates below this (0..1) are not worth rendering.
    pub min_clip_score: f64,
    /// A clip for a client request must be at least this relevant (0..1) to be kept.
    pub min_relevance: f64,
    /// Analyse laughter/applause/loudness and use it for clip selection.
    /// Use the source's own subtitles as the transcript when available.
    pub use_subtitles: bool,
    /// Subtitle language preference, comma separated (e.g. `hi,en`).
    pub sub_langs: String,
    pub audio_events: bool,
    /// HF audio-classification model (empty = sidecar default AST/AudioSet).
    pub audio_model: String,
    /// Weight (0..1) of the audio score in the final clip score.
    pub audio_weight: f64,
    /// Seconds captions are shown earlier than the ASR word times (ASR timestamps
    /// run slightly late; measured ≈0.15s for the Hinglish model).
    pub caption_lead_s: f64,
    /// Whether to skip active-speaker resolution (single speaker).
    pub assume_single_speaker: bool,
    pub llm_log_path: PathBuf,
}

impl Config {
    pub fn default_with_workdir(workdir: PathBuf) -> Config {
        let root = env_path("AICLIPPER_ROOT", PathBuf::from);
        let llm_log_path = workdir.join("llm_prompts.jsonl");
        Config {
            workdir,
            yt_dlp: env("AICLIPPER_YTDLP", "yt-dlp"),
            ffmpeg: env("AICLIPPER_FFMPEG", "ffmpeg"),
            whisper: env("AICLIPPER_WHISPER", "whisper-cli"),
            whisper_model: env(
                "AICLIPPER_WHISPER_MODEL",
                "ggml-small.bin",
            ),
            whisper_threads: env_usize("AICLIPPER_WHISPER_THREADS", 4),
            whisper_lang: env("AICLIPPER_WHISPER_LANG", "auto"),
            whisper_backend: env("AICLIPPER_WHISPER_BACKEND", "cpp"),
            whisper_hf_model: env(
                "AICLIPPER_WHISPER_HF_MODEL",
                "Oriserve/Whisper-Hindi2Hinglish-Apex",
            ),
            sidecar_url: env("AICLIPPER_SIDECAR_URL", "http://127.0.0.1:8931"),
            sidecar_start: env_bool("AICLIPPER_SIDECAR_START", true),
            remotion_dir: root.join("remotion"),
            vision_shim: root.join("vision_shim").join("target").join("release").join("libVisionShim.dylib"),
            // Default: the PRD's Gemma on MLX — the 4-bit instruct build already in the
            // HuggingFace cache. Thinking is disabled in the sidecar so it answers
            // directly instead of burning the budget on a `<|think|>` channel.
            llm_model: env(
                "AICLIPPER_LLM_MODEL",
                "mlx-community/gemma-4-e2b-it-4bit",
            ),
            llm_max_tokens: env_u32("AICLIPPER_LLM_MAX_TOKENS", 4096),
            source_height_cap: env_u32("AICLIPPER_SRC_HEIGHT_CAP", 1080),
            sample_interval: 0.5,
            silence_threshold_ms: 400,
            max_cut_fraction: 0.25,
            coalesce_gap_s: 0.35,
            scene_threshold: 0.25,
            face_confidence: 0.6,
            face_coverage: 0.5,
            framing_padding: 0.08,
            speaker_switch_dwell_s: 1.5,
            join_crossfade_ms: 35,
            render_concurrency: 2,
            max_clip_duration: 90.0,
            min_clip_duration: 5.0,
            top_n: 8,
            min_clip_score: env_f64("AICLIPPER_MIN_CLIP_SCORE", 0.3),
            min_relevance: env_f64("AICLIPPER_MIN_RELEVANCE", 0.6),
            use_subtitles: env_bool("AICLIPPER_SUBTITLES", true),
            sub_langs: env("AICLIPPER_SUB_LANGS", "hi,en"),
            audio_events: env_bool("AICLIPPER_AUDIO_EVENTS", true),
            audio_model: env("AICLIPPER_AUDIO_MODEL", ""),
            audio_weight: env_f64("AICLIPPER_AUDIO_WEIGHT", 0.35),
            caption_lead_s: env_f64(
                "AICLIPPER_CAPTION_LEAD_S",
                if env("AICLIPPER_WHISPER_BACKEND", "cpp") == "hf" { 0.15 } else { 0.0 },
            ),
            assume_single_speaker: false,
            llm_log_path,
        }
        .resolve_root(&root)
    }

    fn resolve_root(mut self, root: &Path) -> Config {
        if !self.remotion_dir.as_os_str().is_empty() && !self.remotion_dir.is_absolute() {
            self.remotion_dir = root.join(&self.remotion_dir);
        }
        if !self.vision_shim.as_os_str().is_empty() && !self.vision_shim.is_absolute() {
            self.vision_shim = root.join(&self.vision_shim);
        }
        self
    }

    /// Directory layout under workdir.
    pub fn stage_dir(&self) -> PathBuf { self.workdir.join("stages") }
    pub fn media_dir(&self) -> PathBuf { self.workdir.join("media") }
    pub fn clips_dir(&self) -> PathBuf { self.workdir.join("clips") }
    pub fn renders_dir(&self) -> PathBuf { self.workdir.join("renders") }
    pub fn exports_dir(&self) -> PathBuf { self.workdir.join("exports") }
    pub fn frames_dir(&self) -> PathBuf { self.workdir.join("frames") }

    pub fn ensure_dirs(&self) -> Result<()> {
        for d in [
            self.stage_dir(),
            self.media_dir(),
            self.clips_dir(),
            self.renders_dir(),
            self.exports_dir(),
            self.frames_dir(),
        ] {
            std::fs::create_dir_all(&d)
                .with_context(|| format!("creating work dir {}", d.display()))?;
        }
        if let Some(p) = self.llm_log_path.parent() {
            std::fs::create_dir_all(p).ok();
        }
        Ok(())
    }
}

/// Persisted export-profile configuration. Loaded from a JSON/toml file if present,
/// else built-in defaults are used (single default profile is safe everywhere §5.11).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileTable {
    pub profiles: Vec<ExportProfile>,
}

impl ProfileTable {
    pub fn load(path: Option<&Path>) -> Result<Vec<ExportProfile>> {
        match path {
            Some(p) if p.exists() => {
                let raw = std::fs::read_to_string(p)
                    .with_context(|| format!("reading {}", p.display()))?;
                let table: ProfileTable = serde_json::from_str(&raw)
                    .with_context(|| format!("parsing {}", p.display()))?;
                if table.profiles.is_empty() {
                    anyhow::bail!("profile table {} has no profiles", p.display());
                }
                Ok(table.profiles)
            }
            _ => Ok(ExportProfile::defaults()),
        }
    }

    #[allow(dead_code)]
    pub fn find(&self, name: &str) -> Option<&ExportProfile> {
        self.profiles.iter().find(|p| p.name == name)
    }
}

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
fn env_path(key: &str, f: impl FnOnce(String) -> PathBuf) -> PathBuf {
    std::env::var(key).map(f).unwrap_or_else(|_| PathBuf::from("."))
}
fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => v == "1" || v == "true" || v == "yes",
        Err(_) => default,
    }
}
fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}