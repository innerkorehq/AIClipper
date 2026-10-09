//! Data contracts — mirrors `ai-clipper-prd.md` §7.
//!
//! These structs are the shared intermediate format between pipeline stages.
//! Everything is serde-serializable to JSON so each stage's output can be
//! persisted to disk for resumability and debugging.

use serde::{Deserialize, Serialize};

/// A single word with timing, in seconds from start of source video.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// Word-level transcription output (stage 5.2).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub words: Vec<Word>,
}

impl Transcript {
    /// Duration of a word-index range (inclusive of end index, exclusive logic handled by caller).
    pub fn range_start(&self, start_idx: usize) -> f64 {
        self.words
            .get(start_idx)
            .map(|w| w.start)
            .unwrap_or_else(|| self.words.last().map(|w| w.end).unwrap_or(0.0))
    }
    pub fn range_end(&self, end_idx: usize) -> f64 {
        self.words
            .get(end_idx)
            .map(|w| w.end)
            .unwrap_or_else(|| self.words.last().map(|w| w.end).unwrap_or(0.0))
    }
    /// The contiguous slice of words in [start_idx, end_idx) — `end_idx` exclusive.
    pub fn slice(&self, start_idx: usize, end_idx: usize) -> &[Word] {
        let end = end_idx.min(self.words.len());
        &self.words[start_idx.min(self.words.len())..end]
    }
}

/// A scored candidate clip (stage 5.3). Times are seconds in source-video time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipCandidate {
    pub start: f64,
    pub end: f64,
    pub score: f64,
    pub rationale: String,
    /// Word-index span of the candidate (computed from transcript words).
    #[serde(default)]
    pub word_start: Option<usize>,
    #[serde(default)]
    pub word_end: Option<usize>,
}

/// A contiguous range of words to cut. `end_word_idx` is exclusive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CutRange {
    pub start_word_idx: u32,
    pub end_word_idx: u32,
    pub reason: CutReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CutReason {
    Filler,
    Redundant,
    Tangent,
    FalseStart,
    Silence,
}

/// What kind of framing a shot wants (narrative direction from Gemma).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShotFrame {
    Speaker,
    Wide,
    PunchIn,
}

/// One entry in Gemma's shot list (stage 5.5). Times are seconds in source-video time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotListEntry {
    pub start: f64,
    pub end: f64,
    pub frame: ShotFrame,
    #[serde(default)]
    pub target_track_id: Option<String>,
    pub reason: String,
}

/// Output of stage 5.5: content edits + shot list in a single call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContentEdit {
    pub candidate_idx: usize,
    /// Inclusive/exclusive word-index cuts. Silence is added deterministically later.
    pub cut_ranges: Vec<CutRange>,
    pub shot_list: Vec<ShotListEntry>,
    pub rationale: String,
}

/// Per-shot framing strategy (stage 5.6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShotStrategy {
    FaceCrop,
    SaliencyCrop,
    BlurFill,
}

/// A normalized bounding box (0.0–1.0 relative to frame width/height).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
    #[allow(dead_code)]
    pub fn scale(&self, s: f32) -> BBox {
        BBox {
            x: self.x,
            y: self.y,
            w: (self.w * s).min(1.0),
            h: (self.h * s).min(1.0),
        }
    }
}

/// A per-frame detection from the Vision pass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Detection {
    pub frame: u32,
    pub bbox: BBox,
    pub confidence: f32,
    /// Saliency mass inside bbox, for saliency crops (0..1).
    #[serde(default)]
    pub saliency_mass: f32,
}

/// A keyframe for the crop trajectory (sparse, normalized 0–1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBoxKeyframe {
    pub frame: u32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Per-shot render segment (PRD §5.8 output, merged into shot segment artifacts).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotSegment {
    /// Output-time frames (post-remap).
    pub start_frame: u32,
    pub end_frame: u32,
    pub strategy: ShotStrategy,
    pub keyframes: Vec<BBoxKeyframe>,
    /// Secondary subject track (normalized boxes) for two-speaker shots —
    /// the renderer stacks them top/bottom. Empty unless a second confident
    /// face persisted for the shot.
    #[serde(default)]
    pub second_keyframes: Vec<BBoxKeyframe>,
    /// True when this shot carries a second subject (→ split-screen framing).
    #[serde(default)]
    pub split: bool,
    /// Source (rough-cut) frames shown during this shot — `start_frame`/`end_frame`
    /// in the rough cut's own timeline. Populated during time remapping so
    /// Remotion never needs the inverse mapping itself.
    #[serde(default)]
    pub src_start_frame: u32,
    #[serde(default)]
    pub src_end_frame: u32,
    /// Which output frame the rough cut should fast-forward to (skipped cuts).
    #[serde(default)]
    pub video_trim_frame: u32,
}

/// A diarized speech turn from the sidecar (per elocution, source-video time).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

/// A time span in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeSpan {
    pub start: f64,
    pub end: f64,
}

/// Single source of truth for one clip's render. Output of time remapping (§5.9);
/// the *only* input to Remotion. All times are output-time (already remapped).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderManifest {
    pub clip_id: String,
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Rough-cut source video path for this clip (16:9 source segment).
    pub source: String,
    /// Absolute path of the input asset usable by Remotion.
    pub source_abs: String,
    /// Small, silent, short-GOP proxy of the source used only for the blurred
    /// background layer (so the renderer doesn't decode 4K just to blur it).
    #[serde(default)]
    pub blur_src: String,
    /// Natural dimensions of the source (rough-cut) video.
    pub source_width: u32,
    pub source_height: u32,
    /// Shot segments ordered by start_frame (frames in output time).
    pub shots: Vec<ShotSegment>,
    /// Time-remapped caption words (word reveal timing).
    pub captions: Vec<Word>,
    /// Time-remapped cut ranges (for audio crossfades at joins).
    pub cut_ranges: Vec<CutRange>,
    /// Output-time spans of dropped material (post remap) — the renderer's
    /// crossfade/join keys.
    pub cuts_s: Vec<TimeSpan>,
    /// Output-time boundary markers where a *content* cut joins kept material
    /// (for punch-in masking and crossfade placement).
    pub joins: Vec<f64>,
}

impl RenderManifest {
    /// Output-time boundary of the last shot (clip duration in frames).
    #[allow(dead_code)]
    pub fn frame_duration(&self) -> u32 {
        self.shots
            .iter()
            .map(|s| s.end_frame)
            .max()
            .unwrap_or(self.captions.last().map(|w| (w.end * self.fps as f64) as u32).unwrap_or(0))
    }
}

/// The whole run, as persisted in `<workdir>/run.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunState {
    pub source: Option<String>,
    pub duration: Option<f64>,
    pub fps: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub transcript: Option<Transcript>,
    pub candidates: Vec<ClipCandidate>,
    pub content_edits: Vec<ContentEdit>,
    pub manifests: Vec<RenderManifest>,
}