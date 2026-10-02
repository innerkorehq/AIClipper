# AI Clipper — Product Requirements Document

**Status:** Draft v1
**Owner:** Gagan
**Scope:** Fully local, offline short-form video clipping pipeline — long video in, multiple platform-ready short clips out.

---

## 1. Summary

AI Clipper ingests a long-form video (via URL or local file) and autonomously produces multiple short-form clips, each with speaker-aware or saliency-aware reframing to vertical, styled word-level captions, and edited-down narration (filler/redundancy/tangent removal). All inference — transcription, clip scoring, editorial decisions, and framing direction — runs locally on Apple Silicon. No cloud API dependency for the core pipeline.

This is a standalone tool, architecturally separate from [[vidgen]] (Innerkore's script-to-video generation pipeline). AI Clipper solves the inverse problem: long video → short clips, not script → generated video. The two may share downstream primitives (Remotion, caption styling conventions) but not a codebase or runtime.

## 2. Goals

- Take a video URL (via `yt-dlp`) or local file and produce N ranked, ready-to-post short clips with no manual editing required for the common case.
- Handle both talking-head content (podcasts, interviews, single-speaker commentary) and non-talking-head content (anime, action, b-roll-heavy) with appropriate, distinct reframing strategies.
- Autonomously edit narration: remove dead air, filler, false starts, redundant restatement, and off-topic tangents from the spoken track without degrading meaning or pacing.
- Autonomously reframe 16:9 source to 9:16 with time-varying, content-aware crop (not fixed center-crop) where appropriate, and fall back to blur-fill composition where a single crop target would lose information.
- Render burned-in, word-timed captions via Remotion + remocn components — not ffmpeg subtitle filters.
- Export clips compliant with the tightest real-world constraint among target platforms (Instagram Graph API), so a single default profile is safe everywhere without per-platform branching in the common case.
- Every stage's output is inspectable, deterministic given the same inputs (except the two LLM judgment calls, which are logged with their rationale), and resumable — a failed run doesn't require re-transcribing or re-scoring from scratch.

## 3. Non-Goals

- Not a general-purpose video editor or timeline UI. No human-in-the-loop editing surface in v1 (may be added later as an optional review step).
- Not a cloud service. No multi-tenant, no hosted API, no billing.
- Not attempting perfect shot classification or perfect editorial judgment — target is "good enough to ship without a human pass most of the time," not zero-defect.
- Not building a new face/object detector — relies entirely on Apple's Vision framework for CV primitives.
- Not responsible for actually publishing to platforms (no Graph API posting integration in v1) — output is a compliant file ready for manual or separately-automated upload.

## 4. High-Level Architecture

```
 [yt-dlp]                                                 SOURCE VIDEO
     │
     ▼
 [whisper.cpp]  ──────────────────────────────────────►  word-level transcript JSON
     │                                                     (word, start, end)
     ▼
 [Gemma: clip scoring]  ◄── transcript                     ranked ClipCandidate[]
     │                                                      {start, end, score, rationale}
     ▼
 [ffmpeg: rough cut]  ──────────────────────────────────►  isolated per-candidate segment
     │                                                      (stream-copy, no re-encode)
     ▼
 [Gemma: content-edit + shot-list]  ◄── transcript slice    {cut_ranges[], shot_list[]}
     │  (single call, shares context)                       per candidate clip
     ▼
 [Shot classifier]  ◄── Vision face detector                per-shot strategy:
     │                  + saliency request                  face_crop | saliency_crop | blur_fill
     ▼
 [Trajectory builder]  ◄── Vision detections (dense)        BBoxKeyframe[] per shot
     │                     + active-speaker mapping
     ▼
 [Time remapper]  ◄── coalesced cut_ranges                  original_time → output_time
     │                                                       applied to captions, keyframes,
     │                                                       shot boundaries
     ▼
 [Remotion render]  ◄── remocn caption components            single composed clip,
     │                  + per-shot Sequence blocks            native resolution, target aspect
     │                  (captions + crop/blur-fill + cuts)
     ▼
 [ffmpeg: platform export]  ────────────────────────────►   final MP4 per profile
                                                              (instagram_reels / tiktok / shorts)
```

Persistent local services (not spawned per-call):
- **Python ML sidecar** (long-running process, hit over localhost HTTP or Unix socket): hosts the MLX Gemma model and pyannote.audio diarization. Both Gemma calls (clip scoring; content-edit + shot-list) and diarization go through this one process — avoids Python/model cold-start cost per invocation.
- **Vision shim** (Swift/Obj-C compiled to a dylib, called from Rust via a thin C ABI or `objc2` bindings): face detection (`VNDetectFaceRectanglesRequest`), saliency (`VNGenerateAttentionBasedSaliencyImageRequest`, `VNGenerateObjectnessBasedSaliencyImageRequest`). Hardware-accelerated via the Neural Engine, no model download.
- **Remotion renderer**: one warm Node process per run, headless Chromium opened once (`openBrowser()`), reused across all clip candidates in that run rather than a fresh `npx remotion render` CLI invocation per clip.

Orchestration: Rust, async (tokio), each pipeline stage a function over shared intermediate structs, all intermediate artifacts serialized to disk as JSON between stages for resumability and debugging.

## 5. Pipeline Stages — Detail

### 5.1 Ingest — `yt-dlp`
- Input: URL or local file path.
- Cap source resolution to what's actually needed for the target export (e.g. `-f "bv*[height<=1080]+ba"`) — no reason to pull 4K to produce a 1080-wide vertical clip.
- Output: source video file + metadata (duration, original resolution/fps).

### 5.2 Transcription — `whisper.cpp` via `whisper-rs`
- Word-level timestamps required (not just segment-level) — this is the timing backbone for everything downstream: clip boundaries, caption word-reveal, silence-gap detection, and content-cut snapping.
- Output: `Transcript { words: Vec<Word { text, start, end }> }`, persisted as JSON.

### 5.3 Clip Scoring — Gemma (MLX, via Python sidecar)
- Sliding window over transcript segments; rubric-driven prompt (hook strength, self-containedness, single clear point, emotional/curiosity pull).
- Output: `ClipCandidate[] { start, end, score, rationale }`.

### 5.4 Rough Cut — `ffmpeg` (stream-copy)
- Isolates each candidate's time range from the source into its own file, no re-encode.
- Purpose: keeps later stages (Remotion, Vision passes) operating on tight per-clip files instead of seeking into the full-length source repeatedly.

### 5.5 Content Edit + Shot List — Gemma (single combined call)
- Input: word-level transcript slice for this candidate + its stage-5.3 rationale (so the model doesn't cut the material the clip was selected for) + (for multi-shot candidates) scene-cut timestamps from an `ffmpeg` scene-detection pre-pass (`select='gt(scene,0.4)'`).
- Two outputs from one call, sharing context:
  - `cut_ranges: [{ start_word_idx, end_word_idx, reason: filler | redundant | tangent | false_start }]`
  - `shot_list: [{ start, end, frame: speaker | wide | punch_in, target_track_id?, reason }]`
- Guardrails enforced in post-processing, not trusted to the model: cap total cut duration at ~25% of clip length; coalesce cuts within a small gap into one contiguous range.
- Silence/dead-air removal is **not** part of this call — computed deterministically from word-gap thresholds (>400ms) in the same post-processing step, no model involved.

### 5.6 Shot Classification
- Per shot (not per clip): run Vision face detector across sampled frames.
- Classify `talking_head` if face confidence + screen-time coverage over the shot exceeds a threshold; otherwise fall through to saliency-based classification.
- For non-face shots: run `VNGenerateAttentionBasedSaliencyImageRequest`; compute heatmap concentration (entropy or max/mean ratio). Concentrated → `saliency_crop`; diffuse → `blur_fill`.
- Output per shot: `strategy: face_crop | saliency_crop | blur_fill`.

### 5.7 Active Speaker Resolution (multi-speaker `face_crop` shots only)
- If speaker positions are fixed (known camera layout): map diarized turns (pyannote, via sidecar) directly to known track positions.
- Otherwise: correlate each face track's mouth-region motion (Vision face landmarks, mouth-aspect-ratio time series) against the audio energy envelope for the current diarized turn; highest-correlation track wins.
- Single-speaker shots skip this entirely — one track, no resolution needed.

### 5.8 Trajectory Builder (deterministic — see §6 for ownership rationale)
- Input: dense, per-sampled-frame detections (face or saliency bbox + confidence), plus resolved active-speaker track per turn.
- Responsibilities:
  1. Gap-fill between sampled frames (linear or Kalman-predicted).
  2. Smooth centroid/size (EMA or Kalman) to remove detector jitter.
  3. Downsample to sparse control points (path-simplification, e.g. Ramer–Douglas–Peucker over the smoothed path) — Remotion interpolates the rest at render time.
  4. Enforce shot-level constraints: reset trajectory at every shot cut; minimum dwell time before an active-speaker switch; emit nothing for `blur_fill` shots.
- Output: `BBoxKeyframe[]` per shot, merged with the shot-list strategy/timing into one per-shot artifact: `{ start_frame, end_frame, strategy, keyframes }`.

### 5.9 Time Remapping (applies after cut decision, before Remotion)
- Single shared utility, not reimplemented per consumer.
- Builds a monotonic `original_time → output_time` function from the coalesced `cut_ranges` list (output time = original time minus total preceding cut duration; anything inside a cut range maps to "dropped").
- Applied uniformly to: caption words, `BBoxKeyframe` timestamps, shot-list segment boundaries. Any keyframe/boundary landing inside a cut range is dropped or re-anchored to the nearest surviving boundary.
- This is the step most likely to silently break sync if skipped or duplicated inconsistently across consumers — treat as a hard dependency all downstream artifacts pass through exactly once.

### 5.10 Remotion Render
- Composition built from a **sequence of per-shot segments** (`<Sequence>` blocks positioned at shot frame offsets), not one continuous `<Video>` with a single transform — required because strategy and trajectory reset per shot.
- `face_crop` / `saliency_crop` shots: simple transform (translate/scale) on the video track, driven by interpolated `BBoxKeyframe`s.
- `blur_fill` shots: composite branch — full frame blurred and stretched to fill 9:16, original content overlaid centered, nothing cropped out.
- Captions: remocn components, word-timed from the (time-remapped) transcript, karaoke-style word reveal via ASS-equivalent timing data passed as props (not an ffmpeg subtitle burn).
- Content cuts: short crossfade (20–50ms) on audio at the join; for `face_crop` shots, cuts land at a `punch_in` framing moment from the shot-list where possible, so the crop change itself masks the jump cut. Not needed for shots already cutting on their own cadence (action/anime).
- One warm render process handles all clip candidates for a run.

### 5.11 Platform Export — `ffmpeg`
- One default profile targeting the tightest real constraint (Instagram Graph API): H.264, progressive scan, **closed GOP** (`-flags +cgop` / paired `-g`/`-keyint_min` — not ffmpeg's default), 4:2:0 chroma, 23–60fps, max width 1920px, AAC ≤48kHz stereo at 128kbps, video bitrate ≤25Mbps, file size ≤300MB.
- This single profile also satisfies TikTok and YouTube Shorts. Only add a named per-platform profile if a real quality/format divergence is found — not preemptively.
- Named profiles kept as a small config table (`instagram_reels`, `tiktok`, `shorts`) mapping to ffmpeg arg sets, so adding a platform later is additive, not a redesign.

## 6. Component Ownership — Deterministic vs. LLM

A recurring design principle throughout this pipeline: **the LLM directs, deterministic code computes.** Two places this was explicitly decided against giving the LLM more scope:

| Decision | Owner | Why |
|---|---|---|
| Which moments make good clips | Gemma | Genuine semantic/narrative judgment |
| Which words are filler/redundant/tangential | Gemma | Genuine semantic judgment |
| Which words are silence/dead air | Deterministic (word-gap threshold) | Pure signal processing, no judgment needed |
| Per-shot framing strategy & timing (shot list) | Gemma | Directorial judgment — narrative shaped |
| Per-frame crop/bbox coordinates | Deterministic (Vision + trajectory builder) | Solved CV problem; LLM-computed geometry is slow, non-deterministic, and worse than a detector |
| Which face is the active speaker | Deterministic (diarization + correlation) | Signal correlation problem, not language |
| Semantic override of saliency (e.g. "the punchline is the small text, not the big motion") | Gemma (coarse, keyframe-sampled) | Saliency heatmaps don't know narrative importance; used as an override on the deterministic default, not the primary decision-maker |

## 7. Data Contracts

```ts
Word          { text: string, start: f64, end: f64 }
Transcript    { words: Word[] }

ClipCandidate { start: f64, end: f64, score: f64, rationale: string }

CutRange      { start_word_idx: u32, end_word_idx: u32,
                reason: "filler" | "redundant" | "tangent" | "false_start" | "silence" }

ShotListEntry { start: f64, end: f64,
                frame: "speaker" | "wide" | "punch_in",
                target_track_id?: string, reason: string }

BBoxKeyframe  { frame: u32, x: f32, y: f32, w: f32, h: f32 }   // normalized 0-1

ShotSegment   { start_frame: u32, end_frame: u32,
                strategy: "face_crop" | "saliency_crop" | "blur_fill",
                keyframes: BBoxKeyframe[] }

RenderManifest{ clip_id: string, shots: ShotSegment[],
                captions: Word[] (time-remapped),
                cut_ranges: CutRange[] }
```

`RenderManifest` (after time remapping) is the sole input to Remotion — no consumer downstream of §5.9 touches original-video time.

## 8. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | Rust + tokio | Stage functions over shared structs; JSON artifacts on disk between stages |
| Download | yt-dlp | Resolution-capped |
| Transcription | whisper.cpp via `whisper-rs` | Word-level timestamps on |
| Video cutting/export | `ffmpeg`, invoked via `ffmpeg-sidecar` crate | Stream-copy for rough cuts; re-encode only at final export |
| LLM inference | Gemma (MLX) via persistent Python sidecar | Local HTTP/socket API; also hosts diarization |
| Diarization | pyannote.audio | Same Python sidecar |
| Face/saliency detection | Apple Vision framework | Via Swift/Obj-C shim → C ABI, called from Rust |
| Caption + composite rendering | Remotion + remocn components | Warm Node process, `renderMedia()`, reused browser instance |

## 9. Resumability & Failure Handling

- Every stage writes its output to disk as JSON before the next stage begins; a re-run detects existing stage output and skips re-computation unless explicitly forced.
- Gemma calls (stages 5.3, 5.5) are the only non-deterministic steps — their outputs are logged with full rationale so a run can be audited or replayed without re-invoking the model.
- A failed Remotion render for one clip candidate should not abort the whole run — candidates render independently; partial success (N of M clips produced) is an acceptable outcome, not a failure state.

## 10. Open Questions

- **`mlx-rs` vs. subprocess/server for the Gemma boundary**: subprocess/server assumed as the default in this design (safer against MLX API churn); worth revisiting if `mlx-rs` maturity improves.
- **Silence threshold tuning**: 400ms is a starting default: needs validation against real source material — Hinglish/conversational content may have different natural pause patterns than the scripted content this threshold is usually tuned for elsewhere.
- **Review step**: v1 has no human-in-the-loop gate before export. Worth deciding whether to add an optional review step (per the source article's recommendation) once real output quality is assessed — likely a v1.1 addition rather than blocking v1.
- **Anime/action content volume**: shot classification and saliency-driven blur-fill are designed but unvalidated against real anime source material — needs a dedicated test pass before considered production-ready for that content type specifically.

## 11. Phased Rollout

| Phase | Scope |
|---|---|
| 1 | yt-dlp → whisper.cpp → Gemma scoring → ffmpeg rough cut → fixed center-crop reframe → Remotion captions only → single export profile. Talking-head content only. |
| 2 | Content-edit + shot-list Gemma call; silence removal; time remapping utility; face-tracking trajectory builder replacing fixed center-crop. |
| 3 | Shot classification (saliency path, blur-fill); active speaker resolution for multi-speaker content; anime/action content support. |
| 4 | Multi-platform export profiles (if divergence found); optional human review step. |
