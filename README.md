# AI Clipper

Fully local, offline short-form video clipping pipeline — long video in, multiple platform-ready short clips out.

Implements `ai-clipper-prd.md`. All inference (transcription, clip scoring, editorial decisions, framing direction) runs locally on Apple Silicon.

## Pipeline

```
[yt-dlp]                         SOURCE VIDEO
    ▼
[whisper.cpp]                    word-level transcript JSON
    ▼
[Gemma: clip scoring]            ranked ClipCandidate[]
    ▼
[ffmpeg: rough cut]              isolated per-candidate segment (stream-copy)
    ▼
[Gemma: content-edit + shot-list]{cut_ranges[], shot_list[]}
    ▼
[Shot classifier] (Apple Vision) per-shot strategy: face_crop | saliency_crop | blur_fill
    ▼
[Active speaker resolution]      diarization + mouth-motion/audio correlation
    ▼
[Trajectory builder]             BBoxKeyframe[] per shot
    ▼
[Time remapper]                  original_time → output_time
    ▼
[Remotion render]                composed clips (remocn captions + per-shot sequences)
    ▼
[ffmpeg: platform export]        final MP4 per profile
```

## Architecture

| Layer | Choice |
|---|---|
| Orchestration | Rust + tokio |
| Download | yt-dlp (resolution-capped) |
| Transcription | whisper.cpp (`whisper-cli`) |
| Video cut/export | ffmpeg (`ffmpeg-sidecar`) |
| LLM inference | Gemma (MLX) via persistent Python sidecar |
| Diarization | pyannote.audio (same sidecar) |
| Face/saliency | Apple Vision → Swift dylib → C ABI |
| Render | Remotion + remocn |

The core design rule: **the LLM directs, deterministic code computes.** Gemma makes clip/word/shot-directorial judgment; silence removal, crop geometry, speaker resolution, and time remapping are deterministic.

## Usage

```bash
cargo run --release -- --url <URL> [--workdir work/run1]
cargo run --release -- --file /path/video.mp4 [--profiles instagram_reels,tiktok,shorts]
```

### Choosing the clip

- **AI picks** (default): the transcript is split into overlapping windows, Gemma proposes
  hook → payoff clips per window, then every merged candidate is rated on a fixed rubric
  (hook 35%, standalone 25%, payoff 25%, emotion 15%). Clip edges are snapped to sentence
  boundaries and lightly padded.
- **Exact range**: `--from 1:30 --to 2:10` (seconds, `MM:SS` or `HH:MM:SS`) skips the AI picker and
  clips exactly that range. `--from` alone runs to the end of the video. Works with no speech.
- **`--no-cuts`**: keep the clip continuous — no filler/silence/tangent removal. Use for action
  scenes. (Clips with no speech are always continuous.)

```bash
cargo run --release -- --file movie.mp4 --from 12:05 --to 12:40 --no-cuts
```

Changing the range or cut mode in an existing `--workdir` automatically recomputes the clip stages.

Requires:
- `yt-dlp`, `ffmpeg`, `whisper-cli` on PATH
- Python sidecar deps: `python_sidecar/requirements.txt`
- Remotion deps: `cd remotion && npm install`

## Stages & resumability

Every stage writes a JSON artifact into `<workdir>/stages/<stage>.json`. A re-run skips any stage whose artifact already exists (unless `--force`). Gemma stages record rationale for auditability.

## Directories

- `src/` — Rust orchestration + stages + sidecar client + vision ABI
- `python_sidecar/` — persistent Python sidecar (Gemma MLX + pyannote.audio)
- `vision_shim/` — Swift → C ABI dylib over Apple Vision
- `remotion/` — Remotion composition + warm render worker
- `config/` — platform export profiles, defaults