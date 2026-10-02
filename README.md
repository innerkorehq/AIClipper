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