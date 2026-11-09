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

### Framing: how each shot is cropped to vertical

There is a menu of framing algorithms and an AI chooser that picks one **per shot**. Deterministic code
measures each shot (faces, how much the main face moves, motion energy and activity, saliency focus, how
much is speech) and works out which algorithms can physically run on it; the model then picks the one that
fits the footage, the request and what the video is, and says why. Code only guards the answer (an unknown
or impossible pick is replaced by the default and the reason recorded) and covers for a model that is
unavailable.

| algorithm | what it does | can run when |
|---|---|---|
| `face_follow` | follows the main face, full-height vertical window | a face is present in at least 40% of the shot |
| `speaker_follow` | follows whoever is speaking among several faces | two or more faces appear |
| `split_two` | two people stacked, each fixed to their own half | exactly two people consistently (a split would drop a third) |
| `group_fit` | a window just large enough to contain every face | two or more people consistently |
| `punch_in` | tight zoom on one face (emphasis/reaction) | a face is present and the shot is 6 s or shorter |
| `locked` | one fixed window, no camera motion | what it frames holds still (face movement under 0.08 of the frame width) |
| `saliency_follow` | follows the visually salient region (no faces needed) | attention has a clear focal point |
| `motion_follow` | follows where the motion is (action, sports, dance) | the shot is busy, or a distinct moving region is present in at least half of it |
| `wide_blur` | whole scene centred over a blurred fill (nothing cropped) | always |

`group_fit` never crops anyone out. A group wider than a vertical window is shown letterboxed over a
blurred fill, so three people across a 16:9 frame come out small. That is expected, not a failure.

**What counts as a person.** The chooser only sees faces that are people. Face detection also fires on
printed photos (a WANTED poster, a framed portrait) and on cardboard cutouts, and counting those would
send a two-person shot to a group fit. Two conservative filters remove them before any measurement
(`src/vision/decoys.rs`):

- **Printed photos** are nearly colourless (colour saturation about 0.1, against about 0.5 for a living
  face). A face is dropped only when it is both nearly colourless and much less colourful than another
  face in the same frame, so a black-and-white film loses nothing.
- **Cutouts** do not move at all, while real faces drift. A face whose box stays pixel-identical across a
  shot, in a picture where other faces do move, is dropped. If everyone is still, nothing is dropped.

Both tests keep a face when in doubt (a lone face, a short shot, or an unknown colour value). The colour
values are cached with the face detections, so work folders from before this change need a fresh
`--workdir` to get the printed-photo filter.

`--framing auto` (default, the AI chooses) · `--framing rules` (deterministic, no model) ·
`--framing <algorithm>` (force one where it can run; for debugging/comparison). What was chosen for every
shot, by whom and why, with the measurements, is in `<workdir>/framing/<clip>.md`. Changing `--framing`
recomputes the clip stages.

### Asking for specific clips

Tell the pipeline what you want instead of letting it pick. One or many requests; they replace the
automatic picking, and a request with no good match produces **no clip** (it is never padded with an
unrelated moment).

```bash
cargo run --release -- --file talk.mp4 \
  --ask "the funniest exchange about the movie poster, 20-40s" \
  --ask "2 clips where they explain the gym routine, after 10:00"
cargo run --release -- --file talk.mp4 --brief clips.toml
```

Count, duration and a time window can be written in the sentence — `2 clips`, `30-45s`, `under 40 seconds`,
`about a minute`, `after 10:00`, `between 5:00 and 12:30`, `in the first 10 minutes`. For anything
explicit use a brief (explicit fields beat the sentence):

```toml
[[clip]]
ask = "a funny moment about the movie poster"
duration = [20, 40]            # seconds, of the FINISHED clip (selection leaves room for trimming)

[[clip]]
ask = "the part about steroids or the gym"
count = 2
window = ["2:00", "4:00"]      # only search inside this span
cuts = false                   # keep this clip continuous (no filler/silence removal)
label = "gym story"            # used in the file name: clip_001_gym-story.mp4
tone = "intense"               # optional: any | funny | emotional | intense (steers the audio signals)
```

The parsed requests are printed before any model work, and a report (`<workdir>/clips_report.md`) lists,
per request, the clips found (time, relevance, why, transcript excerpt) or how close the nearest moment
came. Clips get names like `clip_000_funny-movie-poster`. A clip must rate at least 60% relevant
(`AICLIPPER_MIN_RELEVANCE`). `--ask/--brief` cannot be combined with `--from/--to`.

Limits: selection is text-based (transcript/subtitles plus laughter/loudness), so purely visual asks
("when he falls off the horse") cannot be matched — use `--from/--to` for those. A request naming a
speaker only works if the name is spoken in the transcript.

### Transcript source

If the video comes with subtitles they are the transcript, and speech recognition is skipped:
a `.srt`/`.vtt` next to a local file (`movie.srt`, `movie.hi.srt`), an embedded text track, or the
uploader's own subtitles fetched by yt-dlp for a URL. Auto-generated captions are never used —
they are just another ASR. Subtitle cues are phrase-level, so words get timings spread across each
cue by length. Language preference: `AICLIPPER_SUB_LANGS` (default `hi,en`). A subtitle track that
looks too sparse to be a transcript falls back to ASR.

- `--subtitles FILE` use this subtitle file
- `--no-subtitles` ignore subtitles and transcribe the audio
- `AICLIPPER_SUBTITLES=0` turn the feature off

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