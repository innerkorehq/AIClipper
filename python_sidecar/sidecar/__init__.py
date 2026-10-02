# Local ML sidecar — Gemma (MLX) for clip scoring / content-edit, pyannote.audio
# for diarization. Runs as a persistent HTTP process so Rust never pays Python
# cold-start per call (PRD §4).