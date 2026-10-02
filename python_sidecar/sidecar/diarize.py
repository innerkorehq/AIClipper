"""Diarization via pyannote.audio (optional / lazily loaded)."""

from __future__ import annotations

import os

_pipeline = None
_available: bool | None = None


def diarization_available() -> bool:
    global _available
    if _available is None:
        import importlib.util

        _available = importlib.util.find_spec("pyannote.audio") is not None
    return _available


def _get_pipeline(model_id: str | None):
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline

        model_id = model_id or os.environ.get("AICLIPPER_DIAR_MODEL") or "pyannote/speaker-diarization-3.1"
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            _pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
        else:
            # Ask like v3.1 does; requires env token normally.
            _pipeline = Pipeline.from_pretrained(model_id)
    return _pipeline


def diarize(wav_path: str, diar_model: str | None = None) -> list[dict]:
    if not diarization_available():
        raise RuntimeError("pyannote.audio not installed (pip install -r python_sidecar/requirements.txt)")
    pipeline = _get_pipeline(diar_model)
    diarization = pipeline(wav_path, num_speakers=None)

    # Collapse adjacent turns from the same speaker into contiguous tuples,
    # labeled SPEAKER_00/01/... in order of first appearance.
    raw: list[dict] = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        raw.append({"start": float(turn.start), "end": float(turn.end), "speaker": label})
    raw.sort(key=lambda t: t["start"])

    label_order: dict[str, str] = {}
    out: list[dict] = []
    for tr in raw:
        if tr["speaker"] not in label_order:
            label_order[tr["speaker"]] = f"SPEAKER_{len(label_order):02d}"
        out.append({"start": tr["start"], "end": tr["end"], "speaker": label_order[tr["speaker"]]})
    return out