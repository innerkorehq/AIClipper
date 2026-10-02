"""HuggingFace whisper transcription via transformers.

Used when `AICLIPPER_WHISPER_BACKEND=hf` — lets the pipeline run Hinglish
ASR models like ``Oriserve/Whisper-Hindi2Hinglish-Apex`` that only ship as
transformers checkpoints (no ggml build). Mirrors whisper.cpp's `-ojf -sow`
word shape consumed by the Rust transcribe stage.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "Oriserve/Whisper-Hindi2Hinglish-Apex"

_pipe: object | None = None
_pipe_model: str | None = None
_pipe_meta: dict | None = None


def transcription_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def _pick_device_dtype():
    """Device + torch dtype from env (or sensible Apple-Silicon defaults)."""
    import torch

    device = os.environ.get("AICLIPPER_WHISPER_DEVICE", "auto")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype_name = os.environ.get("AICLIPPER_WHISPER_DTYPE", "auto").lower()
    if dtype_name == "auto":
        dtype_name = "float16" if device.startswith(("mps", "cuda")) else "float32"
    torch_dtype = getattr(torch, dtype_name, torch.float32)
    return device, torch_dtype


def _load_pipe(model_id: str):
    global _pipe, _pipe_model, _pipe_meta
    if _pipe is not None and _pipe_model == model_id and _pipe_meta is not None:
        return _pipe, _pipe_meta

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device, torch_dtype = _pick_device_dtype()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Oriserve's checkpoint ships `alignment_heads: null`, breaking word-level
    # timestamp extraction. Its base is whisper-large-v3-turbo — restore those.
    heads = getattr(model.generation_config, "alignment_heads", None)
    if not heads:
        model.generation_config.alignment_heads = [
            [2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14],
        ]

    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        generate_kwargs={
            "task": "transcribe",
            "language": "en",
            "output_attentions": True,
        },
    )
    _pipe_model = model_id
    _pipe_meta = {"device": device, "torch_dtype": str(torch_dtype)}
    return _pipe, _pipe_meta


def transcribe(wav: str, model_id: str | None = None) -> dict:
    if not transcription_available():
        raise RuntimeError(
            "transformers/torch not installed — run: pip3 install -r python_sidecar/requirements.txt"
        )

    import numpy as np
    from scipy.io import wavfile

    model_id = model_id or DEFAULT_MODEL
    pipe, meta = _load_pipe(model_id)

    sr, pcm = wavfile.read(wav)
    if sr != 16000:
        raise ValueError(f"expected 16 kHz mono WAV, got {sr} Hz")
    audio = pcm.astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    result = pipe(
        {"array": audio, "sampling_rate": sr},
        return_timestamps="word",
        chunk_length_s=30,
        stride_length_s=(5, 2),
    )

    words = []
    for chunk in result.get("chunks", []):
        ts = chunk.get("timestamp")
        start = float(ts[0]) if ts else None
        end = float(ts[1]) if ts else None
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        words.append({"text": text, "start": start if start is not None else 0.0, "end": end if end is not None else 0.0})

    if not words:
        raise ValueError("no words transcribed")
    return {"ok": True, "words": words, "model": model_id, "device": meta["device"]}