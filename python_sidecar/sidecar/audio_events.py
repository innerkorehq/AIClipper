"""Audio event analysis: laughter / applause probabilities + loudness peaks.

Slides a short window over the 16 kHz mono WAV. Each window gets:
  * laughter  - max probability over the AudioSet laughter classes
  * applause  - max probability over applause / cheering / clapping classes
  * loud_z    - loudness (dB RMS) as a robust z-score against the surrounding
                minute, so "loud for this video" rather than absolute volume

The classifier is an Audio Spectrogram Transformer fine-tuned on AudioSet
(default MIT/ast-finetuned-audioset-10-10-0.4593); labels are multi-label
sigmoid scores, so laughter over speech is still detected.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

LAUGH_LABELS = ("laughter", "giggle", "snicker", "belly laugh", "chuckle, chortle", "baby laughter")
CROWD_LABELS = ("applause", "cheering", "clapping")

_clf = None
_clf_model: str | None = None


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def _device() -> str:
    import torch

    d = os.environ.get("AICLIPPER_AUDIO_DEVICE", "auto")
    if d == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return d


def _load(model_id: str):
    global _clf, _clf_model
    if _clf is not None and _clf_model == model_id:
        return _clf
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    device = _device()
    extractor = ASTFeatureExtractor.from_pretrained(model_id)
    model = ASTForAudioClassification.from_pretrained(model_id).to(device).eval()
    id2label = {int(i): str(l).lower() for i, l in model.config.id2label.items()}
    laugh_ids = [i for i, l in id2label.items() if l in LAUGH_LABELS]
    crowd_ids = [i for i, l in id2label.items() if l in CROWD_LABELS]
    if not laugh_ids:
        raise RuntimeError(f"model {model_id} has no laughter class")
    _clf = (extractor, model, device, laugh_ids, crowd_ids)
    _clf_model = model_id
    return _clf


def _read_wav(path: str):
    import numpy as np
    from scipy.io import wavfile

    sr, pcm = wavfile.read(path)
    if sr != 16000:
        raise ValueError(f"expected 16 kHz mono WAV, got {sr} Hz")
    audio = pcm.astype(np.float32)
    if pcm.dtype == np.int16:
        audio /= 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def loudness_z(db, span: int = 41):
    """Robust z-score of loudness (dB) against a rolling median/MAD (~1 minute)."""
    import numpy as np

    n = len(db)
    out = np.zeros(n, dtype=np.float32)
    half = span // 2
    for i in range(n):
        seg = db[max(0, i - half): i + half + 1]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) * 1.4826
        out[i] = (db[i] - med) / max(mad, 1.5)  # floor: 1.5 dB so quiet rooms don't explode
    return out


def analyze(wav: str, model_id: str | None = None, win: float = 3.0, hop: float = 1.5, batch: int = 16) -> dict:
    if not available():
        raise RuntimeError("transformers/torch not installed - run: pip3 install -r python_sidecar/requirements.txt")
    import numpy as np
    import torch

    model_id = model_id or DEFAULT_MODEL
    extractor, model, device, laugh_ids, crowd_ids = _load(model_id)
    audio, sr = _read_wav(wav)
    wlen, hlen = int(win * sr), int(hop * sr)
    starts = list(range(0, max(1, len(audio) - wlen + hlen), hlen))
    chunks = [audio[s: s + wlen] for s in starts]

    laughter, applause, db = [], [], []
    for b in range(0, len(chunks), batch):
        part = chunks[b: b + batch]
        # pad the (possibly short) tail chunk to full length for stable features
        part = [np.pad(c, (0, max(0, wlen - len(c)))) for c in part]
        inputs = extractor(part, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
        probs = torch.sigmoid(logits).float().cpu().numpy()
        laughter += probs[:, laugh_ids].max(axis=1).tolist()
        applause += (probs[:, crowd_ids].max(axis=1).tolist() if crowd_ids else [0.0] * len(part))
        for c in part:
            rms = float(np.sqrt(np.mean(np.square(c)) + 1e-12))
            db.append(20.0 * np.log10(rms + 1e-9))

    z = loudness_z(np.array(db, dtype=np.float32))
    return {
        "ok": True,
        "model": model_id,
        "device": device,
        "win": win,
        "hop": hop,
        "duration": len(audio) / sr,
        "laughter": [round(float(x), 3) for x in laughter],
        "applause": [round(float(x), 3) for x in applause],
        "loud_z": [round(float(x), 2) for x in z],
    }
