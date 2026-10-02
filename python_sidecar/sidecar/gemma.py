"""Gemma (MLX) generation helpers — lazy model load, JSON-only responses."""

from __future__ import annotations

import json
import re

from . import prompts

_gemma_available: bool | None = None
_model = None
_tokenizer = None


def gemma_available() -> bool:
    global _gemma_available
    if _gemma_available is None:
        try:
            import importlib.util

            _gemma_available = (
                importlib.util.find_spec("mlx_lm") is not None
                and importlib.util.find_spec("mlx") is not None
            )
        except Exception:
            _gemma_available = False
    return _gemma_available


def _load(model_id: str):
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load

        _model, _tokenizer = load(model_id)
    return _model, _tokenizer


def _generate(model_id: str, system: str, user: str, max_tokens: int) -> str:
    model, tokenizer = _load(model_id)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    from mlx_lm import generate

    # Reasoning models (Gemma 4 / 3n family) default to emitting a
    # `<|think|>` channel that can consume the whole token budget before any
    # JSON appears. Disable thinking when the template supports it; fall back
    # for templates (Qwen, non-thinking models) that reject the kwarg.
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def _extract_json(text: str) -> dict | list:
    """Best-effort JSON extraction (reasoning-stripped, first balanced block)."""
    t = text.strip()
    # Drop reasoning channels emitted by thinking models (e.g. Gemma 3n/4):
    #   <|channel>thought ... <channel|>
    t = re.sub(r"<\|channel>.*?<channel\|>", "", t, flags=re.S)
    t = re.sub(r"<\|channel>.*$", "", t, flags=re.S)
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Grab the first balanced { ... } or [ ... ] block.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = t.find(open_ch)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(t)):
            if t[i] == open_ch:
                depth += 1
            elif t[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no JSON found in model output")


def score_clips(model_id: str, words: list[dict], max_clips: int, min_duration: float, max_duration: float, max_tokens: int) -> tuple[list[dict], str, str]:
    system = prompts.CLIP_SCORE_SYSTEM.format(
        min_duration=min_duration, max_duration=max_duration, max_clips=max_clips
    )
    transcript = {"words": words}
    user = (
        prompts.CLIP_SCORE_EXAMPLE
        + "\n\nTranscript:\n"
        + json.dumps(transcript)
        + "\nOutput the JSON now."
    )
    raw = _generate(model_id, system, user, max_tokens)
    data = _extract_json(raw)
    if not isinstance(data, dict) or "clips" not in data:
        raise ValueError("score_clips: model output missing `clips`")
    clips = data["clips"]
    if not isinstance(clips, list):
        raise ValueError("score_clips: `clips` is not a list")
    candidates = [c for c in (prompts.resolve_clip_times(c, words) for c in clips) if c is not None]
    return candidates, raw, system + "\n\n" + user


def content_edit(
    model_id: str,
    candidate: dict,
    words: list[dict],
    scene_cuts: list[float],
    max_cut_fraction: float,
    sample_interval: float,
    max_tokens: int,
) -> tuple[dict, str, str]:
    system = prompts.CONTENT_EDIT_SYSTEM.format(max_cut_fraction=max_cut_fraction)
    payload = {
        "candidate": candidate,
        "clip_words": words,
        "scene_cuts": scene_cuts,
        "sample_interval_s": sample_interval,
    }
    user = (
        prompts.CONTENT_EDIT_EXAMPLE
        + "\n\nClip data:\n"
        + json.dumps(payload)
        + "\nOutput the JSON now."
    )
    raw = _generate(model_id, system, user, max_tokens)
    data = _extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("content_edit: model output is not an object")
    if "cut_ranges" not in data or "shot_list" not in data:
        raise ValueError("content_edit: output missing `cut_ranges`/`shot_list`")
    return {
        "cut_ranges": data.get("cut_ranges", []),
        "shot_list": data.get("shot_list", []),
        "rationale": str(data.get("rationale", "")),
    }, raw, system + "\n\n" + user