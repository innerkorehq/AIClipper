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


# ---- clip scoring: chunked proposals + rubric rating --------------------------

WINDOW_WORDS = 450       # words per proposal window (keeps a small model sharp)
WINDOW_OVERLAP = 90      # overlap so a clip on a seam is seen whole by one window
MAX_RATED = 12           # cap on rubric-rated candidates (one LLM call each)
MERGE_IOU = 0.5          # proposals overlapping more than this are duplicates


def _format_words(words: list[dict], lo: int, hi: int) -> str:
    """`idx:text` stream with a `<t=..>` marker every ~6 words (global indexes)."""
    out = []
    for i in range(lo, hi):
        if (i - lo) % 6 == 0:
            out.append(f"<t={words[i]['start']:.1f}>")
        out.append(f"{i}:{words[i]['text']}")
    return " ".join(out)


def _windows(n: int) -> list[tuple[int, int]]:
    if n <= WINDOW_WORDS:
        return [(0, n)]
    step = WINDOW_WORDS - WINDOW_OVERLAP
    spans, lo = [], 0
    while lo < n:
        hi = min(n, lo + WINDOW_WORDS)
        spans.append((lo, hi))
        if hi >= n:
            break
        lo += step
    return spans


def _words_in(words: list[dict], start: float, end: float) -> tuple[int, int] | None:
    """Word index span [lo, hi) overlapping [start, end], or None if empty."""
    idx = [i for i, w in enumerate(words) if w["end"] > start and w["start"] < end]
    return (idx[0], idx[-1] + 1) if idx else None


def _iou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def merge_proposals(cands: list[dict]) -> list[dict]:
    """Drop duplicates (same moment proposed by overlapping windows), best first."""
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: c["score"], reverse=True):
        if all(_iou(c, k) <= MERGE_IOU for k in kept):
            kept.append(c)
    return kept


def rubric_score(r: dict) -> float | None:
    """Weighted 0..1 score from a rubric response, or None if unusable."""
    total = 0.0
    for key, weight in prompts.RUBRIC_WEIGHTS.items():
        try:
            v = float(r[key])
        except (KeyError, TypeError, ValueError):
            return None
        total += weight * max(0.0, min(10.0, v)) / 10.0
    return total


def _rate(model_id: str, c: dict, words: list[dict], max_tokens: int) -> tuple[float, str, str]:
    """Rate one candidate on the rubric. Falls back to the proposal score."""
    text = " ".join(w["text"] for w in words[c["lo"]:c["hi"]])
    user = (
        prompts.CLIP_RATE_EXAMPLE
        + f"\n\nClip ({c['end'] - c['start']:.1f}s):\n{text}\nOutput the JSON now."
    )
    raw = _generate(model_id, prompts.CLIP_RATE_SYSTEM, user, min(max_tokens, 512))
    try:
        r = _extract_json(raw)
        score = rubric_score(r) if isinstance(r, dict) else None
        if score is None:
            raise ValueError("incomplete rubric")
        why = (
            f"hook {r['hook']}/standalone {r['standalone']}/payoff {r['payoff']}/"
            f"emotion {r['emotion']} - {r.get('reason', '')}"
        )
        return score, why, raw
    except (ValueError, TypeError):
        # unusable rating: keep the proposal, discounted so rated clips outrank it
        return c["score"] * 0.8, c["rationale"] + " (unrated)", raw


def _generate_json(model_id: str, system: str, user: str, max_tokens: int, attempts: int = 2):
    """Generate and parse a JSON reply, retrying once when the model fails to
    produce one (reasoning that never closes, prose instead of JSON, truncation).
    The retry names the problem and doubles the token budget (capped).
    Returns (parsed, raw_of_last_attempt)."""
    last = ""
    for i in range(attempts):
        u = user
        budget = max_tokens
        if i > 0:
            u = (
                user
                + "\n\nYour previous reply was not valid JSON (it began: "
                + json.dumps(last.strip()[:240])
                + "). Reply again with ONLY the JSON object - no reasoning, no explanation, no markdown."
            )
            budget = min(max_tokens * 2, 8192)
        last = _generate(model_id, system, u, budget)
        try:
            return _extract_json(last), last
        except ValueError:
            continue
    raise ValueError(
        f"no JSON found in model output after {attempts} attempts; last output began: {last.strip()[:200]!r}"
    )


def score_clips(model_id: str, words: list[dict], max_clips: int, min_duration: float, max_duration: float, max_tokens: int, extra: list[dict] | None = None) -> tuple[list[dict], str, str]:
    """Two passes. (1) Propose clips per overlapping transcript window.
    (2) Rate each merged candidate on a fixed rubric; the weighted rubric
    (not the model's free-form number) is the final score."""
    system = prompts.CLIP_SCORE_SYSTEM.format(
        min_duration=min_duration, max_duration=max_duration, max_clips=min(max_clips, 5)
    )
    raws: list[str] = []
    prompt_log: list[str] = [system]
    proposals: list[dict] = []
    for lo, hi in _windows(len(words)):
        user = (
            prompts.CLIP_SCORE_EXAMPLE
            + "\n\nTranscript:\n"
            + _format_words(words, lo, hi)
            + "\nOutput the JSON now."
        )
        prompt_log.append(user)
        raw = _generate(model_id, system, user, max_tokens)
        raws.append(raw)
        try:
            data = _extract_json(raw)
        except ValueError:
            continue  # one bad window must not sink the whole run
        clips = data.get("clips") if isinstance(data, dict) else None
        if not isinstance(clips, list):
            continue
        for c in clips:
            r = prompts.resolve_clip_times(c, words)
            if r is None:
                continue
            # keep the word span so the rating pass can read the exact text
            r["lo"] = int(round(float(c["start_word_idx"])))
            r["hi"] = int(round(float(c["end_word_idx"]))) + 1
            proposals.append(r)

    # Moments found by other means (e.g. audio laughter peaks): rate them with
    # the same rubric. They start with a high proposal score so they survive
    # the MAX_RATED cut; the rubric then decides.
    for e in extra or []:
        span = _words_in(words, float(e["start"]), float(e["end"]))
        if span is not None:
            proposals.append({
                "start": words[span[0]]["start"], "end": words[span[1] - 1]["end"],
                "score": 0.85, "rationale": str(e.get("source", "audio")), "lo": span[0], "hi": span[1],
            })

    if not proposals:
        raise ValueError("score_clips: model proposed no valid clips")

    merged = merge_proposals(proposals)[:MAX_RATED]
    out = []
    for c in merged:
        score, why, raw = _rate(model_id, c, words, max_tokens)
        raws.append(raw)
        out.append({"start": c["start"], "end": c["end"], "score": score, "rationale": why})
    out.sort(key=lambda c: c["score"], reverse=True)
    return out, "\n---\n".join(raws), "\n\n".join(prompt_log)


# ---- client-requested clips ----------------------------------------------------------

MAX_RATED_PER_REQUEST = 6


def _request_line(r: dict) -> str:
    lo, hi = r.get("min_s"), r.get("max_s")
    dur = ""
    if lo and hi:
        dur = f" (duration {lo:.0f}-{hi:.0f}s)"
    elif hi:
        dur = f" (at most {hi:.0f}s)"
    elif lo:
        dur = f" (at least {lo:.0f}s)"
    return f'{r["id"]}: "{r["text"]}"{dur}'


def request_rubric_score(r: dict) -> float | None:
    total = 0.0
    for key, weight in prompts.REQUEST_RUBRIC_WEIGHTS.items():
        try:
            v = float(r[key])
        except (KeyError, TypeError, ValueError):
            return None
        total += weight * max(0.0, min(10.0, v)) / 10.0
    return total


def _rate_request(model_id: str, c: dict, req: dict, words: list[dict], max_tokens: int) -> tuple[float, float, str, str]:
    """(score 0..1, relevance 0..1, reason, raw). Falls back to the proposal's own
    relevance when the rating cannot be parsed."""
    text = " ".join(w["text"] for w in words[c["lo"]:c["hi"]])
    system = prompts.CLIP_REQUEST_RATE_SYSTEM.format(request=req["text"].replace('"', "'"))
    user = f"Clip ({c['end'] - c['start']:.1f}s):\n{text}\nOutput the JSON now."
    raw = _generate(model_id, system, user, min(max_tokens, 512))
    try:
        r = _extract_json(raw)
        score = request_rubric_score(r) if isinstance(r, dict) else None
        if score is None:
            raise ValueError("incomplete rubric")
        rel = max(0.0, min(10.0, float(r["relevance"]))) / 10.0
        why = (
            f"relevance {r['relevance']}/hook {r['hook']}/standalone {r['standalone']}/payoff {r['payoff']}/"
            f"emotion {r['emotion']} - {r.get('reason', '')}"
        )
        return score, rel, why, raw
    except (ValueError, TypeError):
        # unusable rating: trust the proposal, discounted
        rel = float(c.get("relevance", 0.0)) * 0.8
        return rel * 0.8, rel, str(c.get("why", "")) + " (unrated)", raw


def select_clips(
    model_id: str,
    words: list[dict],
    requests: list[dict],
    max_tokens: int,
    extra: list[dict] | None = None,
) -> tuple[list[dict], list[float], str, str]:
    """Find clips for client requests. One proposal call per transcript window covers
    ALL requests (cost does not grow with their number); every merged candidate is
    then rated against its own request. Returns (candidates, best relevance per
    request, raw output, prompt log). Candidates carry request_id, relevance (0..1),
    score (0..1) and a rationale."""
    raws: list[str] = []
    prompt_log: list[str] = []
    by_id = {r["id"]: r for r in requests}
    best = {r["id"]: 0.0 for r in requests}
    proposals: list[dict] = []

    def propose(active_all: list[dict]) -> None:
        """One proposal call per transcript window for the given requests."""
        for lo, hi in _windows(len(words)):
            w_start, w_end = words[lo]["start"], words[hi - 1]["end"]
            # a request with a time window only sees the windows that overlap it
            active = [
                r for r in active_all
                if not r.get("window") or (r["window"][0] < w_end and r["window"][1] > w_start)
            ]
            if not active:
                continue
            system = prompts.CLIP_REQUEST_SYSTEM.format(max_per_request=min(3, max(r["count"] for r in active) + 1))
            user = (
                prompts.CLIP_REQUEST_EXAMPLE
                + "\n\nREQUESTS:\n" + "\n".join(_request_line(r) for r in active)
                + "\n\nTranscript:\n" + _format_words(words, lo, hi)
                + "\nOutput the JSON now."
            )
            prompt_log.append(user)
            try:
                data, raw = _generate_json(model_id, system, user, max_tokens)
            except ValueError:
                continue  # one bad window must not sink the run
            raws.append(raw)
            matches = data.get("matches") if isinstance(data, dict) else None
            if not isinstance(matches, list):
                continue
            for m in matches:
                try:
                    rid = int(m["request"])
                    rel = max(0.0, min(10.0, float(m["relevance"]))) / 10.0
                except (KeyError, TypeError, ValueError):
                    continue
                if rid not in by_id or rel < 0.5:
                    continue
                r = prompts.resolve_clip_times({**m, "score": rel}, words)
                if r is None:
                    continue
                req = by_id[rid]
                win = req.get("window")
                if win and not (r["start"] >= win[0] - 1.0 and r["end"] <= win[1] + 1.0):
                    continue
                best[rid] = max(best[rid], rel)
                proposals.append({
                    **r, "request_id": rid, "relevance": rel, "why": str(m.get("why", "")),
                    "lo": int(round(float(m["start_word_idx"]))), "hi": int(round(float(m["end_word_idx"]))) + 1,
                })

    # Pass 1: every request together (cost does not grow with their number).
    propose(requests)

    # Pass 2: small models answer the first request and drop the rest. Any request
    # that came back with fewer distinct moments than it asked for gets its own
    # focused pass, so it cannot be missed just because of its position in the list.
    def distinct(rid: int) -> int:
        return len(merge_proposals([{**p, "score": p["relevance"]} for p in proposals if p["request_id"] == rid]))

    for r in requests:
        if distinct(r["id"]) < r["count"]:
            propose([r])

    # Moments found by other means (audio laughter/loudness peaks) for specific requests.
    for e in extra or []:
        rid = e.get("request")
        if rid not in by_id:
            continue
        span = _words_in(words, float(e["start"]), float(e["end"]))
        if span is not None:
            proposals.append({
                "start": words[span[0]]["start"], "end": words[span[1] - 1]["end"], "score": 0.5,
                "request_id": rid, "relevance": 0.5, "why": str(e.get("source", "audio")),
                "lo": span[0], "hi": span[1], "rationale": str(e.get("source", "audio")),
            })

    out: list[dict] = []
    for rid, req in by_id.items():
        mine = merge_proposals([{**p, "score": p["relevance"]} for p in proposals if p["request_id"] == rid])
        for c in mine[: max(MAX_RATED_PER_REQUEST, req["count"] * 2)]:
            score, rel, why, raw = _rate_request(model_id, c, req, words, max_tokens)
            raws.append(raw)
            best[rid] = max(best[rid], rel)
            out.append({"start": c["start"], "end": c["end"], "score": score, "relevance": rel, "request_id": rid, "rationale": why})
    out.sort(key=lambda c: (c["request_id"], -c["score"]))
    return out, [best[r["id"]] for r in requests], "\n---\n".join(raws), "\n\n".join(prompt_log)


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
    data, raw = _generate_json(model_id, system, user, max_tokens)
    if not isinstance(data, dict):
        raise ValueError("content_edit: model output is not an object")
    if "cut_ranges" not in data or "shot_list" not in data:
        raise ValueError("content_edit: output missing `cut_ranges`/`shot_list`")
    return {
        "cut_ranges": data.get("cut_ranges", []),
        "shot_list": data.get("shot_list", []),
        "rationale": str(data.get("rationale", "")),
    }, raw, system + "\n\n" + user