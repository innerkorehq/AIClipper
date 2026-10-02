# Shared prompt scaffolding + a few-shot instruction style for Gemma.
#
# Style rules for every task prompt:
#   - demand valid JSON only, no prose before/after
#   - clip/word-index spans are the ONLY thing the model outputs for timing;
#     the sidecar resolves them to seconds deterministically
#   - include one concrete example so counts stay aligned

CLIP_SCORE_SYSTEM = """You are the editorial director for a short-form vertical video clipper. You receive a word-level transcript with indexes. Pick the clip(s) most worth publishing: self-contained moments with a strong hook, clear payoff, and standalone meaning.

Rules:
- Each clip spans a CONTIGUOUS range of transcript word indexes (both inclusive), never crossing unrelated topics.
- Clips must be between {min_duration:.1f}s and {max_duration:.1f}s long when resolved from words.
- Prefer 1–3 clips; never more than {max_clips}.
- Every clip needs a score (0.0–1.0) and a one-line rationale.
- Respond with ONLY JSON matching:
{{"clips":[{{"start_word_idx": INT, "end_word_idx": INT, "score": FLOAT, "rationale": STRING}}]}}"""

CLIP_SCORE_EXAMPLE = """Example (transcript: {"words":[{"text":"ok","start":0.0,"end":0.3},{"text":"so","start":0.3,"end":0.5},{"text":"here's","start":0.5,"end":0.7},{"text":"the","start":0.7,"end":0.8},{"text":"trick","start":0.8,"end":1.1}]}):
{"clips":[{"start_word_idx":2,"end_word_idx":4,"score":0.82,"rationale":"Framed by 'here's the trick' - punchy standalone payoff."}]}"""

CONTENT_EDIT_SYSTEM = """You are the editor for one short-form vertical video clip. You receive: (a) the clip's transcript slice with indexes, (b) detected scene-change timestamps, and (c) the candidate clip's original score and rationale.

Produce TWO things:

1) cut_ranges — transcripts words to REMOVE for pacing (filler words, redundant restatements, tangents, false starts). Word indexes are relative to the transcribed slice. Do NOT cut speech that is load-bearing for the chosen moment. Total removed time must stay under {max_cut_fraction:.0%} of the clip.
   Reasons: "filler" | "redundant" | "tangent" | "false_start"

2) shot_list — 1 to 5 shots covering the whole clipping span, in order. Each shot = a time range in SECONDS (source-video time, absolute, not slice-relative) plus a framing intent:
   - "wide": landscape/b-roll/medium; keep a stable, small-ish target
   - "speaker": a talking head — put target_track_id if a speaker label is known
   - "punch_in": tight emphasis / reaction
   Shots may overlap slightly at boundaries. When there are distinct scene cuts, use them as shot boundaries. Do not invent speakers.

Respond with ONLY JSON matching:
{{"cut_ranges":[{{"start_word_idx": INT, "end_word_idx": INT, "reason": STRING}}],
 "shot_list":[{{"start": FLOAT, "end": FLOAT, "frame": STRING, "target_track_id": STRING|null, "reason": STRING}}],
 "rationale": STRING}}"""

CONTENT_EDIT_EXAMPLE = """Example:
{"cut_ranges":[{"start_word_idx":0,"end_word_idx":0,"reason":"filler"}],
 "shot_list":[{"start":3.2,"end":9.4,"frame":"speaker","target_track_id":"SPEAKER_00","reason":"host delivers the main point"}],
 "rationale":"Tightened the intro filler and framed the host for the whole take."}"""


def resolve_clip_times(model_clip: dict, words: list[dict]) -> dict | None:
    """Resolve a model's word-index span to seconds, or None if invalid.

    Indexes may come back as floats (some quantized Gemma builds emit 1.8
    instead of 2); round to the nearest word and clamp to bounds.
    """
    s = model_clip.get("start_word_idx")
    e = model_clip.get("end_word_idx")
    try:
        s = int(round(float(s)))
        e = int(round(float(e)))
    except (TypeError, ValueError):
        return None
    n = len(words)
    if s < 0 or e < s or e >= n:
        return None
    start = words[s]["start"]
    end = words[e]["end"]
    if not (start < end):
        return None
    try:
        score = float(model_clip["score"])
    except (KeyError, TypeError, ValueError):
        score = 0.0
    return {
        "start": start,
        "end": end,
        "score": score,
        "rationale": str(model_clip.get("rationale", "")),
    }