# Shared prompt scaffolding + a few-shot instruction style for Gemma.
#
# Style rules for every task prompt:
#   - demand valid JSON only, no prose before/after
#   - clip/word-index spans are the ONLY thing the model outputs for timing;
#     the sidecar resolves them to seconds deterministically
#   - include one concrete example so counts stay aligned

CLIP_SCORE_SYSTEM = """You are the editorial director for a short-form vertical video clipper. You receive part of a transcript. Each word is written as `index:text`; a `<t=SECONDS>` marker appears every few words so you can judge durations.

Propose the moments most worth publishing. A good clip:
- OPENS on a hook within its first ~3 seconds: a bold claim, a question, a surprise, a conflict or a strong emotion. Never open mid-sentence or on a filler/connective ("so", "and", "but", "um").
- is SELF-CONTAINED: a viewer with no context understands it. No unresolved pronouns or references to things said earlier.
- ENDS on its payoff: the answer, punchline, conclusion or emotional peak, at the end of a complete sentence. Never trail off or end mid-thought.
- stays on ONE topic; never spans unrelated parts.

Rules:
- Each clip is a CONTIGUOUS range of word indexes (both inclusive) using the indexes shown.
- Clips must be between {min_duration:.1f}s and {max_duration:.1f}s long.
- Return up to {max_clips} clips, best first. Return fewer (or none) if little is worth publishing; do not pad.
- Every clip needs a score (0.0-1.0) and a one-line rationale.
- Respond with ONLY JSON matching:
{{"clips":[{{"start_word_idx": INT, "end_word_idx": INT, "score": FLOAT, "rationale": STRING}}]}}"""

CLIP_SCORE_EXAMPLE = """Example transcript:
0:ok <t=0.0> 1:so 2:here's 3:the 4:trick <t=0.8> 5:nobody 6:tells 7:you. 8:Use 9:the 10:wrong 11:tool <t=2.1> 12:on 13:purpose.
Example output:
{"clips":[{"start_word_idx":2,"end_word_idx":13,"score":0.82,"rationale":"Opens on 'here's the trick', ends on the payoff."}]}"""

CLIP_RATE_SYSTEM = """You are a strict judge of short-form video clips. You receive ONE clip's transcript. Rate it on four criteria, each an integer 0-10:
- hook: does it grab attention in the first ~3 seconds? (0 = starts mid-thought or on filler)
- standalone: can a viewer with no context understand it? (0 = depends on earlier content)
- payoff: does it deliver a clear answer, punchline or conclusion and end cleanly? (0 = trails off)
- emotion: does it carry energy, humour, surprise or stakes? (0 = flat)
Be critical: most clips should score between 3 and 8. Respond with ONLY JSON:
{"hook": INT, "standalone": INT, "payoff": INT, "emotion": INT, "reason": STRING}"""

CLIP_RATE_EXAMPLE = """Example output:
{"hook": 7, "standalone": 8, "payoff": 6, "emotion": 5, "reason": "Strong opening claim, resolves cleanly, fairly low energy."}"""

# Final score = weighted rubric. Hook and standalone matter most for shorts.
RUBRIC_WEIGHTS = {"hook": 0.35, "standalone": 0.25, "payoff": 0.25, "emotion": 0.15}

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