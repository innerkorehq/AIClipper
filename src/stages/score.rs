//! Stage 5.3 — Clip scoring via Gemma (MLX, through the sidecar).
//!
//! The model doesn't output timestamps reliably; it returns word-index-aligned
//! clip spans. We post the word-level transcript and let the sidecar resolve
//! spans from word offsets. Deterministic dedupe/ranking happens here.

use crate::config::Config;
use crate::stages::audio_events::{candidate_spans, extended_end, AudioEvents, EventKind};
use crate::stages::requests::{ClipRequest, Tone};
use crate::sidecar::{ScoreClipsResp, SidecarClient};
use crate::types::{ClipCandidate, Transcript, Word};
use anyhow::{Context, Result};

/// Score the full transcript and return the top `n` non-overlapping candidates.
pub async fn score(
    cfg: &Config,
    client: &SidecarClient,
    transcript: &Transcript,
    audio: Option<&AudioEvents>,
) -> Result<Vec<ClipCandidate>> {
    // Moments the audio says are worth a look (laughter / cheering / loud peaks)
    // are rated alongside the model's own proposals.
    let extra = audio
        .map(|a| candidate_spans(&transcript.words, a, cfg.min_clip_duration, cfg.max_clip_duration, cfg.top_n * 2, &[EventKind::Laughter, EventKind::Loud]))
        .unwrap_or_default();
    if !extra.is_empty() {
        tracing::info!("{} audio-driven candidates (laughter/loudness peaks)", extra.len());
    }
    let resp: ScoreClipsResp = client
        .score_clips(&transcript.words, cfg, &extra)
        .await
        .context("Gemma clip scoring")?;

    // Sanity: clip the model to actual transcript bounds, keep minimum duration.
    let mut candidates: Vec<ClipCandidate> = resp
        .candidates
        .into_iter()
        .filter(|c| c.score > 0.0 && c.end > c.start)
        .map(|c| finalize_candidate(c, &transcript.words, audio, cfg.min_clip_duration, cfg.max_clip_duration, cfg.audio_weight))
        .filter(|c| c.score >= cfg.min_clip_score)
        .collect();

    if candidates.is_empty() {
        tracing::warn!(
            "no clip rated at least {:.2} — the transcript may be too poor to judge (try a larger whisper model); \
             lower AICLIPPER_MIN_CLIP_SCORE or use --from/--to to override",
            cfg.min_clip_score
        );
    }
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let within = |c: &ClipCandidate| {
        (c.end - c.start) >= cfg.min_clip_duration && (c.end - c.start) <= cfg.max_clip_duration
    };
    let (ok, bad): (Vec<_>, Vec<_>) = candidates.into_iter().partition(within);
    if !bad.is_empty() {
        tracing::warn!("dropped {} candidates outside [{:.0}s, {:.0}s]", bad.len(), cfg.min_clip_duration, cfg.max_clip_duration);
    }

    // Greedy non-overlap, keeping the highest-scoring first.
    let mut kept: Vec<ClipCandidate> = Vec::new();
    for cand in ok {
        let overlaps = kept.iter().any(|k| {
            cand.start < k.end + 0.25 && k.start < cand.end + 0.25
        });
        if !overlaps {
            kept.push(cand);
        }
        if kept.len() >= cfg.top_n {
            break;
        }
    }
    kept.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    tracing::info!("{} clip candidates selected", kept.len());
    Ok(kept)
}

/// Default length of a requested clip when the ask gives none: a short-form
/// sized clip. (Without a stated length the model proposes 2-second or 10-minute
/// "moments".) Still bounded by the global limits.
const REQUEST_DEFAULT_MIN_S: f64 = 15.0;
const REQUEST_DEFAULT_MAX_S: f64 = 60.0;

/// Duration limits for a request: its own, else short-form defaults within the
/// global limits; never inverted.
pub fn bounds(cfg: &Config, r: &ClipRequest) -> (f64, f64) {
    let min_d = r.min_s.unwrap_or(REQUEST_DEFAULT_MIN_S.max(cfg.min_clip_duration));
    let max_d = r.max_s.unwrap_or(REQUEST_DEFAULT_MAX_S.min(cfg.max_clip_duration)).max(min_d + 1.0);
    (min_d, max_d)
}

/// For asks about a *kind* of moment (funny / emotional / intense) the audio is
/// evidence of that kind — evidence the text model, often reading garbled
/// Hinglish, cannot give. So for those asks:
///   * clearly audible laughter makes a clip that is not garbage (relevance ≥ 0.2)
///     relevant in its own right (a floor that rises with the laughter), and
///   * any on-topic clip (≥ 0.4) is lifted toward 1 in proportion to the reaction.
/// Never lowers relevance; plain asks are judged on the text alone.
pub fn lift_relevance(rel: f64, tone: Tone, a: &crate::stages::audio_events::ClipAudio) -> f64 {
    let reaction = match tone {
        Tone::Any => return rel,
        Tone::Funny => ((a.laugh_max / 0.6).min(1.0)).max(0.8 * a.crowd_max.min(1.0)) as f64,
        Tone::Emotional | Tone::Intense => a.loud as f64,
    };
    let mut out = rel;
    if rel >= 0.4 {
        out = rel + (1.0 - rel) * 0.6 * reaction;
    }
    if tone == Tone::Funny && rel >= 0.2 && a.laugh_max >= 0.3 {
        out = out.max(0.45 + 0.4 * reaction);
    }
    out
}

/// Share of a clip that filler/silence trimming may typically remove. A client who
/// asks for "at least 20s" means the finished clip, so the source span is chosen
/// longer when trimming will happen.
const TYPICAL_TRIM: f64 = 0.20;

/// The minimum source span to select for a request: its minimum, plus room for
/// the trimming that will shrink it — only when a minimum was asked for and cuts
/// are on. Never above the maximum.
pub fn selection_min(r: &ClipRequest, min_d: f64, max_d: f64, global_allow_cuts: bool) -> f64 {
    let cuts_on = global_allow_cuts && r.no_cuts != Some(true);
    if r.min_s.is_some() && cuts_on {
        (min_d / (1.0 - TYPICAL_TRIM)).min(max_d)
    } else {
        min_d
    }
}

/// Pick the clips for the client's requests. Returns the kept clips (ordered by
/// request, then time) and, per request, the best relevance seen — so "no match"
/// can say how close the nearest moment came. Requests with nothing relevant
/// enough yield no clip; nothing is ever padded in.
pub async fn select_for_requests(
    cfg: &Config,
    client: &SidecarClient,
    transcript: &Transcript,
    audio: Option<&AudioEvents>,
    requests: &[ClipRequest],
    allow_cuts: bool,
    context: &str,
) -> Result<(Vec<ClipCandidate>, Vec<f64>)> {
    if transcript.words.is_empty() {
        anyhow::bail!("clip requests need a transcript, but there are no words (no speech/subtitles) — use --from/--to instead");
    }
    // Funny / emotional / intense asks also look at laughter and loudness peaks.
    let mut extra: Vec<(usize, f64, f64, String)> = Vec::new();
    if let Some(a) = audio {
        for r in requests.iter().filter(|r| r.tone != Tone::Any) {
            let (min_d, max_d) = bounds(cfg, r);
            // what each kind of ask listens for: funny → laughter; intense → loud
            // peaks; emotional → either
            let kinds: &[EventKind] = match r.tone {
                Tone::Funny => &[EventKind::Laughter],
                Tone::Intense => &[EventKind::Loud],
                _ => &[EventKind::Laughter, EventKind::Loud],
            };
            let limit = cfg.top_n.max(r.count * 6);
            for (s, e, src) in candidate_spans(&transcript.words, a, min_d, max_d, limit, kinds) {
                if r.window.map(|(ws, we)| s >= ws - 1.0 && e <= we + 1.0).unwrap_or(true) {
                    extra.push((r.id, s, e, src));
                }
            }
        }
    }
    let resp = client
        .select_clips(&transcript.words, requests, &extra, context, cfg)
        .await
        .context("Gemma clip selection for requests")?;

    let mut kept: Vec<ClipCandidate> = Vec::new();
    for c in resp.candidates {
        let Some(req) = c.request_id.and_then(|id| requests.get(id)) else { continue };
        let (min_d, max_d) = bounds(cfg, req);
        // audio only matters in proportion to what the ask is about
        let audio_w = match req.tone {
            Tone::Funny => cfg.audio_weight,
            Tone::Emotional | Tone::Intense => cfg.audio_weight * 0.7,
            Tone::Any => cfg.audio_weight * 0.3,
        };
        let mut c = finalize_candidate(c, &transcript.words, audio, selection_min(req, min_d, max_d, allow_cuts), max_d, audio_w);
        c.no_cuts = req.no_cuts;
        if let Some(a) = audio {
            let lifted = lift_relevance(c.relevance, req.tone, &a.stats(c.start, c.end));
            if lifted > c.relevance + 1e-9 {
                c.rationale = format!("{} | relevance {:.0}%→{:.0}% on audio reaction", c.rationale, c.relevance * 100.0, lifted * 100.0);
                c.relevance = lifted;
            }
        }
        let dur = c.end - c.start;
        let in_window = req.window.map(|(a, b)| c.start >= a - 1.0 && c.end <= b + 1.0).unwrap_or(true);
        if c.relevance >= cfg.min_relevance && dur >= min_d - 0.5 && dur <= max_d + 0.5 && in_window {
            kept.push(c);
        }
    }
    let kept = allocate(kept, requests);
    for r in requests {
        let n = kept.iter().filter(|c| c.request_id == Some(r.id)).count();
        let best = resp.best.get(r.id).copied().unwrap_or(0.0);
        if n == 0 {
            tracing::warn!("request #{} \"{}\": no good match (closest moment {:.0}% relevant, needs {:.0}%)", r.id + 1, r.text, best * 100.0, cfg.min_relevance * 100.0);
        } else if n < r.count {
            tracing::info!("request #{} \"{}\": found {n} of up to {}", r.id + 1, r.text, r.count);
        } else {
            tracing::info!("request #{} \"{}\": {n} clip(s)", r.id + 1, r.text);
        }
    }
    Ok((kept, resp.best))
}

/// Choose up to `count` clips per request, none overlapping another. The most
/// relevant moment wins a contested stretch (request order breaks ties). Output is
/// ordered by request, then by time, so clip numbering follows the asks.
pub fn allocate(mut cands: Vec<ClipCandidate>, requests: &[ClipRequest]) -> Vec<ClipCandidate> {
    cands.sort_by(|a, b| {
        b.relevance
            .partial_cmp(&a.relevance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.request_id.cmp(&b.request_id))
    });
    let mut used = vec![0usize; requests.len()];
    let mut taken: Vec<ClipCandidate> = Vec::new();
    for c in cands {
        let Some(id) = c.request_id.filter(|id| *id < requests.len()) else { continue };
        if used[id] >= requests[id].count {
            continue;
        }
        if taken.iter().any(|k| c.start < k.end + 0.25 && k.start < c.end + 0.25) {
            continue;
        }
        used[id] += 1;
        taken.push(c);
    }
    taken.sort_by(|a, b| {
        a.request_id.cmp(&b.request_id).then(a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal))
    });
    taken
}

/// Turn a model-proposed span into a publishable clip: clamp to the transcript,
/// end an over-long clip on a sentence boundary, grow a too-short one, snap both
/// edges to sentence boundaries, let a cut-off laugh land, and fold the audio
/// score in (boost only). Shared by the automatic and the requested-clip paths.
fn finalize_candidate(
    mut c: ClipCandidate,
    words: &[Word],
    audio: Option<&AudioEvents>,
    min_d: f64,
    max_d: f64,
    audio_weight: f64,
) -> ClipCandidate {
    let last_end = words.last().map(|w| w.end).unwrap_or(0.0);
    let min_bridge_s = 1.0;
    if c.end > last_end + 0.5 {
        c.end = last_end;
    }
    c.start = c.start.max(0.0);
    c.end = c.end.max(c.start + 1.0);
    // Resolve word indices into transcript time.
    c.word_start = words.iter().position(|w| w.end >= c.start);
    c.word_end = words.iter().rposition(|w| w.start <= c.end).map(|i| i + 1);
    // Too long: end it on the last sentence boundary that fits, rather
    // than discarding a good moment.
    if c.end - c.start > max_d {
        if let (Some(ws), Some(we0)) = (c.word_start, c.word_end) {
            let fits = |we: usize| words[we - 1].end - words[ws].start <= max_d;
            let best = (ws + 1..=we0.min(words.len()))
                .rev()
                .filter(|&we| fits(we))
                .find(|&we| ends_sentence(words, we - 1))
                .or_else(|| (ws + 1..=we0.min(words.len())).rev().find(|&we| fits(we)));
            if let Some(we) = best {
                c.word_end = Some(we);
                c.end = words[we - 1].end;
            }
        }
    }
    // Small, quantized models (Gemma-4-MLX 4-bit) often choose spans
    // shorter than the minimum clip. Extend the end forward, word by
    // word, until it meets min_clip_duration — but never jump a >1s
    // silence gap (which usually separates topics).
    let mut we = c.word_end.unwrap_or(0);
    while we > 0 && we < words.len() {
        let dur = c.end - c.start;
        if dur >= min_d {
            break;
        }
        let next = &words[we];
        let prev = &words[we - 1];
        if (next.start - prev.end) > min_bridge_s {
            break;
        }
        we += 1;
        c.end = c.end.max(next.end);
        if c.end - c.start > max_d {
            c.end = c.start + max_d;
            break;
        }
    }
    c.word_end = Some(we);
    snap_to_sentences(words, &mut c, min_d, max_d);
    if let Some(a) = audio {
        // Let a laugh land: a clip that cuts one off is extended past it.
        if let Some(t) = extended_end(a, c.start, c.end, max_d) {
            c.end = t;
            c.word_end = words.iter().rposition(|w| w.start <= t).map(|i| i + 1);
        }
        // Blend the model's rubric score with the audio score.
        let stats = a.stats(c.start, c.end);
        let w = audio_weight.clamp(0.0, 1.0);
        let text_score = c.score;
        // Audio can only help: no laughter in a calm stretch is not evidence
        // against the clip, so never score below the text rubric.
        c.score = text_score.max((1.0 - w) * text_score + w * stats.score());
        c.rationale = format!("{} | audio {:.2} ({})", c.rationale, stats.score(), stats.describe());
    }
    c
}

/// Max words a boundary may move to land on a sentence edge.
const SNAP_MAX_WORDS: usize = 8;
/// Silence (s) after a word that also counts as a sentence end.
const SENTENCE_GAP_S: f64 = 0.6;
/// Breathing room added around the first/last word (clamped to neighbours).
const PAD_START_S: f64 = 0.15;
const PAD_END_S: f64 = 0.30;

/// True when a sentence ends right after word `i`.
pub(crate) fn ends_sentence(words: &[Word], i: usize) -> bool {
    if i + 1 >= words.len() {
        return true;
    }
    let t = words[i].text.trim_end();
    t.ends_with(['.', '?', '!', '…', '。', '！', '？', '۔'])
        || words[i + 1].start - words[i].end >= SENTENCE_GAP_S
}

/// True when a sentence starts at word `i`.
pub(crate) fn sentence_starts(words: &[Word], i: usize) -> bool {
    i == 0 || ends_sentence(words, i - 1)
}

/// Move a candidate's edges onto sentence boundaries (within `SNAP_MAX_WORDS`),
/// without leaving [min, max] duration, then pad slightly so speech isn't
/// clipped. A clip should open on a sentence start and close on a sentence end.
fn snap_to_sentences(words: &[Word], c: &mut ClipCandidate, min_d: f64, max_d: f64) {
    let (Some(ws0), Some(we0)) = (c.word_start, c.word_end) else { return };
    if words.is_empty() || we0 == 0 || ws0 >= we0 || we0 > words.len() {
        return;
    }
    let dur_ok = |ws: usize, we: usize| {
        let d = words[we - 1].end - words[ws].start;
        d >= min_d && d <= max_d
    };

    // End: boundary after word `we-1`. Prefer the nearest; ties extend.
    let mut we = we0;
    if !ends_sentence(words, we0 - 1) {
        for shift in 1..=SNAP_MAX_WORDS {
            let fwd = we0 - 1 + shift;
            if fwd < words.len() && ends_sentence(words, fwd) && dur_ok(ws0, fwd + 1) {
                we = fwd + 1;
                break;
            }
            if shift < we0 && we0 - 1 - shift > ws0 {
                let back = we0 - 1 - shift;
                if ends_sentence(words, back) && dur_ok(ws0, back + 1) {
                    we = back + 1;
                    break;
                }
            }
        }
    }

    // Start: a sentence starts at word i when the previous word ends one.
    let starts = |i: usize| i == 0 || ends_sentence(words, i - 1);
    let mut ws = ws0;
    if !starts(ws0) {
        for shift in 1..=SNAP_MAX_WORDS {
            let fwd = ws0 + shift;
            if fwd + 1 < we && starts(fwd) && dur_ok(fwd, we) {
                ws = fwd;
                break;
            }
            if shift <= ws0 {
                let back = ws0 - shift;
                if starts(back) && dur_ok(back, we) {
                    ws = back;
                    break;
                }
            }
        }
    }

    c.word_start = Some(ws);
    c.word_end = Some(we);
    let prev_end = if ws > 0 { words[ws - 1].end } else { 0.0 };
    let next_start = words.get(we).map(|w| w.start).unwrap_or(f64::MAX);
    c.start = (words[ws].start - PAD_START_S).max(prev_end).max(0.0);
    c.end = (words[we - 1].end + PAD_END_S).min(next_start);
}

// Keep Word import used for potential future needs; silence warning.
#[allow(dead_code)]
fn _word_span(words: &[Word], start: f64, end: f64) -> (usize, usize) {
    let s = words.iter().position(|w| w.end >= start).unwrap_or(0);
    let e = words.iter().rposition(|w| w.start <= end).map(|i| i + 1).unwrap_or(words.len());
    (s, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(text: &str, start: f64, end: f64) -> Word {
        Word { text: text.into(), start, end }
    }

    fn cand(ws: usize, we: usize) -> ClipCandidate {
        ClipCandidate { score: 1.0, word_start: Some(ws), word_end: Some(we), ..Default::default() }
    }

    fn sentence_words() -> Vec<Word> {
        // "A b c. D e f g. H i."  (0.5s per word, no gaps)
        let t = ["A", "b", "c.", "D", "e", "f", "g.", "H", "i."];
        t.iter().enumerate().map(|(i, x)| w(x, i as f64 * 0.5, i as f64 * 0.5 + 0.5)).collect()
    }

    #[test]
    fn snaps_midsentence_edges() {
        let words = sentence_words();
        // starts mid-sentence at "e"(4), ends mid-sentence after "f"(6)
        let mut c = cand(4, 6);
        snap_to_sentences(&words, &mut c, 0.5, 100.0);
        assert_eq!(c.word_start, Some(3)); // "D"
        assert_eq!(c.word_end, Some(7)); // after "g."
    }

    #[test]
    fn keeps_clean_boundaries() {
        let words = sentence_words();
        let mut c = cand(3, 7);
        snap_to_sentences(&words, &mut c, 0.5, 100.0);
        assert_eq!((c.word_start, c.word_end), (Some(3), Some(7)));
    }

    #[test]
    fn respects_max_duration() {
        let words = sentence_words();
        // extending to the sentence end would exceed 1.6s, so end must not move forward
        let mut c = cand(3, 6);
        snap_to_sentences(&words, &mut c, 0.5, 1.6);
        assert!(words[c.word_end.unwrap() - 1].end - words[c.word_start.unwrap()].start <= 1.6);
    }

    #[test]
    fn silence_gap_counts_as_boundary() {
        let words = vec![w("a", 0.0, 0.4), w("b", 0.4, 0.8), w("c", 2.0, 2.4), w("d", 2.4, 2.8)];
        assert!(ends_sentence(&words, 1));
        assert!(!ends_sentence(&words, 0));
    }

    fn req(id: usize, count: usize) -> ClipRequest {
        crate::stages::requests::parse_ask(id, "a joke", 1000.0).with_count(count)
    }

    fn rc(id: usize, start: f64, end: f64, relevance: f64) -> ClipCandidate {
        ClipCandidate { start, end, score: relevance, relevance, request_id: Some(id), ..Default::default() }
    }

    #[test]
    fn allocation_respects_counts_and_never_overlaps() {
        let reqs = vec![req(0, 2), req(1, 1)];
        let c = vec![
            rc(0, 10.0, 40.0, 0.9),
            rc(0, 12.0, 42.0, 0.8),  // overlaps the 0.9 clip of the same request
            rc(0, 100.0, 130.0, 0.7),
            rc(0, 200.0, 230.0, 0.65), // request 0 already has 2
            rc(1, 15.0, 35.0, 0.95), // contested: more relevant than request 0's 0.9
            rc(1, 300.0, 330.0, 0.7),
        ];
        let out = allocate(c, &reqs);
        let ids: Vec<(usize, f64)> = out.iter().map(|c| (c.request_id.unwrap(), c.start)).collect();
        // request 1 wins the contested stretch; request 0 falls back to its next best
        assert_eq!(ids, vec![(0, 100.0), (0, 200.0), (1, 15.0)]);
        for (i, a) in out.iter().enumerate() {
            for b in &out[i + 1..] {
                assert!(a.end + 0.25 <= b.start || b.end + 0.25 <= a.start, "overlap {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn allocation_can_return_fewer_than_asked_and_orders_by_request() {
        let reqs = vec![req(0, 3), req(1, 1)];
        let out = allocate(vec![rc(1, 500.0, 520.0, 0.8), rc(0, 900.0, 930.0, 0.7)], &reqs);
        assert_eq!(out.iter().map(|c| c.request_id.unwrap()).collect::<Vec<_>>(), vec![0, 1]);
        assert!(allocate(vec![], &reqs).is_empty());
    }

    #[test]
    fn selection_leaves_room_for_trimming_only_when_a_minimum_was_asked() {
        let cfg = Config::default_with_workdir(std::path::PathBuf::from("w"));
        let mut r = req(0, 1);
        r.min_s = Some(20.0);
        r.max_s = Some(40.0);
        let (a, b) = bounds(&cfg, &r);
        assert!((selection_min(&r, a, b, true) - 25.0).abs() < 1e-9, "20s / 0.8");
        assert_eq!(selection_min(&r, a, b, false), 20.0, "no cuts: exact");
        r.no_cuts = Some(true);
        assert_eq!(selection_min(&r, a, b, true), 20.0, "per-request no-cuts: exact");
        r.no_cuts = None;
        r.min_s = None; // no minimum asked: default bounds, no slack
        let (a, b) = bounds(&cfg, &r);
        assert_eq!(selection_min(&r, a, b, true), a);
        r.min_s = Some(38.0);
        r.max_s = Some(40.0);
        let (a, b) = bounds(&cfg, &r);
        assert_eq!(selection_min(&r, a, b, true), 40.0, "capped at the maximum");
    }

    #[test]
    fn audio_lifts_relevance_only_for_tone_asks_and_only_when_on_topic() {
        use crate::stages::audio_events::ClipAudio;
        let laugh = ClipAudio { laugh_max: 0.6, laugh_frac: 0.5, crowd_max: 0.0, loud: 0.2 };
        let quiet = ClipAudio::default();
        // funny ask, on-topic clip, strong laughter: lift gives 0.5 + 0.5*0.6*1.0 = 0.8,
        // the laughter floor gives 0.45 + 0.4*1.0 = 0.85 — the higher wins
        assert!((lift_relevance(0.5, Tone::Funny, &laugh) - 0.85).abs() < 1e-9);
        assert_eq!(lift_relevance(0.5, Tone::Funny, &quiet), 0.5, "no reaction, no lift");
        assert!(lift_relevance(0.3, Tone::Funny, &laugh) >= 0.6, "strong laughter alone carries a not-garbage clip");
        assert_eq!(lift_relevance(0.1, Tone::Funny, &laugh), 0.1, "garbage stays garbage even with laughter");
        let faint = ClipAudio { laugh_max: 0.2, laugh_frac: 0.1, crowd_max: 0.0, loud: 0.0 };
        assert_eq!(lift_relevance(0.3, Tone::Funny, &faint), 0.3, "faint laughter is not evidence");
        assert_eq!(lift_relevance(0.5, Tone::Any, &laugh), 0.5, "plain asks are judged on text alone");
        assert!(lift_relevance(0.9, Tone::Funny, &laugh) <= 1.0);
    }

    #[test]
    fn request_bounds_fall_back_and_never_invert() {
        let cfg = Config::default_with_workdir(std::path::PathBuf::from("w"));
        let mut r = req(0, 1);
        assert_eq!(bounds(&cfg, &r), (15.0, 60.0), "short-form defaults when the ask gives no length");
        r.min_s = Some(30.0);
        r.max_s = Some(45.0);
        assert_eq!(bounds(&cfg, &r), (30.0, 45.0));
        r.min_s = Some(200.0);
        r.max_s = None; // min above the global max
        let (a, b) = bounds(&cfg, &r);
        assert!(b > a);
    }
}
