//! Stage 5.3 — Clip scoring via Gemma (MLX, through the sidecar).
//!
//! The model doesn't output timestamps reliably; it returns word-index-aligned
//! clip spans. We post the word-level transcript and let the sidecar resolve
//! spans from word offsets. Deterministic dedupe/ranking happens here.

use crate::config::Config;
use crate::sidecar::{ScoreClipsResp, SidecarClient};
use crate::types::{ClipCandidate, Transcript, Word};
use anyhow::{Context, Result};

/// Score the full transcript and return the top `n` non-overlapping candidates.
pub async fn score(cfg: &Config, client: &SidecarClient, transcript: &Transcript) -> Result<Vec<ClipCandidate>> {
    let resp: ScoreClipsResp = client
        .score_clips(&transcript.words, cfg)
        .await
        .context("Gemma clip scoring")?;

    // Sanity: clip the model to actual transcript bounds, keep minimum duration.
    let last_end = transcript.words.last().map(|w| w.end).unwrap_or(0.0);
    let min_bridge_s = 1.0;
    let mut candidates: Vec<ClipCandidate> = resp
        .candidates
        .into_iter()
        .filter(|c| c.score >= cfg.min_clip_score && c.end > c.start)
        .map(|mut c| {
            if c.end > last_end + 0.5 {
                c.end = last_end;
            }
            c.start = c.start.max(0.0);
            c.end = c.end.max(c.start + 1.0);
            // Resolve word indices into transcript time.
            c.word_start = transcript.words.iter().position(|w| w.end >= c.start);
            c.word_end = transcript.words.iter().rposition(|w| w.start <= c.end).map(|i| i + 1);
            // Too long: end it on the last sentence boundary that fits, rather
            // than discarding a good moment.
            if c.end - c.start > cfg.max_clip_duration {
                if let (Some(ws), Some(we0)) = (c.word_start, c.word_end) {
                    let fits = |we: usize| transcript.words[we - 1].end - transcript.words[ws].start <= cfg.max_clip_duration;
                    let best = (ws + 1..=we0.min(transcript.words.len()))
                        .rev()
                        .filter(|&we| fits(we))
                        .find(|&we| ends_sentence(&transcript.words, we - 1))
                        .or_else(|| (ws + 1..=we0.min(transcript.words.len())).rev().find(|&we| fits(we)));
                    if let Some(we) = best {
                        c.word_end = Some(we);
                        c.end = transcript.words[we - 1].end;
                    }
                }
            }
            // Small, quantized models (Gemma-4-MLX 4-bit) often choose spans
            // shorter than the minimum clip. Extend the end forward, word by
            // word, until it meets min_clip_duration — but never jump a >1s
            // silence gap (which usually separates topics).
            let mut we = c.word_end.unwrap_or(0);
            while we > 0 && we < transcript.words.len() {
                let dur = c.end - c.start;
                if dur >= cfg.min_clip_duration {
                    break;
                }
                let next = &transcript.words[we];
                let prev = &transcript.words[we - 1];
                if (next.start - prev.end) > min_bridge_s {
                    break;
                }
                we += 1;
                c.end = c.end.max(next.end);
                if c.end - c.start > cfg.max_clip_duration {
                    c.end = c.start + cfg.max_clip_duration;
                    break;
                }
            }
            c.word_end = Some(we);
            snap_to_sentences(&transcript.words, &mut c, cfg.min_clip_duration, cfg.max_clip_duration);
            c
        })
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

/// Max words a boundary may move to land on a sentence edge.
const SNAP_MAX_WORDS: usize = 8;
/// Silence (s) after a word that also counts as a sentence end.
const SENTENCE_GAP_S: f64 = 0.6;
/// Breathing room added around the first/last word (clamped to neighbours).
const PAD_START_S: f64 = 0.15;
const PAD_END_S: f64 = 0.30;

/// True when a sentence ends right after word `i`.
fn ends_sentence(words: &[Word], i: usize) -> bool {
    if i + 1 >= words.len() {
        return true;
    }
    let t = words[i].text.trim_end();
    t.ends_with(['.', '?', '!', '…', '。', '！', '？', '۔'])
        || words[i + 1].start - words[i].end >= SENTENCE_GAP_S
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
        ClipCandidate { start: 0.0, end: 0.0, score: 1.0, rationale: String::new(), word_start: Some(ws), word_end: Some(we) }
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
}
