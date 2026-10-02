//! Stage 5.5 — Content edit + shot list via Gemma (single combined call), with
//! deterministic guardrails.
//!
//! Silence/dead-air removal is NOT part of the model call — it is computed
//! deterministically from word-gap thresholds (>400ms) here (PRD §5.5).
//! Guardrails (cut cap, coalescing, bounds clamping) are enforced here, not
//! trusted to the model.

use crate::config::Config;
use crate::sidecar::{ContentEditResp, RawCutRange, SidecarClient};
use crate::types::{
    ClipCandidate, ContentEdit, CutRange, CutReason, ShotFrame, ShotListEntry, Transcript,
};
use anyhow::{Context, Result};

pub struct ContentEditInput<'a> {
    pub candidate_idx: usize,
    pub candidate: &'a ClipCandidate,
    pub transcript: &'a Transcript,
    /// Scene-cut timestamps within the *rough-cut segment* (source-relative).
    pub scene_cuts: Vec<f64>,
    /// Source-file offset of this candidate's start (add to scene-cut times).
    pub source_offset: f64,
    /// When false the clip is kept continuous: no model cuts, no silence cuts,
    /// and the shot list is a single covering shot.
    pub allow_cuts: bool,
}

/// Run 5.5 for one candidate and enforce guardrails.
///
/// `words` passed to the model are the transcript slice for this candidate.
/// The model's cut ranges are in *slice-relative* word indices; we rebase to
/// global transcript indices here.
pub async fn content_edit(
    cfg: &Config,
    client: &SidecarClient,
    input: &ContentEditInput<'_>,
) -> Result<ContentEdit> {
    let (slice_start_idx, slice_end_idx) = word_span(input.transcript, input.candidate.start, input.candidate.end);
    let slice = input.transcript.slice(slice_start_idx, slice_end_idx);
    // No cuts requested, or nothing was said (action footage): there is nothing
    // for the model to edit. Layouts are decided per scene by the shot
    // classifier, so a single covering shot is enough.
    if !input.allow_cuts || slice.is_empty() {
        let why = if !input.allow_cuts { "cuts disabled" } else { "no speech in clip" };
        return Ok(ContentEdit {
            candidate_idx: input.candidate_idx,
            cut_ranges: Vec::new(),
            shot_list: vec![ShotListEntry {
                start: input.candidate.start,
                end: input.candidate.end,
                frame: ShotFrame::Speaker,
                target_track_id: None,
                reason: format!("continuous clip ({why})"),
            }],
            rationale: format!("Kept continuous: {why}."),
        });
    }

    let mut scene_cuts_input = input
        .scene_cuts
        .iter()
        .map(|t| input.source_offset + t)
        .filter(|t| *t > input.candidate.start && *t < input.candidate.end)
        .collect::<Vec<_>>();
    scene_cuts_input.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let resp: ContentEditResp = client
        .content_edit(input.candidate, slice, &scene_cuts_input, cfg)
        .await
        .with_context(|| format!("Gemma content-edit for candidate #{}", input.candidate_idx))?;

    // ---- rebase + validate model cut ranges -------------------------------
    let mut cut_ranges = rebase_cut_ranges(&resp.cut_ranges, slice_start_idx, slice.len())?;
    // ---- silence cuts (deterministic, no model) ---------------------------
    cut_ranges.extend(silence_cuts(&input.transcript.words, slice_start_idx, slice_end_idx, cfg.silence_threshold_ms));

    // ---- guardrails -------------------------------------------------------
    coalesce_cuts(&mut cut_ranges, input.transcript, cfg.coalesce_gap_s);
    enforce_cut_cap(&mut cut_ranges, input.transcript, input.candidate, cfg.max_cut_fraction);
    sort_and_validate(&mut cut_ranges, input.transcript);

    // ---- shot list: validate + clamp to candidate bounds ------------------
    let mut shot_list = resp
        .shot_list
        .into_iter()
        .filter_map(|s| {
            let frame = match s.frame.as_str() {
                "speaker" => ShotFrame::Speaker,
                "wide" => ShotFrame::Wide,
                "punch_in" => ShotFrame::PunchIn,
                _ => {
                    tracing::warn!("shot list: unknown frame `{}`, treating as wide", s.frame);
                    ShotFrame::Wide
                }
            };
            let start = s.start.max(input.candidate.start);
            let mut end = s.end.min(input.candidate.end);
            if end - start < 0.5 {
                return None;
            }
            if start > input.candidate.end {
                return None;
            }
            end = end.max(start + 0.5);
            Some(ShotListEntry { start, end, frame, target_track_id: s.target_track_id, reason: s.reason })
        })
        .collect::<Vec<_>>();
    shot_list.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    tile_shot_list(&mut shot_list, input.candidate.start, input.candidate.end);
    if shot_list.is_empty() {
        // Fallback: one shot covering the whole candidate (talking head default).
        shot_list.push(ShotListEntry {
            start: input.candidate.start,
            end: input.candidate.end,
            frame: ShotFrame::Speaker,
            target_track_id: None,
            reason: "fallback: single covering shot".into(),
        });
    }

    Ok(ContentEdit {
        candidate_idx: input.candidate_idx,
        cut_ranges,
        shot_list,
        rationale: resp.rationale.clone(),
    })
}

/// Make the shot list cover [start, end] exactly: no gaps, no overlaps.
///
/// The model is asked for shots that cover the clip, but nothing guarantees it
/// (especially once scene cuts are in the prompt). Any moment not covered by a
/// shot renders as black frames, so geometry is enforced here, not trusted:
/// a gap is absorbed by extending the earlier shot, an overlap is resolved in
/// favour of the later shot, and the ends are stretched to the clip bounds.
/// Shots that end up shorter than 50 ms are dropped.
pub fn tile_shot_list(shots: &mut Vec<ShotListEntry>, start: f64, end: f64) {
    shots.retain(|s| s.end > s.start);
    if shots.is_empty() {
        return;
    }
    shots.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    for i in 0..shots.len() - 1 {
        let next_start = shots[i + 1].start;
        shots[i].end = next_start; // closes a gap or trims an overlap
    }
    shots.first_mut().unwrap().start = start;
    shots.last_mut().unwrap().end = end;
    shots.retain(|s| s.end - s.start >= 0.05);
    // dropping a sliver reopens a gap: re-close it
    for i in 0..shots.len().saturating_sub(1) {
        shots[i].end = shots[i + 1].start;
    }
    if let Some(l) = shots.last_mut() {
        l.end = end;
    }
}

/// Word-index span (global) covering [start, end] source time.
pub fn word_span(t: &Transcript, start: f64, end: f64) -> (usize, usize) {
    let s = t.words.iter().position(|w| w.end >= start).unwrap_or(0);
    let e = t.words.iter().rposition(|w| w.start <= end).map(|i| i + 1).unwrap_or(t.words.len());
    (s, e)
}

/// Rebase slice-relative word indices to global, dropping out-of-bounds.
fn rebase_cut_ranges(
    raw: &[RawCutRange],
    offset: usize,
    slice_len: usize,
) -> Result<Vec<CutRange>> {
    let mut out = Vec::new();
    for r in raw {
        let start = r.start_word_idx as usize;
        let end = r.end_word_idx as usize;
        if end > slice_len || start >= end || end <= start {
            tracing::warn!("dropping invalid model cut range ({start}, {end})");
            continue;
        }
        let reason = match r.reason.as_str() {
            "filler" => CutReason::Filler,
            "redundant" => CutReason::Redundant,
            "tangent" => CutReason::Tangent,
            "false_start" => CutReason::FalseStart,
            other => {
                tracing::warn!("dropping cut with unknown reason `{other}`");
                continue;
            }
        };
        out.push(CutRange {
            start_word_idx: (offset + start) as u32,
            end_word_idx: (offset + end) as u32,
            reason,
        });
    }
    Ok(out)
}

/// Deterministic silence/dead-air cuts from word gaps > `threshold_ms`.
fn silence_cuts(words: &[crate::types::Word], start_idx: usize, end_idx: usize, threshold_ms: u64) -> Vec<CutRange> {
    let mut cuts = Vec::new();
    let thr = threshold_ms as f64 / 1000.0;
    let slice = &words[start_idx..end_idx];
    for (i, pair) in slice.windows(2).enumerate() {
        let gap = pair[1].start - pair[0].end;
        if gap >= thr {
            let global_a = start_idx + i;
            let global_b = start_idx + i + 1;
            let gap_s = pair[1].start - pair[0].end;
            if gap_s >= thr {
                cuts.push(CutRange {
                    start_word_idx: global_b as u32,
                    end_word_idx: global_b as u32, // empty word span ⇒ pure gap
                    reason: CutReason::Silence,
                });
                let _ = global_a;
            }
        }
    }
    cuts
}

/// Durations of each cut in source seconds.
fn cut_time(cuts: &[CutRange], words: &[crate::types::Word]) -> Vec<f64> {
    cuts.iter()
        .map(|c| {
            let idx = c.start_word_idx as usize;
            match c.reason {
                CutReason::Silence => {
                    // gap between idx-1 and idx
                    let prev_end = if idx > 0 { words[idx - 1].end } else { words[idx].start };
                    let next_start = words.get(idx).map(|w| w.start).unwrap_or(prev_end);
                    (next_start - prev_end).max(0.0)
                }
                _ => {
                    let s = words.get(idx).map(|w| w.start).unwrap_or(0.0);
                    let e = words
                        .get((c.end_word_idx as usize).saturating_sub(1))
                        .map(|w| w.end)
                        .unwrap_or(s);
                    (e - s).max(0.0)
                }
            }
        })
        .collect()
}

fn coalesce_cuts(cuts: &mut Vec<CutRange>, t: &Transcript, gap_s: f64) {
    // Merge any two cuts whose time gap is < gap_s into one contiguous range.
    cuts.sort_by_key(|c| c.start_word_idx);
    let mut merged: Vec<CutRange> = Vec::new();
    for c in cuts.drain(..) {
        if let Some(last) = merged.last_mut() {
            let last_end_time = cut_time(std::slice::from_ref(last), &t.words)[0]
                .max(t.range_end(last.end_word_idx as usize));
            let this_start_time = cut_time(std::slice::from_ref(&c), &t.words)[0]
                .min(t.range_start(c.start_word_idx as usize));
            // Approximate adjacency by word indices too, since gaps matter for silence.
            let idx_gap =
                (c.start_word_idx as i64) - (last.end_word_idx as i64);
            let close_time = (this_start_time - last_end_time).abs() < gap_s || (this_start_time - t.range_end(last.end_word_idx as usize)).abs() < gap_s;
            let close_idx = idx_gap <= 1;
            if close_time || (close_idx && c.reason != CutReason::Silence && last.reason != CutReason::Silence) {
                last.end_word_idx = last.end_word_idx.max(c.end_word_idx);
                last.reason = merge_reason(&last.reason, &c.reason);
                continue;
            }
        }
        merged.push(c);
    }
    *cuts = merged;
}

fn merge_reason(a: &CutReason, b: &CutReason) -> CutReason {
    if a == b {
        return a.clone();
    }
    // Primary semantic reasons win over Silence.
    if *a == CutReason::Silence { b.clone() } else { a.clone() }
}

/// Trim cuts so total removed time ≤ max_cut_fraction of clip duration.
fn enforce_cut_cap(cuts: &mut Vec<CutRange>, t: &Transcript, candidate: &ClipCandidate, max_frac: f64) {
    let cap = (candidate.end - candidate.start) * max_frac;
    let all = std::mem::take(cuts);
    let (silences, word_cuts): (Vec<_>, Vec<_>) =
        all.into_iter().partition(|c| c.reason == CutReason::Silence);

    let mut keep: Vec<CutRange> = Vec::new();
    let mut total = 0.0;
    // Silence first: safest, cheapest cuts.
    for c in silences {
        let dur = cut_time(std::slice::from_ref(&c), &t.words)[0];
        if total + dur <= cap {
            keep.push(c);
            total += dur;
        } else {
            tracing::warn!("silence cuts alone exceed {:.0}% of clip — dropping tail", max_frac * 100.0);
        }
    }
    // Then word cuts, shortest first so a near-cap list still fits.
    let mut wc = word_cuts;
    wc.sort_by_key(|c| ordered_float(cut_time(std::slice::from_ref(c), &t.words)[0]));
    for c in wc {
        let dur = cut_time(std::slice::from_ref(&c), &t.words)[0];
        if total + dur <= cap {
            keep.push(c);
            total += dur;
        } else {
            break; // sorted ascending — the rest cannot fit either
        }
    }
    *cuts = keep;
}

fn ordered_float(f: f64) -> std::cmp::Reverse<u64> {
    std::cmp::Reverse((f * 1000.0) as u64)
}

fn sort_and_validate(cuts: &mut Vec<CutRange>, _t: &Transcript) {
    cuts.sort_by_key(|c| c.start_word_idx);
    cuts.dedup_by_key(|c| (c.start_word_idx, c.end_word_idx));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shot(start: f64, end: f64) -> ShotListEntry {
        ShotListEntry { start, end, frame: ShotFrame::Speaker, target_track_id: None, reason: String::new() }
    }

    fn assert_tiles(v: &[ShotListEntry], start: f64, end: f64) {
        assert_eq!(v.first().unwrap().start, start);
        assert_eq!(v.last().unwrap().end, end);
        for w in v.windows(2) {
            assert_eq!(w[0].end, w[1].start, "gap or overlap between shots");
        }
    }

    #[test]
    fn closes_gaps_trims_overlaps_and_stretches_ends() {
        // gap 2.6–4.1, overlap 6.0–6.4, ragged start/end
        let mut v = vec![shot(1.0, 2.6), shot(4.1, 6.4), shot(6.0, 10.3), shot(10.3, 11.4)];
        tile_shot_list(&mut v, 0.0, 12.0);
        assert_tiles(&v, 0.0, 12.0);
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn single_and_unsorted_shots() {
        let mut one = vec![shot(3.0, 4.0)];
        tile_shot_list(&mut one, 0.0, 10.0);
        assert_tiles(&one, 0.0, 10.0);
        let mut un = vec![shot(5.0, 8.0), shot(0.0, 3.0)];
        tile_shot_list(&mut un, 0.0, 9.0);
        assert_tiles(&un, 0.0, 9.0);
    }

    #[test]
    fn drops_slivers_without_reopening_gaps() {
        let mut v = vec![shot(0.0, 3.0), shot(3.0, 3.02), shot(3.02, 6.0)];
        tile_shot_list(&mut v, 0.0, 6.0);
        assert_tiles(&v, 0.0, 6.0);
    }
}
