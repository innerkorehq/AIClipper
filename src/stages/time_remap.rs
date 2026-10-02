//! Stage 5.9 — Time remapping (applies after cut decision, before Remotion).
//!
//! Single shared utility: builds a monotonic `original_time → output_time`
//! function from the coalesced cut ranges (output time = original time minus
//! total preceding cut duration; anything inside a cut range maps to "dropped").
//!
//! Per PRD §5.9 this is the step most likely to silently break sync if skipped
//! or duplicated inconsistently across consumers — every downstream artifact
//! must pass through here exactly once.

use crate::types::{
    ContentEdit, CutRange, CutReason, RenderManifest, ShotSegment, ShotStrategy, TimeSpan,
    Transcript, Word,
};
use anyhow::Result;

/// A monotonic original→output time mapper.
#[derive(Debug, Clone)]
pub struct TimeRemap {
    /// Ascending, non-overlapping cut spans in absolute source seconds.
    spans: Vec<(f64, f64)>,
}

#[allow(dead_code)]
impl TimeRemap {
    /// Build from fractional-f64 spans; sorts and merges overlaps.
    pub fn from_spans(mut spans: Vec<(f64, f64)>) -> Self {
        spans.retain(|(s, e)| e > s && *s >= 0.0);
        spans.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (s, e) in spans {
            if let Some(last) = merged.last_mut() {
                if s <= last.1 {
                    last.1 = last.1.max(e);
                    continue;
                }
            }
            merged.push((s, e));
        }
        Self { spans: merged }
    }

    /// Resolve a ContentEdit's word-index cuts into absolute source-time spans.
    /// `clip_start`/`clip_end` bound the candidate so edits can't leak out.
    pub fn from_content_edit(transcript: &Transcript, edit: &ContentEdit, clip_start: f64, clip_end: f64) -> Self {
        let mut spans = Vec::new();
        for cut in &edit.cut_ranges {
            if let Some(span) = cut_span(transcript, cut) {
                spans.push(span);
            }
        }
        Self::from_spans(spans)
            .clamp(clip_start, clip_end)
    }

    pub fn clamp(mut self, lo: f64, hi: f64) -> Self {
        self.spans = self
            .spans
            .into_iter()
            .map(|(s, e)| (s.max(lo), e.min(hi)))
            .filter(|(s, e)| e > s)
            .collect();
        self
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// true when `t` (original time) falls inside a cut span (dropped).
    pub fn is_dropped(&self, t: f64) -> bool {
        self.spans.iter().any(|(s, e)| t > *s && t < *e)
    }

    /// Map original → output; `None` when the point is inside a cut span.
    pub fn map(&self, original: f64) -> Option<f64> {
        let mut shift = 0.0f64;
        for (s, e) in &self.spans {
            if original >= *e {
                shift += e - s;
            } else if original > *s {
                return None;
            } else {
                break;
            }
        }
        Some(original - shift)
    }

    /// Inverse mapping: output time → original time. `None` inside a dropped span.
    pub fn inverse(&self, output: f64) -> Option<f64> {
        let mut shift = 0.0f64;
        for (s, e) in &self.spans {
            let os = self.map(*s).unwrap_or(*s);
            let oe = self.map(*e).unwrap_or(*e);
            if output >= oe {
                shift += e - s;
            } else if output > os {
                return None; // inside dropped span
            } else {
                break;
            }
        }
        Some(output + shift)
    }

    /// Original time for output time, snapped to the nearest surviving edge.
    pub fn inverse_snap(&self, output: f64) -> f64 {
        match self.inverse(output) {
            Some(t) => t,
            None => {
                let mut best = (f64::INFINITY, 0.0f64);
                for (s, e) in &self.spans {
                    for edge in [self.map(*s).unwrap_or(*s), self.map(*e).unwrap_or(*e)] {
                        let orig = if edge <= output { *e } else { *s };
                        let d = (edge - output).abs();
                        if d < best.0 {
                            best = (d, orig);
                        }
                    }
                }
                best.1
            }
        }
    }

    /// Output-time duration contributed by original interval [a, b] (inside
    /// a cut contributes 0). Safe even when a or b is dropped.
    pub fn surviving_duration(&self, a: f64, b: f64) -> f64 {
        let mut dur = 0.0;
        let mut t = a;
        let end = b.max(a);
        let steps = 400.max(((end - a) / 0.02) as usize); // 20ms resolution
        let h = (end - a) / steps as f64;
        for _ in 0..steps {
            if !self.is_dropped(t + h / 2.0) {
                dur += h;
            }
            t += h;
        }
        dur
    }

    /// Map a frame number at source `fps` to an output frame number. Points
    /// inside a cut are resolved to the nearest surviving boundary.
    pub fn map_frame(&self, frame: u32, fps: f64) -> u32 {
        let t = frame as f64 / fps;
        match self.map(t) {
            Some(ot) => (ot * fps).round().max(0.0) as u32,
            None => {
                // inside a cut: nearest surviving boundary (left or right edge)
                let mut best = (f64::INFINITY, 0.0f64); // (dist, out_time)
                for (s, e) in &self.spans {
                    for edge in [*s, *e] {
                        let out_edge = self.map(edge).unwrap_or(edge);
                        let d = (t - edge).abs();
                        if d < best.0 {
                            best = (d, out_edge);
                        }
                    }
                }
                (best.1 * fps).round().max(0.0) as u32
            }
        }
    }

    /// Source-time intervals not covered by any cut span (`[lo, hi]` clipped).
    pub fn surviving_intervals(&self, lo: f64, hi: f64) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        let mut cur = lo;
        for (s, e) in &self.spans {
            let (s, e) = (s.max(lo), e.min(hi));
            if e <= s {
                continue;
            }
            if s > cur {
                out.push((cur, s));
            }
            cur = cur.max(e);
        }
        if cur < hi {
            out.push((cur, hi));
        }
        out
    }

    /// The output-time spans of dropped material (for the render manifest).
    pub fn dropped_spans(&self) -> Vec<TimeSpan> {
        self.spans
            .iter()
            .map(|(s, e)| TimeSpan { start: self.map(*s).unwrap_or(*s), end: self.map(*e).unwrap_or(*e) })
            .collect()
    }

    /// Join edges in output time where kept material meets (crossfade targets).
    pub fn joins(&self) -> Vec<f64> {
        let mut edges = Vec::new();
        for (s, e) in &self.spans {
            if let Some(a) = self.map(*s) {
                edges.push(a);
            }
            if let Some(b) = self.map(*e) {
                edges.push(b);
            }
        }
        // For directly adjacent cuts the shared edge collapses — that's fine.
        edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
        edges
    }
}

/// Convert a word-index CutRange to an absolute span.
pub fn cut_span(transcript: &Transcript, cut: &CutRange) -> Option<(f64, f64)> {
    let words = &transcript.words;
    let i = cut.start_word_idx as usize;
    match cut.reason {
        CutReason::Silence => {
            // gap between word[i-1] and word[i]
            let prev_end = words.get(i.wrapping_sub(1)).map(|w| w.end).unwrap_or(words.first().map(|w| w.start).unwrap_or(0.0));
            let next_start = words.get(i).map(|w| w.start).unwrap_or(prev_end);
            if next_start > prev_end {
                Some((prev_end, next_start))
            } else {
                None
            }
        }
        _ => {
            let s = words.get(i).map(|w| w.start)?;
            let e = words.get(cut.end_word_idx as usize - 1).map(|w| w.end).unwrap_or(s);
            if e > s {
                Some((s, e))
            } else {
                None
            }
        }
    }
}

/// Value of the nearest keyframe to time `t` (raw keyframe payload).
fn nearest_keyframe(kfs: &[(f64, f32, f32, f32, f32)], t: f64) -> Option<(f32, f32, f32, f32)> {
    kfs.iter()
        .min_by(|a, b| (a.0 - t).abs().partial_cmp(&(b.0 - t).abs()).unwrap())
        .map(|k| (k.1, k.2, k.3, k.4))
}

/// Produce the fully time-remapped RenderManifest for one candidate.
///
/// Everything downstream touches only output time after this point.
#[allow(clippy::too_many_arguments)]
pub fn build_manifest(
    clip_id: &str,
    candidate_start: f64,
    candidate_end: f64,
    fps: u32,
    source: &std::path::Path,
    source_abs: &std::path::Path,
    source_width: u32,
    source_height: u32,
    transcript: &Transcript,
    edit: &ContentEdit,
    segments: &[ShotSegment],
    width: u32,
    height: u32,
) -> Result<RenderManifest> {
    let remap = TimeRemap::from_content_edit(transcript, edit, candidate_start, candidate_end);

    // ---- captions ----------------------------------------------------------
    let mut captions: Vec<Word> = Vec::new();
    // Output time is clip-relative (0 at the candidate's first frame).
    let clip_origin = remap.map(candidate_start).unwrap_or(candidate_start);
    for w in &transcript.words {
        if w.start < candidate_start - 0.01 || w.start >= candidate_end {
            continue;
        }
        if w.end > candidate_end + 0.05 {
            continue;
        }
        if let (Some(s), Some(e)) = (remap.map(w.start), remap.map(w.end)) {
            let (s, e) = (s - clip_origin, e - clip_origin);
            if e > s && s >= -0.001 {
                captions.push(Word { text: w.text.clone(), start: s.max(0.0), end: e.max(0.0) });
            }
        }
    }

    // ---- shots: split every source shot at surviving-interval boundaries ----
    // Each piece is a contiguous original-time run, so the rough-cut video plays
    // linearly inside it. Keyframes are clipped + remapped per piece.
    let fps_f = fps as f64;
    let candidate_rel = |t: f64| (t - candidate_start) * fps_f; // frame into rough cut
    let survivors = remap.surviving_intervals(candidate_start, candidate_end);
    let mut shots: Vec<ShotSegment> = Vec::new();
    for seg in segments {
        let seg_start = seg.start_frame as f64 / fps_f;
        let seg_end = seg.end_frame as f64 / fps_f;
        let seg_keyframes: Vec<(f64, f32, f32, f32, f32)> = seg
            .keyframes
            .iter()
            .map(|k| {
                (k.frame as f64 / fps_f, k.x, k.y, k.w, k.h)
            })
            .collect();
        let seg_second: Vec<(f64, f32, f32, f32, f32)> = seg
            .second_keyframes
            .iter()
            .map(|k| {
                (k.frame as f64 / fps_f, k.x, k.y, k.w, k.h)
            })
            .collect();
        for (iv_start, iv_end) in &survivors {
            let o = seg_start.max(*iv_start);
            let oe = seg_end.min(*iv_end);
            if oe - o < 1.0 / fps_f {
                continue;
            }
            let out_s = remap.map(o).unwrap_or(0.0);
            let out_e = remap.map(oe).unwrap_or(0.0);
            let start_frame = (out_s * fps_f).round().max(0.0) as u32;
            let end_frame = (out_e * fps_f).round().max(0.0) as u32;
            if end_frame <= start_frame {
                continue;
            }
            // Where the piece sits on the OUTPUT timeline (clip-relative frames, cuts
            // removed). The renderer places each shot's <Sequence> at these frames.
            let src_start = ((out_s - clip_origin) * fps_f).round().max(0.0) as u32;
            let src_end = (((out_e - clip_origin) * fps_f).round().max(0.0) as u32).max(src_start + 1);
            // Where to start playing inside the rough-cut video: its own position,
            // which differs from the output position once earlier material is cut.
            let rough_trim = candidate_rel(o).round().max(0.0) as u32;

            // keyframes inside this piece, map their frame through remap
            let mut kfs: Vec<crate::types::BBoxKeyframe> = Vec::new();
            for (t, x, y, w, h) in seg_keyframes.iter().filter(|(t, ..)| *t >= o && *t <= oe) {
                let f = remap.map(*t).map(|ot| (ot * fps_f).round() as u32).unwrap_or(start_frame);
                kfs.push(crate::types::BBoxKeyframe { frame: f, x: *x, y: *y, w: *w, h: *h });
            }
            // anchor piece edges with neighboring keyframe positions (interp/de-dup later)
            if kfs.is_empty() && seg.strategy != ShotStrategy::BlurFill {
                let pos = nearest_keyframe(&seg_keyframes, (o + oe) / 2.0)
                    .unwrap_or((0.18, 0.15, 0.64, 0.7));
                kfs.push(crate::types::BBoxKeyframe { frame: start_frame, x: pos.0, y: pos.1, w: pos.2, h: pos.3 });
            }
            kfs.sort_by_key(|k| k.frame);
            kfs.dedup_by(|a, b| a.frame == b.frame);

            // Second-subject track (split-screen two-speaker shots).
            let mut kfs2: Vec<crate::types::BBoxKeyframe> = Vec::new();
            if seg.split && !seg_second.is_empty() {
                for (t, x, y, w, h) in seg_second.iter().filter(|(t, ..)| *t >= o && *t <= oe) {
                    let f = remap.map(*t).map(|ot| (ot * fps_f).round() as u32).unwrap_or(start_frame);
                    kfs2.push(crate::types::BBoxKeyframe { frame: f, x: *x, y: *y, w: *w, h: *h });
                }
                if kfs2.is_empty() {
                    let pos = nearest_keyframe(&seg_second, (o + oe) / 2.0)
                        .unwrap_or((0.18, 0.15, 0.64, 0.7));
                    kfs2.push(crate::types::BBoxKeyframe { frame: start_frame, x: pos.0, y: pos.1, w: pos.2, h: pos.3 });
                }
                kfs2.sort_by_key(|k| k.frame);
                kfs2.dedup_by(|a, b| a.frame == b.frame);
            }

            shots.push(ShotSegment {
                start_frame,
                end_frame,
                strategy: seg.strategy.clone(),
                keyframes: kfs,
                second_keyframes: kfs2.clone(),
                split: seg.split && !kfs2.is_empty(),
                src_start_frame: src_start,
                src_end_frame: src_end,
                video_trim_frame: rough_trim,
            });
        }
    }
    shots.sort_by_key(|s| s.start_frame);

    let output_duration = remap.surviving_duration(candidate_start, candidate_end);
    if output_duration < 1.0 {
        anyhow::bail!("clip {clip_id} has no surviving material after edits");
    }
    let total_frames = (output_duration * fps_f).round() as u32;
    let repaired = ensure_full_coverage(&mut shots, total_frames);
    if repaired > 0 {
        tracing::warn!("{clip_id}: shots left {repaired} output frames uncovered (would render black) — extended neighbouring shots");
    }

    Ok(RenderManifest {
        clip_id: clip_id.to_string(),
        duration: output_duration,
        width,
        height,
        fps,
        source: source.display().to_string(),
        source_abs: source_abs.display().to_string(),
        source_width,
        source_height,
        shots,
        captions,
        cut_ranges: edit.cut_ranges.clone(),
        cuts_s: remap.dropped_spans(),
        joins: remap.joins(),
    })
}

/// Make the shots (ordered by output position) cover output frames
/// `0..total_frames` with no gaps. Any frame outside every shot renders black,
/// so a gap is closed by extending the earlier shot (the video simply keeps
/// playing), and a late first shot is pulled back to frame 0. Returns the
/// number of frames that were uncovered.
pub fn ensure_full_coverage(shots: &mut [ShotSegment], total_frames: u32) -> u32 {
    shots.sort_by_key(|s| s.src_start_frame);
    let mut repaired = 0u32;
    if let Some(first) = shots.first_mut() {
        let d = first.src_start_frame;
        if d > 0 {
            repaired += d;
            first.src_start_frame = 0;
            first.start_frame = first.start_frame.saturating_sub(d);
            first.video_trim_frame = first.video_trim_frame.saturating_sub(d);
        }
    }
    for i in 0..shots.len().saturating_sub(1) {
        let next_start = shots[i + 1].src_start_frame;
        if shots[i].src_end_frame < next_start {
            repaired += next_start - shots[i].src_end_frame;
        }
        shots[i].src_end_frame = next_start;
    }
    if let Some(last) = shots.last_mut() {
        if last.src_end_frame < total_frames {
            repaired += total_frames - last.src_end_frame;
            last.src_end_frame = total_frames;
        }
    }
    repaired
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    use crate::types::ShotStrategy;

    fn seg(a: u32, b: u32) -> ShotSegment {
        ShotSegment {
            start_frame: a + 1000,
            end_frame: b + 1000,
            strategy: ShotStrategy::FaceCrop,
            keyframes: vec![],
            second_keyframes: vec![],
            split: false,
            src_start_frame: a,
            src_end_frame: b,
            video_trim_frame: a,
        }
    }

    #[test]
    fn covers_start_gaps_and_end() {
        let mut v = vec![seg(16, 40), seg(60, 100), seg(100, 120)];
        let n = ensure_full_coverage(&mut v, 130);
        assert_eq!(v[0].src_start_frame, 0);
        assert_eq!(v[0].video_trim_frame, 0);
        assert_eq!(v[0].start_frame, 1000); // keyframe origin moves with the start
        assert_eq!(v[0].src_end_frame, v[1].src_start_frame);
        assert_eq!(v[1].src_end_frame, v[2].src_start_frame);
        assert_eq!(v.last().unwrap().src_end_frame, 130);
        assert_eq!(n, 16 + 20 + 10);
    }

    #[test]
    fn already_covered_is_untouched() {
        let mut v = vec![seg(0, 50), seg(50, 100)];
        assert_eq!(ensure_full_coverage(&mut v, 100), 0);
    }
}
