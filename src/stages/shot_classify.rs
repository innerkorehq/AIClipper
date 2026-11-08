//! Stage 5.6 — Shot classification.
//!
//! Per shot (not per clip): sample frames, run Vision face detector, and pick
//! one of three layouts for every stretch of the shot:
//!
//!   * `Single` — one clear actor            → face_crop (full-bleed crop)
//!   * `Split`  — two clear actors           → face_crop + stacked halves
//!   * `Wide`   — no clear face (action etc.) → blur_fill (whole scene centred)
//!
//! The layout is decided per sampled frame, smoothed (min dwell) and the shot
//! is cut wherever the layout changes or a hard scene cut occurs, so the
//! layout follows the scene instead of being fixed per LLM shot.

use crate::config::Config;
use crate::types::{BBox, ShotFrame, ShotListEntry, ShotStrategy};
use crate::vision::{saliency_target_bbox, FaceRect, SaliencyGrid, VisionService};
use anyhow::Result;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ShotSample {
    pub time: f64,
    pub frame_idx: u32,
    pub faces: Vec<FaceRect>,
    pub saliency: Option<SaliencyGrid>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ShotClassified {
    pub shot_list_idx: usize,
    pub start: f64,
    pub end: f64,
    pub frame: ShotFrame,
    pub target_track_id: Option<String>,
    pub strategy: ShotStrategy,
    /// Layout this segment is framed with.
    pub layout: Layout,
    /// Dense samples this shot was classified from (sampled at sample_interval).
    pub samples: Vec<ShotSample>,
    /// Deterministic fallback target (mid-frame) for shots with no detections.
    pub fallback_bbox: BBox,
}

/// Framing layout of a shot segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Single,
    Split,
    Wide,
}

/// A second face only counts (two-actor layout) if its area is at least this
/// fraction of the primary face — ignores background bystanders.
pub const SECOND_FACE_MIN_RATIO: f32 = 0.35;
/// Layout must hold at least this long (s) or it is absorbed by a neighbour —
/// stops detector flicker from toggling layouts.
const MIN_LAYOUT_DWELL_S: f64 = 1.5;
/// Ignore scene cuts this close (s) to a shot edge.
const CUT_EDGE_MARGIN_S: f64 = 0.4;

/// Classify every shot in a clip's shot list. `rough` is the candidate's rough
/// cut (its timeline starts at 0), `offset_s` is the candidate's start in
/// absolute source time. Vision extraction uses rough-relative times; sample
/// metadata stays in absolute source time for the trajectory/remap stages.
pub async fn classify_shots(
    cfg: &Config,
    vision: &VisionService,
    rough: &std::path::Path,
    offset_s: f64,
    fps: f64,
    shot_list: &[ShotListEntry],
    scene_cuts: &[f64],
) -> Result<Vec<ShotClassified>> {
    if !vision.available() {
        anyhow::bail!("shot classification requires the Vision shim (build with `cargo build`)");
    }
    let mut out = Vec::with_capacity(shot_list.len());
    for shot in shot_list.iter() {
        let segments = classify_one(cfg, vision, rough, offset_s, fps, shot, scene_cuts).await?;
        out.extend(segments);
    }
    // Segments get unique sequential ids (speaker/trajectory stages key on them).
    for (i, seg) in out.iter_mut().enumerate() {
        seg.shot_list_idx = i;
    }
    Ok(out)
}

async fn classify_one(
    cfg: &Config,
    vision: &VisionService,
    rough: &std::path::Path,
    offset_s: f64,
    fps: f64,
    shot: &ShotListEntry,
    scene_cuts: &[f64],
) -> Result<Vec<ShotClassified>> {
    let mut t = shot.start;
    let mut times = Vec::new();
    while t < shot.end {
        times.push(t);
        t += cfg.sample_interval;
    }
    if times.is_empty() {
        times.push((shot.start + shot.end) / 2.0);
    }

    // Vision reads from the rough-cut file, whose timeline starts at 0.
    let rough_times: Vec<f64> = times.iter().map(|t| (t - offset_s).max(0.0)).collect();
    let face_sets = vision.faces_at(rough, &rough_times).await?;
    let saliencies = vision.saliency_at(rough, &rough_times).await?;

    let mut samples = Vec::with_capacity(times.len());
    for (i, time) in times.iter().enumerate() {
        samples.push(ShotSample {
            time: *time,
            frame_idx: (time * fps).round() as u32,
            faces: face_sets[i].clone(),
            saliency: saliencies[i].clone(),
        });
    }

    // Per-sample layout, smoothed within each hard-cut region.
    let cuts: Vec<f64> = scene_cuts
        .iter()
        .copied()
        .filter(|c| *c > shot.start + CUT_EDGE_MARGIN_S && *c < shot.end - CUT_EDGE_MARGIN_S)
        .collect();
    let region_of = |t: f64| cuts.iter().filter(|c| **c <= t).count();
    let raw: Vec<Layout> = samples.iter().map(|s| sample_layout(cfg, s)).collect();
    let mut labels = raw.clone();
    let mut i = 0;
    while i < samples.len() {
        let r = region_of(samples[i].time);
        let mut j = i;
        while j < samples.len() && region_of(samples[j].time) == r {
            j += 1;
        }
        smooth_layouts(&mut labels[i..j], cfg.sample_interval);
        i = j;
    }

    // Cut into segments at region and layout changes.
    let mut out = Vec::new();
    let mut seg_start = 0usize;
    for k in 1..=samples.len() {
        let boundary = k == samples.len()
            || labels[k] != labels[seg_start]
            || region_of(samples[k].time) != region_of(samples[seg_start].time);
        if !boundary {
            continue;
        }
        let layout = labels[seg_start];
        let start = if seg_start == 0 {
            shot.start
        } else if region_of(samples[seg_start].time) != region_of(samples[seg_start - 1].time) {
            // snap to the hard cut itself
            cuts[region_of(samples[seg_start].time) - 1]
        } else {
            samples[seg_start].time
        };
        let end = if k == samples.len() {
            shot.end
        } else if region_of(samples[k].time) != region_of(samples[seg_start].time) {
            cuts[region_of(samples[k].time) - 1]
        } else {
            samples[k].time
        };
        let seg_samples = samples[seg_start..k].to_vec();
        let strategy = match layout {
            Layout::Wide => ShotStrategy::BlurFill,
            Layout::Single | Layout::Split => ShotStrategy::FaceCrop,
        };
        tracing::info!(
            "shot {}: [{:.1}s–{:.1}s] frame={:?} → {:?} ({:?})",
            out.len(),
            start,
            end,
            shot.frame,
            layout,
            strategy
        );
        out.push(ShotClassified {
            shot_list_idx: 0,
            start,
            end,
            frame: shot.frame.clone(),
            target_track_id: shot.target_track_id.clone(),
            strategy,
            layout,
            fallback_bbox: saliency_target_bbox_or_center(seg_samples.first().and_then(|s| s.saliency.as_ref())),
            samples: seg_samples,
        });
        seg_start = k;
    }
    Ok(out)
}

/// Layout implied by one sampled frame's faces.
fn sample_layout(cfg: &Config, s: &ShotSample) -> Layout {
    let mut areas: Vec<f32> = s
        .faces
        .iter()
        .filter(|f| f.confidence >= cfg.face_confidence)
        .map(|f| f.bbox.w * f.bbox.h)
        .collect();
    areas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    match areas.as_slice() {
        [] => Layout::Wide,
        [_] => Layout::Single,
        [a, b, ..] => {
            if *b >= *a * SECOND_FACE_MIN_RATIO {
                Layout::Split
            } else {
                Layout::Single
            }
        }
    }
}

/// Absorb layout runs shorter than the minimum dwell into a neighbour
/// (preferring the longer one) until every run is long enough.
fn smooth_layouts(labels: &mut [Layout], step: f64) {
    let min_run = ((MIN_LAYOUT_DWELL_S / step.max(0.05)).ceil() as usize).max(1);
    loop {
        // collect runs
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut s0 = 0;
        for k in 1..=labels.len() {
            if k == labels.len() || labels[k] != labels[s0] {
                runs.push((s0, k));
                s0 = k;
            }
        }
        if runs.len() <= 1 {
            return;
        }
        // shortest run below the minimum
        let Some((ri, &(a, b))) = runs
            .iter()
            .enumerate()
            .filter(|(_, (a, b))| b - a < min_run)
            .min_by_key(|(_, (a, b))| b - a)
        else {
            return;
        };
        let prev = if ri > 0 { Some(runs[ri - 1]) } else { None };
        let next = runs.get(ri + 1).copied();
        let to = match (prev, next) {
            (Some(p), Some(n)) => if p.1 - p.0 >= n.1 - n.0 { labels[p.0] } else { labels[n.0] },
            (Some(p), None) => labels[p.0],
            (None, Some(n)) => labels[n.0],
            (None, None) => return,
        };
        for l in labels[a..b].iter_mut() {
            *l = to;
        }
    }
}

fn saliency_target_bbox_or_center(saliency: Option<&SaliencyGrid>) -> BBox {
    match saliency {
        Some(g) => saliency_target_bbox(g, 0.5, 0.5),
        None => BBox { x: 0.18, y: 0.15, w: 0.64, h: 0.7 }, // center-ish portrait
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use Layout::*;

    #[test]
    fn flicker_is_absorbed() {
        // 0.5s samples: a single-frame "Split" blip inside Single is removed.
        let mut l = vec![Single, Single, Single, Split, Single, Single, Single];
        smooth_layouts(&mut l, 0.5);
        assert!(l.iter().all(|x| *x == Single));
    }

    #[test]
    fn sustained_change_is_kept() {
        let mut l = vec![Single, Single, Single, Single, Split, Split, Split, Split];
        smooth_layouts(&mut l, 0.5);
        assert_eq!(&l[..4], &[Single; 4]);
        assert_eq!(&l[4..], &[Split; 4]);
    }

    #[test]
    fn short_leading_run_joins_next() {
        let mut l = vec![Wide, Single, Single, Single, Single];
        smooth_layouts(&mut l, 0.5);
        assert!(l.iter().all(|x| *x == Single));
    }

    #[test]
    fn all_short_runs_collapse_to_one() {
        let mut l = vec![Single, Split, Wide];
        smooth_layouts(&mut l, 0.5);
        assert!(l.windows(2).all(|w| w[0] == w[1]));
    }
}
