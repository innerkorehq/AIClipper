//! Stage 5.8 — Trajectory builder (deterministic).
//!
//! Consumes dense, per-sampled-frame detections (face/active-speaker or
//! saliency bbox) for a shot and emits sparse `BBoxKeyframe`s. Remotion
//! interpolates the rest at render time.
//!
//! Responsibilities (PRD §5.8):
//!   1. gap-fill between sampled frames
//!   2. smooth centroid/size (EMA) to remove detector jitter
//!   3. downsample to sparse control points (Ramer–Douglas–Peucker)
//!   4. shot-level constraints (reset per shot; nothing for `blur_fill`)
//!
//! This stage must stay pure and deterministic — it is the "LLM directs,
//! deterministic code computes" enforcement of crop geometry.

use crate::config::Config;
use crate::stages::shot_classify::{Layout, ShotClassified, SECOND_FACE_MIN_RATIO};
use crate::stages::speaker::ShotSpeaker;
use crate::types::{BBox, BBoxKeyframe, ShotSegment, ShotStrategy};

/// EMA smoothing window factor (fraction of series) for jitter removal.
const EMA_ALPHA: f32 = 0.45;
/// Max gap (s) between detections that we will linearly bridge.
const MAX_BRIDGE_GAP_S: f64 = 2.0;
/// RDP tolerance on normalized centroid + size.
const RDP_EPSILON: f32 = 0.012;

/// Build the render segment list for one clip's classified shots.
#[allow(clippy::too_many_arguments)]
pub fn build_segments_for_clip(
    cfg: &Config,
    fps: f64,
    clip_start: f64,
    shots: &[ShotClassified],
    speakers: &[ShotSpeaker],
) -> Vec<ShotSegment> {
    shots
        .iter()
        .map(|shot| build_one(cfg, fps, clip_start, shot, speakers_for_shot(speakers, shot.shot_list_idx)))
        .collect()
}

fn speakers_for_shot(speakers: &[ShotSpeaker], shot_list_idx: usize) -> Option<&ShotSpeaker> {
    speakers.iter().find(|s| s.shot_list_idx == shot_list_idx)
}

fn build_one(
    cfg: &Config,
    fps: f64,
    clip_start: f64,
    shot: &ShotClassified,
    speaker: Option<&ShotSpeaker>,
) -> ShotSegment {
    let start_frame = to_frame(shot.start, fps);
    let end_frame = to_frame(shot.end, fps);

    let mut seg = ShotSegment {
        start_frame,
        end_frame,
        strategy: shot.strategy.clone(),
        keyframes: Vec::new(),
        second_keyframes: Vec::new(),
        split: false,
        src_start_frame: 0,
        src_end_frame: 0,
        video_trim_frame: 0,
    };
    if shot.strategy == ShotStrategy::BlurFill {
        return seg; // §5.8.4: emit nothing for blur-fill shots
    }

    // Gather (time, target bbox) detections for this shot.
    let mut dets: Vec<(f64, BBox)> = match shot.strategy {
        ShotStrategy::FaceCrop => face_detections(shot, speaker),
        ShotStrategy::SaliencyCrop => saliency_detections(shot),
        ShotStrategy::BlurFill => Vec::new(),
    };

    if dets.is_empty() {
        // Deterministic fallback: drift gently into the shot's fallback region.
        let (ms, me) = (shot.start, shot.end);
        let a = fallback_wobble(&shot.fallback_bbox, 0.0);
        let b = fallback_wobble(&shot.fallback_bbox, 0.03);
        dets = vec![(ms, a), (me, b)];
    }

    seg.keyframes = keyframes_from_dets(cfg, shot, fps, clip_start, &dets);

    // Two actors: each person gets a fixed half of the screen for the whole
    // segment (left of frame → top, right → bottom). Slots follow the people,
    // not the active speaker, so nobody jumps between halves mid-shot.
    if shot.strategy == ShotStrategy::FaceCrop && shot.layout == Layout::Split {
        let (top, bottom) = split_tracks(cfg, shot);
        if top.len() >= 2 && bottom.len() >= 2 {
            let mut kfs1 = keyframes_from_dets(cfg, shot, fps, clip_start, &top);
            let mut kfs2 = keyframes_from_dets(cfg, shot, fps, clip_start, &bottom);
            if kfs1.len() >= 2 && kfs2.len() >= 2 {
                ensure_endpoints(&mut kfs1, shot.start, shot.end, fps, &shot.fallback_bbox);
                ensure_endpoints(&mut kfs2, shot.start, shot.end, fps, &shot.fallback_bbox);
                seg.keyframes = kfs1;
                seg.second_keyframes = kfs2;
                seg.split = true;
            }
        }
    }

    // Ensure endpoints exist (renderer expects a path covering the whole shot).
    ensure_endpoints(&mut seg.keyframes, shot.start, shot.end, fps, &shot.fallback_bbox);
    seg
}

/// Convert a (time, bbox) detection series into sparse remapped keyframes:
/// padding, uniform grid, EMA smoothing, RDP simplification, frame rounding.
fn keyframes_from_dets(
    cfg: &Config,
    shot: &ShotClassified,
    fps: f64,
    clip_start: f64,
    dets: &[(f64, BBox)],
) -> Vec<BBoxKeyframe> {
    if dets.is_empty() {
        return Vec::new();
    }
    let mut pts: Vec<(f64, f32, f32, f32, f32)> = Vec::new();
    for (t, b) in dets {
        let checked = expand_with_padding(*b, cfg.framing_padding);
        pts.push((*t, checked.center().0, checked.center().1, checked.w, checked.h));
    }
    // Gap-fill + EMA smoothing over the time grid.
    let smoothed = smooth_series(&pts, shot.start, shot.end, cfg.sample_interval.clamp(0.1, 1.0), MAX_BRIDGE_GAP_S);
    // RDP simplification down to sparse control points.
    let simplified = rdp(&smoothed, RDP_EPSILON);

    let mut kfs = Vec::new();
    for (t, cx, cy, w, h) in simplified {
        let f = to_frame(t.max(shot.start).min(shot.end), fps);
        if f < clip_start as u32 {
            continue;
        }
        let half_w = (w / 2.0).min(cx).min(1.0 - cx);
        let half_h = (h / 2.0).min(cy).min(1.0 - cy);
        kfs.push(BBoxKeyframe {
            frame: f,
            x: (cx - half_w).clamp(0.0, 1.0),
            y: (cy - half_h).clamp(0.0, 1.0),
            w: (half_w * 2.0).max(0.05),
            h: (half_h * 2.0).max(0.05),
        });
    }
    kfs
}

/// Two stable person tracks for a split shot: `(top, bottom)` detections.
///
/// Per sample the two largest confident faces are taken (the second must be at
/// least `SECOND_FACE_MIN_RATIO` of the first's area). The first assignment puts
/// the left person on top; afterwards each face keeps the slot whose previous
/// position it is closest to, so the assignment never flips when people talk.
fn split_tracks(cfg: &Config, shot: &ShotClassified) -> (Vec<(f64, BBox)>, Vec<(f64, BBox)>) {
    let area = |b: &BBox| b.w * b.h;
    let dist = |a: &BBox, c: (f32, f32)| {
        let (cx, cy) = a.center();
        (cx - c.0).abs() + (cy - c.1).abs()
    };
    let mut slots: Option<[(f32, f32); 2]> = None;
    let (mut top, mut bottom) = (Vec::new(), Vec::new());
    for s in &shot.samples {
        let mut faces: Vec<BBox> =
            s.faces.iter().filter(|f| f.confidence >= cfg.face_confidence).map(|f| f.bbox).collect();
        faces.sort_by(|a, b| area(b).partial_cmp(&area(a)).unwrap_or(std::cmp::Ordering::Equal));
        if faces.len() < 2 || area(&faces[1]) < SECOND_FACE_MIN_RATIO * area(&faces[0]) {
            continue;
        }
        let (f0, f1) = (faces[0], faces[1]);
        let (t, b) = match slots {
            None => {
                if f0.center().0 <= f1.center().0 { (f0, f1) } else { (f1, f0) }
            }
            Some([st, sb]) => {
                if dist(&f0, st) + dist(&f1, sb) <= dist(&f1, st) + dist(&f0, sb) { (f0, f1) } else { (f1, f0) }
            }
        };
        slots = Some([t.center(), b.center()]);
        top.push((s.time, t));
        bottom.push((s.time, b));
    }
    (top, bottom)
}

fn face_detections(shot: &ShotClassified, speaker: Option<&ShotSpeaker>) -> Vec<(f64, BBox)> {
    if let Some(sp) = speaker {
        if !sp.detections.is_empty() {
            return sp.detections.clone();
        }
    }
    // No resolved track: use the largest detections per sample.
    shot.samples
        .iter()
        .filter_map(|s| {
            s.faces
                .iter()
                .max_by(|a, b| (a.bbox.w * a.bbox.h).partial_cmp(&(b.bbox.w * b.bbox.h)).unwrap())
                .map(|f| (s.time, f.bbox))
        })
        .collect()
}

fn saliency_detections(shot: &ShotClassified) -> Vec<(f64, BBox)> {
    shot.samples
        .iter()
        .filter_map(|s| s.saliency.as_ref().map(|g| (s.time, crate::vision::saliency_target_bbox(g, 0.5, 0.5))))
        .collect()
}

fn fallback_wobble(bbox: &BBox, amount: f32) -> BBox {
    // Tiny deterministic drift so the "static" fallback path is still a defined motion.
    BBox {
        x: bbox.x,
        y: bbox.y,
        w: bbox.w * (1.0 - amount),
        h: bbox.h * (1.0 - amount),
    }
}

/// Expand a raw detection box by `pad` on each side, kept in [0,1].
fn expand_with_padding(b: BBox, pad: f32) -> BBox {
    let pw = (b.w * pad).min(0.15);
    let ph = (b.h * pad).min(0.15);
    let mut out = BBox {
        x: (b.x - pw).max(0.0),
        y: (b.y - ph).max(0.0),
        w: (b.w + 2.0 * pw).min(1.0),
        h: (b.h + 2.0 * ph).min(1.0),
    };
    // Keep at least a minimal visible region.
    out.w = out.w.max(0.12);
    out.h = out.h.max(0.12);
    out
}

/// Interpolate the raw series onto a uniform time grid (gap-filled + EMA-smoothed).
fn smooth_series(
    pts: &[(f64, f32, f32, f32, f32)],
    t0: f64,
    t1: f64,
    step: f64,
    max_gap: f64,
) -> Vec<(f64, f32, f32, f32, f32)> {
    if pts.is_empty() {
        return Vec::new();
    }
    let grid_ts: Vec<f64> = {
        let mut v = Vec::new();
        let mut t = t0;
        while t <= t1 {
            v.push(t);
            t += step;
        }
        if *v.last().unwrap_or(&t0) < t1 {
            v.push(t1);
        }
        v
    };

    // Interpolate value at grid point t (linear between samples, clamp edges).
    let interp = |t: f64| -> (f32, f32, f32, f32) {
        let first = pts.first().unwrap();
        let last = pts.last().unwrap();
        if t <= pts[0].0 {
            return (first.1, first.2, first.3, first.4);
        }
        if t >= last.0 {
            return (last.1, last.2, last.3, last.4);
        }
        for (i, p) in pts.windows(2).enumerate() {
            let (a, b) = (&p[0], &p[1]);
            if t >= a.0 && t <= b.0 {
                if b.0 - a.0 > max_gap {
                    // Don't bridge huge gaps — hold at endpoint instead.
                    return (a.1, a.2, a.3, a.4);
                }
                let f = ((t - a.0) / (b.0 - a.0)) as f32;
                return lerp5(a, b, f);
            }
            let _ = i;
        }
        (last.1, last.2, last.3, last.4)
    };

    let raw: Vec<(f64, f32, f32, f32, f32)> = grid_ts.iter().map(|t| (*t, interp(*t).0, interp(*t).1, interp(*t).2, interp(*t).3)).collect();

    // EMA smoothing.
    let mut out: Vec<(f64, f32, f32, f32, f32)> = Vec::with_capacity(raw.len());
    let alpha = EMA_ALPHA;
    if let Some(first) = raw.first() {
        out.push(*first);
        for p in raw.iter().skip(1) {
            let prev = out.last().unwrap();
            out.push((
                p.0,
                ema(prev.1, p.1, alpha),
                ema(prev.2, p.2, alpha),
                ema(prev.3, p.3, alpha),
                ema(prev.4, p.4, alpha),
            ));
        }
    }
    out
}

fn lerp5(a: &(f64, f32, f32, f32, f32), b: &(f64, f32, f32, f32, f32), f: f32) -> (f32, f32, f32, f32) {
    (a.1 + (b.1 - a.1) * f, a.2 + (b.2 - a.2) * f, a.3 + (b.3 - a.3) * f, a.4 + (b.4 - a.4) * f)
}

fn ema(prev: f32, cur: f32, alpha: f32) -> f32 {
    prev + alpha * (cur - prev)
}

/// Ramer–Douglas–Peucker over (t, cx, cy, w, h) with normalized epsilon.
fn rdp(points: &[(f64, f32, f32, f32, f32)], epsilon: f32) -> Vec<(f64, f32, f32, f32, f32)> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    rdp_rec(points, 0, points.len() - 1, epsilon, &mut keep);
    points
        .iter()
        .zip(keep.iter())
        .filter(|(_, k)| **k)
        .map(|(p, _)| *p)
        .collect()
}

fn rdp_rec(
    pts: &[(f64, f32, f32, f32, f32)],
    first: usize,
    last: usize,
    epsilon: f32,
    keep: &mut Vec<bool>,
) {
    if last <= first + 1 {
        return;
    }
    let seg_dist = perp_dist(&pts[first], &pts[last]);
    let (mut dmax, mut index) = (f32::MIN, 0usize);
    for i in (first + 1)..last {
        let d = perp_dist(&pts[i], &pts[first]);
        let _ = seg_dist;
        if d > dmax {
            dmax = d;
            index = i;
        }
    }
    if dmax > epsilon {
        keep[index] = true;
        rdp_rec(pts, first, index, epsilon, keep);
        rdp_rec(pts, index, last, epsilon, keep);
    }
}

/// Perpendicular-ish distance treating all 4 dims (t normalized by 60s).
fn perp_dist(p: &(f64, f32, f32, f32, f32), base: &(f64, f32, f32, f32, f32)) -> f32 {
    let tn = ((p.0 - base.0).abs() / 60.0) as f32;
    let dx = p.1 - base.1;
    let dy = p.2 - base.2;
    let dw = p.3 - base.3;
    let dh = p.4 - base.4;
    (dx * dx + dy * dy + dw * dw + dh * dh + tn * tn).sqrt()
}

fn ensure_endpoints(kfs: &mut Vec<BBoxKeyframe>, shot_start: f64, shot_end: f64, fps: f64, fallback: &BBox) {
    let (sf, ef) = (to_frame(shot_start, fps), to_frame(shot_end, fps));
    let has_start = kfs.iter().any(|k| k.frame == sf);
    let has_end = kfs.iter().any(|k| k.frame == ef);
    let first = kfs.first().copied();
    let last = kfs.last().copied();
    let fb = BBoxKeyframe { frame: sf, x: fallback.x, y: fallback.y, w: fallback.w, h: fallback.h };

    if !has_start {
        let anchor = first.map(|k| BBoxKeyframe { frame: sf, ..k }).unwrap_or(fb);
        kfs.insert(0, anchor);
    }
    if !has_end {
        let mut endk = last.unwrap_or(fb);
        endk.frame = ef;
        kfs.push(endk);
    }
}

fn to_frame(t: f64, fps: f64) -> u32 {
    (t.max(0.0) * fps).round() as u32
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use crate::stages::shot_classify::ShotSample;
    use crate::types::ShotFrame;
    use crate::vision::FaceRect;

    fn face(x: f32) -> FaceRect {
        FaceRect { bbox: BBox { x, y: 0.3, w: 0.12, h: 0.2 }, confidence: 0.9 }
    }

    fn shot(per_sample: Vec<Vec<FaceRect>>) -> ShotClassified {
        ShotClassified {
            shot_list_idx: 0,
            start: 0.0,
            end: per_sample.len() as f64,
            frame: ShotFrame::Speaker,
            target_track_id: None,
            strategy: ShotStrategy::FaceCrop,
            layout: Layout::Split,
            samples: per_sample
                .into_iter()
                .enumerate()
                .map(|(i, faces)| ShotSample { time: i as f64, frame_idx: i as u32, faces, saliency: None })
                .collect(),
            fallback_bbox: BBox { x: 0.2, y: 0.2, w: 0.6, h: 0.6 },
        }
    }

    #[test]
    fn people_keep_their_half_regardless_of_who_is_larger() {
        let cfg = Config::default_with_workdir(std::path::PathBuf::from("w"));
        // left person ~x0.2, right person ~x0.7; detector order and relative size
        // flip from sample to sample (as when the active speaker changes).
        let mut a = face(0.2);
        let mut b = face(0.7);
        a.bbox.w = 0.15; // left larger
        let s1 = vec![a, b];
        b.bbox.w = 0.16; // right larger
        let s2 = vec![b, face(0.2)];
        let s3 = vec![face(0.21), b];
        let (top, bottom) = split_tracks(&cfg, &shot(vec![s1, s2, s3.clone(), s3]));
        assert_eq!(top.len(), 4);
        assert!(top.iter().all(|(_, b)| b.x < 0.4), "top is always the left person");
        assert!(bottom.iter().all(|(_, b)| b.x > 0.5), "bottom is always the right person");
    }

    #[test]
    fn ignores_small_bystander_faces() {
        let cfg = Config::default_with_workdir(std::path::PathBuf::from("w"));
        let mut tiny = face(0.8);
        tiny.bbox.w = 0.03;
        tiny.bbox.h = 0.04;
        let (top, _) = split_tracks(&cfg, &shot(vec![vec![face(0.2), tiny]]));
        assert!(top.is_empty());
    }
}
