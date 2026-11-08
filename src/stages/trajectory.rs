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
use crate::stages::framing::{Algo, Subject, Tracker, Window};
use crate::stages::motion::MotionTrack;
use crate::stages::shot_classify::{Layout, ShotClassified, SECOND_FACE_MIN_RATIO};
use crate::stages::speaker::ShotSpeaker;
use crate::types::{BBox, BBoxKeyframe, ShotSegment, ShotStrategy};

/// Max gap (s) between detections that we will linearly bridge.
const MAX_BRIDGE_GAP_S: f64 = 2.0;
/// RDP tolerance on normalized centroid + size.
const RDP_EPSILON: f32 = 0.012;

/// Build the render segment list for one clip's classified shots. `algos[i]` is
/// the framing algorithm chosen for `shots[i]` (see `stages::chooser`); a shot
/// without one is framed the pre-chooser way. `motion` is the clip's motion
/// analysis (rough-cut timeline, so `clip_start` converts shot times to it).
#[allow(clippy::too_many_arguments)]
pub fn build_segments_for_clip(
    cfg: &Config,
    fps: f64,
    clip_start: f64,
    shots: &[ShotClassified],
    speakers: &[ShotSpeaker],
    algos: &[Algo],
    motion: Option<&MotionTrack>,
) -> Vec<ShotSegment> {
    shots
        .iter()
        .enumerate()
        .map(|(i, shot)| {
            let algo = algos.get(i).copied().unwrap_or_else(|| legacy_algo(shot));
            build_one(cfg, fps, clip_start, shot, speakers_for_shot(speakers, shot.shot_list_idx), algo, motion)
        })
        .collect()
}

/// What the classifier's layout implied before there was a chooser.
fn legacy_algo(shot: &ShotClassified) -> Algo {
    match (shot.strategy.clone(), shot.layout) {
        (ShotStrategy::BlurFill, _) | (_, Layout::Wide) => Algo::WideBlur,
        (_, Layout::Split) => Algo::SplitTwo,
        _ => Algo::SpeakerFollow,
    }
}

fn speakers_for_shot(speakers: &[ShotSpeaker], shot_list_idx: usize) -> Option<&ShotSpeaker> {
    speakers.iter().find(|s| s.shot_list_idx == shot_list_idx)
}

fn has_faces(cfg: &Config, shot: &ShotClassified) -> bool {
    shot.samples.iter().any(|s| s.faces.iter().any(|f| f.confidence >= cfg.face_confidence))
}

fn build_one(
    cfg: &Config,
    fps: f64,
    clip_start: f64,
    shot: &ShotClassified,
    speaker: Option<&ShotSpeaker>,
    algo: Algo,
    motion: Option<&MotionTrack>,
) -> ShotSegment {
    let start_frame = to_frame(shot.start, fps);
    let end_frame = to_frame(shot.end, fps);
    let faces = has_faces(cfg, shot);

    let strategy = match algo {
        Algo::WideBlur => ShotStrategy::BlurFill,
        Algo::SaliencyFollow | Algo::MotionFollow | Algo::GroupFit => ShotStrategy::SaliencyCrop,
        Algo::Locked if !faces => ShotStrategy::SaliencyCrop,
        _ => ShotStrategy::FaceCrop,
    };
    let (window, zoom) = match algo.window() {
        Window::Max => ("max", 1.0),
        Window::Fit => ("fit", 1.0),
        Window::Zoom(z) => ("zoom", z),
        Window::Split => ("split", 1.0),
        Window::Contain => ("contain", 1.0),
    };
    let mut seg = ShotSegment {
        start_frame,
        end_frame,
        strategy,
        algorithm: algo.name().to_string(),
        window: window.to_string(),
        zoom,
        subject: match algo.subject(faces) {
            Subject::Face => "face",
            Subject::Region => "region",
        }
        .to_string(),
        keyframes: Vec::new(),
        second_keyframes: Vec::new(),
        split: false,
        src_start_frame: 0,
        src_end_frame: 0,
        video_trim_frame: 0,
    };
    if algo == Algo::WideBlur {
        return seg; // nothing to track: the whole frame is shown
    }

    // What this algorithm follows.
    let mut dets: Vec<(f64, BBox)> = match algo.tracker() {
        Tracker::LargestFace => largest_face_detections(cfg, shot),
        Tracker::ActiveSpeaker | Tracker::TwoFaces => face_detections(shot, speaker),
        Tracker::FaceUnion => union_detections(cfg, shot),
        Tracker::FacesOrSaliency => {
            let d = largest_face_detections(cfg, shot);
            if d.is_empty() { saliency_detections(shot) } else { d }
        }
        Tracker::Saliency => saliency_detections(shot),
        Tracker::Motion => motion_detections(motion, shot, clip_start),
        Tracker::None => Vec::new(),
    };
    if algo == Algo::Locked {
        dets = collapse_to_median(shot, &dets);
    }

    if dets.is_empty() {
        // Deterministic fallback: drift gently into the shot's fallback region.
        let (ms, me) = (shot.start, shot.end);
        let a = fallback_wobble(&shot.fallback_bbox, 0.0);
        let b = fallback_wobble(&shot.fallback_bbox, 0.03);
        dets = vec![(ms, a), (me, b)];
    }

    seg.keyframes = keyframes_from_dets(cfg, shot, fps, clip_start, &dets, algo.ema_alpha());

    // Two actors: each person gets a fixed half of the screen for the whole
    // segment (left of frame → top, right → bottom). Slots follow the people,
    // not the active speaker, so nobody jumps between halves mid-shot.
    if algo == Algo::SplitTwo {
        let (top, bottom) = split_tracks(cfg, shot);
        if top.len() >= 2 && bottom.len() >= 2 {
            let mut kfs1 = keyframes_from_dets(cfg, shot, fps, clip_start, &top, algo.ema_alpha());
            let mut kfs2 = keyframes_from_dets(cfg, shot, fps, clip_start, &bottom, algo.ema_alpha());
            if kfs1.len() >= 2 && kfs2.len() >= 2 {
                ensure_endpoints(&mut kfs1, shot.start, shot.end, fps, &shot.fallback_bbox);
                ensure_endpoints(&mut kfs2, shot.start, shot.end, fps, &shot.fallback_bbox);
                seg.keyframes = kfs1;
                seg.second_keyframes = kfs2;
                seg.split = true;
            }
        }
        if !seg.split {
            // not enough two-person data to split: frame the main person instead
            seg.algorithm = Algo::FaceFollow.name().to_string();
            seg.window = "max".to_string();
        }
    }

    // Ensure endpoints exist (renderer expects a path covering the whole shot).
    ensure_endpoints(&mut seg.keyframes, shot.start, shot.end, fps, &shot.fallback_bbox);
    seg
}

/// The largest confident face per sampled frame.
fn largest_face_detections(cfg: &Config, shot: &ShotClassified) -> Vec<(f64, BBox)> {
    shot.samples
        .iter()
        .filter_map(|s| {
            s.faces
                .iter()
                .filter(|f| f.confidence >= cfg.face_confidence)
                .max_by(|a, b| (a.bbox.w * a.bbox.h).partial_cmp(&(b.bbox.w * b.bbox.h)).unwrap_or(std::cmp::Ordering::Equal))
                .map(|f| (s.time, f.bbox))
        })
        .collect()
}

/// One box around *all* confident faces per sampled frame (group framing).
fn union_detections(cfg: &Config, shot: &ShotClassified) -> Vec<(f64, BBox)> {
    shot.samples
        .iter()
        .filter_map(|s| {
            let fs: Vec<&BBox> = s.faces.iter().filter(|f| f.confidence >= cfg.face_confidence).map(|f| &f.bbox).collect();
            if fs.is_empty() {
                return None;
            }
            let x0 = fs.iter().map(|b| b.x).fold(f32::MAX, f32::min);
            let y0 = fs.iter().map(|b| b.y).fold(f32::MAX, f32::min);
            let x1 = fs.iter().map(|b| b.x + b.w).fold(0.0, f32::max);
            let y1 = fs.iter().map(|b| b.y + b.h).fold(0.0, f32::max);
            Some((s.time, BBox { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }))
        })
        .collect()
}

/// Boxes around the moving area during the shot (motion is on the rough-cut
/// timeline, shots are in absolute source time).
fn motion_detections(motion: Option<&MotionTrack>, shot: &ShotClassified, clip_start: f64) -> Vec<(f64, BBox)> {
    motion
        .map(|m| m.boxes(shot.start - clip_start, shot.end - clip_start).into_iter().map(|(t, b)| (t + clip_start, b)).collect())
        .unwrap_or_default()
}

/// Replace a track by its median box held for the whole shot: a fixed window.
fn collapse_to_median(shot: &ShotClassified, dets: &[(f64, BBox)]) -> Vec<(f64, BBox)> {
    if dets.is_empty() {
        return Vec::new();
    }
    let med = |f: fn(&BBox) -> f32| {
        let mut v: Vec<f32> = dets.iter().map(|(_, b)| f(b)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };
    let b = BBox { x: med(|b| b.x), y: med(|b| b.y), w: med(|b| b.w), h: med(|b| b.h) };
    vec![(shot.start, b), (shot.end, b)]
}

/// Convert a (time, bbox) detection series into sparse remapped keyframes:
/// padding, uniform grid, EMA smoothing, RDP simplification, frame rounding.
fn keyframes_from_dets(
    cfg: &Config,
    shot: &ShotClassified,
    fps: f64,
    clip_start: f64,
    dets: &[(f64, BBox)],
    alpha: f32,
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
    let smoothed = smooth_series(&pts, shot.start, shot.end, cfg.sample_interval.clamp(0.1, 1.0), MAX_BRIDGE_GAP_S, alpha);
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
    alpha: f32,
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

    // Zero-phase smoothing: an EMA forward, then the same EMA backward over its own
    // output. The render is offline, so the camera need not be causal; a one-way EMA
    // makes the crop trail a moving subject (measured: a fast mover was fully in view
    // only 40% of the time), the forward-backward pass keeps the smoothness but
    // removes the lag.
    let ema_pass = |input: &[(f64, f32, f32, f32, f32)]| -> Vec<(f64, f32, f32, f32, f32)> {
        let mut out: Vec<(f64, f32, f32, f32, f32)> = Vec::with_capacity(input.len());
        if let Some(first) = input.first() {
            out.push(*first);
            for p in input.iter().skip(1) {
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
    };
    let forward = ema_pass(&raw);
    let mut backward: Vec<(f64, f32, f32, f32, f32)> = forward.iter().rev().cloned().collect();
    let times: Vec<f64> = forward.iter().map(|p| p.0).collect();
    backward = ema_pass(&backward);
    backward.reverse();
    // restore the original time stamps (reversal moved them with the samples)
    backward.iter_mut().zip(times).for_each(|(p, t)| p.0 = t);
    backward
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

#[cfg(test)]
mod algo_tests {
    use super::*;
    use crate::stages::motion::MotionSample;
    use crate::stages::shot_classify::ShotSample;
    use crate::types::ShotFrame;
    use crate::vision::FaceRect;

    fn face(x: f32, y: f32, w: f32) -> FaceRect {
        FaceRect { bbox: BBox { x, y, w, h: w * 1.4 }, confidence: 0.9 }
    }

    fn shot(per_sample: Vec<Vec<FaceRect>>) -> ShotClassified {
        ShotClassified {
            shot_list_idx: 0,
            start: 10.0,
            end: 10.0 + per_sample.len() as f64 * 0.5,
            frame: ShotFrame::Speaker,
            target_track_id: None,
            strategy: ShotStrategy::FaceCrop,
            layout: Layout::Single,
            samples: per_sample
                .into_iter()
                .enumerate()
                .map(|(i, faces)| ShotSample { time: 10.0 + i as f64 * 0.5, frame_idx: i as u32, faces, saliency: None })
                .collect(),
            fallback_bbox: BBox { x: 0.2, y: 0.2, w: 0.6, h: 0.6 },
        }
    }

    fn cfg() -> Config {
        Config::default_with_workdir(std::path::PathBuf::from("w"))
    }

    #[test]
    fn union_box_contains_every_face() {
        let s = shot(vec![vec![face(0.1, 0.3, 0.1), face(0.45, 0.35, 0.1), face(0.8, 0.3, 0.1)]; 4]);
        let u = union_detections(&cfg(), &s);
        assert_eq!(u.len(), 4);
        let b = u[0].1;
        assert!(b.x <= 0.1 + 1e-6 && b.x + b.w >= 0.9 - 1e-6, "spans first to last face: {b:?}");
    }

    #[test]
    fn largest_face_wins_and_low_confidence_is_ignored() {
        let mut weak = face(0.7, 0.3, 0.3);
        weak.confidence = 0.1; // biggest but not a face
        let s = shot(vec![vec![face(0.2, 0.3, 0.1), face(0.5, 0.3, 0.15), weak]]);
        let d = largest_face_detections(&cfg(), &s);
        assert_eq!(d.len(), 1);
        assert!((d[0].1.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn locked_holds_one_median_box_for_the_whole_shot() {
        let s = shot(vec![vec![face(0.20, 0.3, 0.1)], vec![face(0.30, 0.3, 0.1)], vec![face(0.80, 0.3, 0.1)], vec![face(0.31, 0.3, 0.1)], vec![face(0.29, 0.3, 0.1)]]);
        let held = collapse_to_median(&s, &largest_face_detections(&cfg(), &s));
        assert_eq!(held.len(), 2);
        assert_eq!(held[0].1, held[1].1, "no motion");
        assert!((held[0].1.x - 0.30).abs() < 1e-6, "the median ignores the 0.80 outlier: {:?}", held[0].1);
        assert!((held[0].0 - s.start).abs() < 1e-9 && (held[1].0 - s.end).abs() < 1e-9);
        assert!(collapse_to_median(&s, &[]).is_empty());
    }

    #[test]
    fn motion_boxes_are_mapped_into_the_shots_time() {
        let bx = BBox { x: 0.6, y: 0.3, w: 0.2, h: 0.2 };
        let m = MotionTrack {
            samples: (0..8)
                .map(|i| MotionSample { time: i as f64 * 0.5, energy: 0.1, bbox: Some(bx), cut: false })
                .collect(),
        };
        // the clip starts at source 100s; this shot is source 101..103 = rough 1..3
        let mut s = shot(vec![vec![]; 4]);
        s.start = 101.0;
        s.end = 103.0;
        let d = motion_detections(Some(&m), &s, 100.0);
        assert_eq!(d.len(), 4);
        assert!((d[0].0 - 101.0).abs() < 1e-9, "back in absolute time: {}", d[0].0);
        assert!(motion_detections(None, &s, 100.0).is_empty());
    }

    #[test]
    fn each_algorithm_produces_the_manifest_fields_the_renderer_needs() {
        let one = vec![vec![face(0.4, 0.3, 0.12)]; 6];
        let c = cfg();
        let seg = |a: Algo, s: &ShotClassified| build_one(&c, 25.0, 10.0, s, None, a, None);

        let s = shot(one.clone());
        let f = seg(Algo::FaceFollow, &s);
        assert_eq!((f.algorithm.as_str(), f.window.as_str(), f.subject.as_str()), ("face_follow", "max", "face"));
        assert!(!f.split && f.keyframes.len() >= 2);

        let p = seg(Algo::PunchIn, &s);
        assert_eq!((p.window.as_str(), p.zoom > 1.0), ("zoom", true));

        let l = seg(Algo::Locked, &s);
        assert_eq!(l.keyframes.first().map(|k| (k.x, k.w)), l.keyframes.last().map(|k| (k.x, k.w)), "locked: first == last keyframe");

        let w = seg(Algo::WideBlur, &s);
        assert_eq!((w.strategy.clone(), w.keyframes.len()), (ShotStrategy::BlurFill, 0));

        let g = seg(Algo::GroupFit, &shot(vec![vec![face(0.1, 0.3, 0.1), face(0.5, 0.3, 0.1), face(0.8, 0.3, 0.1)]; 6]));
        assert_eq!((g.window.as_str(), g.subject.as_str(), g.strategy), ("fit", "region", ShotStrategy::SaliencyCrop));

        // split_two with only one face cannot split: it degrades to following that person
        let d = seg(Algo::SplitTwo, &s);
        assert!(!d.split);
        assert_eq!((d.algorithm.as_str(), d.window.as_str()), ("face_follow", "max"));
    }

    #[test]
    fn missing_choices_fall_back_to_the_legacy_framing() {
        let s = shot(vec![vec![face(0.4, 0.3, 0.12)]; 6]);
        let segs = build_segments_for_clip(&cfg(), 25.0, 10.0, &[s], &[], &[], None);
        assert_eq!(segs[0].algorithm, "speaker_follow");
    }

    /// Tracking quality, measured: simulate a subject that sweeps across the frame
    /// (period 4.8 s, ±0.15 of the frame width), run the real smoothing/keyframe path and
    /// count the frames in which the whole subject is inside the 9:16 crop window.
    fn coverage(alpha: f32) -> f64 {
        let c = cfg();
        let fps = 25.0;
        let n = 44; // 22 s sampled every 0.5 s, like the motion boxes
        // stays inside the frame (a subject that leaves it cannot be kept in view by any camera)
        let truth = |t: f64| 0.70 + 0.15 * (1.3 * t).sin();
        let mut s = shot(vec![vec![]; n]);
        s.start = 10.0;
        s.end = 10.0 + n as f64 * 0.5;
        let obj_w = 0.117; // a 150px object in a 1280px frame
        let dets: Vec<(f64, BBox)> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.5;
                (10.0 + t, BBox { x: (truth(t) - obj_w as f64 / 2.0) as f32, y: 0.35, w: obj_w, h: 0.2 })
            })
            .collect();
        let kfs = keyframes_from_dets(&c, &s, fps, 0.0, &dets, alpha);
        let at = |f: f64| {
            let k: Vec<(f64, f32)> = kfs.iter().map(|k| (k.frame as f64, k.x + k.w / 2.0)).collect();
            if f <= k[0].0 {
                return k[0].1;
            }
            for w in k.windows(2) {
                if f >= w[0].0 && f <= w[1].0 {
                    return w[0].1 + (w[1].1 - w[0].1) * ((f - w[0].0) / (w[1].0 - w[0].0).max(1.0)) as f32;
                }
            }
            k[k.len() - 1].1
        };
        let half = (9.0 / 16.0 * 720.0 / 1280.0 / 2.0) as f32; // half the crop window, in frame widths
        let (mut inside, mut total) = (0, 0);
        for f in (10.0 * fps) as i64..((10.0 + n as f64 * 0.5) * fps) as i64 {
            let t = f as f64 / fps - 10.0;
            let centre = at(f as f64).clamp(half, 1.0 - half); // the window is clamped to the frame
            total += 1;
            if (truth(t) as f32 - centre).abs() <= half - obj_w / 2.0 {
                inside += 1;
            }
        }
        inside as f64 / total as f64
    }

    #[test]
    fn motion_follow_keeps_a_fast_subject_in_the_crop() {
        let got = coverage(Algo::MotionFollow.ema_alpha());
        assert!(got >= 0.95, "the subject should be fully in view at least 95% of the time, got {:.0}%", got * 100.0);
        // the earlier, much heavier smoothing (α 0.18, one-way) is what let it walk out of frame
        assert!(coverage(0.18) < got, "heavier smoothing must not do better ({:.0}% vs {:.0}%)", coverage(0.18) * 100.0, got * 100.0);
    }

    #[test]
    fn smoothing_has_no_lag() {
        // a ramp: zero-phase smoothing keeps its midpoint where it is (a one-way EMA would trail it)
        let c = cfg();
        let n = 20;
        let mut s = shot(vec![vec![]; n]);
        s.start = 10.0;
        s.end = 10.0 + n as f64 * 0.5;
        let dets: Vec<(f64, BBox)> = (0..n)
            .map(|i| (10.0 + i as f64 * 0.5, BBox { x: 0.1 + 0.03 * i as f32, y: 0.3, w: 0.1, h: 0.2 }))
            .collect();
        let kfs = keyframes_from_dets(&c, &s, 25.0, 0.0, &dets, 0.45);
        let mid = |t: f64| {
            let f = t * 25.0;
            kfs.windows(2)
                .find(|w| f >= w[0].frame as f64 && f <= w[1].frame as f64)
                .map(|w| {
                    let r = (f - w[0].frame as f64) / (w[1].frame - w[0].frame).max(1) as f64;
                    let (a, b) = (w[0].x + w[0].w / 2.0, w[1].x + w[1].w / 2.0);
                    a + (b - a) * r as f32
                })
                .unwrap()
        };
        let t = 10.0 + 5.0; // halfway along the ramp
        let truth = 0.1 + 0.03 * 10.0 + 0.05; // centre of the box at i = 10
        assert!((mid(t) - truth).abs() < 0.02, "the smoothed path should pass through the middle of the ramp: {} vs {}", mid(t), truth);
    }
}
