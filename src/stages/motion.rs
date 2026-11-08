//! Motion analysis: how much is moving in a clip, and where.
//!
//! Face detection says nothing about action footage (fights, sports, dancing,
//! vehicles). This measures frame-to-frame change on a tiny grayscale version of
//! the video (96×54 at 4 fps via ffmpeg — a 90 s clip is ~2 MB) and reports, per
//! sample, the overall motion energy and a box around the area that is changing.
//! Both feed the framing chooser (is this shot action?) and the `motion_follow`
//! algorithm (point the crop at where the action is).

use crate::config::Config;
use crate::types::BBox;
use anyhow::{Context, Result};
use std::path::Path;
use tokio::process::Command;

pub const W: usize = 96;
pub const H: usize = 54;
pub const FPS: f64 = 4.0;

/// Mean absolute change above this (of 255) marks a hard cut, not motion.
const CUT_ENERGY: f32 = 0.30;
/// A pixel counts as "moving" above this absolute change (of 255).
const MOVE_THRESHOLD: u8 = 25;
/// Fewer moving pixels than this fraction → the frame is effectively static.
const MIN_MOVING_FRACTION: f32 = 0.003;

#[derive(Debug, Clone, PartialEq)]
pub struct MotionSample {
    /// Seconds from the start of the analysed video.
    pub time: f64,
    /// Mean absolute frame difference, 0..1.
    pub energy: f32,
    /// Normalised box around the moving area; None when static or at a hard cut.
    pub bbox: Option<BBox>,
    /// True when this frame is a hard cut (the whole picture changed).
    pub cut: bool,
}

#[derive(Debug, Clone, Default)]
pub struct MotionTrack {
    pub samples: Vec<MotionSample>,
}

impl MotionTrack {
    /// Mark the frames at known scene cuts (rough-cut seconds) as cuts. A cut between
    /// two similar-looking angles changes less than the hard-cut threshold, so it
    /// would otherwise read as a burst of motion in the shot that follows it.
    pub fn mask_cuts(&mut self, cuts: &[f64]) {
        let window = 1.5 / FPS; // the cut frame and the one after it
        for s in &mut self.samples {
            if cuts.iter().any(|c| s.time >= *c - 0.5 / FPS && s.time < *c + window) {
                s.cut = true;
                s.bbox = None;
            }
        }
    }

    fn range(&self, t0: f64, t1: f64) -> impl Iterator<Item = &MotionSample> {
        self.samples.iter().filter(move |s| s.time >= t0 && s.time < t1)
    }

    /// Mean motion energy over [t0, t1), ignoring hard-cut frames. None if no data.
    pub fn mean_energy(&self, t0: f64, t1: f64) -> Option<f32> {
        let v: Vec<f32> = self.range(t0, t1).filter(|s| !s.cut).map(|s| s.energy).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
    }

    /// Fraction of the (non-cut) samples in [t0, t1) where a distinct moving region was
    /// found. Unlike mean energy this catches a *small* subject moving fast — a person in
    /// a wide shot, a ball, a car — which barely moves the whole-frame average.
    pub fn activity(&self, t0: f64, t1: f64) -> Option<f32> {
        let v: Vec<bool> = self.range(t0, t1).filter(|s| !s.cut).map(|s| s.bbox.is_some()).collect();
        (!v.is_empty()).then(|| v.iter().filter(|b| **b).count() as f32 / v.len() as f32)
    }

    /// (time, box) of the moving area over [t0, t1) — one entry per sample that has one.
    pub fn boxes(&self, t0: f64, t1: f64) -> Vec<(f64, BBox)> {
        self.range(t0, t1).filter_map(|s| s.bbox.map(|b| (s.time, b))).collect()
    }
}

/// Analyse already-decoded grayscale frames (each W*H bytes, sampled at `fps`).
pub fn analyze_frames(frames: &[Vec<u8>], fps: f64) -> Vec<MotionSample> {
    let mut out = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        let time = i as f64 / fps;
        if i == 0 || f.len() != W * H || frames[i - 1].len() != W * H {
            out.push(MotionSample { time, energy: 0.0, bbox: None, cut: false });
            continue;
        }
        let prev = &frames[i - 1];
        let mut sum = 0u64;
        let (mut mass, mut cx, mut cy) = (0.0f64, 0.0f64, 0.0f64);
        let mut moving = 0usize;
        let mut diffs = vec![0u8; W * H];
        for p in 0..W * H {
            let d = f[p].abs_diff(prev[p]);
            diffs[p] = d;
            sum += d as u64;
            if d > MOVE_THRESHOLD {
                moving += 1;
                let (x, y) = ((p % W) as f64, (p / W) as f64);
                let w = d as f64;
                mass += w;
                cx += w * x;
                cy += w * y;
            }
        }
        let energy = sum as f32 / (W * H) as f32 / 255.0;
        if energy > CUT_ENERGY {
            out.push(MotionSample { time, energy, bbox: None, cut: true });
            continue;
        }
        let bbox = if mass > 0.0 && (moving as f32 / (W * H) as f32) >= MIN_MOVING_FRACTION {
            let (mx, my) = (cx / mass, cy / mass);
            let (mut vx, mut vy) = (0.0f64, 0.0f64);
            for p in 0..W * H {
                if diffs[p] > MOVE_THRESHOLD {
                    let (x, y) = ((p % W) as f64, (p / W) as f64);
                    let w = diffs[p] as f64;
                    vx += w * (x - mx).powi(2);
                    vy += w * (y - my).powi(2);
                }
            }
            let (sx, sy) = ((vx / mass).sqrt() / W as f64, (vy / mass).sqrt() / H as f64);
            // ±1.8σ around the centroid, at least 15% of the frame each way
            let bw = (3.6 * sx).clamp(0.15, 1.0) as f32;
            let bh = (3.6 * sy).clamp(0.15, 1.0) as f32;
            let (nx, ny) = ((mx / W as f64) as f32, (my / H as f64) as f32);
            Some(BBox {
                x: (nx - bw / 2.0).clamp(0.0, 1.0 - bw),
                y: (ny - bh / 2.0).clamp(0.0, 1.0 - bh),
                w: bw,
                h: bh,
            })
        } else {
            None
        };
        out.push(MotionSample { time, energy, bbox, cut: false });
    }
    out
}

/// Decode `video` to 96×54 gray at 4 fps with ffmpeg and analyse it.
pub async fn analyze(cfg: &Config, video: &Path) -> Result<MotionTrack> {
    let out = Command::new(&cfg.ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(video)
        .args(["-vf", &format!("fps={FPS},scale={W}:{H}:flags=area,format=gray")])
        .args(["-an", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .output()
        .await
        .context("running ffmpeg for motion analysis")?;
    if !out.status.success() {
        anyhow::bail!("ffmpeg motion analysis failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let frames: Vec<Vec<u8>> = out.stdout.chunks_exact(W * H).map(|c| c.to_vec()).collect();
    Ok(MotionTrack { samples: analyze_frames(&frames, FPS) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank() -> Vec<u8> {
        vec![40u8; W * H]
    }

    /// A bright square of side `s` with its top-left at (x, y).
    fn with_square(x: usize, y: usize, s: usize) -> Vec<u8> {
        let mut f = blank();
        for yy in y..(y + s).min(H) {
            for xx in x..(x + s).min(W) {
                f[yy * W + xx] = 220;
            }
        }
        f
    }

    #[test]
    fn static_video_has_no_motion() {
        let frames = vec![blank(); 6];
        let m = analyze_frames(&frames, FPS);
        assert!(m.iter().all(|s| s.energy == 0.0 && s.bbox.is_none() && !s.cut));
    }

    #[test]
    fn a_moving_object_is_located() {
        // a 10x10 square jumping across the right half of the frame
        let frames = vec![with_square(50, 20, 10), with_square(58, 20, 10), with_square(66, 20, 10)];
        let m = analyze_frames(&frames, FPS);
        let b = m[2].bbox.expect("motion found");
        let cx = b.x + b.w / 2.0;
        assert!(cx > 0.6 && cx < 0.85, "centre x {cx} should be on the right where the square moves");
        assert!(m[2].energy > 0.0 && m[2].energy < CUT_ENERGY);
        assert!(m[0].bbox.is_none(), "first frame has nothing to compare with");
    }

    #[test]
    fn a_hard_cut_is_not_motion() {
        let dark = vec![0u8; W * H];
        let bright = vec![255u8; W * H];
        let m = analyze_frames(&[dark.clone(), dark, bright], FPS);
        assert!(m[2].cut && m[2].bbox.is_none());
    }

    #[test]
    fn a_small_fast_object_shows_as_activity_even_when_energy_is_tiny() {
        // 8x8 square (1.2% of the frame) jumping 5px per sample
        let frames: Vec<Vec<u8>> = (0..8).map(|i| with_square(5 + i * 5, 20, 8)).collect();
        let t = MotionTrack { samples: analyze_frames(&frames, FPS) };
        let energy = t.mean_energy(0.0, 2.0).unwrap();
        let activity = t.activity(0.0, 2.0).unwrap();
        assert!(energy < 0.03, "whole-frame energy underrates it: {energy}");
        assert!(activity > 0.8, "but a moving region is found in nearly every sample: {activity}");
        let still = MotionTrack { samples: analyze_frames(&vec![blank(); 8], FPS) };
        assert_eq!(still.activity(0.0, 2.0), Some(0.0));
        assert!(still.activity(10.0, 11.0).is_none());
    }

    #[test]
    fn known_scene_cuts_are_masked_out_of_the_measurement() {
        // static scene, then a cut at t=1.0 between two similar angles (a modest change),
        // then static again
        let a = vec![60u8; W * H];
        let b = vec![60u8 + 40; W * H]; // +40 everywhere: energy 0.157, below the hard-cut threshold
        let frames = vec![a.clone(), a.clone(), a.clone(), a.clone(), b.clone(), b.clone(), b.clone()];
        let mut t = MotionTrack { samples: analyze_frames(&frames, FPS) };
        assert!(t.samples[4].energy > 0.1 && !t.samples[4].cut, "unmasked, the cut reads as motion");
        assert!(t.mean_energy(0.0, 2.0).unwrap() > 0.02);
        t.mask_cuts(&[1.0]);
        assert!(t.samples[4].cut && t.samples[4].bbox.is_none());
        assert!(t.mean_energy(0.0, 2.0).unwrap() < 0.001, "masked, the shot is the static scene it is");
        assert!(!t.samples[1].cut && !t.samples[6].cut, "only the frames at the cut are masked");
    }

    #[test]
    fn track_queries_ignore_cuts_and_report_boxes() {
        let dark = vec![0u8; W * H];
        let bright = vec![255u8; W * H];
        let mut frames = vec![with_square(10, 10, 8), with_square(18, 10, 8), with_square(26, 10, 8)];
        frames.push(dark);
        frames.push(bright); // cut frame at t=1.0
        let t = MotionTrack { samples: analyze_frames(&frames, FPS) };
        let e = t.mean_energy(0.0, 2.0).unwrap();
        assert!(e < CUT_ENERGY, "the cut frame must not inflate the mean: {e}");
        assert_eq!(t.boxes(0.0, 0.75).len(), 2);
        assert!(t.mean_energy(10.0, 11.0).is_none());
    }
}
