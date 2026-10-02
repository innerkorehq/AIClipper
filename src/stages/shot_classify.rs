//! Stage 5.6 — Shot classification.
//!
//! Per shot (not per clip): sample frames, run Vision face detector. Classify
//! `talking_head` (face_crop) if face confidence + screen-time coverage exceed
//! thresholds; otherwise run attention saliency and measure heatmap
//! concentration → `saliency_crop` (concentrated) or `blur_fill` (diffuse).

use crate::config::Config;
use crate::types::{BBox, ShotFrame, ShotListEntry, ShotStrategy};
use crate::vision::{saliency_concentration, saliency_target_bbox, FaceRect, SaliencyGrid, VisionService};
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
    /// Dense samples this shot was classified from (sampled at sample_interval).
    pub samples: Vec<ShotSample>,
    /// Deterministic fallback target (mid-frame) for shots with no detections.
    pub fallback_bbox: BBox,
}

/// Concentration threshold (normalized entropy) below which → saliency_crop.
pub const CONCENTRATED_ENTROPY: f64 = 0.8;

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
) -> Result<Vec<ShotClassified>> {
    if !vision.available() {
        anyhow::bail!("shot classification requires the Vision shim (build with `cargo build`)");
    }
    let mut out = Vec::with_capacity(shot_list.len());
    for (idx, shot) in shot_list.iter().enumerate() {
        let classified = classify_one(cfg, vision, rough, offset_s, fps, shot, idx).await?;
        out.push(classified);
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
    idx: usize,
) -> Result<ShotClassified> {
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

    // Talking-head test: face confidence + coverage.
    let confident_faces = samples
        .iter()
        .filter(|s| s.faces.iter().any(|f| f.confidence >= cfg.face_confidence))
        .count();
    let coverage = confident_faces as f64 / samples.len() as f64;

    let strategy = if coverage >= f64::from(cfg.face_coverage) {
        ShotStrategy::FaceCrop
    } else {
        // Saliency path: measure concentration of the attention heatmap.
        let grids: Vec<&SaliencyGrid> = samples.iter().filter_map(|s| s.saliency.as_ref()).collect();
        if grids.is_empty() {
            ShotStrategy::BlurFill
        } else {
            let mean_conc: f64 =
                grids.iter().map(|g| saliency_concentration(g)).sum::<f64>() / grids.len() as f64;
            if mean_conc < CONCENTRATED_ENTROPY {
                ShotStrategy::SaliencyCrop
            } else {
                ShotStrategy::BlurFill
            }
        }
    };

    tracing::info!(
        "shot {idx}: [{:.1}s–{:.1}s] frame={:?} faces={:.0}% → {:?}",
        shot.start,
        shot.end,
        shot.frame,
        coverage * 100.0,
        strategy
    );

    Ok(ShotClassified {
        shot_list_idx: idx,
        start: shot.start,
        end: shot.end,
        frame: shot.frame.clone(),
        target_track_id: shot.target_track_id.clone(),
        strategy,
        fallback_bbox: saliency_target_bbox_or_center(samples.first().and_then(|s| s.saliency.as_ref())),
        samples,
    })
}

fn saliency_target_bbox_or_center(saliency: Option<&SaliencyGrid>) -> BBox {
    match saliency {
        Some(g) => saliency_target_bbox(g, 0.5, 0.5),
        None => BBox { x: 0.18, y: 0.15, w: 0.64, h: 0.7 }, // center-ish portrait
    }
}