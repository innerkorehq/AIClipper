//! Faces that are not people: a printed photo (a WANTED poster, a framed portrait) and a
//! cardboard cutout. They are detected like faces, and they used to be counted as people,
//! which pushed the framing towards a split or a group fit for the wrong reasons.
//!
//! Two cheap, independent tests, both conservative (when in doubt the face is kept):
//!
//! * **Printed photo** — a black-and-white or sepia print has almost no colour, a living
//!   face does. A face is dropped only when it is both nearly colourless *and* much less
//!   colourful than another face in the same frame, so a black-and-white film (where every
//!   face is colourless) loses nothing.
//! * **Cutout** — a face that does not move at all, in a picture where other faces do.
//!   The camera is therefore still and the people are alive; a real person never holds
//!   perfectly still for seconds (their box drifts a percent or more), while a cutout's box
//!   is pixel-identical frame to frame. If everyone is still, nothing is dropped.

use super::shim::FaceRect;

/// Below this a face is "colourless" (a print reads ~0.1, a living face ~0.5).
const PRINT_MAX_SATURATION: f32 = 0.25;
/// ...and it must also be at most this fraction of the most colourful face in the frame.
const PRINT_RELATIVE: f32 = 0.5;

/// Track matching: a face continues a track if its centre is within this distance.
const MATCH_DIST: f32 = 0.04;
/// A track needs this many observations before it can be judged.
const MIN_OBS: usize = 5;
/// ...and must be present in at least this fraction of the frames.
const MIN_COVERAGE: f32 = 0.6;
/// A cutout's box spread (std-dev of centre, per axis) stays under this...
const FROZEN_STD: f32 = 0.0015;
/// ...while a live face in the same picture spreads at least this much.
const ALIVE_STD: f32 = 0.004;

/// Remove printed-photo faces and frozen cutout faces from per-frame detections of one
/// continuous stretch of footage. Returns how many faces were removed.
pub fn drop_decoys(frames: &mut [Vec<FaceRect>]) -> usize {
    let before: usize = frames.iter().map(Vec::len).sum();
    drop_prints(frames);
    drop_frozen(frames);
    before - frames.iter().map(Vec::len).sum::<usize>()
}

fn drop_prints(frames: &mut [Vec<FaceRect>]) {
    for faces in frames.iter_mut() {
        let best = faces.iter().map(|f| f.saturation).fold(-1.0f32, f32::max);
        if best < 0.0 {
            continue; // saturation unknown
        }
        faces.retain(|f| {
            f.saturation < 0.0 || !(f.saturation < PRINT_MAX_SATURATION && f.saturation < best * PRINT_RELATIVE)
        });
    }
}

struct Track {
    last: (f32, f32),
    obs: Vec<(usize, f32, f32, f32)>, // (frame, cx, cy, w)
}

fn std_dev(v: impl Iterator<Item = f32> + Clone) -> f32 {
    let n = v.clone().count() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let m = v.clone().sum::<f32>() / n;
    (v.map(|x| (x - m).powi(2)).sum::<f32>() / n).sqrt()
}

fn drop_frozen(frames: &mut [Vec<FaceRect>]) {
    if frames.len() < MIN_OBS {
        return;
    }
    let mut tracks: Vec<Track> = Vec::new();
    // which (frame, face index) belongs to which track
    let mut owner: Vec<Vec<usize>> = frames.iter().map(|f| vec![usize::MAX; f.len()]).collect();
    for (fi, faces) in frames.iter().enumerate() {
        let mut taken = vec![false; tracks.len()];
        for (k, face) in faces.iter().enumerate() {
            let c = face.bbox.center();
            let best = tracks
                .iter()
                .enumerate()
                .filter(|(t, _)| !taken[*t])
                .map(|(t, tr)| (t, ((tr.last.0 - c.0).powi(2) + (tr.last.1 - c.1).powi(2)).sqrt()))
                .filter(|(_, d)| *d <= MATCH_DIST)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let t = match best {
                Some((t, _)) => t,
                None => {
                    tracks.push(Track { last: c, obs: Vec::new() });
                    taken.push(false);
                    tracks.len() - 1
                }
            };
            taken[t] = true;
            tracks[t].last = c;
            tracks[t].obs.push((fi, c.0, c.1, face.bbox.w));
            owner[fi][k] = t;
        }
    }
    let judged: Vec<usize> = (0..tracks.len())
        .filter(|&t| {
            let o = &tracks[t].obs;
            o.len() >= MIN_OBS && o.len() as f32 / frames.len() as f32 >= MIN_COVERAGE
        })
        .collect();
    let spread = |t: usize| {
        let o = &tracks[t].obs;
        std_dev(o.iter().map(|x| x.1)).max(std_dev(o.iter().map(|x| x.2)))
    };
    let someone_alive = judged.iter().any(|&t| spread(t) >= ALIVE_STD);
    if !someone_alive {
        return;
    }
    let frozen: Vec<usize> = judged.iter().copied().filter(|&t| spread(t) < FROZEN_STD).collect();
    if frozen.is_empty() {
        return;
    }
    for (fi, faces) in frames.iter_mut().enumerate() {
        let mut k = 0;
        faces.retain(|_| {
            let t = owner[fi][k];
            k += 1;
            !frozen.contains(&t)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BBox;

    fn face(x: f32, y: f32, sat: f32) -> FaceRect {
        FaceRect { bbox: BBox { x, y, w: 0.05, h: 0.07 }, confidence: 0.9, saturation: sat }
    }

    /// Two people who drift a little, and a cutout whose box never changes.
    fn couch(n: usize) -> Vec<Vec<FaceRect>> {
        (0..n)
            .map(|i| {
                let d = (i as f32 * 0.9).sin() * 0.015;
                vec![face(0.35 + d, 0.53, 0.52), face(0.52, 0.17, 0.36), face(0.67 - d * 0.5, 0.56, 0.52)]
            })
            .collect()
    }

    #[test]
    fn a_cutout_that_never_moves_is_dropped_and_people_are_kept() {
        let mut f = couch(12);
        assert_eq!(drop_decoys(&mut f), 12);
        assert!(f.iter().all(|fr| fr.len() == 2), "the two living faces stay in every frame");
        assert!(f.iter().all(|fr| fr.iter().all(|x| (x.bbox.x - 0.52).abs() > 0.05)));
    }

    #[test]
    fn when_everyone_is_still_nothing_is_dropped() {
        let mut f: Vec<Vec<FaceRect>> = (0..12).map(|_| vec![face(0.3, 0.5, 0.5), face(0.6, 0.5, 0.5)]).collect();
        assert_eq!(drop_decoys(&mut f), 0, "a still picture cannot tell a person from a cutout");
    }

    #[test]
    fn a_short_shot_is_never_judged() {
        let mut f = couch(3);
        assert_eq!(drop_decoys(&mut f), 0);
    }

    #[test]
    fn a_printed_photo_is_dropped_but_a_black_and_white_film_is_not() {
        // a poster held next to two real people
        let mut f = vec![vec![face(0.2, 0.3, 0.5), face(0.5, 0.3, 0.55), face(0.8, 0.3, 0.12)]];
        assert_eq!(drop_decoys(&mut f), 1);
        assert_eq!(f[0].len(), 2);
        // everyone colourless (a monochrome film): all kept
        let mut bw = vec![vec![face(0.2, 0.3, 0.05), face(0.5, 0.3, 0.04)]];
        assert_eq!(drop_decoys(&mut bw), 0);
        // saturation unknown (older cache): all kept
        let mut unk = vec![vec![face(0.2, 0.3, -1.0), face(0.5, 0.3, -1.0)]];
        assert_eq!(drop_decoys(&mut unk), 0);
    }

    #[test]
    fn a_lone_face_is_never_dropped() {
        let mut f: Vec<Vec<FaceRect>> = (0..12).map(|_| vec![face(0.5, 0.3, 0.1)]).collect();
        assert_eq!(drop_decoys(&mut f), 0);
    }
}
