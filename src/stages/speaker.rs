//! Stage 5.7 — Active speaker resolution for multi-speaker `face_crop` shots.
//!
//! - Fixed camera layout: when the shot carries a `target_track_id` (a diarized
//!   label from Gemma), restrict correlation to that speaker's turns.
//! - Otherwise: correlate each face track's mouth-region motion (mouth-aspect
//!   ratio from Vision landmarks) against the audio energy envelope for the
//!   current diarized turns; highest-correlation track wins.
//! - Single-track shots skip resolution entirely.

use crate::config::Config;
use crate::sidecar::SidecarClient;
use crate::stages::shot_classify::ShotClassified;
use crate::types::{BBox, ShotStrategy, SpeechTurn};
use anyhow::{Context, Result};
use std::path::Path;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ShotSpeaker {
    pub cand_idx: usize,
    pub shot_list_idx: usize,
    /// Which face track is speaking (0-based). None if unresolved.
    pub track_id: Option<usize>,
    /// Diarized speaker label assigned to the active track, if known.
    pub speaker: Option<String>,
    /// Time-synced detections (source-video time, normalized bbox) of the active track.
    pub detections: Vec<(f64, BBox)>,
}

/// Face track built by associating detections across sampled frames.
#[derive(Debug, Clone)]
struct FaceTrack {
    id: usize,
    /// (sample_index, bbox)
    hits: Vec<(usize, BBox)>,
}

/// Diarize the whole source audio once (cached by caller via artifact store).
pub async fn diarize_global(client: &SidecarClient, wav: &Path) -> Result<Vec<SpeechTurn>> {
    let turns = client.diarize(wav).await.context("diarization failed")?;
    tracing::info!("{} diarized turns", turns.len());
    Ok(turns)
}

#[allow(clippy::too_many_arguments)]
pub async fn resolve(
    cfg: &Config,
    wav: &Path,
    turns: &[SpeechTurn],
    rough: &Path,
    offset_s: f64,
    cand_idx: usize,
    shots: &[ShotClassified],
    vision: &crate::vision::VisionService,
) -> Result<Vec<ShotSpeaker>> {
    let envelope = load_audio_envelope(wav)?;

    let mut out = Vec::new();
    for shot in shots {
        if shot.strategy != ShotStrategy::FaceCrop {
            continue;
        }
        out.push(resolve_one(cfg, rough, offset_s, shot, turns, &envelope, cand_idx, vision).await?);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
async fn resolve_one(
    cfg: &Config,
    rough: &Path,
    offset_s: f64,
    shot: &ShotClassified,
    turns: &[SpeechTurn],
    envelope: &AudioEnvelope,
    cand_idx: usize,
    vision: &crate::vision::VisionService,
) -> Result<ShotSpeaker> {
    let times: Vec<f64> = shot.samples.iter().map(|s| s.time).collect();
    // Landmarks run against the rough cut (timeline starts at 0).
    let rough_times: Vec<f64> = times.iter().map(|t| (t - offset_s).max(0.0)).collect();
    let landmark_sets = vision.landmarks_at(rough, &rough_times).await?;

    // Build face tracks by greedy IoU association.
    let tracks = associate_tracks(&shot.samples);
    if tracks.len() <= 1 {
        return Ok(ShotSpeaker {
            cand_idx,
            shot_list_idx: shot.shot_list_idx,
            track_id: tracks.first().map(|t| t.id).or(Some(0)),
            speaker: diarized_speaker_at(turns, (shot.start + shot.end) / 2.0),
            detections: active_detections(tracks.first(), &shot.samples),
        });
    }
    if cfg.assume_single_speaker {
        return Ok(ShotSpeaker {
            cand_idx,
            shot_list_idx: shot.shot_list_idx,
            track_id: Some(0),
            speaker: None,
            detections: active_detections(tracks.first(), &shot.samples),
        });
    }

    // Multiple tracks: measure MAR per track and correlate with audio energy.
    // MAR = mouth openness ratio (vertical/horizontal extent) of the mouth region.
    let shot_mid = (shot.start + shot.end) / 2.0;
    let active_speaker = diarized_speaker_at(turns, shot_mid);

    // Restrict correlation windows to the active speaker's turns if labels exist.
    let active_turn_windows = if let Some(name) = shot.target_track_id.clone() {
        turn_windows_for(turns, Some(name.as_str()), shot.start, shot.end)
    } else {
        turn_windows_for(turns, active_speaker.as_deref(), shot.start, shot.end)
            .into_iter()
            .filter(|(s, e)| *e - *s > 0.3)
            .collect()
    };
    // If no usable windows (label missing/silence), use the whole shot.
    let windows: Vec<(f64, f64)> = if active_turn_windows.is_empty() {
        vec![(shot.start, shot.end)]
    } else {
        active_turn_windows
    };

    let mut best_track: Option<usize> = None;
    let mut best_corr = -2.0f64;
    for track in &tracks {
        // MAR series
        let mut mar: Vec<f64> = Vec::new();
        for (samp, _bbox) in &track.hits {
            let lms = landmark_sets.get(*samp).map(|v| v.as_slice()).unwrap_or(&[]);
            mar.push(if lms.is_empty() { f64::NAN } else { mouth_openness(&lms[0]) });
        }
        let times_hit: Vec<f64> = track.hits.iter().map(|(si, _)| times[*si]).collect();
        let audio: Vec<f64> = times_hit.iter().map(|t| envelope.at(*t)).collect();
        let mut corr_sum = 0.0;
        let mut corr_n = 0.0;
        for (s, e) in &windows {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for (i, t) in times_hit.iter().enumerate() {
                if t >= s && t <= e && !mar[i].is_nan() && audio[i] > 1e-4 {
                    a.push(mar[i]);
                    b.push(audio[i]);
                }
            }
            if a.len() >= 3 {
                corr_sum += pearson(&a, &b).unwrap_or(f64::NAN);
                corr_n += 1.0;
            }
        }
        let corr = if corr_n > 0.0 { corr_sum / corr_n } else { f64::NAN };
        let size = track.hits.iter().map(|(_, b)| b.w * b.h).sum::<f32>() / track.hits.len() as f32;
        // Prefer positive correlation; break ties (incl. NaN) by face size.
        let corr_ok = if corr.is_nan() { -1.0 } else { corr };
        let score = corr_ok + 0.02 * f64::from(size);
        if score > best_corr {
            best_corr = score;
            best_track = Some(track.id);
        }
    }

    let best = tracks.iter().find(|t| Some(t.id) == best_track).cloned();
    let track_id = best.as_ref().map(|t| t.id);
    tracing::info!(
        "speaker resolved (cand {cand_idx}, shot {}): track {:?} windows={}",
        shot.shot_list_idx,
        track_id,
        windows.len()
    );
    Ok(ShotSpeaker {
        cand_idx,
        shot_list_idx: shot.shot_list_idx,
        track_id,
        speaker: active_speaker.or_else(|| shot.target_track_id.clone()),
        detections: active_detections(best.as_ref(), &shot.samples),
    })
}

/// Greedy track association across sampled frames by IoU.
fn associate_tracks(samples: &[crate::stages::shot_classify::ShotSample]) -> Vec<FaceTrack> {
    let mut tracks: Vec<FaceTrack> = Vec::new();
    for (si, sample) in samples.iter().enumerate() {
        let mut matched = vec![false; sample.faces.len()];
        // Extend existing tracks: each track grabs the closest unmatched face.
        for sup in &mut tracks {
            let prev = sup.hits.last().map(|(_, b)| *b);
            let mut best: Option<(usize, f64)> = None;
            for (fi, face) in sample.faces.iter().enumerate() {
                if matched[fi] {
                    continue;
                }
                let sc = iou(prev, face.bbox);
                if sc >= 0.3 && best.is_none_or(|(_, bc)| sc > bc) {
                    best = Some((fi, sc));
                }
            }
            if let Some((fi, _)) = best {
                matched[fi] = true;
                sup.hits.push((si, sample.faces[fi].bbox));
            }
        }
        // Unmatched faces start new tracks.
        for (fi, face) in sample.faces.iter().enumerate() {
            if !matched[fi] {
                tracks.push(FaceTrack { id: tracks.len(), hits: vec![(si, face.bbox)] });
            }
        }
    }
    // Drop ultra-short tracks (single stray detection) when we have real samples.
    tracks.retain(|t| t.hits.len() >= 2 || samples.len() <= 1);
    tracks
}

fn iou(a_opt: Option<BBox>, b: BBox) -> f64 {
    match a_opt {
        None => 0.0,
        Some(a) => {
            let ix = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
            let iy = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
            if ix <= 0.0 || iy <= 0.0 {
                return 0.0;
            }
            let inter = ix * iy;
            let union = a.w * a.h + b.w * b.h - inter;
            if union <= 0.0 { 0.0 } else { f64::from(inter / union) }
        }
    }
}

fn active_detections(track: Option<&FaceTrack>, samples: &[crate::stages::shot_classify::ShotSample]) -> Vec<(f64, BBox)> {
    match track {
        None => Vec::new(),
        Some(t) => t
            .hits
            .iter()
            .filter_map(|(si, b)| samples.get(*si).map(|s| (s.time, *b)))
            .collect(),
    }
}

/// Mouth openness (vertical/horizontal extent) from full-image-normalized mouth points.
fn mouth_openness(lm: &crate::vision::MouthLandmarks) -> f64 {
    if lm.points.len() < 4 {
        return f64::NAN;
    }
    let xs: Vec<f32> = lm.points.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = lm.points.iter().map(|p| p.1).collect();
    let minx = xs.iter().cloned().fold(f32::MAX, f32::min);
    let maxx = xs.iter().cloned().fold(f32::MIN, f32::max);
    let miny = ys.iter().cloned().fold(f32::MAX, f32::min);
    let maxy = ys.iter().cloned().fold(f32::MIN, f32::max);
    let w = (maxx - minx).max(1e-4);
    let h = (maxy - miny).max(1e-6);
    f64::from(h / w)
}

fn diarized_speaker_at(turns: &[SpeechTurn], t: f64) -> Option<String> {
    turns
        .iter()
        .find(|tr| t >= tr.start && t <= tr.end)
        .map(|tr| tr.speaker.clone())
}

fn turn_windows_for(turns: &[SpeechTurn], label: Option<&str>, start: f64, end: f64) -> Vec<(f64, f64)> {
    turns
        .iter()
        .filter(|tr| label.map(|l| tr.speaker == l).unwrap_or(true))
        .map(|tr| (tr.start.max(start), tr.end.min(end)))
        .filter(|(s, e)| e > s)
        .collect()
}

fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len().min(b.len());
    if n < 2 {
        return None;
    }
    let ma = a.iter().take(n).sum::<f64>() / n as f64;
    let mb = b.iter().take(n).sum::<f64>() / n as f64;
    let (mut num, mut da, mut db) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let x = a[i] - ma;
        let y = b[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    let den = (da * db).sqrt();
    if den <= 0.0 { None } else { Some(num / den) }
}

// ---- Audio envelope (16 kHz mono PCM WAV) ----------------------------------

#[allow(dead_code)]
struct AudioEnvelope {
    rate: u32,
    hop: f64,
    rms: Vec<f64>,
}

impl AudioEnvelope {
    fn at(&self, t: f64) -> f64 {
        let idx = (t / self.hop) as usize;
        self.rms.get(idx).copied().or_else(|| self.rms.last().copied()).unwrap_or(0.0)
    }
}

fn load_audio_envelope(wav: &Path) -> Result<AudioEnvelope> {
    let (rate, samples) = read_wav_pcm16(wav)?;
    let win = (rate as f64 * 0.025) as usize; // 25 ms
    let hop = rates_to_hop(rate, 0.010); // 10 ms
    let mut rms = Vec::new();
    let mut i = 0usize;
    while i < samples.len() {
        let end = (i + win).min(samples.len());
        if end > i {
            let sum: f64 = samples[i..end].iter().map(|s| (*s as f64).powi(2)).sum();
            let n = (end - i) as f64;
            rms.push((sum / n).sqrt());
        }
        i += hop.max(1);
    }
    Ok(AudioEnvelope { rate, hop: 0.010, rms })
}

fn rates_to_hop(rate: u32, secs: f64) -> usize {
    (rate as f64 * secs) as usize
}

/// Read a standard PCM 16-bit mono WAV (as produced by `extract_wav`).
fn read_wav_pcm16(path: &Path) -> Result<(u32, Vec<i16>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("{} is not a RIFF/WAVE file", path.display());
    }
    let fmt_off = find_chunk(&bytes, b"fmt ")?.ok_or_else(|| anyhow::anyhow!("missing fmt chunk"))?;
    let data_off = find_chunk(&bytes, b"data")?
        .ok_or_else(|| anyhow::anyhow!("missing data chunk"))?;
    let audio_format = u16::from_le_bytes([bytes[fmt_off + 8], bytes[fmt_off + 9]]);
    let channels = u16::from_le_bytes([bytes[fmt_off + 10], bytes[fmt_off + 11]]);
    let rate = u32::from_le_bytes([bytes[fmt_off + 12], bytes[fmt_off + 13], bytes[fmt_off + 14], bytes[fmt_off + 15]]);
    let bits = u16::from_le_bytes([bytes[fmt_off + 22], bytes[fmt_off + 23]]);
    if audio_format != 1 {
        anyhow::bail!("unsupported WAV compression ({audio_format}), expected PCM");
    }
    if bits != 16 {
        anyhow::bail!("unsupported WAV bit depth ({bits}), expected 16");
    }
    let data_len = u32::from_le_bytes([bytes[data_off + 4], bytes[data_off + 5], bytes[data_off + 6], bytes[data_off + 7]]) as usize;
    let data_start = data_off + 8;
    let mut samples = Vec::with_capacity(data_len / 2);
    if channels == 1 {
        for b in bytes[data_start..data_start + data_len].chunks_exact(2) {
            samples.push(i16::from_le_bytes([b[0], b[1]]));
        }
    } else {
        // Downmix to mono (average channels).
        let mut acc: i64 = 0;
        let mut n = 0usize;
        for b in bytes[data_start..data_start + data_len].chunks_exact(2) {
            acc += i16::from_le_bytes([b[0], b[1]]) as i64;
            n += 1;
            if n == channels as usize {
                samples.push((acc / channels as i64) as i16);
                acc = 0;
                n = 0;
            }
        }
    }
    Ok((rate, samples))
}

fn find_chunk(bytes: &[u8], id: &[u8; 4]) -> Result<Option<usize>> {
    let mut off = 12usize;
    while off + 8 <= bytes.len() {
        if &bytes[off..off + 4] == id {
            return Ok(Some(off));
        }
        let size = u32::from_le_bytes([bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]]) as usize;
        off += 8 + size + (size % 2);
    }
    Ok(None)
}