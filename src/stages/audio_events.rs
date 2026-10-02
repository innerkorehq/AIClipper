//! Stage 5.2b — audio events: laughter, applause and loudness peaks.
//!
//! The sidecar slides an AudioSet classifier (laughter / applause) plus a
//! loudness measure over the audio. Text alone cannot tell a funny moment from
//! a flat one; these signals can. They are used three ways (see `score.rs`):
//!   1. laughter / cheering peaks become extra clip *candidates*,
//!   2. every clip's score is blended with its audio score,
//!   3. a clip's end is extended so a laugh is allowed to land.

use crate::config::Config;
use crate::sidecar::SidecarClient;
use crate::types::Word;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A window counts as "laughing" at or above this probability.
pub const LAUGH_ON: f32 = 0.25;
/// A laughter run needs at least one window this strong to become an event.
pub const LAUGH_PEAK_MIN: f32 = 0.40;
/// Loudness z-score (vs the surrounding minute) that counts as a loud peak.
pub const LOUD_PEAK_Z: f32 = 2.5;
/// Laughter probabilities rarely approach 1; this is "very clear laughter".
const LAUGH_FULL: f32 = 0.6;
/// How long past a laugh the clip should run.
pub const LAUGH_TAIL_S: f64 = 0.8;
/// Preferred total clip length when building a clip around an event.
const TARGET_CLIP_S: f64 = 25.0;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioEvents {
    #[serde(default)]
    pub model: String,
    pub win: f64,
    pub hop: f64,
    #[serde(default)]
    pub duration: f64,
    pub laughter: Vec<f32>,
    pub applause: Vec<f32>,
    pub loud_z: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventKind {
    Laughter,
    Loud,
}

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub start: f64,
    pub end: f64,
    pub peak: f32,
    pub kind: EventKind,
}

/// What the audio looks like inside one clip.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClipAudio {
    pub laugh_max: f32,
    pub laugh_frac: f32,
    pub crowd_max: f32,
    /// 0..1 — how loud the loudest moments are relative to the video.
    pub loud: f32,
}

impl ClipAudio {
    /// 0..1 audio interest.
    pub fn score(&self) -> f64 {
        let reaction = (self.laugh_max / LAUGH_FULL).min(1.0).max(0.8 * self.crowd_max.min(1.0));
        (0.5 * reaction as f64 + 0.2 * (self.laugh_frac * 3.0).min(1.0) as f64 + 0.3 * self.loud as f64).min(1.0)
    }

    pub fn describe(&self) -> String {
        format!(
            "laughter {:.2} ({:.0}% of clip), crowd {:.2}, loudness {:.2}",
            self.laugh_max,
            self.laugh_frac * 100.0,
            self.crowd_max,
            self.loud
        )
    }
}

/// Run the sidecar analysis for `wav`.
pub async fn analyze(cfg: &Config, client: &SidecarClient, wav: &Path) -> Result<AudioEvents> {
    let ev = client.audio_events(wav, &cfg.audio_model).await?;
    tracing::info!(
        "audio events: {} windows, {} with laughter ≥{LAUGH_ON}",
        ev.laughter.len(),
        ev.laughter.iter().filter(|p| **p >= LAUGH_ON).count()
    );
    Ok(ev)
}

impl AudioEvents {
    /// Center time of window `i`.
    fn center(&self, i: usize) -> f64 {
        i as f64 * self.hop + self.win / 2.0
    }

    /// Indices of windows whose centre falls inside [start, end).
    fn indices(&self, start: f64, end: f64) -> std::ops::Range<usize> {
        let n = self.laughter.len();
        let lo = (((start - self.win / 2.0) / self.hop).ceil().max(0.0)) as usize;
        let hi = (((end - self.win / 2.0) / self.hop).ceil().max(0.0)) as usize;
        lo.min(n)..hi.min(n)
    }

    pub fn stats(&self, start: f64, end: f64) -> ClipAudio {
        let r = self.indices(start, end);
        if r.is_empty() {
            return ClipAudio::default();
        }
        let slice = |v: &Vec<f32>| -> Vec<f32> { v[r.clone()].iter().copied().filter(|x| x.is_finite()).collect() };
        let laugh = slice(&self.laughter);
        let crowd = slice(&self.applause);
        let mut loud = slice(&self.loud_z);
        loud.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top3 = loud.iter().take(3).map(|z| z.max(0.0)).sum::<f32>() / 3.0;
        ClipAudio {
            laugh_max: laugh.iter().cloned().fold(0.0, f32::max),
            laugh_frac: laugh.iter().filter(|p| **p >= LAUGH_ON).count() as f32 / laugh.len().max(1) as f32,
            crowd_max: crowd.iter().cloned().fold(0.0, f32::max),
            loud: (top3 / 3.0).min(1.0),
        }
    }

    /// Laughter/cheering runs, then loud peaks that are not near a laugh.
    pub fn events(&self) -> Vec<Event> {
        let n = self.laughter.len();
        let react = |i: usize| self.laughter[i].max(self.applause.get(i).copied().unwrap_or(0.0));
        let mut out = Vec::new();
        let mut i = 0;
        while i < n {
            if react(i) < LAUGH_ON {
                i += 1;
                continue;
            }
            let mut j = i;
            let mut last_on = i;
            // merge across a single quiet window
            while j + 1 < n && (react(j + 1) >= LAUGH_ON || (j + 2 < n && react(j + 2) >= LAUGH_ON)) {
                j += 1;
                if react(j) >= LAUGH_ON {
                    last_on = j;
                }
            }
            let peak = (i..=last_on).map(react).fold(0.0, f32::max);
            if peak >= LAUGH_PEAK_MIN {
                out.push(Event {
                    start: (self.center(i) - self.hop / 2.0).max(0.0),
                    end: self.center(last_on) + self.hop / 2.0,
                    peak,
                    kind: EventKind::Laughter,
                });
            }
            i = last_on + 1;
        }
        let laughs: Vec<Event> = out.clone();
        let mut i = 0;
        while i < n {
            if self.loud_z.get(i).copied().unwrap_or(0.0) < LOUD_PEAK_Z {
                i += 1;
                continue;
            }
            let mut j = i;
            while j + 1 < n && self.loud_z[j + 1] >= LOUD_PEAK_Z {
                j += 1;
            }
            let (s, e) = (self.center(i) - self.hop / 2.0, self.center(j) + self.hop / 2.0);
            let near_laugh = laughs.iter().any(|l| l.start - 8.0 < e && s < l.end + 8.0);
            if !near_laugh {
                out.push(Event { start: s.max(0.0), end: e, peak: (self.loud_z[i..=j].iter().cloned().fold(0.0, f32::max) / 5.0).min(1.0), kind: EventKind::Loud });
            }
            i = j + 1;
        }
        out
    }
}

/// Sentence-aware clip spans around the strongest events: they end just after
/// the laugh and begin at the sentence start that gives a ~25s clip.
pub fn candidate_spans(
    words: &[Word],
    audio: &AudioEvents,
    min_d: f64,
    max_d: f64,
    limit: usize,
) -> Vec<(f64, f64, String)> {
    use crate::stages::score::{ends_sentence, sentence_starts};
    let mut events = audio.events();
    events.sort_by(|a, b| b.peak.partial_cmp(&a.peak).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::new();
    for ev in events.into_iter().take(limit) {
        let end_t = (ev.end + LAUGH_TAIL_S).min(audio.duration.max(ev.end));
        let Some(we) = words.iter().rposition(|w| w.start <= end_t) else { continue };
        // prefer to end on a sentence end at/before the laugh begins, but never
        // cut off the reaction: the span end stays at end_t.
        let _ = ends_sentence;
        let best = (0..=we)
            .filter(|&i| sentence_starts(words, i))
            .filter_map(|i| {
                let dur = end_t - words[i].start;
                (dur >= min_d && dur <= max_d).then_some((i, (dur - TARGET_CLIP_S).abs()))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((i, _)) = best {
            let what = match ev.kind {
                EventKind::Laughter => format!("laughter peak {:.2}", ev.peak),
                EventKind::Loud => format!("loud peak {:.2}", ev.peak),
            };
            out.push((words[i].start, end_t, format!("audio: {what}")));
        }
    }
    out
}

/// If a laugh starts inside the clip (or just after it) and would be cut off,
/// return the time the clip should end so the laugh lands.
pub fn extended_end(audio: &AudioEvents, start: f64, end: f64, max_d: f64) -> Option<f64> {
    audio
        .events()
        .into_iter()
        .filter(|e| e.kind == EventKind::Laughter)
        .filter(|e| e.start < end + 2.0 && e.end > end && e.start > start)
        .map(|e| (e.end + LAUGH_TAIL_S).min(start + max_d))
        .filter(|t| *t > end)
        .fold(None, |acc: Option<f64>, t| Some(acc.map_or(t, |a| a.max(t))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(laugh: &[f32], loud: &[f32]) -> AudioEvents {
        AudioEvents {
            model: "m".into(),
            win: 3.0,
            hop: 1.5,
            duration: laugh.len() as f64 * 1.5 + 1.5,
            laughter: laugh.to_vec(),
            applause: vec![0.0; laugh.len()],
            loud_z: loud.to_vec(),
        }
    }

    #[test]
    fn finds_laughter_runs_and_ignores_weak_ones() {
        let mut l = vec![0.01; 40];
        l[10] = 0.3;
        l[11] = 0.7;
        l[12] = 0.5; // strong run
        l[30] = 0.3; // weak blip below LAUGH_PEAK_MIN
        let a = ev(&l, &vec![0.0; 40]);
        let events: Vec<_> = a.events().into_iter().filter(|e| e.kind == EventKind::Laughter).collect();
        assert_eq!(events.len(), 1);
        assert!((events[0].peak - 0.7).abs() < 1e-6);
        assert!(events[0].start < 16.0 && events[0].end > 18.0);
    }

    #[test]
    fn loud_peak_near_laughter_is_not_double_counted() {
        let mut l = vec![0.0; 40];
        l[10] = 0.8;
        let mut z = vec![0.0; 40];
        z[10] = 4.0; // coincides with the laugh
        z[30] = 4.0; // alone
        let a = ev(&l, &z);
        let loud: Vec<_> = a.events().into_iter().filter(|e| e.kind == EventKind::Loud).collect();
        assert_eq!(loud.len(), 1);
        assert!(loud[0].start > 40.0);
    }

    #[test]
    fn clip_audio_stats_and_score() {
        let mut l = vec![0.0; 40];
        l[20] = 0.9;
        l[21] = 0.8;
        let mut z = vec![0.0; 40];
        z[20] = 6.0;
        let a = ev(&l, &z);
        let inside = a.stats(25.0, 40.0);
        let outside = a.stats(0.0, 10.0);
        assert!(inside.laugh_max > 0.8 && inside.score() > 0.7);
        assert_eq!(outside.score(), 0.0);
    }

    #[test]
    fn extends_a_clip_that_cuts_off_a_laugh() {
        let mut l = vec![0.0; 40];
        l[14] = 0.8; // windows 14–15: centres 22.5s / 24.0s → event 21.75–24.75s
        l[15] = 0.7;
        let a = ev(&l, &vec![0.0; 40]);
        // clip ends at 22.5 — mid-laugh
        let t = extended_end(&a, 5.0, 22.5, 90.0).expect("should extend");
        assert!((t - 25.55).abs() < 1e-6, "{t}"); // event end 24.75 + tail 0.8
        // clip that already ends after the laugh is untouched
        assert!(extended_end(&a, 5.0, 40.0, 90.0).is_none());
    }
}
