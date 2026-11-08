//! The framing chooser: per shot, the AI picks a framing algorithm.
//!
//! Deterministic code *measures* each shot (`framing::compute_features`) and works
//! out which algorithms can physically run on it. The model — told the request and
//! what the video is — then decides which one fits the footage and says why. Code
//! only guards the answer (an unknown or infeasible pick is replaced by the
//! deterministic default, with the reason recorded) and covers for a model that
//! is unavailable. `--framing rules` skips the model, `--framing <algorithm>`
//! forces one (for debugging and comparison).

use crate::config::Config;
use crate::sidecar::SidecarClient;
use crate::stages::framing::{feasible, resolve, rules_default, Algo, ShotFeatures};
use crate::stages::shot_classify::ShotClassified;
use serde::Deserialize;

/// How framing algorithms are chosen for a run.
#[derive(Debug, Clone, PartialEq)]
pub enum FramingMode {
    /// The model chooses per shot (default).
    Auto,
    /// Deterministic rules only; no model call.
    Rules,
    /// Always this algorithm where it can run (rules elsewhere).
    Fixed(Algo),
}

impl FramingMode {
    pub fn parse(s: &str) -> Result<FramingMode, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "ai" => Ok(FramingMode::Auto),
            "rules" => Ok(FramingMode::Rules),
            other => Algo::parse(other).map(FramingMode::Fixed).ok_or_else(|| {
                format!(
                    "unknown framing `{s}`: use auto, rules, or one of {}",
                    Algo::ALL.iter().map(|a| a.name()).collect::<Vec<_>>().join(", ")
                )
            }),
        }
    }

    pub fn label(&self) -> String {
        match self {
            FramingMode::Auto => "auto".into(),
            FramingMode::Rules => "rules".into(),
            FramingMode::Fixed(a) => a.name().into(),
        }
    }
}

/// Where a choice came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The model picked it.
    Ai,
    /// Deterministic rules (by request, or by design in `rules` mode).
    Rules,
    /// Forced with `--framing <algorithm>`.
    Fixed,
    /// The model's answer was missing or unusable; the rules covered for it.
    Fallback,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Ai => "ai",
            Source::Rules => "rules",
            Source::Fixed => "fixed",
            Source::Fallback => "fallback",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Choice {
    pub algo: Algo,
    pub source: Source,
    pub reason: String,
}

/// One entry of the model's answer.
#[derive(Debug, Clone, Deserialize)]
pub struct RawChoice {
    pub shot: usize,
    pub algorithm: String,
    #[serde(default)]
    pub reason: String,
}

/// Names of the algorithms that can run on this shot.
pub fn feasible_names(f: &ShotFeatures) -> Vec<&'static str> {
    Algo::ALL.into_iter().filter(|a| feasible(*a, f)).map(|a| a.name()).collect()
}

/// The menu the model reads: (name, when to use it).
pub fn menu() -> Vec<(&'static str, &'static str)> {
    Algo::ALL.into_iter().map(|a| (a.name(), a.describe())).collect()
}

/// Validate the model's answers against the shots. A shot with no answer, an
/// unknown algorithm or an infeasible one falls back to the deterministic default.
pub fn apply_answers(feats: &[ShotFeatures], answers: &[RawChoice]) -> Vec<Choice> {
    feats
        .iter()
        .map(|f| match answers.iter().find(|a| a.shot == f.idx) {
            None => Choice {
                algo: rules_default(f),
                source: Source::Fallback,
                reason: "the model gave no answer for this shot".into(),
            },
            Some(a) => {
                let (algo, note) = resolve(&a.algorithm, f);
                match note {
                    None => Choice { algo, source: Source::Ai, reason: a.reason.trim().to_string() },
                    Some(n) => Choice { algo, source: Source::Fallback, reason: format!("{n}; used the default instead") },
                }
            }
        })
        .collect()
}

/// Choose an algorithm for every shot of a clip.
pub async fn choose(
    cfg: &Config,
    client: &SidecarClient,
    mode: &FramingMode,
    feats: &[ShotFeatures],
    context: &str,
    ask: &str,
) -> Vec<Choice> {
    match mode {
        FramingMode::Rules => feats
            .iter()
            .map(|f| Choice { algo: rules_default(f), source: Source::Rules, reason: "deterministic rules".into() })
            .collect(),
        FramingMode::Fixed(a) => feats
            .iter()
            .map(|f| {
                if feasible(*a, f) {
                    Choice { algo: *a, source: Source::Fixed, reason: "forced with --framing".into() }
                } else {
                    Choice {
                        algo: rules_default(f),
                        source: Source::Fallback,
                        reason: format!("`{}` cannot run on this shot; used the default", a.name()),
                    }
                }
            })
            .collect(),
        FramingMode::Auto => {
            if feats.is_empty() {
                return Vec::new();
            }
            match client.choose_framing(feats, context, ask, cfg).await {
                Ok(raw) => apply_answers(feats, &raw),
                Err(e) => {
                    tracing::warn!("framing chooser unavailable ({e:#}); using the deterministic rules for this clip");
                    feats
                        .iter()
                        .map(|f| Choice { algo: rules_default(f), source: Source::Fallback, reason: "the model was unavailable".into() })
                        .collect()
                }
            }
        }
    }
}

/// Markdown for `work/framing/clip_NNN.md`: what was chosen for every shot, by
/// whom, why, and the measurements behind it.
pub fn report_section(clip: &str, shots: &[ShotClassified], feats: &[ShotFeatures], choices: &[Choice], mode: &FramingMode) -> String {
    let mut md = format!("# Framing — {clip} (mode: {})\n\n| shot | time | algorithm | by | faces | motion | speaking | reason |\n|---|---|---|---|---|---|---|---|\n", mode.label());
    for ((s, f), c) in shots.iter().zip(feats).zip(choices) {
        md.push_str(&format!(
            "| {} | {:.1}–{:.1}s | **{}** | {} | {:.0}/{} | {} | {:.0}% | {} |\n",
            f.idx,
            s.start,
            s.end,
            c.algo.name(),
            c.source.label(),
            f.faces_median,
            f.faces_max,
            f.motion_energy.map(|m| format!("{m:.3}")).unwrap_or_else(|| "–".into()),
            f.speaking * 100.0,
            c.reason.replace('|', "/"),
        ));
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(idx: usize, faces: usize, motion: f32) -> ShotFeatures {
        ShotFeatures {
            idx,
            duration_s: 4.0,
            layout: if faces >= 2 { "split" } else if faces == 1 { "single" } else { "wide" },
            faces_median: faces as f32,
            faces_max: faces,
            face_presence: if faces > 0 { 1.0 } else { 0.0 },
            face_area: 0.04,
            face_motion: 0.01,
            group_width: 0.0,
            saliency_focus: Some(0.5),
            motion_energy: Some(motion),
            motion_activity: Some(if motion > 0.05 { 1.0 } else { 0.0 }),
            speaking: 0.5,
        }
    }

    fn raw(shot: usize, algorithm: &str) -> RawChoice {
        RawChoice { shot, algorithm: algorithm.into(), reason: "because".into() }
    }

    #[test]
    fn modes_parse_and_reject_typos_helpfully() {
        assert_eq!(FramingMode::parse("auto"), Ok(FramingMode::Auto));
        assert_eq!(FramingMode::parse("AI"), Ok(FramingMode::Auto));
        assert_eq!(FramingMode::parse("rules"), Ok(FramingMode::Rules));
        assert_eq!(FramingMode::parse("punch-in"), Ok(FramingMode::Fixed(Algo::PunchIn)));
        let e = FramingMode::parse("teleport").unwrap_err();
        assert!(e.contains("teleport") && e.contains("face_follow") && e.contains("auto"), "{e}");
        assert_eq!(FramingMode::Fixed(Algo::Locked).label(), "locked");
    }

    #[test]
    fn menu_and_feasible_lists_come_from_the_registry() {
        assert_eq!(menu().len(), Algo::ALL.len());
        let one = feat(0, 1, 0.01);
        let names = feasible_names(&one);
        assert!(names.contains(&"face_follow") && names.contains(&"wide_blur"));
        assert!(!names.contains(&"split_two") && !names.contains(&"group_fit"));
    }

    #[test]
    fn good_answers_are_kept_bad_ones_are_repaired_missing_ones_covered() {
        let feats = vec![feat(0, 1, 0.01), feat(1, 1, 0.01), feat(2, 0, 0.2), feat(3, 2, 0.01)];
        let answers = vec![
            raw(0, "punch_in"),        // fine
            raw(1, "split_two"),       // needs two faces: infeasible
            raw(2, "hyperdrive"),      // unknown
            // shot 3: no answer
        ];
        let c = apply_answers(&feats, &answers);
        assert_eq!((c[0].algo, c[0].source), (Algo::PunchIn, Source::Ai));
        assert_eq!(c[0].reason, "because");
        assert_eq!((c[1].algo, c[1].source), (rules_default(&feats[1]), Source::Fallback));
        assert!(c[1].reason.contains("cannot run"), "{}", c[1].reason);
        assert_eq!(c[2].source, Source::Fallback);
        assert!(c[2].reason.contains("unknown"));
        assert_eq!((c[3].algo, c[3].source), (rules_default(&feats[3]), Source::Fallback));
        assert!(c[3].reason.contains("no answer"));
    }

    #[test]
    fn answers_are_matched_by_shot_id_not_position() {
        let feats = vec![feat(7, 1, 0.01), feat(9, 0, 0.2)];
        let answers = vec![raw(9, "motion_follow"), raw(7, "locked")];
        let c = apply_answers(&feats, &answers);
        assert_eq!((c[0].algo, c[1].algo), (Algo::Locked, Algo::MotionFollow));
    }

    #[test]
    fn report_lists_every_shot_with_its_choice() {
        use crate::stages::shot_classify::Layout;
        use crate::types::{BBox, ShotFrame, ShotStrategy};
        let shot = |idx: usize, a: f64, b: f64| ShotClassified {
            shot_list_idx: idx,
            start: a,
            end: b,
            frame: ShotFrame::Speaker,
            target_track_id: None,
            strategy: ShotStrategy::FaceCrop,
            layout: Layout::Single,
            samples: vec![],
            fallback_bbox: BBox { x: 0.0, y: 0.0, w: 1.0, h: 1.0 },
        };
        let feats = vec![feat(0, 1, 0.01), feat(1, 0, 0.2)];
        let choices = apply_answers(&feats, &[raw(0, "face_follow"), raw(1, "motion_follow")]);
        let md = report_section("clip_000", &[shot(0, 0.0, 4.0), shot(1, 4.0, 9.0)], &feats, &choices, &FramingMode::Auto);
        assert!(md.contains("clip_000") && md.contains("mode: auto"));
        assert!(md.contains("**face_follow**") && md.contains("**motion_follow**") && md.contains("| ai |"));
    }
}
