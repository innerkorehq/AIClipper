//! Framing algorithms and the features used to choose between them.
//!
//! "Framing" = which part of a 16:9 shot ends up in the vertical frame. Different
//! footage needs different answers — a talking head, a two-person conversation, a
//! group, a fight scene, a tripod shot where nobody moves — so instead of one
//! hard-coded rule there is a menu of algorithms and a chooser (`chooser.rs`)
//! that picks one per shot.
//!
//! Each algorithm says *what to track* (`Tracker`), *how the crop window is
//! placed* (`Window`) and *how to treat the box* (`Subject`). This module also
//! holds the measurements the chooser reasons over (`ShotFeatures`), the
//! feasibility rules (an algorithm that needs two faces is never run on a shot
//! with none) and a deterministic default used when the AI is unavailable.

use crate::config::Config;
use crate::stages::motion::MotionTrack;
use crate::stages::shot_classify::{Layout, ShotClassified};
use crate::types::Word;
use crate::vision::saliency_concentration;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Algo {
    /// Follow the most prominent face, full-height 9:16 window.
    FaceFollow,
    /// Follow whoever is speaking among several faces.
    SpeakerFollow,
    /// Two people, stacked half/half, each in their own fixed half.
    SplitTwo,
    /// Frame everyone: a window just large enough for the whole group.
    GroupFit,
    /// Tight, zoomed-in face for emphasis or reaction.
    PunchIn,
    /// Hold one fixed window for the whole shot (no camera motion).
    Locked,
    /// Follow the visually salient region (no faces needed).
    SaliencyFollow,
    /// Follow where the motion is (action, sports, dance).
    MotionFollow,
    /// Whole scene centred with a blurred fill above and below.
    WideBlur,
}

/// What a tracker follows, i.e. where the keyframe boxes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tracker {
    LargestFace,
    ActiveSpeaker,
    TwoFaces,
    FaceUnion,
    FacesOrSaliency,
    Saliency,
    Motion,
    None,
}

/// How the crop window is derived from the tracked box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Window {
    /// Largest 9:16 window centred on the subject.
    Max,
    /// Smallest window that contains the whole box (never crops the subject).
    Fit,
    /// `Max` shrunk by this factor (> 1 = tighter).
    Zoom(f32),
    /// Two halves, see the split solver.
    Split,
    /// Whole frame, blurred fill.
    Contain,
}

/// Whether the tracked box is a face (grown to head-and-shoulders) or a region
/// (used as is).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subject {
    Face,
    Region,
}

pub const PUNCH_IN_ZOOM: f32 = 1.45;

impl Algo {
    pub const ALL: [Algo; 9] = [
        Algo::FaceFollow,
        Algo::SpeakerFollow,
        Algo::SplitTwo,
        Algo::GroupFit,
        Algo::PunchIn,
        Algo::Locked,
        Algo::SaliencyFollow,
        Algo::MotionFollow,
        Algo::WideBlur,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Algo::FaceFollow => "face_follow",
            Algo::SpeakerFollow => "speaker_follow",
            Algo::SplitTwo => "split_two",
            Algo::GroupFit => "group_fit",
            Algo::PunchIn => "punch_in",
            Algo::Locked => "locked",
            Algo::SaliencyFollow => "saliency_follow",
            Algo::MotionFollow => "motion_follow",
            Algo::WideBlur => "wide_blur",
        }
    }

    pub fn parse(s: &str) -> Option<Algo> {
        let s = s.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        Algo::ALL.into_iter().find(|a| a.name() == s)
    }

    /// One line the chooser reads: what it does and when it is the right pick.
    pub fn describe(self) -> &'static str {
        match self {
            Algo::FaceFollow => "Follows the main face with a full-height vertical window. The default for one person talking to camera; adapts to head movement.",
            Algo::SpeakerFollow => "Follows whoever is speaking when 2+ people are on screen, switching only after a short dwell. For interviews and panels where the camera stays wide but one person talks at a time.",
            Algo::SplitTwo => "Stacks two people, one per half of the screen, each fixed to their half. For conversations where both matter and are on screen together.",
            Algo::GroupFit => "A window just large enough to contain every face. For 3+ people together (groups, panels, families) where cropping anyone would be wrong.",
            Algo::PunchIn => "A tight zoom on one face. For short emphasis or reaction shots, and strong emotional moments; too tight to hold for long.",
            Algo::Locked => "Holds a single fixed window for the whole shot, no camera motion. For tripod shots where the subject barely moves, or when tracking would just add jitter.",
            Algo::SaliencyFollow => "Follows the visually most interesting region; needs no faces. For scenery, objects, products or demos with one clear focal point.",
            Algo::MotionFollow => "Follows where the motion is. For action, sports, dancing, fights, vehicles — anything where the subject moves a lot and may not be a face.",
            Algo::WideBlur => "Shows the whole scene centred with a blurred fill above and below, nothing cropped. For wide establishing shots, crowds, landscapes, screens/slides, or anything that loses meaning when cropped.",
        }
    }

    pub fn tracker(self) -> Tracker {
        match self {
            Algo::FaceFollow | Algo::PunchIn => Tracker::LargestFace,
            Algo::SpeakerFollow => Tracker::ActiveSpeaker,
            Algo::SplitTwo => Tracker::TwoFaces,
            Algo::GroupFit => Tracker::FaceUnion,
            Algo::Locked => Tracker::FacesOrSaliency,
            Algo::SaliencyFollow => Tracker::Saliency,
            Algo::MotionFollow => Tracker::Motion,
            Algo::WideBlur => Tracker::None,
        }
    }

    pub fn window(self) -> Window {
        match self {
            Algo::FaceFollow | Algo::SpeakerFollow | Algo::Locked | Algo::SaliencyFollow | Algo::MotionFollow => Window::Max,
            Algo::GroupFit => Window::Fit,
            Algo::PunchIn => Window::Zoom(PUNCH_IN_ZOOM),
            Algo::SplitTwo => Window::Split,
            Algo::WideBlur => Window::Contain,
        }
    }

    /// Face-based algorithms grow the face box to head-and-shoulders; the rest
    /// track a region that is already the right size.
    pub fn subject(self, has_faces: bool) -> Subject {
        match self {
            Algo::FaceFollow | Algo::SpeakerFollow | Algo::PunchIn | Algo::SplitTwo => Subject::Face,
            Algo::Locked if has_faces => Subject::Face,
            _ => Subject::Region,
        }
    }

    /// Smoothing strength for the tracked box: lower = steadier camera. A subject that
    /// moves fast needs a responsive camera — smoothing it hard just lets it walk out
    /// of the frame (measured: 0.18 kept a moving object fully in view only 40% of
    /// the time). Slow-changing targets (saliency, a group) tolerate a steadier one.
    pub fn ema_alpha(self) -> f32 {
        match self {
            Algo::MotionFollow => 0.60,
            Algo::SaliencyFollow => 0.30,
            Algo::GroupFit => 0.30,
            _ => 0.45,
        }
    }
}

// ---- features -----------------------------------------------------------------

/// Measurements of one shot, computed by deterministic code. The chooser (AI)
/// reasons over these; nothing here is a decision.
#[derive(Debug, Clone, Serialize)]
pub struct ShotFeatures {
    pub idx: usize,
    pub duration_s: f64,
    /// What the classifier saw: "single", "split" or "wide".
    pub layout: &'static str,
    /// Median / maximum number of confident faces per sampled frame.
    pub faces_median: f32,
    pub faces_max: usize,
    /// Fraction of sampled frames with at least one confident face.
    pub face_presence: f32,
    /// Median area of the largest face (fraction of the frame).
    pub face_area: f32,
    /// How much the main face moves: std-dev of its centre, in frame widths.
    pub face_motion: f32,
    /// Median width of the box around all faces (fraction of frame width).
    pub group_width: f32,
    /// 0 = attention is spread over the frame, 1 = one clear focal point.
    pub saliency_focus: Option<f32>,
    /// Mean frame-to-frame change, 0..1 (≈0.01 static, >0.05 busy).
    pub motion_energy: Option<f32>,
    /// Fraction of the shot in which a distinct moving region exists (0..1). Catches a
    /// small subject moving fast, which `motion_energy` underrates.
    pub motion_activity: Option<f32>,
    /// Fraction of the shot covered by speech.
    pub speaking: f32,
}

fn median(v: &mut Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Measure a classified shot. `clip_start` converts the shot's absolute times to
/// the rough cut's own timeline, which is what `motion` is indexed by.
pub fn compute_features(
    cfg: &Config,
    shot: &ShotClassified,
    words: &[Word],
    motion: Option<&MotionTrack>,
    clip_start: f64,
) -> ShotFeatures {
    let n = shot.samples.len().max(1);
    let mut counts: Vec<f32> = Vec::new();
    let mut areas: Vec<f32> = Vec::new();
    let mut widths: Vec<f32> = Vec::new();
    let mut centres: Vec<(f32, f32)> = Vec::new();
    let mut with_face = 0usize;
    for s in &shot.samples {
        let faces: Vec<_> = s.faces.iter().filter(|f| f.confidence >= cfg.face_confidence).collect();
        counts.push(faces.len() as f32);
        if let Some(big) = faces.iter().max_by(|a, b| (a.bbox.w * a.bbox.h).partial_cmp(&(b.bbox.w * b.bbox.h)).unwrap_or(std::cmp::Ordering::Equal)) {
            with_face += 1;
            areas.push(big.bbox.w * big.bbox.h);
            centres.push(big.bbox.center());
        }
        if faces.len() >= 2 {
            let lo = faces.iter().map(|f| f.bbox.x).fold(f32::MAX, f32::min);
            let hi = faces.iter().map(|f| f.bbox.x + f.bbox.w).fold(0.0, f32::max);
            widths.push(hi - lo);
        }
    }
    let faces_max = counts.iter().cloned().fold(0.0, f32::max) as usize;
    let face_motion = if centres.len() >= 2 {
        let m = centres.iter().map(|c| c.0).sum::<f32>() / centres.len() as f32;
        (centres.iter().map(|c| (c.0 - m).powi(2)).sum::<f32>() / centres.len() as f32).sqrt()
    } else {
        0.0
    };

    let grids: Vec<f64> = shot.samples.iter().filter_map(|s| s.saliency.as_ref()).map(saliency_concentration).collect();
    let saliency_focus = (!grids.is_empty()).then(|| (1.0 - grids.iter().sum::<f64>() / grids.len() as f64).clamp(0.0, 1.0) as f32);

    let motion_energy = motion.and_then(|m| m.mean_energy(shot.start - clip_start, shot.end - clip_start));
    let motion_activity = motion.and_then(|m| m.activity(shot.start - clip_start, shot.end - clip_start));

    let dur = (shot.end - shot.start).max(0.01);
    let spoken: f64 = words
        .iter()
        .map(|w| (w.end.min(shot.end) - w.start.max(shot.start)).max(0.0))
        .sum();

    ShotFeatures {
        idx: shot.shot_list_idx,
        duration_s: (dur * 10.0).round() / 10.0,
        layout: match shot.layout {
            Layout::Single => "single",
            Layout::Split => "split",
            Layout::Wide => "wide",
        },
        faces_median: median(&mut counts),
        faces_max,
        face_presence: with_face as f32 / n as f32,
        face_area: median(&mut areas),
        face_motion,
        group_width: median(&mut widths),
        saliency_focus,
        motion_energy,
        motion_activity,
        speaking: (spoken / dur).clamp(0.0, 1.0) as f32,
    }
}

impl ShotFeatures {
    /// The same measurements in plain words, for a model that reads words better
    /// than decimals. Derived from the numbers; adds no information of its own.
    pub fn describe(&self) -> String {
        let people = match self.faces_max {
            0 => "no faces".to_string(),
            1 => "one face".to_string(),
            2 => "two faces".to_string(),
            n => format!("{n} faces"),
        };
        let face_move = if self.face_presence < 0.4 {
            ""
        } else if self.face_motion < 0.02 {
            ", barely moves"
        } else if self.face_motion < LOCKED_MAX_FACE_MOTION {
            ", moves a little"
        } else {
            ", moves around a lot"
        };
        let small_mover = self.motion_activity.map(|a| a >= MIN_ACTIVITY_TO_FOLLOW).unwrap_or(false);
        let motion = match self.motion_energy {
            None => "motion unknown".to_string(),
            Some(e) if e >= BUSY_MOTION => "the picture is busy with movement".to_string(),
            Some(_) if small_mover => "something moves through part of an otherwise calm picture".to_string(),
            Some(e) if e < 0.015 => "the picture is static".to_string(),
            Some(_) => "the picture is calm".to_string(),
        };
        let length = if self.duration_s < 2.0 {
            "very short"
        } else if self.duration_s < 5.0 {
            "short"
        } else if self.duration_s < 15.0 {
            "medium length"
        } else {
            "long"
        };
        let speech = if self.speaking < 0.1 {
            "no speech"
        } else if self.speaking < 0.7 {
            "some speech"
        } else {
            "speech throughout"
        };
        format!("{length} shot ({:.1}s); {people}{face_move}; {motion}; {speech}", self.duration_s)
    }

    /// Compact, rounded JSON for the chooser prompt.
    pub fn to_json(&self) -> serde_json::Value {
        let r = |x: f32| (x * 1000.0).round() / 1000.0;
        serde_json::json!({
            "shot": self.idx,
            "in_words": self.describe(),
            "duration_s": self.duration_s,
            "classifier_layout": self.layout,
            "faces_median": self.faces_median,
            "faces_max": self.faces_max,
            "face_presence": r(self.face_presence),
            "face_area": r(self.face_area),
            "face_motion": r(self.face_motion),
            "group_width": r(self.group_width),
            "saliency_focus": self.saliency_focus.map(r),
            "motion_energy": self.motion_energy.map(r),
            "motion_activity": self.motion_activity.map(r),
            "speaking": r(self.speaking),
        })
    }
}

// ---- feasibility and the deterministic default --------------------------------

/// Motion energy above which a shot counts as busy.
pub const BUSY_MOTION: f32 = 0.04;
/// Following motion needs some motion to follow: busy overall, or a distinct moving
/// region present for at least this share of the shot.
pub const MIN_MOTION_TO_FOLLOW: f32 = 0.03;
pub const MIN_ACTIVITY_TO_FOLLOW: f32 = 0.5;
/// Following a focal point needs a clear one (0 = attention is everywhere).
pub const MIN_FOCUS_TO_FOLLOW: f32 = 0.35;
/// A punch-in is an emphasis device: too tight to hold for long.
pub const PUNCH_IN_MAX_S: f64 = 6.0;
/// A locked window crops the subject out if they move more than this (std-dev of the
/// face centre, in frame widths).
pub const LOCKED_MAX_FACE_MOTION: f32 = 0.08;

/// Can `algo` physically run on a shot with these measurements? Guards the AI:
/// it may pick anything it likes among what is possible.
pub fn feasible(algo: Algo, f: &ShotFeatures) -> bool {
    let has_faces = f.face_presence >= 0.4;
    match algo {
        Algo::WideBlur => true,
        Algo::FaceFollow => has_faces,
        Algo::PunchIn => has_faces && f.duration_s <= PUNCH_IN_MAX_S,
        Algo::SpeakerFollow | Algo::SplitTwo => f.faces_max >= 2 && f.face_presence >= 0.4,
        Algo::GroupFit => f.faces_max >= 2 && f.face_presence >= 0.4,
        // fixed window: only if what it frames holds still
        Algo::Locked => {
            if has_faces {
                f.face_motion <= LOCKED_MAX_FACE_MOTION
            } else {
                f.saliency_focus.is_some() || f.motion_energy.is_some()
            }
        }
        Algo::SaliencyFollow => f.saliency_focus.map(|x| x >= MIN_FOCUS_TO_FOLLOW).unwrap_or(false),
        Algo::MotionFollow => has_motion_to_follow(f),
    }
}

fn has_motion_to_follow(f: &ShotFeatures) -> bool {
    f.motion_energy.map(|e| e >= MIN_MOTION_TO_FOLLOW).unwrap_or(false)
        || f.motion_activity.map(|a| a >= MIN_ACTIVITY_TO_FOLLOW).unwrap_or(false)
}

/// The deterministic default (and the fallback when the AI answer is unusable):
/// the pre-chooser behaviour, extended to the new algorithms.
pub fn rules_default(f: &ShotFeatures) -> Algo {
    if f.layout == "split" && feasible(Algo::SplitTwo, f) {
        return Algo::SplitTwo;
    }
    if f.faces_median >= 3.0 && feasible(Algo::GroupFit, f) {
        return Algo::GroupFit;
    }
    if f.face_presence >= 0.5 {
        return if f.faces_max >= 2 { Algo::SpeakerFollow } else { Algo::FaceFollow };
    }
    if f.motion_energy.map(|e| e >= BUSY_MOTION).unwrap_or(false) || f.motion_activity.map(|a| a >= 0.6).unwrap_or(false) {
        return Algo::MotionFollow;
    }
    Algo::WideBlur
}

/// Turn a model's answer into an algorithm that can run. Unknown names and
/// infeasible picks fall back to the default, with a note saying why.
pub fn resolve(name: &str, f: &ShotFeatures) -> (Algo, Option<String>) {
    match Algo::parse(name) {
        None => (rules_default(f), Some(format!("unknown algorithm `{name}`"))),
        Some(a) if !feasible(a, f) => (
            rules_default(f),
            Some(format!("`{}` cannot run on this shot (faces {}, presence {:.0}%)", a.name(), f.faces_max, f.face_presence * 100.0)),
        ),
        Some(a) => (a, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f() -> ShotFeatures {
        ShotFeatures {
            idx: 0,
            duration_s: 5.0,
            layout: "single",
            faces_median: 1.0,
            faces_max: 1,
            face_presence: 1.0,
            face_area: 0.04,
            face_motion: 0.01,
            group_width: 0.0,
            saliency_focus: Some(0.5),
            motion_energy: Some(0.01),
            motion_activity: Some(0.0),
            speaking: 0.9,
        }
    }

    #[test]
    fn names_round_trip_and_parse_loosely() {
        for a in Algo::ALL {
            assert_eq!(Algo::parse(a.name()), Some(a));
            assert!(!a.describe().is_empty());
        }
        assert_eq!(Algo::parse("Face Follow"), Some(Algo::FaceFollow));
        assert_eq!(Algo::parse(" motion-follow "), Some(Algo::MotionFollow));
        assert_eq!(Algo::parse("teleport"), None);
    }

    #[test]
    fn every_algorithm_has_a_consistent_window_and_subject() {
        assert_eq!(Algo::SplitTwo.window(), Window::Split);
        assert_eq!(Algo::WideBlur.window(), Window::Contain);
        assert_eq!(Algo::GroupFit.window(), Window::Fit);
        assert!(matches!(Algo::PunchIn.window(), Window::Zoom(z) if z > 1.0));
        assert_eq!(Algo::GroupFit.subject(true), Subject::Region);
        assert_eq!(Algo::MotionFollow.subject(false), Subject::Region);
        assert_eq!(Algo::Locked.subject(true), Subject::Face);
        assert_eq!(Algo::Locked.subject(false), Subject::Region);
        assert!(Algo::MotionFollow.ema_alpha() >= Algo::FaceFollow.ema_alpha(), "a fast mover needs a responsive camera");
        assert!(Algo::GroupFit.ema_alpha() <= Algo::FaceFollow.ema_alpha(), "a group can be framed steadily");
    }

    #[test]
    fn feasibility_blocks_impossible_picks() {
        let one = f();
        assert!(feasible(Algo::FaceFollow, &one) && feasible(Algo::PunchIn, &one) && feasible(Algo::Locked, &one));
        assert!(!feasible(Algo::SplitTwo, &one) && !feasible(Algo::GroupFit, &one) && !feasible(Algo::SpeakerFollow, &one));
        let mut none = f();
        none.face_presence = 0.0;
        none.faces_max = 0;
        none.faces_median = 0.0;
        assert!(!feasible(Algo::FaceFollow, &none));
        // nothing is moving and attention is spread: there is nothing to follow
        none.saliency_focus = Some(0.2);
        assert!(feasible(Algo::WideBlur, &none) && !feasible(Algo::SaliencyFollow, &none) && !feasible(Algo::MotionFollow, &none));
        none.motion_energy = Some(0.12);
        none.saliency_focus = Some(0.6);
        assert!(feasible(Algo::SaliencyFollow, &none) && feasible(Algo::MotionFollow, &none));
        none.saliency_focus = None;
        none.motion_energy = None;
        assert!(!feasible(Algo::SaliencyFollow, &none) && !feasible(Algo::MotionFollow, &none) && !feasible(Algo::Locked, &none));
        assert!(feasible(Algo::WideBlur, &none), "wide always works");
    }

    #[test]
    fn common_sense_limits_on_the_dramatic_choices() {
        let mut calm = f(); // one still face, static picture, 5s
        assert!(feasible(Algo::PunchIn, &calm));
        calm.duration_s = 20.0;
        assert!(!feasible(Algo::PunchIn, &calm), "a punch-in is not held for 20 seconds");
        let mut moving = f();
        moving.face_motion = 0.15;
        assert!(!feasible(Algo::Locked, &moving), "a fixed window would crop a subject who moves around");
        assert!(feasible(Algo::FaceFollow, &moving));
        let mut action = f();
        action.face_presence = 0.0;
        action.faces_max = 0;
        action.motion_energy = Some(0.021);
        assert!(!feasible(Algo::MotionFollow, &action), "0.021 is not enough motion to follow");
        action.motion_energy = Some(0.06);
        assert!(feasible(Algo::MotionFollow, &action));
        // a small subject moving fast: little whole-frame energy, but a mover is there
        action.motion_energy = Some(0.004);
        action.motion_activity = Some(0.9);
        assert!(feasible(Algo::MotionFollow, &action));
        assert_eq!(rules_default(&action), Algo::MotionFollow);
        assert!(action.describe().contains("something moves"), "{}", action.describe());
    }

    #[test]
    fn plain_language_reading_matches_the_numbers() {
        let mut x = f();
        x.duration_s = 4.0;
        let d = x.describe();
        assert!(d.contains("short shot") && d.contains("one face") && d.contains("barely moves") && d.contains("static") && d.contains("speech throughout"), "{d}");
        x.duration_s = 1.4;
        x.faces_max = 3;
        x.face_motion = 0.2;
        x.motion_energy = Some(0.09);
        x.speaking = 0.0;
        let d = x.describe();
        assert!(d.contains("very short") && d.contains("3 faces") && d.contains("moves around a lot") && d.contains("busy") && d.contains("no speech"), "{d}");
        x.motion_energy = None;
        assert!(x.describe().contains("motion unknown"));
        assert!(x.to_json()["in_words"].as_str().unwrap().contains("very short"));
    }

    #[test]
    fn rules_default_covers_each_kind_of_shot() {
        assert_eq!(rules_default(&f()), Algo::FaceFollow);
        let mut two = f();
        two.layout = "split";
        two.faces_median = 2.0;
        two.faces_max = 2;
        assert_eq!(rules_default(&two), Algo::SplitTwo);
        let mut group = f();
        group.faces_median = 4.0;
        group.faces_max = 5;
        assert_eq!(rules_default(&group), Algo::GroupFit);
        let mut panel = f();
        panel.faces_max = 2; // two faces but not classified split: follow the speaker
        panel.faces_median = 1.0;
        assert_eq!(rules_default(&panel), Algo::SpeakerFollow);
        let mut action = f();
        action.layout = "wide";
        action.face_presence = 0.0;
        action.faces_max = 0;
        action.faces_median = 0.0;
        action.motion_energy = Some(0.12);
        assert_eq!(rules_default(&action), Algo::MotionFollow);
        action.motion_energy = Some(0.005);
        assert_eq!(rules_default(&action), Algo::WideBlur);
    }

    #[test]
    fn resolve_accepts_good_picks_and_falls_back_on_bad_ones() {
        let one = f();
        assert_eq!(resolve("punch_in", &one), (Algo::PunchIn, None));
        let (a, note) = resolve("split_two", &one);
        assert_eq!(a, rules_default(&one));
        assert!(note.unwrap().contains("cannot run"));
        let (a, note) = resolve("hyperdrive", &one);
        assert_eq!(a, rules_default(&one));
        assert!(note.unwrap().contains("unknown"));
    }
}
