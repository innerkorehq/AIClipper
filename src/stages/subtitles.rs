//! Source subtitles as the transcript's source of truth.
//!
//! When the video ships with subtitles — a sidecar `.srt`/`.vtt`, an embedded
//! text track, or (for URLs) the uploader's own subtitles fetched by yt-dlp — they
//! were written or checked by a person and beat any ASR guess, especially on
//! code-mixed speech (Hinglish) where ASR is weakest. They also save the whole
//! transcription step.
//!
//! Subtitles are phrase-level, the pipeline is word-level, so each cue's words
//! are given timings spread across the cue in proportion to their length.
//! Only uploader-provided subtitles are used; YouTube's auto-generated captions
//! are just another ASR and are never fetched.

use crate::config::Config;
use crate::types::Word;
use anyhow::Result;
use std::path::{Path, PathBuf};
use tokio::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Parse `HH:MM:SS,mmm`, `HH:MM:SS.mmm` or `MM:SS.mmm`.
fn parse_ts(s: &str) -> Option<f64> {
    let s = s.trim().replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }
    let mut total = 0.0;
    for p in &parts {
        total = total * 60.0 + p.parse::<f64>().ok()?;
    }
    Some(total)
}

/// Remove markup (`<i>`, `<c.colorE5E5E5>`, `<00:00:01.000>`, ASS `{\an8}`) and
/// decode the handful of HTML entities subtitle files use.
fn clean_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut depth_angle = 0u32;
    let mut depth_brace = 0u32;
    for ch in raw.chars() {
        match ch {
            '<' => depth_angle += 1,
            '>' if depth_angle > 0 => depth_angle -= 1,
            '{' => depth_brace += 1,
            '}' if depth_brace > 0 => depth_brace -= 1,
            _ if depth_angle > 0 || depth_brace > 0 => {}
            _ => out.push(ch),
        }
    }
    let out = out
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
        .replace('♪', " ");
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// True for cues that only describe a sound — `[Music]`, `(laughs)` — not speech.
fn is_annotation(text: &str) -> bool {
    let t = text.trim();
    t.is_empty()
        || (t.starts_with('[') && t.ends_with(']'))
        || (t.starts_with('(') && t.ends_with(')'))
}

/// Parse SRT or WebVTT text into cues, dropping sound annotations and merging
/// back-to-back duplicates (rolling captions repeat the previous line).
pub fn parse_subtitles(text: &str) -> Vec<Cue> {
    let text = text.trim_start_matches('\u{feff}').replace("\r\n", "\n").replace('\r', "\n");
    let mut cues: Vec<Cue> = Vec::new();
    for block in text.split("\n\n") {
        let lines: Vec<&str> = block.lines().collect();
        let Some(ti) = lines.iter().position(|l| l.contains("-->")) else { continue };
        let (a, b) = match lines[ti].split_once("-->") {
            Some(p) => p,
            None => continue,
        };
        // the end timestamp may be followed by VTT cue settings ("align:start")
        let b = b.split_whitespace().next().unwrap_or("");
        let (Some(start), Some(end)) = (parse_ts(a), parse_ts(b)) else { continue };
        if end <= start {
            continue;
        }
        let body = lines[ti + 1..].join(" ");
        let mut cleaned = clean_text(&body);
        // dialogue dashes ("- Hello") are speaker turns, not words
        while let Some(rest) = cleaned.strip_prefix("- ").or_else(|| cleaned.strip_prefix("— ")) {
            cleaned = rest.to_string();
        }
        if is_annotation(&cleaned) {
            continue;
        }
        if let Some(prev) = cues.last_mut() {
            if prev.text == cleaned && start <= prev.end + 0.1 {
                prev.end = prev.end.max(end);
                continue;
            }
        }
        cues.push(Cue { start, end, text: cleaned });
    }
    cues.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    cues
}

/// Seconds a person needs to say `chars` characters, as an upper bound on how
/// long a cue's speech really lasts (cues usually linger after the last word).
fn speech_seconds(chars: usize) -> f64 {
    0.3 + 0.1 * chars as f64
}

/// Word-level transcript from cues. Within a cue each word gets time in
/// proportion to its length; a cue's speech is capped at a plausible duration so
/// the last word isn't stretched across the cue's trailing silence, and never
/// runs into the next cue.
pub fn cues_to_words(cues: &[Cue]) -> Vec<Word> {
    let mut words = Vec::new();
    for (i, c) in cues.iter().enumerate() {
        // bare dashes separate speakers in a cue; they are not spoken words
        let toks: Vec<&str> = c
            .text
            .split_whitespace()
            .filter(|t| !t.chars().all(|ch| matches!(ch, '-' | '–' | '—')))
            .collect();
        if toks.is_empty() {
            continue;
        }
        let chars: usize = toks.iter().map(|t| t.chars().count()).sum();
        let next_start = cues.get(i + 1).map(|n| n.start).unwrap_or(f64::MAX);
        let end = c.end.min(next_start).max(c.start + 0.05);
        let span = (end - c.start).min(speech_seconds(chars));
        let mut t = c.start;
        for tok in toks {
            let w = tok.chars().count().max(1) as f64 / chars.max(1) as f64;
            let dur = span * w;
            words.push(Word { text: tok.to_string(), start: t, end: t + dur });
            t += dur;
        }
    }
    words
}

/// Is this subtitle track a believable transcript for a video of `duration`?
/// (A few forced-subtitle lines for foreign phrases are not.)
pub fn is_usable(words: &[Word], duration: f64) -> bool {
    let minutes = (duration / 60.0).max(0.1);
    words.len() >= 30 && words.len() as f64 / minutes >= 20.0
}

pub fn transcript_from_file(path: &Path) -> Result<Vec<Word>> {
    let raw = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&raw);
    Ok(cues_to_words(&parse_subtitles(&text)))
}

fn is_sub_file(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| matches!(e.to_ascii_lowercase().as_str(), "srt" | "vtt"))
        .unwrap_or(false)
}

/// Language token of `name.<lang>.srt` (None for plain `name.srt`).
fn lang_of(p: &Path) -> Option<String> {
    let stem = p.file_stem()?.to_str()?;
    let (_, lang) = stem.rsplit_once('.')?;
    Some(lang.to_ascii_lowercase())
}

/// Choose among subtitle files: the first preferred language that matches
/// (`hi` matches `hi`, `hi-IN`), else an untagged file, else the first by name.
pub fn pick_subtitle(mut files: Vec<PathBuf>, prefs: &[String]) -> Option<PathBuf> {
    files.sort();
    files.retain(|p| is_sub_file(p));
    for pref in prefs {
        let pref = pref.to_ascii_lowercase();
        if let Some(f) = files
            .iter()
            .find(|p| lang_of(p).map(|l| l == pref || l.starts_with(&format!("{pref}-"))).unwrap_or(false))
        {
            return Some(f.clone());
        }
    }
    files.first().cloned()
}

fn list_dir(dir: &Path, keep: impl Fn(&Path) -> bool) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| keep(p)).collect())
        .unwrap_or_default()
}

/// Find subtitles for `input` (a local path or a URL), best source first:
/// an explicit `--subtitles` file, a sidecar file next to a local video, an
/// embedded text track, then files yt-dlp downloaded beside the source.
pub async fn locate(cfg: &Config, input: &str, source: &Path, explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return p.is_file().then(|| p.to_path_buf());
    }
    let prefs: Vec<String> = cfg.sub_langs.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();

    let local = Path::new(input);
    if local.is_file() {
        // sidecar: `<video stem>.srt`, `<video stem>.<lang>.srt`
        if let (Some(dir), Some(stem)) = (local.parent(), local.file_stem().and_then(|s| s.to_str())) {
            let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
            let stem = stem.to_string();
            let found = list_dir(dir, |p| {
                is_sub_file(p)
                    && p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with(&stem)).unwrap_or(false)
            });
            if let Some(p) = pick_subtitle(found, &prefs) {
                return Some(p);
            }
        }
        // embedded text track
        let out = cfg.media_dir().join("embedded_subs.srt");
        let status = Command::new(&cfg.ffmpeg)
            .args(["-y", "-v", "error", "-i"])
            .arg(source)
            .args(["-map", "0:s:0", "-c:s", "srt"])
            .arg(&out)
            .output()
            .await;
        if let Ok(o) = status {
            if o.status.success() && std::fs::metadata(&out).map(|m| m.len() > 0).unwrap_or(false) {
                return Some(out);
            }
        }
        let _ = std::fs::remove_file(&out);
        return None;
    }

    // URL: yt-dlp wrote `source.<lang>.srt` beside `source.mp4`
    let found = list_dir(&cfg.media_dir(), |p| {
        is_sub_file(p) && p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with("source.")).unwrap_or(false)
    });
    pick_subtitle(found, &prefs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRT: &str = "\u{feff}1\r\n00:00:01,000 --> 00:00:03,000\r\n<i>Namaste</i> doston, aaj ka topic\r\n\r\n2\r\n00:00:03,200 --> 00:00:04,000\r\n[Music]\r\n\r\n3\r\n00:00:04,000 --> 00:00:06,500\r\n- Haan &amp; yeh bahut zaroori hai\r\n- Bilkul.\r\n";

    #[test]
    fn parses_srt_with_markup_annotations_and_entities() {
        let c = parse_subtitles(SRT);
        assert_eq!(c.len(), 2, "the [Music] cue is dropped: {c:?}");
        assert_eq!(c[0].text, "Namaste doston, aaj ka topic");
        assert_eq!((c[0].start, c[0].end), (1.0, 3.0));
        assert_eq!(c[1].text, "Haan & yeh bahut zaroori hai - Bilkul.");
        assert_eq!(c[1].start, 4.0);
    }

    #[test]
    fn parses_vtt_with_settings_and_short_timestamps() {
        let vtt = "WEBVTT\n\nNOTE comment\n\n00:01.500 --> 00:03.000 align:start position:0%\n<c.yellow>Hello</c> there\n\n00:00:03.000 --> 00:00:04.000\nHello there\n\n00:00:03.050 --> 00:00:05.000\nHello there\n";
        let c = parse_subtitles(vtt);
        assert_eq!(c.len(), 1, "duplicates merge: {c:?}");
        assert_eq!(c[0].text, "Hello there");
        assert_eq!((c[0].start, c[0].end), (1.5, 5.0));
    }

    #[test]
    fn words_are_timed_inside_their_cue_and_proportionally() {
        let c = vec![
            Cue { start: 10.0, end: 12.0, text: "go to the market".into() },
            Cue { start: 12.0, end: 20.0, text: "ok".into() }, // lingers 8s for 2 chars
        ];
        let w = cues_to_words(&c);
        assert_eq!(w.len(), 5);
        assert_eq!(w[0].start, 10.0);
        assert!(w[3].end <= 12.0 + 1e-9, "stays inside its cue");
        for p in w.windows(2) {
            assert!(p[0].end <= p[1].start + 1e-9, "ordered, no overlap");
        }
        assert!(w[3].end - w[3].start > w[0].end - w[0].start, "longer words get longer");
        let ok = &w[4];
        assert!(ok.end - ok.start <= speech_seconds(2) + 1e-9, "a lingering cue does not stretch its word");
    }

    #[test]
    fn speaker_dashes_are_not_words() {
        let c = vec![Cue { start: 0.0, end: 3.0, text: "haan - nahin - theek hai".into() }];
        let t: Vec<_> = cues_to_words(&c).into_iter().map(|w| w.text).collect();
        assert_eq!(t, ["haan", "nahin", "theek", "hai"]);
    }

    #[test]
    fn cue_never_runs_into_the_next_one() {
        let c = vec![
            Cue { start: 0.0, end: 5.0, text: "one two three four five six seven".into() },
            Cue { start: 1.0, end: 2.0, text: "next".into() },
        ];
        let w = cues_to_words(&c);
        assert!(w[..7].iter().all(|x| x.end <= 1.0 + 1e-9));
    }

    #[test]
    fn usability_rejects_forced_subtitle_tracks() {
        let few: Vec<Word> = (0..6).map(|i| Word { text: "x".into(), start: i as f64, end: i as f64 + 0.5 }).collect();
        assert!(!is_usable(&few, 1200.0));
        let many: Vec<Word> = (0..400).map(|i| Word { text: "x".into(), start: i as f64, end: i as f64 + 0.5 }).collect();
        assert!(is_usable(&many, 600.0));
    }

    #[test]
    fn picks_preferred_language() {
        let files = vec![PathBuf::from("v.en.srt"), PathBuf::from("v.hi-IN.srt"), PathBuf::from("v.fr.vtt"), PathBuf::from("v.txt")];
        assert_eq!(pick_subtitle(files.clone(), &["hi".into(), "en".into()]).unwrap(), PathBuf::from("v.hi-IN.srt"));
        assert_eq!(pick_subtitle(files.clone(), &["en".into()]).unwrap(), PathBuf::from("v.en.srt"));
        assert_eq!(pick_subtitle(files, &["de".into()]).unwrap(), PathBuf::from("v.en.srt")); // first by name
    }
}
