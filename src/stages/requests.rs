//! Clip requests: telling the pipeline *which* clips you want.
//!
//! A request is one plain-language ask ("the funniest exchange about the DM
//! mix-up, 30-45s, after 10:00"), optionally with a count, duration limits, a
//! time window and a cut mode. Requests come from repeated `--ask` flags and/or a
//! `--brief` file. Hints written in the sentence are pulled out with
//! deterministic parsing — no model involved, so what you typed is what you get —
//! and explicit brief fields always win. The parsed requests are echoed before any
//! model work so a misreading is visible immediately.

use crate::cli::parse_time;
use crate::types::{ClipCandidate, Transcript};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MAX_COUNT: usize = 20;

/// What kind of moment is wanted; only steers how much the audio signals
/// (laughter / loudness) count, never what the text says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tone {
    #[default]
    Any,
    Funny,
    Emotional,
    Intense,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipRequest {
    pub id: usize,
    /// The ask as written.
    pub text: String,
    /// Short slug used in the output file name.
    pub label: String,
    /// Up to this many clips (fewer is fine and reported — never padded).
    pub count: usize,
    pub min_s: Option<f64>,
    pub max_s: Option<f64>,
    /// Only search inside this span of the source (seconds).
    pub window: Option<(f64, f64)>,
    pub tone: Tone,
    /// Some(true): keep this clip continuous (no filler/silence cuts).
    pub no_cuts: Option<bool>,
}

impl ClipRequest {
    #[cfg(test)]
    pub fn with_count(mut self, n: usize) -> Self {
        self.count = n;
        self
    }

    pub fn describe(&self) -> String {
        let mut parts = vec![format!("up to {} clip{}", self.count, if self.count == 1 { "" } else { "s" })];
        match (self.min_s, self.max_s) {
            (Some(a), Some(b)) => parts.push(format!("{a:.0}–{b:.0}s")),
            (Some(a), None) => parts.push(format!("≥{a:.0}s")),
            (None, Some(b)) => parts.push(format!("≤{b:.0}s")),
            _ => {}
        }
        if let Some((a, b)) = self.window {
            parts.push(format!("within {}–{}", fmt_clock(a), fmt_clock(b)));
        }
        if self.tone != Tone::Any {
            parts.push(format!("tone {:?}", self.tone).to_lowercase());
        }
        if self.no_cuts == Some(true) {
            parts.push("no cuts".into());
        }
        format!("#{} [{}] \"{}\" — {}", self.id + 1, self.label, self.text, parts.join(", "))
    }
}

pub fn fmt_clock(t: f64) -> String {
    let t = t.max(0.0).round() as u64;
    if t >= 3600 {
        format!("{}:{:02}:{:02}", t / 3600, (t / 60) % 60, t % 60)
    } else {
        format!("{}:{:02}", t / 60, t % 60)
    }
}

// ---- parsing hints out of a sentence ---------------------------------------

fn unit_secs(u: &str) -> Option<f64> {
    match u.trim_end_matches(['.', ',', '?', '!']) {
        "s" | "sec" | "secs" | "second" | "seconds" => Some(1.0),
        "m" | "min" | "mins" | "minute" | "minutes" => Some(60.0),
        _ => None,
    }
}

fn number_word(w: &str) -> Option<f64> {
    Some(match w {
        "a" | "an" | "one" => 1.0,
        "two" => 2.0,
        "three" => 3.0,
        "four" => 4.0,
        "five" => 5.0,
        "six" => 6.0,
        "seven" => 7.0,
        "eight" => 8.0,
        "nine" => 9.0,
        "ten" => 10.0,
        _ => return None,
    })
}

fn num(tok: &str) -> Option<f64> {
    let t = tok.trim_end_matches(['.', ',', '?', '!']);
    t.parse::<f64>().ok().or_else(|| number_word(t))
}

/// `30s`, `45sec`, `2min` → (value, unit seconds). `30` → (30, None).
fn num_unit(tok: &str) -> Option<(f64, Option<f64>)> {
    let t = tok.trim_end_matches(['.', ',', '?', '!']);
    let split = t.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(t.len());
    if split == 0 {
        return number_word(t).map(|v| (v, None));
    }
    let v: f64 = t[..split].parse().ok()?;
    let u = &t[split..];
    if u.is_empty() {
        Some((v, None))
    } else {
        unit_secs(u).map(|k| (v, Some(k)))
    }
}

/// A clock value: `10:00`, `1:02:03` (only tokens with ':').
fn clock(tok: &str) -> Option<f64> {
    let t = tok.trim_end_matches(['.', ',', '?', '!']);
    if t.contains(':') {
        parse_time(t).ok()
    } else {
        None
    }
}

fn split_range(tok: &str) -> Option<(&str, &str)> {
    for sep in ['-', '–', '—'] {
        if let Some((a, b)) = tok.split_once(sep) {
            if !a.is_empty() && !b.is_empty() {
                return Some((a, b));
            }
        }
    }
    None
}

#[derive(Default)]
struct Hints {
    count: Option<usize>,
    min_s: Option<f64>,
    max_s: Option<f64>,
    window: Option<(f64, f64)>,
}

fn parse_hints(text: &str, video_duration: f64) -> Hints {
    let cleaned: String = text
        .to_lowercase()
        .chars()
        .map(|c| if matches!(c, '(' | ')' | ';' | '"' | '\u{201c}' | '\u{201d}') { ' ' } else { c })
        .collect();
    let toks: Vec<&str> = cleaned.split_whitespace().collect();
    let mut used = vec![false; toks.len()];
    let mut h = Hints::default();
    let get = |i: usize| toks.get(i).copied();

    // -- time window --------------------------------------------------------
    for i in 0..toks.len() {
        if used[i] {
            continue;
        }
        let t = toks[i];
        // "between X and Y" / "from X to Y"
        if matches!(t, "between" | "from") {
            if let (Some(a), Some(sep), Some(b)) = (get(i + 1).and_then(clock), get(i + 2), get(i + 3).and_then(clock)) {
                if matches!(sep, "and" | "to" | "-" | "–") {
                    h.window = Some((a, b));
                    used[i..=i + 3].iter_mut().for_each(|u| *u = true);
                    continue;
                }
            }
        }
        // "10:00-15:00"
        if let Some((a, b)) = split_range(t) {
            if let (Some(a), Some(b)) = (clock(a), clock(b)) {
                h.window = Some((a, b));
                used[i] = true;
                continue;
            }
        }
        // "after X" / "before X" (X a clock or "N minutes")
        if matches!(t, "after" | "before" | "past") {
            let (val, n) = match get(i + 1) {
                Some(x) if clock(x).is_some() => (clock(x), 1),
                Some(x) => match (num(x), get(i + 2).and_then(unit_secs)) {
                    (Some(v), Some(k)) => (Some(v * k), 2),
                    _ => (None, 0),
                },
                None => (None, 0),
            };
            if let Some(v) = val {
                h.window = Some(if t == "before" { (0.0, v) } else { (v, video_duration) });
                used[i..=i + n].iter_mut().for_each(|u| *u = true);
                continue;
            }
        }
        // "first N minutes" / "last N minutes"
        if matches!(t, "first" | "last") {
            if let (Some(v), Some(k)) = (get(i + 1).and_then(num), get(i + 2).and_then(unit_secs)) {
                let span = v * k;
                h.window = Some(if t == "first" { (0.0, span) } else { ((video_duration - span).max(0.0), video_duration) });
                used[i..=i + 2].iter_mut().for_each(|u| *u = true);
            }
        }
    }

    // -- duration ------------------------------------------------------------
    for i in 0..toks.len() {
        if used[i] || toks[i].contains(':') {
            continue;
        }
        let t = toks[i];
        // "30-45s" / "30-45 seconds"
        if let Some((a, b)) = split_range(t) {
            if let (Some((lo, ua)), Some((hi, ub))) = (num_unit(a), num_unit(b)) {
                let unit = ub.or(ua).or_else(|| get(i + 1).and_then(unit_secs));
                if let Some(k) = unit {
                    h.min_s = Some(lo * k);
                    h.max_s = Some(hi * k);
                    used[i] = true;
                    if ub.is_none() && ua.is_none() {
                        used[i + 1] = true;
                    }
                    continue;
                }
            }
        }
        // "30 to 45 seconds"
        if let (Some((lo, ua)), Some("to"), Some((hi, ub))) = (num_unit(t), get(i + 1), get(i + 2).and_then(num_unit)) {
            let unit = ub.or(ua).or_else(|| get(i + 3).and_then(unit_secs));
            if let Some(k) = unit {
                h.min_s = Some(lo * k);
                h.max_s = Some(hi * k);
                used[i..=i + 2].iter_mut().for_each(|u| *u = true);
                continue;
            }
        }
        // comparatives: "under 40 seconds", "at least 20s", "about a minute"
        let (kind, skip) = match t {
            "under" | "below" | "max" | "maximum" | "within" => (1, 1),
            "less" if get(i + 1) == Some("than") => (1, 2),
            "up" if get(i + 1) == Some("to") => (1, 2),
            "at" if get(i + 1) == Some("most") => (1, 2),
            "over" | "above" | "min" | "minimum" => (2, 1),
            "more" | "longer" if get(i + 1) == Some("than") => (2, 2),
            "at" if get(i + 1) == Some("least") => (2, 2),
            "around" | "about" | "roughly" | "approximately" | "~" => (3, 1),
            _ => (0, 0),
        };
        if kind != 0 {
            // number (+unit) after the keyword
            let j = i + skip;
            if let Some((v, u)) = get(j).and_then(num_unit) {
                let unit = u.or_else(|| get(j + 1).and_then(unit_secs));
                if let Some(k) = unit {
                    let secs = v * k;
                    match kind {
                        1 => h.max_s = Some(secs),
                        2 => h.min_s = Some(secs),
                        _ => {
                            h.min_s = Some(secs * 0.8);
                            h.max_s = Some(secs * 1.25);
                        }
                    }
                    let n = if u.is_some() { j } else { j + 1 };
                    used[i..=n].iter_mut().for_each(|x| *x = true);
                    continue;
                }
            }
        }
    }
    // bare "30 second clip" / "45s": treat as "about N"
    if h.min_s.is_none() && h.max_s.is_none() {
        for i in 0..toks.len() {
            if used[i] || toks[i].contains(':') {
                continue;
            }
            if let Some((v, u)) = num_unit(toks[i]) {
                let unit = u.or_else(|| get(i + 1).and_then(unit_secs));
                // plain numbers need an explicit unit; "a minute" counts, "a clip" does not
                if let Some(k) = unit {
                    if u.is_some() || num(toks[i]).is_some() {
                        h.min_s = Some(v * k * 0.8);
                        h.max_s = Some(v * k * 1.25);
                        used[i] = true;
                        if u.is_none() {
                            used[i + 1] = true;
                        }
                        break;
                    }
                }
            }
        }
    }

    // -- count ---------------------------------------------------------------
    const NOUNS: [&str; 7] = ["clip", "moment", "part", "scene", "highlight", "segment", "exchange"];
    for i in 0..toks.len() {
        let noun = toks[i].trim_end_matches(['.', ',', '?', '!']).trim_end_matches('s');
        if !NOUNS.contains(&noun) || used[i] {
            continue;
        }
        // "2 clips", "three moments", "2 funny clips"
        for back in 1..=2 {
            if i < back {
                break;
            }
            let prev = toks[i - back];
            if unit_secs(prev).is_some() || used[i - back] {
                break; // "30 second clip" is a duration, not a count
            }
            if let Some(v) = num(prev) {
                if v >= 1.0 && v.fract() == 0.0 && (v as usize) <= MAX_COUNT && !(prev == "a" || prev == "an") {
                    h.count = Some(v as usize);
                }
                break;
            }
        }
    }
    for i in 0..toks.len().saturating_sub(1) {
        if toks[i] == "top" && !used[i] {
            if let Some(v) = num(toks[i + 1]) {
                if v >= 1.0 && (v as usize) <= MAX_COUNT {
                    h.count = Some(v as usize);
                }
            }
        }
    }
    h
}

fn tone_of(text: &str) -> Tone {
    let t = text.to_lowercase();
    let has = |ks: &[&str]| ks.iter().any(|k| t.contains(k));
    if has(&["funny", "funniest", "hilarious", "laugh", "humor", "humour", "joke", "comic", "comedy", "banter", "roast"]) {
        Tone::Funny
    } else if has(&["heated", "intense", "angry", "argument", "argue", "fight", "shout", "loud", "passionate", "debate"]) {
        Tone::Intense
    } else if has(&["emotional", "emotion", "heartfelt", "touching", "sad", "cry", "moving", "wholesome"]) {
        Tone::Emotional
    } else {
        Tone::Any
    }
}

/// `the-funniest-exchange-about-the` style slug for file names.
fn slugify(text: &str, id: usize) -> String {
    const STOP: &[&str] = &[
        "the", "a", "an", "of", "to", "and", "in", "on", "at", "for", "with", "where", "when", "part", "clip", "clips", "moment",
        "moments", "scene", "about", "that", "this", "me", "my", "seconds", "second", "sec", "minutes", "minute", "min", "how",
    ];
    let words: Vec<String> = text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1 && !STOP.contains(w) && !w.chars().all(|c| c.is_ascii_digit()))
        .take(4)
        .map(|w| w.to_string())
        .collect();
    let mut slug = words.join("-");
    slug.truncate(32);
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        format!("request-{}", id + 1)
    } else {
        slug
    }
}

/// Build a request from one sentence.
pub fn parse_ask(id: usize, text: &str, video_duration: f64) -> ClipRequest {
    let h = parse_hints(text, video_duration);
    ClipRequest {
        id,
        text: text.trim().to_string(),
        label: slugify(text, id),
        count: h.count.unwrap_or(1),
        min_s: h.min_s,
        max_s: h.max_s,
        window: h.window,
        tone: tone_of(text),
        no_cuts: None,
    }
}

// ---- brief file (a small TOML subset) -------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Val {
    Str(String),
    Num(f64),
    Bool(bool),
    Arr(Vec<Val>),
}

fn strip_comment(line: &str) -> &str {
    let mut in_str: Option<char> = None;
    for (i, c) in line.char_indices() {
        match (in_str, c) {
            (None, '"') | (None, '\'') => in_str = Some(c),
            (Some(q), c) if c == q => in_str = None,
            (None, '#') => return &line[..i],
            _ => {}
        }
    }
    line
}

fn split_top_level(s: &str) -> Vec<String> {
    let (mut out, mut cur, mut in_str) = (Vec::new(), String::new(), None::<char>);
    for c in s.chars() {
        match (in_str, c) {
            (None, '"') | (None, '\'') => {
                in_str = Some(c);
                cur.push(c)
            }
            (Some(q), c) if c == q => {
                in_str = None;
                cur.push(c)
            }
            (None, ',') => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

fn parse_val(raw: &str) -> Result<Val> {
    let s = raw.trim();
    if let Some(inner) = s.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        return Ok(Val::Arr(split_top_level(inner).iter().map(|p| parse_val(p)).collect::<Result<_>>()?));
    }
    if let Some(inner) = s.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
        return Ok(Val::Str(inner.replace("\\\"", "\"").replace("\\\\", "\\")));
    }
    if let Some(inner) = s.strip_prefix('\'').and_then(|r| r.strip_suffix('\'')) {
        return Ok(Val::Str(inner.to_string()));
    }
    match s {
        "true" => return Ok(Val::Bool(true)),
        "false" => return Ok(Val::Bool(false)),
        _ => {}
    }
    s.parse::<f64>().map(Val::Num).map_err(|_| anyhow::anyhow!("cannot read value `{s}` (use \"text\", a number, true/false or [a, b])"))
}

fn as_seconds(v: &Val) -> Result<f64> {
    match v {
        Val::Num(n) => Ok(*n),
        Val::Str(s) => parse_time(s).map_err(|e| anyhow::anyhow!(e)),
        _ => bail!("expected a time like \"10:00\" or a number of seconds"),
    }
}

/// Parse a brief: `[[clip]]` tables with `ask`, `count`, `duration`, `window`,
/// `cuts`, `label`, `tone`. Explicit fields override hints parsed from `ask`.
pub fn parse_brief(text: &str, first_id: usize, video_duration: f64) -> Result<Vec<ClipRequest>> {
    let mut tables: Vec<Vec<(usize, String, Val)>> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = strip_comment(line).trim();
        let lineno = n + 1;
        if line.is_empty() {
            continue;
        }
        if line == "[[clip]]" {
            tables.push(Vec::new());
            continue;
        }
        if line.starts_with('[') && !line.contains('=') {
            bail!("line {lineno}: only `[[clip]]` tables are supported, found `{line}`");
        }
        let (k, v) = line.split_once('=').with_context(|| format!("line {lineno}: expected `key = value`, found `{line}`"))?;
        let table = tables.last_mut().with_context(|| format!("line {lineno}: `{}` appears before the first [[clip]]", k.trim()))?;
        let val = parse_val(v).with_context(|| format!("line {lineno}: `{}`", k.trim()))?;
        table.push((lineno, k.trim().to_string(), val));
    }
    if tables.is_empty() {
        bail!("the brief has no [[clip]] entries");
    }

    let mut out = Vec::new();
    for (ti, table) in tables.iter().enumerate() {
        let ask = table
            .iter()
            .find(|(_, k, _)| k == "ask")
            .and_then(|(_, _, v)| if let Val::Str(s) = v { Some(s.clone()) } else { None })
            .with_context(|| format!("[[clip]] #{}: missing `ask = \"...\"`", ti + 1))?;
        let mut r = parse_ask(first_id + ti, &ask, video_duration);
        for (lineno, key, val) in table {
            let ctx = || format!("line {lineno}: `{key}`");
            match key.as_str() {
                "ask" => {}
                "count" => match val {
                    Val::Num(n) if n.fract() == 0.0 && *n >= 1.0 => r.count = *n as usize,
                    _ => bail!("{}: must be a whole number ≥ 1", ctx()),
                },
                "duration" => match val {
                    Val::Arr(a) if a.len() == 2 => {
                        r.min_s = Some(as_seconds(&a[0]).with_context(ctx)?);
                        r.max_s = Some(as_seconds(&a[1]).with_context(ctx)?);
                    }
                    Val::Num(n) => {
                        r.min_s = Some(n * 0.8);
                        r.max_s = Some(n * 1.25);
                    }
                    _ => bail!("{}: use [min, max] seconds or one number", ctx()),
                },
                "window" => match val {
                    Val::Arr(a) if a.len() == 2 => r.window = Some((as_seconds(&a[0]).with_context(ctx)?, as_seconds(&a[1]).with_context(ctx)?)),
                    _ => bail!("{}: use [\"10:00\", \"15:00\"]", ctx()),
                },
                "cuts" => match val {
                    Val::Bool(b) => r.no_cuts = if *b { None } else { Some(true) },
                    _ => bail!("{}: must be true or false", ctx()),
                },
                "label" => match val {
                    Val::Str(s) => r.label = slugify(s, r.id),
                    _ => bail!("{}: must be text", ctx()),
                },
                "tone" => match val {
                    Val::Str(s) => {
                        r.tone = match s.to_lowercase().as_str() {
                            "any" => Tone::Any,
                            "funny" => Tone::Funny,
                            "emotional" => Tone::Emotional,
                            "intense" => Tone::Intense,
                            other => bail!("{}: unknown tone `{other}` (any, funny, emotional, intense)", ctx()),
                        }
                    }
                    _ => bail!("{}: must be text", ctx()),
                },
                other => bail!("line {lineno}: unknown field `{other}` (ask, count, duration, window, cuts, label, tone)"),
            }
        }
        out.push(r);
    }
    Ok(out)
}

/// Check and normalise requests against the video; unique labels.
pub fn validate(reqs: &mut [ClipRequest], video_duration: f64) -> Result<()> {
    let mut seen: Vec<String> = Vec::new();
    for r in reqs.iter_mut() {
        let who = format!("request #{} (\"{}\")", r.id + 1, r.text);
        if r.text.is_empty() {
            bail!("request #{} is empty", r.id + 1);
        }
        if r.count == 0 || r.count > MAX_COUNT {
            bail!("{who}: count must be 1–{MAX_COUNT}");
        }
        if let (Some(a), Some(b)) = (r.min_s, r.max_s) {
            if a > b {
                bail!("{who}: minimum duration {a:.0}s is longer than maximum {b:.0}s");
            }
        }
        if let Some((a, b)) = r.window {
            if a >= video_duration {
                bail!("{who}: the window starts at {} but the video is only {} long", fmt_clock(a), fmt_clock(video_duration));
            }
            let b = b.min(video_duration);
            if b <= a {
                bail!("{who}: the window ends before it starts");
            }
            r.window = Some((a.max(0.0), b));
        }
        let mut label = r.label.clone();
        let mut n = 2;
        while seen.contains(&label) {
            label = format!("{}-{n}", r.label);
            n += 1;
        }
        r.label = label.clone();
        seen.push(label);
    }
    Ok(())
}

/// Build the requests from `--ask` / `--brief`. Empty when neither is given.
pub fn load(asks: &[String], brief: Option<&Path>, video_duration: f64) -> Result<Vec<ClipRequest>> {
    let mut reqs: Vec<ClipRequest> = Vec::new();
    if let Some(p) = brief {
        let text = std::fs::read_to_string(p).with_context(|| format!("reading brief {}", p.display()))?;
        reqs.extend(parse_brief(&text, 0, video_duration).with_context(|| format!("in brief {}", p.display()))?);
    }
    for a in asks {
        reqs.push(parse_ask(reqs.len(), a, video_duration));
    }
    validate(&mut reqs, video_duration)?;
    Ok(reqs)
}

/// Stable fingerprint of the requests, so a change invalidates cached clip stages.
pub fn fingerprint(reqs: &[ClipRequest]) -> String {
    if reqs.is_empty() {
        return "none".into();
    }
    let mut h: u64 = 0xcbf29ce484222325;
    for b in format!("{reqs:?}").bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

// ---- report ------------------------------------------------------------------

/// Write `clips_report.md` / `.json` next to the stages: for each request, the
/// clips found (time, relevance, why, transcript excerpt) or why there is none.
pub fn write_report(
    dir: &Path,
    reqs: &[ClipRequest],
    cands: &[ClipCandidate],
    best: &[f64],
    min_relevance: f64,
    transcript: &Transcript,
) -> Result<String> {
    let mut md = String::from("# Clip report\n\n");
    let mut json = Vec::new();
    for r in reqs {
        md.push_str(&format!("## {}\n\n", r.describe()));
        let mine: Vec<&ClipCandidate> = cands.iter().filter(|c| c.request_id == Some(r.id)).collect();
        if mine.is_empty() {
            let b = best.get(r.id).copied().unwrap_or(0.0);
            md.push_str(&format!(
                "**No match.** The closest moment rated {:.0}% relevant (needs ≥ {:.0}%). \
                 Text-based search cannot find purely visual moments — for those use `--from/--to`.\n\n",
                b * 100.0,
                min_relevance * 100.0
            ));
        }
        for (i, c) in mine.iter().enumerate() {
            let excerpt: String = transcript
                .words
                .iter()
                .filter(|w| w.start >= c.start && w.end <= c.end)
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let excerpt: String = excerpt.chars().take(220).collect();
            md.push_str(&format!(
                "{}. **{}–{}** ({:.0}s) · relevance {:.0}% · score {:.2}\n   - {}\n   - “{}…”\n",
                i + 1,
                fmt_clock(c.start),
                fmt_clock(c.end),
                c.end - c.start,
                c.relevance * 100.0,
                c.score,
                c.rationale,
                excerpt
            ));
            json.push(serde_json::json!({"request": r.id, "start": c.start, "end": c.end, "relevance": c.relevance, "score": c.score, "why": c.rationale}));
        }
        if !mine.is_empty() && mine.len() < r.count {
            md.push_str(&format!("\n_Asked for up to {}, found {} good match(es)._\n", r.count, mine.len()));
        }
        md.push('\n');
    }
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("clips_report.md"), &md)?;
    std::fs::write(dir.join("clips_report.json"), serde_json::to_vec_pretty(&serde_json::json!({"requests": reqs, "clips": json, "best_relevance": best}))?)?;
    Ok(md)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUR: f64 = 1200.0;

    fn p(text: &str) -> ClipRequest {
        parse_ask(0, text, DUR)
    }

    #[test]
    fn plain_ask_has_defaults() {
        let r = p("the funniest exchange about the DM mix-up");
        assert_eq!((r.count, r.min_s, r.max_s, r.window), (1, None, None, None));
        assert_eq!(r.tone, Tone::Funny);
        assert!(r.label.contains("funniest") && r.label.len() <= 32, "{}", r.label);
    }

    #[test]
    fn labels_do_not_contain_duration_words() {
        assert_eq!(p("how to cook dal, about 30 seconds").label, "cook-dal");
        assert_eq!(p("a one minute clip of the poster joke").label, "one-poster-joke");
    }

    #[test]
    fn counts_from_digits_and_words() {
        assert_eq!(p("2 clips where they argue").count, 2);
        assert_eq!(p("give me three moments about steroids").count, 3);
        assert_eq!(p("2 funny clips").count, 2);
        assert_eq!(p("top 4 highlights").count, 4);
        assert_eq!(p("a clip about the poster").count, 1);
        assert_eq!(p("a 30 second clip about the poster").count, 1, "30 second is a duration");
    }

    #[test]
    fn durations_in_several_phrasings() {
        let r = p("the poster joke, 30-45s");
        assert_eq!((r.min_s, r.max_s), (Some(30.0), Some(45.0)));
        let r = p("the poster joke between 30 to 45 seconds");
        assert_eq!((r.min_s, r.max_s), (Some(30.0), Some(45.0)));
        let r = p("something under 40 seconds");
        assert_eq!((r.min_s, r.max_s), (None, Some(40.0)));
        let r = p("at least 20s of the debate");
        assert_eq!((r.min_s, r.max_s), (Some(20.0), None));
        let r = p("about a minute on the DM story");
        assert_eq!((r.min_s, r.max_s), (Some(48.0), Some(75.0)));
        let r = p("1-2 minutes on the DM story");
        assert_eq!((r.min_s, r.max_s), (Some(60.0), Some(120.0)));
    }

    #[test]
    fn time_windows() {
        assert_eq!(p("the DM story after 10:00").window, Some((600.0, DUR)));
        assert_eq!(p("the DM story before 5:30").window, Some((0.0, 330.0)));
        assert_eq!(p("a joke between 5:00 and 12:30").window, Some((300.0, 750.0)));
        assert_eq!(p("a joke from 1:00 to 2:00").window, Some((60.0, 120.0)));
        assert_eq!(p("a joke in 10:00-15:00").window, Some((600.0, 900.0)));
        assert_eq!(p("a joke in the first 10 minutes").window, Some((0.0, 600.0)));
        assert_eq!(p("a joke in the last 5 minutes").window, Some((900.0, DUR)));
        assert_eq!(p("a joke after 3 minutes").window, Some((180.0, DUR)));
    }

    #[test]
    fn window_times_are_not_mistaken_for_durations() {
        let r = p("2 clips about the DM, 30-45s, after 10:00");
        assert_eq!((r.count, r.min_s, r.max_s, r.window), (2, Some(30.0), Some(45.0), Some((600.0, DUR))));
        let r = p("the first 10 minutes joke");
        assert_eq!((r.min_s, r.max_s), (None, None), "first 10 minutes is a window");
    }

    #[test]
    fn brief_parsing_and_overrides() {
        let toml = r#"
# two requests
[[clip]]
ask = "the funniest exchange about the DM mix-up, 20-30s"   # hint in text
count = 2
duration = [30, 45]          # explicit fields win over the text
window = ["10:00", "15:00"]
cuts = false

[[clip]]
ask = "Bhuvan explaining how he got started"
tone = "emotional"
label = "origin story"
"#;
        let r = parse_brief(toml, 0, DUR).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].count, r[0].min_s, r[0].max_s), (2, Some(30.0), Some(45.0)));
        assert_eq!(r[0].window, Some((600.0, 900.0)));
        assert_eq!(r[0].no_cuts, Some(true));
        assert_eq!(r[0].tone, Tone::Funny);
        assert_eq!((r[1].id, r[1].tone, r[1].label.as_str()), (1, Tone::Emotional, "origin-story"));
    }

    #[test]
    fn brief_errors_are_specific() {
        let e = |t: &str| format!("{:#}", parse_brief(t, 0, DUR).unwrap_err());
        assert!(e("[[clip]]\ncount = 2\n").contains("missing `ask"));
        assert!(e("ask = \"x\"\n").contains("before the first [[clip]]"));
        assert!(e("[[clip]]\nask = \"x\"\nbogus = 1\n").contains("unknown field `bogus`"));
        assert!(e("[[clip]]\nask = \"x\"\ncount = 0\n").contains("count"));
        assert!(e("[[clip]]\nask = \"x\"\nduration = [1]\n").contains("duration"));
        assert!(e("").contains("no [[clip]]"));
    }

    #[test]
    fn validation_catches_bad_requests_and_dedupes_labels() {
        let mut v = vec![p("poster joke"), p("poster joke")];
        validate(&mut v, DUR).unwrap();
        assert_ne!(v[0].label, v[1].label);

        let mut bad = vec![p("x, 50-30s")];
        assert!(validate(&mut bad, DUR).unwrap_err().to_string().contains("longer than maximum"));
        let mut late = vec![p("a joke after 30:00")];
        assert!(validate(&mut late, DUR).unwrap_err().to_string().contains("only"));
        let mut clamp = vec![p("a joke between 5:00 and 99:00")];
        validate(&mut clamp, DUR).unwrap();
        assert_eq!(clamp[0].window, Some((300.0, DUR)));
    }

    #[test]
    fn fingerprint_changes_with_requests() {
        let a = vec![p("poster joke")];
        let b = vec![p("poster joke, 2 clips")];
        assert_eq!(fingerprint(&a), fingerprint(&a.clone()));
        assert_ne!(fingerprint(&a), fingerprint(&b));
        assert_eq!(fingerprint(&[]), "none");
    }

    #[test]
    fn report_lists_matches_and_misses() {
        let reqs = vec![parse_ask(0, "poster joke", DUR), parse_ask(1, "a scene where somebody falls", DUR)];
        let cands = vec![ClipCandidate { start: 65.0, end: 95.0, score: 0.7, relevance: 0.9, rationale: "they joke about the poster".into(), request_id: Some(0), ..Default::default() }];
        let t = Transcript { words: vec![crate::types::Word { text: "poster".into(), start: 70.0, end: 70.5 }] };
        let dir = std::env::temp_dir().join("aiclipper_report_test");
        let md = write_report(&dir, &reqs, &cands, &[0.9, 0.2], 0.6, &t).unwrap();
        assert!(md.contains("1:05–1:35") && md.contains("90%") && md.contains("poster"));
        assert!(md.contains("No match") && md.contains("20%") && md.contains("--from/--to"));
        assert!(dir.join("clips_report.json").exists());
    }
}
