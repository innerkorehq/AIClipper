//! CLI argument parsing (clap derive).

use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "aiclipper",
    about = "Fully local, offline short-form video clipping pipeline",
    long_about = None,
    version
)]
pub struct Cli {
    /// Source video URL (yt-dlp supported) — mutually exclusive with --file.
    #[arg(long, conflicts_with = "file")]
    pub url: Option<String>,

    /// Path to a local source video — mutually exclusive with --url.
    #[arg(long, conflicts_with = "url")]
    pub file: Option<PathBuf>,

    /// Working directory for artifacts and media.
    #[arg(long, default_value = "work")]
    pub workdir: PathBuf,

    /// Export platforms: a comma-separated list (instagram_reels,tiktok,shorts)
    /// — one folder each under exports/ — or a path to a JSON profile table.
    #[arg(long)]
    pub profiles: Option<PathBuf>,

    /// Export profile name (defaults to instagram_reels semantics).
    #[arg(long, default_value = "instagram_reels")]
    pub profile: String,

    /// Recompute every stage, ignoring existing artifacts.
    #[arg(long)]
    pub force: bool,

    /// Stop after manifest generation — do not render or export.
    #[arg(long)]
    pub no_render: bool,

    /// Do not start the Python sidecar (talk to an already-running one).
    #[arg(long)]
    pub no_sidecar: bool,

    /// Skip active-speaker resolution (assume a single speaker).
    #[arg(long)]
    pub single_speaker: bool,

    /// Clip start (seconds, MM:SS or HH:MM:SS). With --from/--to the AI clip
    /// picker is skipped and exactly this range is clipped.
    #[arg(long = "from", value_name = "TIME", value_parser = parse_time)]
    pub range_from: Option<f64>,

    /// Clip end (seconds, MM:SS or HH:MM:SS). Defaults to the end of the source
    /// when only --from is given.
    #[arg(long = "to", value_name = "TIME", value_parser = parse_time)]
    pub range_to: Option<f64>,

    /// Describe a clip you want, in plain words (repeat for several clips), e.g.
    /// `--ask "the funniest exchange about the DM mix-up, 30-45s"`. Count, duration
    /// and time window can be written in the sentence ("2 clips", "30-45s", "after 10:00").
    /// Replaces automatic clip picking. Not combinable with --from/--to.
    #[arg(long = "ask", value_name = "TEXT")]
    pub ask: Vec<String>,

    /// A TOML file of clip requests (`[[clip]]` tables with ask, count, duration,
    /// window, cuts) — see README. May be combined with --ask.
    #[arg(long, value_name = "FILE")]
    pub brief: Option<PathBuf>,

    /// Subtitle file (.srt/.vtt) to use as the transcript instead of ASR.
    /// By default subtitles shipped with the video are found automatically.
    #[arg(long, value_name = "FILE")]
    pub subtitles: Option<PathBuf>,

    /// Ignore any subtitles and transcribe the audio.
    #[arg(long)]
    pub no_subtitles: bool,

    /// How each shot is framed for vertical: `auto` (the AI picks a framing
    /// algorithm per shot, default), `rules` (deterministic, no model), or the
    /// name of one algorithm to force: face_follow, speaker_follow, split_two,
    /// group_fit, punch_in, locked, saliency_follow, motion_follow, wide_blur.
    #[arg(long, value_name = "MODE", default_value = "auto", value_parser = parse_framing)]
    pub framing: String,

    /// Keep the clip continuous: do not cut filler, silences or tangents.
    /// Use for action scenes / footage without speech.
    #[arg(long)]
    pub no_cuts: bool,
}

fn parse_framing(s: &str) -> Result<String, String> {
    crate::stages::chooser::FramingMode::parse(s).map(|m| m.label())
}

/// Parse `90`, `90.5`, `1:30`, `01:02:03.5` into seconds.
pub fn parse_time(s: &str) -> Result<f64, String> {
    let parts: Vec<&str> = s.trim().split(':').collect();
    if parts.is_empty() || parts.len() > 3 {
        return Err(format!("invalid time `{s}` (use seconds, MM:SS or HH:MM:SS)"));
    }
    let mut total = 0.0f64;
    for part in &parts {
        let v: f64 = part.parse().map_err(|_| format!("invalid time `{s}`"))?;
        if v < 0.0 {
            return Err(format!("invalid time `{s}`: negative"));
        }
        total = total * 60.0 + v;
    }
    Ok(total)
}

impl Cli {
    /// The input source — exactly one of --url / --file is set.
    pub fn input(&self) -> String {
        self.url
            .clone()
            .or_else(|| self.file.as_ref().map(|p| p.display().to_string()))
            .expect("either --url or --file is required")
    }
}

#[cfg(test)]
mod tests {
    use super::parse_time;

    #[test]
    fn parses_time_formats() {
        assert_eq!(parse_time("90").unwrap(), 90.0);
        assert_eq!(parse_time("1:30").unwrap(), 90.0);
        assert_eq!(parse_time("01:02:03.5").unwrap(), 3723.5);
        assert!(parse_time("abc").is_err());
        assert!(parse_time("1:2:3:4").is_err());
        assert!(parse_time("-5").is_err());
    }
}
