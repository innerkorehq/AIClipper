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

    /// Path to a JSON export-profile table (see `ExportProfile`).
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