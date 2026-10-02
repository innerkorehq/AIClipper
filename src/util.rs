//! Small shared helpers: subprocess running, media probing, audio extraction,
//! LLM-session logging (PRD §9 auditability).

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

pub struct MediaInfo {
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub audio: bool,
}

/// Resolve an `ffprobe` binary path given the configured `ffmpeg` path/name.
/// Accepts either a bare name on PATH or a full path whose sibling is `ffprobe`.
pub fn ffprobe_path(ffmpeg: &str) -> String {
    if ffmpeg.contains('/') {
        let p = Path::new(ffmpeg);
        let file: String = p.file_name().and_then(|s| s.to_str()).unwrap_or("ffmpeg").to_string();
        // "ffmpeg" (or "ffmpeg-9", ...) → "ffprobe" (or "ffprobe-9", ...) sibling.
        let probe = if let Some(suffix) = file.strip_prefix("ffmpeg") {
            format!("ffprobe{suffix}")
        } else {
            "ffprobe".to_string()
        };
        return p.with_file_name(probe).to_string_lossy().into_owned();
    }
    "ffprobe".to_string()
}

/// Probe duration/resolution/fps with ffprobe (JSON output).
pub async fn probe_media(ffmpeg_bin: &str, path: &Path) -> Result<MediaInfo> {
    let probe = ffprobe_path(ffmpeg_bin);
    let out = Command::new(&probe)
        .args([
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(path.as_os_str())
        .output()
        .await
        .with_context(|| format!(
            "running `{probe}` on {}. Is ffmpeg installed and on PATH?",
            path.display()
        ))?;
    if !out.status.success() {
        anyhow::bail!(
            "ffprobe failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)
        .with_context(|| format!("parsing ffprobe output for {}", path.display()))?;
    let duration = v["format"]["duration"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| v["format"]["duration"].as_f64())
        .unwrap_or(0.0);
    let mut width = 0;
    let mut height = 0;
    let mut fps = 0.0;
    let mut audio = false;
    for s in v["streams"].as_array().cloned().unwrap_or_default() {
        match s["codec_type"].as_str() {
            Some("video") => {
                width = s["width"].as_u64().unwrap_or(0) as u32;
                height = s["height"].as_u64().unwrap_or(0) as u32;
                fps = s["r_frame_rate"]
                    .as_str()
                    .and_then(parse_rate)
                    .or_else(|| s["avg_frame_rate"].as_str().and_then(parse_rate))
                    .unwrap_or(0.0);
            }
            Some("audio") => audio = true,
            _ => {}
        }
    }
    Ok(MediaInfo { duration, width, height, fps, audio })
}

fn parse_rate(s: &str) -> Option<f64> {
    let (n, d) = s.split_once('/')?;
    let n: f64 = n.parse().ok()?;
    let d: f64 = d.parse().ok()?;
    if d == 0.0 { None } else { Some(n / d) }
}

/// Run a subprocess to completion, capturing stdout; fail with stderr on non-zero exit.
pub async fn run_ok(cmd: &mut Command, what: &str) -> Result<Vec<u8>> {
    let out = cmd.output().await.with_context(|| format!("spawning {what}"))?;
    if !out.status.success() {
        anyhow::bail!(
            "{what} exited with {}:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(out.stdout)
}

/// Stream subprocess output lines to a spawned task while waiting for exit code.
pub async fn run_checked(cmd: &mut Command, what: &str) -> Result<()> {
    let mut child = cmd
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {what}"))?;
    let stderr = child.stderr.take().expect("stderr piped");
    let name = what.to_string();
    tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let mut lines = tokio::io::BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!(target: "subprocess", "{name}: {line}");
        }
    });
    let status = child.wait().await.context("waiting for subprocess")?;
    if !status.success() {
        anyhow::bail!("{what} failed with {status}");
    }
    Ok(())
}

/// Write one request+response record to the LLM audit log (JSONL). Called for
/// every Gemma interaction so a run can be audited or replayed without the model.
pub fn log_llm_call(path: &Path, kind: &str, request: &serde_json::Value, response: &serde_json::Value) {
    let rec = serde_json::json!({
        "ts": chrono_now(),
        "kind": kind,
        "request": request,
        "response": response,
    });
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("open llm log");
        use std::io::Write;
        let _ = writeln!(f, "{rec}");
    }
}

fn chrono_now() -> String {
    // Avoid pulling in chrono just for a timestamp.
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
        .to_string()
}