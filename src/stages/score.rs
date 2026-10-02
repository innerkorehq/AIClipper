//! Stage 5.3 — Clip scoring via Gemma (MLX, through the sidecar).
//!
//! The model doesn't output timestamps reliably; it returns word-index-aligned
//! clip spans. We post the word-level transcript and let the sidecar resolve
//! spans from word offsets. Deterministic dedupe/ranking happens here.

use crate::config::Config;
use crate::sidecar::{ScoreClipsResp, SidecarClient};
use crate::types::{ClipCandidate, Transcript, Word};
use anyhow::{Context, Result};

/// Score the full transcript and return the top `n` non-overlapping candidates.
pub async fn score(cfg: &Config, client: &SidecarClient, transcript: &Transcript) -> Result<Vec<ClipCandidate>> {
    let resp: ScoreClipsResp = client
        .score_clips(&transcript.words, cfg)
        .await
        .context("Gemma clip scoring")?;

    // Sanity: clip the model to actual transcript bounds, keep minimum duration.
    let last_end = transcript.words.last().map(|w| w.end).unwrap_or(0.0);
    let min_bridge_s = 1.0;
    let mut candidates: Vec<ClipCandidate> = resp
        .candidates
        .into_iter()
        .filter(|c| c.score > 0.0 && c.end > c.start)
        .map(|mut c| {
            if c.end > last_end + 0.5 {
                c.end = last_end;
            }
            c.start = c.start.max(0.0);
            c.end = c.end.max(c.start + 1.0);
            // Resolve word indices into transcript time.
            c.word_start = transcript.words.iter().position(|w| w.end >= c.start);
            c.word_end = transcript.words.iter().rposition(|w| w.start <= c.end).map(|i| i + 1);
            // Small, quantized models (Gemma-4-MLX 4-bit) often choose spans
            // shorter than the minimum clip. Extend the end forward, word by
            // word, until it meets min_clip_duration — but never jump a >1s
            // silence gap (which usually separates topics).
            let mut we = c.word_end.unwrap_or(0);
            while we > 0 && we < transcript.words.len() {
                let dur = c.end - c.start;
                if dur >= cfg.min_clip_duration {
                    break;
                }
                let next = &transcript.words[we];
                let prev = &transcript.words[we - 1];
                if (next.start - prev.end) > min_bridge_s {
                    break;
                }
                we += 1;
                c.end = c.end.max(next.end);
                if c.end - c.start > cfg.max_clip_duration {
                    c.end = c.start + cfg.max_clip_duration;
                    break;
                }
            }
            c.word_end = Some(we);
            c
        })
        .collect();

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let within = |c: &ClipCandidate| {
        (c.end - c.start) >= cfg.min_clip_duration && (c.end - c.start) <= cfg.max_clip_duration
    };
    let (ok, bad): (Vec<_>, Vec<_>) = candidates.into_iter().partition(within);
    if !bad.is_empty() {
        tracing::warn!("dropped {} candidates outside [{:.0}s, {:.0}s]", bad.len(), cfg.min_clip_duration, cfg.max_clip_duration);
    }

    // Greedy non-overlap, keeping the highest-scoring first.
    let mut kept: Vec<ClipCandidate> = Vec::new();
    for cand in ok {
        let overlaps = kept.iter().any(|k| {
            cand.start < k.end + 0.25 && k.start < cand.end + 0.25
        });
        if !overlaps {
            kept.push(cand);
        }
        if kept.len() >= cfg.top_n {
            break;
        }
    }
    kept.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    tracing::info!("{} clip candidates selected", kept.len());
    Ok(kept)
}

// Keep Word import used for potential future needs; silence warning.
#[allow(dead_code)]
fn _word_span(words: &[Word], start: f64, end: f64) -> (usize, usize) {
    let s = words.iter().position(|w| w.end >= start).unwrap_or(0);
    let e = words.iter().rposition(|w| w.start <= end).map(|i| i + 1).unwrap_or(words.len());
    (s, e)
}