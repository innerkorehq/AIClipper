import React, {useMemo} from "react";
import {AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig} from "remotion";
import type {RenderManifest, Word} from "../lib/manifest";
import {shotLayout, type ShotLayout} from "../ShotLayer";

// Karaoke word-reveal captions. Words are already time-remapped.
//
// Pages: up to MAX_WORDS words, closing on sentence end / long gaps. Each page
// renders in a <Sequence> from its first word until the next page's first word.
// The currently-spoken word is highlighted; completed words stay lit; upcoming
// words are dimmed — a calm, readable social-clip caption style.

const MAX_WORDS_PER_PAGE = 9;
const MAX_CHARS_PER_PAGE = 44;
const SENTENCE_END = /[.!?。！？]$/;
const HOLD_SECONDS = 0.4;

export type CaptionPage = {
  start: number;
  end: number;
  words: Word[];
};

// Two-actor shots keep the caption to a single line, so pages are shorter.
const SPLIT_MAX_WORDS = 4;
const SPLIT_MAX_CHARS = 20;

export function buildCaptionPages(
  words: Word[],
  maxWords = MAX_WORDS_PER_PAGE,
  maxChars = MAX_CHARS_PER_PAGE,
): CaptionPage[] {
  const pages: CaptionPage[] = [];
  let current: Word[] = [];
  let chars = 0;

  const flush = () => {
    if (current.length === 0) {
      return;
    }
    const start = current[0].start;
    const end = current[current.length - 1].end + HOLD_SECONDS;
    // don't overlap the next page's first word
    pages.push({start, end: Math.max(end, start + HOLD_SECONDS), words: current});
    current = [];
    chars = 0;
  };

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const next = words[i + 1] ?? null;
    const gap = next ? next.start - w.end : 0;

    current.push(w);
    chars += w.text.length;

    const sentenceEnd = SENTENCE_END.test(w.text) && gap > 0.28;
    const longGap = gap > 0.7;
    const maxed = current.length >= maxWords || chars >= maxChars;

    if (maxed || sentenceEnd || longGap) {
      flush();
    }
  }
  flush();
  return pages;
}

const FONT_STACK =
  "-apple-system, 'SF Pro Display', 'Helvetica Neue', 'Segoe UI', Roboto, Arial, sans-serif";

/** Layout of the shot playing at output frame `f` (defaults to single). */
function layoutAt(manifest: RenderManifest, f: number): ShotLayout {
  for (const seg of manifest.shots) {
    if (f >= seg.src_start_frame && f < seg.src_end_frame) {
      return shotLayout(seg);
    }
  }
  return "single";
}

export const CaptionStack: React.FC<{words: Word[]; fps: number; manifest: RenderManifest}> = ({
  words,
  fps,
  manifest,
}) => {
  const pages = useMemo(() => {
    // Group consecutive words by the layout of the shot they start in, then
    // page each run with that layout's limits.
    const out: CaptionPage[] = [];
    let run: Word[] = [];
    let runSplit = false;
    const flushRun = () => {
      if (run.length) {
        out.push(
          ...(runSplit
            ? buildCaptionPages(run, SPLIT_MAX_WORDS, SPLIT_MAX_CHARS)
            : buildCaptionPages(run)),
        );
      }
      run = [];
    };
    for (const w of words) {
      const isSplit = layoutAt(manifest, Math.round(w.start * fps)) === "split";
      if (run.length && isSplit !== runSplit) {
        flushRun();
      }
      runSplit = isSplit;
      run.push(w);
    }
    flushRun();
    return out;
  }, [words, manifest, fps]);
  const {height} = useVideoConfig();

  return (
    <AbsoluteFill style={{pointerEvents: "none"}}>
      {/* Bottom scrim only for single-actor shots (full-bleed footage). */}
      {manifest.shots
        .filter((seg) => shotLayout(seg) === "single")
        .map((seg, i) => (
          <Sequence
            key={`scrim-${seg.src_start_frame}-${i}`}
            from={seg.src_start_frame}
            durationInFrames={Math.max(1, seg.src_end_frame - seg.src_start_frame)}
            layout="none"
          >
            <div
              style={{
                position: "absolute",
                left: 0,
                right: 0,
                bottom: 0,
                height: height * 0.34,
                background: "linear-gradient(to top, rgba(0,0,0,0.62), rgba(0,0,0,0) 85%)",
              }}
            />
          </Sequence>
        ))}
      {pages.map((page, i) => {
        const next = pages[i + 1] ?? null;
        const startFrame = Math.round(page.start * fps);
        const nextStartFrame = next ? Math.round(next.start * fps) : null;
        // page holds on screen (faded) until the next page's first word
        const endFrame = nextStartFrame ?? Math.round(page.end * fps);
        const durationInFrames = Math.max(1, Math.round(endFrame) - startFrame);
        return (
          <Sequence
            key={`${startFrame}-${page.words[0].text}`}
            from={startFrame}
            durationInFrames={durationInFrames}
            name="CaptionPage"
          >
            <CaptionPageLayer page={page} fps={fps} isLast={next === null} manifest={manifest} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

const ACCENT = "#ffc50a";

const CaptionPageLayer: React.FC<{
  page: CaptionPage;
  fps: number;
  isLast: boolean;
  manifest: RenderManifest;
}> = ({page, fps, isLast, manifest}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const absoluteMs = (page.start + frame / fps) * 1000;

  // Vertical anchor by shot layout: below (single), middle seam (split),
  // lower gap beneath the centred full scene (wide).
  const layout = layoutAt(manifest, Math.round(page.start * fps) + frame);
  let anchor: React.CSSProperties = {bottom: height * 0.075};
  if (layout === "split") {
    anchor = {top: height / 2, transform: "translateY(-50%)"};
  } else if (layout === "wide") {
    const sceneH = Math.min(height, (width * manifest.source_height) / manifest.source_width);
    anchor = {top: (3 * height + sceneH) / 4, transform: "translateY(-50%)"};
  }

  // quick fade-in (80ms); the final page fades out over its tail hold
  const opacity = Math.min(1, frame / (fps * 0.08)) * (isLast ? Math.max(0, 1 - frame / (fps * HOLD_SECONDS * 9)) : 1);

  const maxWidth = width * 0.88;
  let fontSize = Math.min(width * 0.072, 84);

  if (layout === "split") {
    // single line: shrink to fit the available width (≈0.55em per character)
    const chars = page.words.reduce((n, w) => n + w.text.length + 1, 0);
    fontSize = Math.min(fontSize, (maxWidth - 56) / Math.max(1, chars * 0.55));
  }

  const tokens = page.words.map((w) => {
    const fromMs = w.start * 1000;
    const toMs = w.end * 1000;
    const active = absoluteMs >= fromMs && absoluteMs < toMs;
    const done = absoluteMs >= toMs;
    const color = active ? ACCENT : done ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.45)";
    return {text: ` ${w.text}`, color, active};
  });

  return (
    <div
      style={{
        position: "absolute",
        left: (width - maxWidth) / 2,
        width: maxWidth,
        ...anchor,
        display: "flex",
        justifyContent: "center",
        opacity,
      }}
    >
      <div
        style={{
          padding: "16px 28px",
          borderRadius: 18,
          backgroundColor: "rgba(0,0,0,0.42)",
          boxShadow: "0 2px 18px rgba(0,0,0,0.4)",
          backdropFilter: "blur(10px)",
          WebkitBackdropFilter: "blur(10px)",
          maxWidth: maxWidth,
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontSize,
            fontWeight: 800,
            lineHeight: 1.16,
            color: "white",
            whiteSpace: layout === "split" ? "nowrap" : "pre-wrap",
            wordBreak: "break-word",
            fontFamily: FONT_STACK,
            textShadow: "0 2px 6px rgba(0,0,0,0.5)",
          }}
        >
          {tokens.map((t, i) => (
            <span
              key={`${t.text}-${i}`}
              style={{
                color: t.color,
                ...(t.active
                  ? {
                      textShadow: `0 0 22px ${ACCENT}55`,
                    }
                  : {}),
              }}
            >
              {t.text}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};