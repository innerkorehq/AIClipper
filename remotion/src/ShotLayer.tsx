import React, {useMemo} from "react";
import {staticFile, Video, useCurrentFrame, useVideoConfig} from "remotion";
import type {BBoxKeyframe, RenderManifest, ShotSegment, ShotStrategy} from "./lib/manifest";

/** Volume fades (in/out over the first/last ~2 frames) to soften audio joins. */
const EDGE_FADE_FRAMES = 2;

function interpolateBBox(
  keyframes: BBoxKeyframe[],
  segStart: number,
  targetFrame: number,
): BBoxKeyframe {
  if (keyframes.length === 0) {
    return {frame: targetFrame, x: 0.18, y: 0.15, w: 0.64, h: 0.7};
  }
  if (keyframes.length === 1) {
    const k = keyframes[0];
    return {frame: targetFrame, x: k.x, y: k.y, w: k.w, h: k.h};
  }
  const frame = targetFrame - segStart;

  // clamp before first / after last
  if (frame <= keyframes[0].frame - segStart) {
    const k = keyframes[0];
    return {frame: targetFrame, x: k.x, y: k.y, w: k.w, h: k.h};
  }
  const last = keyframes[keyframes.length - 1];
  if (frame >= last.frame - segStart) {
    return {frame: targetFrame, x: last.x, y: last.y, w: last.w, h: last.h};
  }

  for (let i = 0; i < keyframes.length - 1; i++) {
    const a = keyframes[i];
    const b = keyframes[i + 1];
    const af = a.frame - segStart;
    const bf = b.frame - segStart;
    if (frame >= af && frame <= bf) {
      const t = bf === af ? 0 : (frame - af) / (bf - af);
      return {
        frame: targetFrame,
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        w: a.w + (b.w - a.w) * t,
        h: a.h + (b.h - a.h) * t,
      };
    }
  }
  const k = keyframes[0];
  return {frame: targetFrame, x: k.x, y: k.y, w: k.w, h: k.h};
}

const clamp01 = (v: number): number => Math.min(Math.max(v, 0), 1);

/**
 * Widen the tracked face bbox so the *whole person* stays inside the frame:
 * headroom above the face, and the figure below it down to the frame edge.
 */
function expandSubject(bbox: BBoxKeyframe): BBoxKeyframe {
  const cx = bbox.x + bbox.w / 2;
  const top = clamp01(bbox.y - 0.55 * bbox.h);
  const w = Math.min(2.2 * bbox.w, 1);
  const x = clamp01(cx - w / 2);
  return {
    frame: bbox.frame,
    x,
    y: top,
    w: Math.min(w, 1 - x),
    h: Math.min(1 - top, 1),
  };
}

/**
 * Reframe the source so the subject box fits ENTIRELY within the target rect
 * (contain, not cover) — the full subject is always visible.
 */
function containInRect(
  box: BBoxKeyframe,
  srcW: number,
  srcH: number,
  rectX: number,
  rectY: number,
  rectW: number,
  rectH: number,
): React.CSSProperties {
  const bw = Math.max(0.01, box.w) * srcW;
  const bh = Math.max(0.01, box.h) * srcH;
  const scale = Math.min(rectW / bw, rectH / bh) * 0.995; // hairline margin
  const cx = (box.x + box.w / 2) * srcW;
  const cy = (box.y + box.h / 2) * srcH;
  return {
    position: "absolute",
    left: 0,
    top: 0,
    width: srcW,
    height: srcH,
    transform: `translate3d(${rectX + rectW / 2 - cx * scale}px, ${rectY + rectH / 2 - cy * scale}px, 0) scale(${scale})`,
    transformOrigin: "0 0",
  };
}

/**
 * Cover the target region with the subject (fill + slight overzoom) — used
 * when the subject's aspect is already tall enough to cleanly fill a vertical
 * canvas.
 */
function coverCrop(
  box: BBoxKeyframe,
  srcW: number,
  srcH: number,
  rectX: number,
  rectY: number,
  rectW: number,
  rectH: number,
): React.CSSProperties {
  const bw = Math.max(0.01, box.w) * srcW;
  const bh = Math.max(0.01, box.h) * srcH;
  const scale = Math.max(rectW / bw, rectH / bh) * 1.012; // slight overzoom
  const cx = (box.x + box.w / 2) * srcW;
  const cy = (box.y + box.h / 2) * srcH;
  return {
    position: "absolute",
    left: 0,
    top: 0,
    width: srcW,
    height: srcH,
    transform: `translate3d(${rectX + rectW / 2 - cx * scale}px, ${rectY + rectH / 2 - cy * scale}px, 0) scale(${scale})`,
    transformOrigin: "0 0",
  };
}

export type ShotLayout = "single" | "split" | "wide";

/**
 * single: one actor, full-bleed max crop, captions below.
 * split:  two actors stacked half/half, captions in the middle.
 * wide:   action / fallback — whole scene centred, captions in the gaps.
 */
export function shotLayout(seg: ShotSegment): ShotLayout {
  if (seg.strategy !== "face_crop") {
    return "wide";
  }
  return seg.split && seg.second_keyframes && seg.second_keyframes.length > 0 ? "split" : "single";
}

/**
 * Largest crop window of the canvas aspect that fits inside the source frame,
 * positioned over the subject (clamped to the frame edges). Returns the window
 * as a normalized box so it can be fed to coverCrop.
 */
function maxWindowBox(
  subject: BBoxKeyframe,
  srcW: number,
  srcH: number,
  canvasW: number,
  canvasH: number,
): BBoxKeyframe {
  const target = canvasW / canvasH;
  let winW = srcH * target; // full-height window
  let winH = srcH;
  if (winW > srcW) {
    winW = srcW; // source narrower than target: full-width window
    winH = srcW / target;
  }
  const nw = winW / srcW;
  const nh = winH / srcH;
  const cx = subject.x + subject.w / 2;
  const cy = subject.y + subject.h / 2;
  return {
    frame: subject.frame,
    x: Math.min(Math.max(cx - nw / 2, 0), 1 - nw),
    y: Math.min(Math.max(cy - nh / 2, 0), 1 - nh),
    w: nw,
    h: nh,
  };
}

type Actor = {face: BBoxKeyframe; subject: BBoxKeyframe};

/**
 * Crop windows for the two halves of a two-actor layout, solved TOGETHER so that
 *   1. each window fully contains its own actor,
 *   2. neither window shows any part of the other actor, and
 *   3. the two windows never overlap each other (nobody appears in both halves).
 * Other bystanders may appear in either window.
 *
 * Both windows share one scale (largest first, shrinking until feasible) and are
 * placed as close to centred on their actor as the constraints allow. Actors are
 * ordered by screen position, so the pair is consistent even if they cross. If
 * the actors overlap so much that no valid pair exists, the frame is split at
 * the midpoint between them (windows touch but never overlap).
 */
export function splitWindows(
  top: Actor,
  bottom: Actor,
  srcW: number,
  srcH: number,
  canvasW: number,
  canvasH: number,
): {top: BBoxKeyframe; bottom: BBoxKeyframe} {
  const target = canvasW / canvasH;
  const w0 = Math.min(srcW, srcH * target);
  const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
  const m = srcW * 0.008; // margin at a cut
  const g = srcW * 0.004; // minimum gap between the two windows

  // "core" = the face widened a little (hair/shoulders), in source pixels
  // (clamped inside the frame: an actor at the edge can't extend past it)
  const coreL = (f: BBoxKeyframe) => clamp((f.x - 0.3 * f.w) * srcW, m, srcW - m);
  const coreR = (f: BBoxKeyframe) => clamp((f.x + 1.3 * f.w) * srcW, m, srcW - m);
  const topIsLeft = coreL(top.face) + coreR(top.face) <= coreL(bottom.face) + coreR(bottom.face);
  const left = topIsLeft ? top : bottom;
  const right = topIsLeft ? bottom : top;

  const aL = coreL(left.face);
  const aR = coreR(left.face);
  const bL = coreL(right.face);
  const bR = coreR(right.face);
  const cxLeft = (left.subject.x + left.subject.w / 2) * srcW;
  const cxRight = (right.subject.x + right.subject.w / 2) * srcW;

  const make = (L: number, w: number, who: Actor): BBoxKeyframe => {
    const h = w / target;
    const cy = (who.subject.y + who.subject.h / 2) * srcH;
    const t = clamp(cy - h / 2, 0, srcH - h);
    return {frame: who.subject.frame, x: L / srcW, y: t / srcH, w: w / srcW, h: h / srcH};
  };
  const result = (l1: number, l2: number, w: number) => {
    const lw = make(l1, w, left);
    const rw = make(l2, w, right);
    return topIsLeft ? {top: lw, bottom: rw} : {top: rw, bottom: lw};
  };

  for (let sc = 1; sc >= 0.15; sc -= 0.02) {
    const w = w0 * sc;
    const lo1 = Math.max(0, aR + m - w);
    const hi1 = Math.min(srcW - w, aL - m, bL - m - w);
    const lo2 = Math.max(0, bR + m - w, aR + m);
    const hi2 = Math.min(srcW - w, bL - m);
    if (lo1 > hi1 || lo2 > hi2) {
      continue;
    }
    let l1 = clamp(cxLeft - w / 2, lo1, hi1);
    let l2 = clamp(cxRight - w / 2, lo2, hi2);
    const need = w + g;
    if (l2 - l1 < need) {
      // push the windows apart, symmetrically first, then use any slack left
      const d = need - (l2 - l1);
      l1 = clamp(l1 - d / 2, lo1, hi1);
      l2 = clamp(l2 + d / 2, lo2, hi2);
      let rem = need - (l2 - l1);
      if (rem > 0) {
        const n1 = clamp(l1 - rem, lo1, hi1);
        rem -= l1 - n1;
        l1 = n1;
      }
      if (rem > 0) {
        l2 = clamp(l2 + rem, lo2, hi2);
      }
    }
    if (l2 - l1 >= need - 0.5) {
      return result(l1, l2, w);
    }
  }

  // No equal-size pair fits (actors crowded together or against an edge): split
  // the frame at the gap between them and size each window on its own. The
  // windows still never overlap; each actor is contained as fully as room allows.
  const X = aR < bL ? (aR + bL) / 2 : ((aL + aR) / 2 + (bL + bR) / 2) / 2;
  const roomL = Math.min(w0, X);
  const roomR = Math.min(w0, srcW - X);
  const base = Math.min(roomL, roomR);
  const needL = aR - aL + 2 * m;
  const needR = bR - bL + 2 * m;
  // (never wider than the room on its side of X — that is what keeps them disjoint)
  const wl = Math.min(roomL, Math.max(1, clamp(base, Math.min(needL, roomL), roomL)));
  const wr = Math.min(roomR, Math.max(1, clamp(base, Math.min(needR, roomR), roomR)));
  const l1 = clamp(cxLeft - wl / 2, 0, Math.max(0, X - wl));
  const l2 = clamp(cxRight - wr / 2, X, Math.max(X, srcW - wr));
  const lw = make(l1, wl, left);
  const rw = make(l2, wr, right);
  return topIsLeft ? {top: lw, bottom: rw} : {top: rw, bottom: lw};
}

const Half: React.FC<{top: number; height: number; children: React.ReactNode}> = ({
  top,
  height,
  children,
}) => (
  <div style={{position: "absolute", left: 0, right: 0, top, height, overflow: "hidden"}}>
    {children}
  </div>
);

export const ShotLayer: React.FC<{
  seg: ShotSegment;
  manifest: RenderManifest;
}> = ({seg, manifest}) => {
  const frame = useCurrentFrame();
  const {durationInFrames, width, height} = useVideoConfig();

  // The Rust render stage stages the rough cut into `public/_sources/` and
  // stores the public-relative path in `source_abs`; Chromium can only load
  // assets served by Remotion.
  const src = staticFile(manifest.source_abs || manifest.source);
  const endAt = seg.video_trim_frame + Math.max(1, seg.src_end_frame - seg.src_start_frame);

  const bbox = useMemo(
    () => interpolateBBox(seg.keyframes, seg.start_frame, frame + seg.start_frame),
    [seg, frame],
  );

  const secondBBox = useMemo(
    () =>
      seg.split && seg.second_keyframes && seg.second_keyframes.length > 0
        ? interpolateBBox(seg.second_keyframes, seg.start_frame, frame + seg.start_frame)
        : null,
    [seg, frame],
  );

  // For tracked subjects, expand the face box into a full-person region.
  const subject = useMemo(() => {
    if (shotLayout(seg) === "wide") {
      return null;
    }
    return expandSubject(bbox);
  }, [bbox, seg]);
  const secondSubject = useMemo(
    () => (secondBBox ? expandSubject(secondBBox) : null),
    [secondBBox],
  );

  // volume = 1, fading the first/last EDGE_FADE_FRAMES frames
  const edgeFade = useMemo(() => {
    return Math.min(
      frame / Math.max(1, EDGE_FADE_FRAMES),
      (durationInFrames - 1 - frame) / Math.max(1, EDGE_FADE_FRAMES),
      1,
    );
  }, [frame, durationInFrames]);

  const sourceW = manifest.source_width;
  const sourceH = manifest.source_height;

  // Prefer a clean full-bleed vertical crop over a blur fill whenever the
  // tracked subject is already tall enough to fill the 9:16 canvas (its pixel
  // aspect >= canvas aspect). Otherwise fall back to blurred background.
  const maxWindow =
    subject !== null ? maxWindowBox(subject, sourceW, sourceH, width, height) : null;
  // Single-subject shots always use the maximal window around the subject.
  const canCropClean = subject !== null && maxWindow !== null;
  const split = secondSubject !== null && subject !== null;

  // Two-actor layout: both crop windows are solved together (see splitWindows).
  const splitWins = split
    ? splitWindows(
        {face: bbox, subject: subject!},
        {face: secondBBox!, subject: secondSubject!},
        sourceW,
        sourceH,
        width,
        height / 2,
      )
    : null;

  // Background: blurred cover of the whole frame (fills letterbox edges).
  const BlurBack = (
    <Video
      src={src}
      startFrom={seg.video_trim_frame}
      endAt={endAt}
      volume={edgeFade}
      style={{
        position: "absolute",
        width: "100%",
        height: "100%",
        objectFit: "cover",
        transform: "scale(1.35)",
        filter: "blur(70px) saturate(1.45)",
      }}
    />
  );

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {split ? (
        <>
          {BlurBack}
          <Half top={0} height={height / 2}>
            <Video
              src={src}
              startFrom={seg.video_trim_frame}
              endAt={endAt}
              volume={edgeFade}
              style={coverCrop(
                splitWins!.top,
                sourceW,
                sourceH,
                0,
                0,
                width,
                height / 2,
              )}
            />
          </Half>
          <Half top={height / 2} height={height / 2}>
            <Video
              src={src}
              startFrom={seg.video_trim_frame}
              endAt={endAt}
              volume={0}
              style={coverCrop(
                splitWins!.bottom,
                sourceW,
                sourceH,
                0,
                0,
                width,
                height / 2,
              )}
            />
          </Half>
        </>
      ) : subject === null ? (
        <>
          {BlurBack}
          <Video
            src={src}
            startFrom={seg.video_trim_frame}
            endAt={endAt}
            volume={edgeFade}
            style={{
              position: "absolute",
              width: "100%",
              height: "100%",
              objectFit: "contain",
            }}
          />
        </>
      ) : canCropClean ? (
        <Video
          src={src}
          startFrom={seg.video_trim_frame}
          endAt={endAt}
          volume={edgeFade}
          style={coverCrop(maxWindow!, sourceW, sourceH, 0, 0, width, height)}
        />
      ) : (
        <>
          {BlurBack}
          <Video
            src={src}
            startFrom={seg.video_trim_frame}
            endAt={endAt}
            volume={edgeFade}
            style={containInRect(subject, sourceW, sourceH, 0, 0, width, height)}
          />
        </>
      )}
    </div>
  );
};