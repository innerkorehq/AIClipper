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
                maxWindowBox(subject!, sourceW, sourceH, width, height / 2),
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
                maxWindowBox(secondSubject!, sourceW, sourceH, width, height / 2),
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