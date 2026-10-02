import React from "react";
import {AbsoluteFill, Sequence} from "remotion";
import type {RenderManifest} from "./lib/manifest";
import {ShotLayer} from "./ShotLayer";
import {CaptionStack} from "./captions/CaptionStack";

export const ClipComposition: React.FC<{manifest: RenderManifest}> = ({manifest}) => {
  return (
    <AbsoluteFill
      name="Clip"
      style={{
        backgroundColor: "#000",
        overflow: "hidden",
      }}
    >
      {manifest.shots.map((seg, i) => (
        <Sequence
          key={`${seg.src_start_frame}-${i}`}
          // `src_*_frame` are output-time (relative to the rough cut);
          // `start/end_frame` are absolute source frames used only for
          // locating keyframes.
          from={seg.src_start_frame}
          durationInFrames={Math.max(1, seg.src_end_frame - seg.src_start_frame)}
          layout="none"
        >
          <ShotLayer seg={seg} manifest={manifest} />
        </Sequence>
      ))}
      <CaptionStack words={manifest.captions} fps={manifest.fps || 30} manifest={manifest} />
    </AbsoluteFill>
  );
};