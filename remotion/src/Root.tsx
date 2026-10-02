import React from "react";
import {Composition} from "remotion";
import {ClipComposition} from "./ClipComposition";
import type {RenderManifest} from "./lib/manifest";

const defaultManifest: RenderManifest = {
  clip_id: "preview",
  duration: 5,
  width: 1080,
  height: 1920,
  fps: 30,
  source: "",
  source_abs: "",
  source_width: 1920,
  source_height: 1080,
  shots: [],
  captions: [],
  cut_ranges: [],
  cuts_s: [],
  joins: [],
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="AIClip"
      component={ClipComposition}
      durationInFrames={5 * 30}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={{manifest: defaultManifest}}
      calculateMetadata={({props}) => {
        const m = props.manifest as RenderManifest;
        const durationInFrames = Math.max(
          1,
          Math.ceil(m.duration * (m.fps || 30)) + 1,
        );
        return {
          durationInFrames,
          width: m.width || 1080,
          height: m.height || 1920,
          fps: m.fps || 30,
        };
      }}
    />
  );
};