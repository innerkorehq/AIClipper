// Mirror of `types::RenderManifest` (src/types.rs). All times are *output* time.

export type BBoxKeyframe = {
  frame: number;
  x: number;
  y: number;
  w: number;
  h: number;
};

export type ShotStrategy = "face_crop" | "saliency_crop" | "blur_fill";

export type ShotSegment = {
  start_frame: number;
  end_frame: number;
  strategy: ShotStrategy;
  keyframes: BBoxKeyframe[];
  /** Optional second-subject track for split-screen two-speaker shots. */
  second_keyframes?: BBoxKeyframe[];
  /** True when this shot carries a second subject (split-screen). */
  split?: boolean;
  src_start_frame: number;
  src_end_frame: number;
  video_trim_frame: number;
};

export type Word = {
  text: string;
  start: number;
  end: number;
};

export type TimeSpan = {
  start: number;
  end: number;
};

export type CutRange = {
  start_word_idx: number;
  end_word_idx: number;
  reason: "filler" | "redundant" | "tangent" | "false_start" | "silence";
};

export type RenderManifest = {
  clip_id: string;
  duration: number;
  width: number;
  height: number;
  fps: number;
  source: string;
  source_abs: string;
  /** Small silent proxy used only for the blurred background layer. */
  blur_src?: string;
  source_width: number;
  source_height: number;
  shots: ShotSegment[];
  captions: Word[];
  cut_ranges: CutRange[];
  cuts_s: TimeSpan[];
  joins: number[];
};