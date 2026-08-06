/*
 * This file is part of paged (https://paged.media).
 *
 * paged is free software: you may redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License, version 3, as published by
 * the Free Software Foundation, OR under the Paged Media Enterprise License
 * (PMEL), a commercial license available from And The Next GmbH. Full
 * copyright and license information is available in LICENSE.md, distributed
 * with this source code.
 *
 * paged is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the licenses for details.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

// The typed engine FACADE + boot. The Rust wasm (image-js) does ALL the
// raster work — decode (codec/PSD CPU lanes) and the adjustments chain
// (Engine A, GPU-only WGSL kernels); this is a thin camelCase shape over
// its wasm-bindgen surface so the rest of the bundle codes against a
// stable contract and tests can stub it.
//
// BOOT (BREAKAGE I-07 / the sheets S-10 pattern). The artifact is the
// wasm-bindgen `--target web` glue produced by scripts/build-wasm.sh
// into manifest/wasm/ — the exact path the manifest declares under
// capabilities.wasm[] (so the plugin-cli size gate measures the real
// file). We DON'T use the host's `loadBundleWasm` (it instantiates a
// RAW module with no wbindgen imports and no ambient authority — no
// navigator.gpu); we load the glue in the BUNDLE REALM exactly like
// @paged-media/canvas-wasm does, where WebGPU is reachable and
// `initGpu()` can request the device. Browser vs Node branches mirror
// plugin-sheets' engine.ts. Until the artifact is built the dynamic
// import REJECTS — bootEngine surfaces that honestly so the panel can
// say "engine wasm not built".

/** Levels (the panel's black/white/gamma + output range), composite over
 *  all channels. Identity: in 0/1, gamma 1, out 0/1. */
export interface LevelsParams {
  inBlack: number;
  inWhite: number;
  gamma: number;
  outBlack: number;
  outWhite: number;
}

export const IDENTITY_LEVELS: LevelsParams = {
  inBlack: 0,
  inWhite: 1,
  gamma: 1,
  outBlack: 0,
  outWhite: 1,
};

/** One channel's Levels input remap (`adjust.levels_rgb`). Identity
 *  `{0, 1, 1}` — the OUTPUT range stays composite on [`LevelsParams`]. */
export interface LevelsChannel {
  inBlack: number;
  inWhite: number;
  gamma: number;
}

export const IDENTITY_LEVELS_CHANNEL: LevelsChannel = {
  inBlack: 0,
  inWhite: 1,
  gamma: 1,
};

/** Per-channel Levels (`adjust.levels_rgb`). */
export interface LevelsRgbParams {
  r: LevelsChannel;
  g: LevelsChannel;
  b: LevelsChannel;
}

/** Color balance (`adjust.color_balance`): one offset per opponent axis
 *  — `[cyan↔red, magenta↔green, yellow↔blue]` — per tonal range. All 0
 *  = off. */
export interface ColorBalanceParams {
  shadows: [number, number, number];
  midtones: [number, number, number];
  highlights: [number, number, number];
}

/** Photo filter (`adjust.photo_filter`): a colored gel. `density` 0 =
 *  OFF (the stage is skipped whatever the color is). */
export interface PhotoFilterParams {
  /** Straight RGB in [0,1]. */
  color: [number, number, number];
  density: number;
  preserveLuminosity: boolean;
}

/** Channel mixer (`adjust.channel_mixer`): each row is
 *  `[inR, inG, inB, constant]` for the r/g/b outputs. */
export interface ChannelMixerParams {
  r: [number, number, number, number];
  g: [number, number, number, number];
  b: [number, number, number, number];
}

/** Black & White (`adjust.black_white`): the six hue-sector grayscale
 *  weights behind an explicit `enabled` gate (the conversion looks
 *  destructive, so it never rides a "neutral value" default). */
export interface BlackWhiteParams {
  enabled: boolean;
  /** reds, yellows, greens, cyans, blues, magentas. */
  weights: [number, number, number, number, number, number];
}

/** The committed adjustment parameters. Identity = every field neutral:
 *  exposure 0 / brightness 0 / contrast 1 / saturation 1, white balance
 *  0/0, levels identity, no curve LUT, and every EXTENDED stage gated
 *  off. */
export interface AdjustParams {
  exposureEv: number;
  brightness: number;
  contrast: number;
  saturation: number;
  /** White balance: temp (amber↔blue), tint (green↔magenta); 0/0 = off. */
  temp: number;
  tint: number;
  /** Composite levels (all channels). */
  levels: LevelsParams;
  /** Curves: a 256-byte tone LUT (built from the curve editor's control
   *  points via `engine.curveLut`), or null for the identity curve. */
  curveLut: Uint8Array | null;
  /** FILTER stages (first wasm reach of the T1/T2 kernels): Gaussian
   *  blur sigma px (0 = off), unsharp amount (0 = off), hue rotation
   *  degrees (0 = off), per-color invert. */
  blurSigma: number;
  sharpenAmount: number;
  hueDegrees: number;
  invert: boolean;
  // ── the EXTENDED (kernel-breadth) stages ────────────────────────────
  // Chain order + rationale live on the Rust `ingest::adjust_rgba8`
  // doc; the panel groups them as "Color" / "Effects" / "Levels (per
  // channel)". Every one is mask-aware and identity-short-circuited.
  /** `adjust.vibrance` — saturation weighted by (1 − existing sat). */
  vibrance: number;
  colorBalance: ColorBalanceParams;
  photoFilter: PhotoFilterParams;
  channelMixer: ChannelMixerParams;
  levelsRgb: LevelsRgbParams;
  blackWhite: BlackWhiteParams;
  /** `adjust.posterize` — output levels per channel; `null` = off. */
  posterizeLevels: number | null;
  /** `adjust.threshold` — luma cut in [0,1]; `null` = off. */
  threshold: number | null;
}

/** The Black & White default mix (the conventional reds .4 / yellows .6
 *  / greens .4 / cyans .6 / blues .2 / magentas .8). */
export const DEFAULT_BW_WEIGHTS: BlackWhiteParams["weights"] = [
  0.4, 0.6, 0.4, 0.6, 0.2, 0.8,
];

/** The default photo-filter gel: Warming filter (85). Density 0 keeps
 *  the stage off until the panel raises it. */
export const DEFAULT_PHOTO_FILTER_COLOR: [number, number, number] = [
  0.925, 0.639, 0.365,
];

export const IDENTITY_PARAMS: AdjustParams = {
  exposureEv: 0,
  brightness: 0,
  contrast: 1,
  saturation: 1,
  temp: 0,
  tint: 0,
  levels: { ...IDENTITY_LEVELS },
  curveLut: null,
  blurSigma: 0,
  sharpenAmount: 0,
  hueDegrees: 0,
  invert: false,
  vibrance: 0,
  colorBalance: {
    shadows: [0, 0, 0],
    midtones: [0, 0, 0],
    highlights: [0, 0, 0],
  },
  photoFilter: {
    color: [...DEFAULT_PHOTO_FILTER_COLOR],
    density: 0,
    preserveLuminosity: true,
  },
  channelMixer: {
    r: [1, 0, 0, 0],
    g: [0, 1, 0, 0],
    b: [0, 0, 1, 0],
  },
  levelsRgb: {
    r: { ...IDENTITY_LEVELS_CHANNEL },
    g: { ...IDENTITY_LEVELS_CHANNEL },
    b: { ...IDENTITY_LEVELS_CHANNEL },
  },
  blackWhite: { enabled: false, weights: [...DEFAULT_BW_WEIGHTS] },
  posterizeLevels: null,
  threshold: null,
};

/** A DEEP clone of the identity params — the nested objects/arrays must
 *  never be shared with the constant (a mutation would poison it). */
export function freshIdentityParams(): AdjustParams {
  const p = IDENTITY_PARAMS;
  return {
    ...p,
    levels: { ...p.levels },
    curveLut: null,
    colorBalance: {
      shadows: [...p.colorBalance.shadows],
      midtones: [...p.colorBalance.midtones],
      highlights: [...p.colorBalance.highlights],
    },
    photoFilter: { ...p.photoFilter, color: [...p.photoFilter.color] },
    channelMixer: {
      r: [...p.channelMixer.r],
      g: [...p.channelMixer.g],
      b: [...p.channelMixer.b],
    },
    levelsRgb: {
      r: { ...p.levelsRgb.r },
      g: { ...p.levelsRgb.g },
      b: { ...p.levelsRgb.b },
    },
    blackWhite: { enabled: false, weights: [...p.blackWhite.weights] },
  };
}

function levelsIdentity(l: LevelsParams): boolean {
  return (
    l.inBlack === 0 &&
    l.inWhite === 1 &&
    l.gamma === 1 &&
    l.outBlack === 0 &&
    l.outWhite === 1
  );
}

export function isIdentity(p: AdjustParams): boolean {
  return (
    p.exposureEv === 0 &&
    p.brightness === 0 &&
    p.contrast === 1 &&
    p.saturation === 1 &&
    p.temp === 0 &&
    p.tint === 0 &&
    levelsIdentity(p.levels) &&
    p.curveLut === null &&
    filtersIdentity(p) &&
    extendedIdentity(p)
  );
}

function filtersIdentity(p: AdjustParams): boolean {
  return (
    p.blurSigma === 0 &&
    p.sharpenAmount === 0 &&
    p.hueDegrees === 0 &&
    !p.invert
  );
}

function levelsChannelIdentity(c: LevelsChannel): boolean {
  return c.inBlack === 0 && c.inWhite === 1 && c.gamma === 1;
}

const allZero = (v: readonly number[]) => v.every((n) => n === 0);

/** True when every EXTENDED stage is gated off / neutral. SEMANTIC (it
 *  mirrors the Rust `AdjustParams::has_extended_stage`): a gated-off
 *  stage is identity whatever its other fields hold, so the photo
 *  filter's color and the black & white weights never matter here. */
function extendedIdentity(p: AdjustParams): boolean {
  const m = p.channelMixer;
  return (
    p.vibrance === 0 &&
    allZero(p.colorBalance.shadows) &&
    allZero(p.colorBalance.midtones) &&
    allZero(p.colorBalance.highlights) &&
    p.photoFilter.density === 0 &&
    m.r[0] === 1 &&
    m.r[1] === 0 &&
    m.r[2] === 0 &&
    m.r[3] === 0 &&
    m.g[0] === 0 &&
    m.g[1] === 1 &&
    m.g[2] === 0 &&
    m.g[3] === 0 &&
    m.b[0] === 0 &&
    m.b[1] === 0 &&
    m.b[2] === 1 &&
    m.b[3] === 0 &&
    levelsChannelIdentity(p.levelsRgb.r) &&
    levelsChannelIdentity(p.levelsRgb.g) &&
    levelsChannelIdentity(p.levelsRgb.b) &&
    !p.blackWhite.enabled &&
    p.posterizeLevels === null &&
    p.threshold === null
  );
}

/** True when ONLY the base exposure/brightness/contrast/saturation are set
 *  (no WB / levels / curves / extended) — the legacy `adjust_image` fast
 *  path. */
function isBaseOnly(p: AdjustParams): boolean {
  return (
    p.temp === 0 &&
    p.tint === 0 &&
    levelsIdentity(p.levels) &&
    p.curveLut === null &&
    filtersIdentity(p) &&
    extendedIdentity(p)
  );
}

/** Wire length of the EXTENDED adjust block — MUST match the Rust
 *  `image_js::ingest::ADJUST_EXT_LEN`. */
export const ADJUST_EXT_LEN = 47;

/** Pack the extended stages into the flat `f32` block the
 *  `adjust_image_ext` door reads. The layout is the ONE cross-language
 *  contract (documented on the Rust `ADJUST_EXT_LEN`); keep the two in
 *  lockstep — `packAdjustExt` is unit-tested against it. */
export function packAdjustExt(p: AdjustParams): Float32Array {
  const e = new Float32Array(ADJUST_EXT_LEN);
  e[0] = p.vibrance;
  e.set(p.colorBalance.shadows, 1);
  e.set(p.colorBalance.midtones, 4);
  e.set(p.colorBalance.highlights, 7);
  e[10] = p.blackWhite.enabled ? 1 : 0;
  e.set(p.blackWhite.weights, 11);
  e[17] = p.posterizeLevels === null ? 0 : 1;
  e[18] = p.posterizeLevels ?? 0;
  e[19] = p.threshold === null ? 0 : 1;
  e[20] = p.threshold ?? 0;
  e[21] = p.photoFilter.density;
  e.set(p.photoFilter.color, 22);
  e[25] = p.photoFilter.preserveLuminosity ? 1 : 0;
  e.set(p.channelMixer.r, 26);
  e.set(p.channelMixer.g, 30);
  e.set(p.channelMixer.b, 34);
  const lr = p.levelsRgb;
  e.set([lr.r.inBlack, lr.r.inWhite, lr.r.gamma], 38);
  e.set([lr.g.inBlack, lr.g.inWhite, lr.g.gamma], 41);
  e.set([lr.b.inBlack, lr.b.inWhite, lr.b.gamma], 44);
  return e;
}

/** The RGB + luma histogram of an image (4 × 256 bins; the panel renders
 *  it). Each channel's bins sum to the pixel count. */
export interface ImageHistogram {
  r: Uint32Array;
  g: Uint32Array;
  b: Uint32Array;
  luma: Uint32Array;
}

/** Auto-enhance estimate (spec §6): percentile-clipped auto-levels black/
 *  white points + a gray-world white balance, derived from the image's
 *  histogram. Identity `{0, 1, 0, 0}` for a flat/neutral image (a no-op,
 *  never a wrong-looking correction). Merged into the panel's levels + WB. */
export interface AutoEnhanceParams {
  inBlack: number;
  inWhite: number;
  temp: number;
  tint: number;
}

/** How a selection shape folds into the existing selection (mirrors the
 *  Rust `image_gpu::CombineMode` wire discriminants 0–3). */
export type SelectionMode = "replace" | "add" | "subtract" | "intersect";

/** The wire discriminant for a [`SelectionMode`]. */
export function selectionModeCode(mode: SelectionMode): number {
  switch (mode) {
    case "replace":
      return 0;
    case "add":
      return 1;
    case "subtract":
      return 2;
    case "intersect":
      return 3;
  }
}

/** Selection readout (`selection_stats`): the non-zero coverage bounding
 *  box (image px), the mean coverage fraction (0–1), and the monotone
 *  revision. `null` from the facade = no explicit selection (everything
 *  is implicitly selected; adjustments run unmasked). */
export interface SelectionStats {
  x: number;
  y: number;
  w: number;
  h: number;
  /** Mean coverage over the whole image, 0–1 (an AA/feathered selection
   *  counts fractionally). */
  coverage: number;
  revision: number;
}

/** The 8 crop grips + the body Move (discriminants mirror the Rust
 *  `image_core::Handle`); -1 = a miss (outside the chrome). */
export type CropHandle = number;

/** An axis-aligned crop rectangle in image-pixel space. */
export interface CropRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Aspect-ratio lock encoded for the wasm geometry: `null` (free) or a
 *  `w:h` ratio pair. */
export type AspectLock = { w: number; h: number } | null;

/** A decoded image held ENGINE-SIDE behind a handle (spec §2.1.3 —
 *  pixels stay in wasm between calls). */
export interface DecodedInfo {
  handle: number;
  width: number;
  height: number;
  /** CMS rung 1 — what the RGB display transform did at decode. The
   *  panel states this so "sRGB assumed" is a reported state rather than
   *  a silent one. */
  display: DisplayTreatment;
}

/** How the ingest lane treated the source's colour. `managed` means an
 *  embedded ICC profile compiled and the pixels were transformed into the
 *  working sRGB space; the other two are honest pass-throughs. */
export type DisplayTreatment = "managed" | "assumed-srgb" | "profile-rejected";

/** Map the wasm discriminant (image-js `display_code`) to the name. */
export function displayTreatmentOf(code: number | undefined): DisplayTreatment {
  return code === 0
    ? "managed"
    : code === 2
      ? "profile-rejected"
      : "assumed-srgb";
}

/** A short phrase for the panel's colour row. */
export function displayTreatmentLabel(t: DisplayTreatment): string {
  return t === "managed"
    ? "ICC managed"
    : t === "profile-rejected"
      ? "sRGB assumed (profile rejected)"
      : "sRGB assumed";
}

/** The stable engine contract the bundle codes against. Every method
 *  forwards to the wasm surface; the facade only renames + shapes. */
/** The T1 resample kernels the resize door accepts. */
export type ResampleFilter = "nearest" | "mitchell" | "lanczos3";

/** Which `gen.*` gradient the fill door dispatches (the wire names the
 *  Rust `GradientKind::from_wire` decodes). */
export type GradientKind =
  "linear" | "radial" | "angular" | "reflected" | "diamond";

export const GRADIENT_KINDS: GradientKind[] = [
  "linear",
  "radial",
  "angular",
  "reflected",
  "diamond",
];

/** A straight RGBA colour in [0,1] — the fill door's stop format. */
export type Rgba01 = [number, number, number, number];

/** The raster re-encode formats the non-PSD save-back lane offers. */
export type RasterFormat = "png" | "jpeg";

// ─────────────────────────────── PAINT ───────────────────────────────

/** The three painting tools the engine's `brush_stroke_begin` takes
 *  (mirrors the Rust `StrokeTool` wire names). */
export type StrokeTool = "brush" | "pencil" | "eraser";

export const STROKE_TOOLS: StrokeTool[] = ["brush", "pencil", "eraser"];

/** What a pen's pressure drives (mirrors the Rust `PressureTarget`). */
export type PressureTarget = "none" | "size" | "opacity" | "both";

export const PRESSURE_TARGETS: PressureTarget[] = [
  "none",
  "size",
  "opacity",
  "both",
];

/** Everything a stroke FREEZES at `begin` (a stroke whose size changed
 *  halfway through would not be replayable — the engine says so and this
 *  shape mirrors its `StrokeParams` field for field). Straight RGBA in
 *  [0,1] for `color`; `blend` is a `compose.*` kernel name with the
 *  prefix optional. */
export interface BrushParams {
  /** Tip diameter in IMAGE pixels (before pressure scaling). */
  size: number;
  /** Fully-opaque fraction of the radius, 0..1 (the pencil ignores it —
   *  its binary tip is the point). */
  hardness: number;
  /** The ceiling the whole stroke composites at, 0..1. */
  opacity: number;
  /** How much each dab deposits, 0..1 (forced to 1 for the pencil). */
  flow: number;
  /** Dab spacing as a fraction of the tip DIAMETER (Photoshop's
   *  convention): 0.25 = a dab every quarter-diameter. */
  spacing: number;
  /** The `compose.*` blend the paint goes down through (the eraser
   *  ignores it — erasing is `band.set_alpha`, not a blend). */
  blend: string;
  color: Rgba01;
  pressureTarget: PressureTarget;
}

/** The v0 defaults — the SAME values the Rust `StrokeParams::defaults`
 *  documents (24 px half-hard round tip, full opacity and flow,
 *  quarter-diameter spacing, normal blend, opaque black, pressure
 *  driving size AND flow). Kept in lockstep by
 *  `image_editor_paint_defaults_are_the_documented_v0` on the Rust side
 *  and the brush spec on this one. */
export const DEFAULT_BRUSH_PARAMS: BrushParams = {
  size: 24,
  hardness: 0.5,
  opacity: 1,
  flow: 1,
  spacing: 0.25,
  blend: "normal",
  color: [0, 0, 0, 1],
  pressureTarget: "both",
};

/** The in-flight stroke's readout (`brush_stroke_stats`): the dab count
 *  and the stroke's bounding box in image px. Null when no stroke is in
 *  progress or nothing has landed on the canvas yet. */
export interface BrushStats {
  dabs: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

// ────────────────────────────── LAYERS ───────────────────────────────

/** One row of the engine's layer stack (`layers_list`). BOTTOM-first:
 *  `index` 0 is the bottom-most layer, exactly as the engine composites
 *  and as PSD stores them (the panel renders the list reversed, because
 *  a layers palette reads top-down). */
export interface LayerInfo {
  index: number;
  /** Stable across reorders — the key the UI lists rows by. */
  id: number;
  name: string;
  visible: boolean;
  /** Pixel lock: paint / fill / bake refuse; properties still move. */
  locked: boolean;
  /** 0–1. */
  opacity: number;
  /** A `compose.*` wire name (the prefix dropped). */
  blend: string;
  /** What the layer contributes. An `adjustment` carries no pixels and
   *  transforms everything beneath it; a `smart` layer's pixels are a
   *  cached RENDER of preserved source bytes, so rescaling it is
   *  lossless. */
  kind: "pixels" | "adjustment" | "smart";
  /** Whether a layer mask is attached at all. */
  hasMask: boolean;
  /** Whether that mask APPLIES. Disabled keeps the coverage — the two
   *  are different states and the panel shows both. */
  maskEnabled: boolean;
}

export interface LayerStackInfo {
  /** Index of the layer edits land in; -1 when no stack is open. */
  active: number;
  layers: LayerInfo[];
}

/** The undo readout (`layers_history`). The BOUND is part of it on
 *  purpose: a journal that silently forgets is worse than one that says
 *  how much it holds. `dropped` counts entries the bound evicted. */
export interface LayerHistory {
  canUndo: boolean;
  canRedo: boolean;
  depth: number;
  redoDepth: number;
  bytes: number;
  maxBytes: number;
  maxEntries: number;
  dropped: number;
  generation: number;
  undoLabel: string | null;
  redoLabel: string | null;
  /** Every RETAINED undo step, oldest first — the History panel's list.
   *  Not the whole session: the journal is byte-budgeted and `dropped`
   *  counts what fell off, which the panel states rather than implying a
   *  completeness the bound cannot provide. */
  undoSteps: readonly string[];
  /** Every redo step, next-to-replay first. */
  redoSteps: readonly string[];
}

export const EMPTY_LAYER_STACK: LayerStackInfo = { active: -1, layers: [] };

/** One PSD layer-record row (`psd_layer_list`), record order. */
export interface PsdLayerInfo {
  index: number;
  name: string;
  /** 0–255. */
  opacity: number;
  /** PSD flags bit 1 — hidden (display-only; no visibility edit tier yet). */
  hidden: boolean;
  top: number;
  left: number;
  bottom: number;
  right: number;
}

export interface ImageEngine {
  abiVersion(): number;
  kernelCount(): number;
  /** Request the WebGPU device in the bundle realm. Resolves false when
   *  the environment has no WebGPU — the honest no-GPU state (kernels
   *  are GPU-only; identity adjusts still work, nothing else). */
  initGpu(): Promise<boolean>;
  gpuReady(): boolean;
  /** Decode PSD/PNG/JPEG bytes (magic-sniffed) to an engine-held RGBA8
   *  image. Throws with the engine's honest message on unsupported
   *  inputs (16-bit, CMYK, ZIP composites, …). */
  decode(bytes: Uint8Array): DecodedInfo;
  /** K-3 — register PRE-DECODED straight RGBA8 (from the decode worker
   *  pool, which ran the CPU decode off-thread) as an engine-held image,
   *  returning a handle for the GPU adjust + tile paths. `rgba` must be
   *  `width*height*4` bytes; a mismatch throws. */
  ingestRgba8(width: number, height: number, rgba: Uint8Array): DecodedInfo;
  /** Run the adjustments chain (GPU for the kernel stages + a CPU curve
   *  LUT pass) and return straight RGBA8 — the C-1 Stage-A scene-item
   *  payload. Identity params return the decode verbatim without touching
   *  the GPU; the FULL panel set (WB / levels / curves) routes through the
   *  extended surface. */
  adjust(handle: number, params: AdjustParams): Promise<Uint8Array>;
  /** Compute the RGB + luma 256-bin histogram of an engine-held image
   *  (the LEVELS / CURVES panel readout). Pure CPU reduction; no GPU. */
  histogram(handle: number): ImageHistogram;
  /** Derive an auto-enhance estimate (auto-levels + gray-world white
   *  balance) from the engine-held image's histogram. Pure CPU readout;
   *  no GPU, no kernel dispatch — the values flow through the SAME
   *  adjust pipeline the sliders use (the caller still commits via Apply). */
  autoEnhanceParams(handle: number): AutoEnhanceParams;
  /** Commit a CROP: cut the integer pixel rectangle out of an engine-held
   *  image and register the result as a NEW engine-held image, returning
   *  its handle. The source handle is left intact. Throws on an empty /
   *  out-of-bounds rectangle. */
  crop(handle: number, rect: CropRect): DecodedInfo;
  /** STRAIGHTEN + CROP commit: rotate by `−degrees` about the rect's
   *  centre (`geom.rotate_bilinear` — backward-mapped bilinear,
   *  clamp-to-edge) so the rotated crop FRAME lands upright, then cut
   *  the rect. Returns a NEW engine-held image; the source stays.
   *  `degrees === 0` is the pure windowing path (no GPU, no resample);
   *  any other angle is a resample and needs the WebGPU device. */
  straightenCrop(
    handle: number,
    rect: CropRect,
    degrees: number,
  ): Promise<DecodedInfo>;
  /** Hit-test the crop chrome (the nearest grip within `tol`, else Move
   *  inside the body, else -1). Pure geometry from `image_core::crop`. */
  cropHitHandle(
    rect: CropRect,
    point: [number, number],
    tol: number,
  ): CropHandle;
  /** Apply a pointer drag at `handle` to the crop rect, with the aspect
   *  lock + image-extent clamp. Returns the new rect. */
  cropApplyDrag(
    rect: CropRect,
    handle: CropHandle,
    start: [number, number],
    point: [number, number],
    aspect: AspectLock,
    imageW: number,
    imageH: number,
  ): CropRect;
  /** The four crop-FRAME corners rotated by the straighten `degrees`
   *  (TL, TR, BR, BL) — the closed polyline the overlay draws. */
  cropFrameCorners(rect: CropRect, degrees: number): Array<[number, number]>;
  /** Build a 256-byte tone LUT from the curve editor's `(input, output)`
   *  control points in [0,1] (the LUT `adjust` consumes for curves). */
  curveLut(points: Array<[number, number]>): Uint8Array;
  /** RESAMPLE to a new size through the T1 kernels (GPU-only; requires
   *  initGpu). Returns a NEW engine-held image; the source stays. */
  resize(
    handle: number,
    outW: number,
    outH: number,
    filter: ResampleFilter,
  ): Promise<DecodedInfo>;
  /** PSD structural session — the mutatable tier's wasm reach ("Paged
   *  never destroys a PSD" was a Rust-test-only property until these).
   *  `psdOpen` retains the PARSED file behind a handle; edits accumulate
   *  on it; `psdSave` re-emits with full carry-through preservation. */
  psdOpen(bytes: Uint8Array): number;
  psdLayers(handle: number): PsdLayerInfo[];
  psdSetLayerOpacity(handle: number, layer: number, opacity: number): void;
  psdSetLayerName(handle: number, layer: number, name: string): void;
  psdRemoveLayer(handle: number, layer: number): void;
  psdSave(handle: number): Uint8Array;
  psdClose(handle: number): void;
  /** PSD SAVE-BACK: write the ADJUSTED full-resolution RGBA8 into the
   *  retained parse (the merged composite is always rewritten) and
   *  answer the HONEST description of what happened to the layer
   *  structure — "…written into the single content layer (structure
   *  preserved)" or "…FLATTENED into a new single-layer PSD…". Call
   *  `psdSave` afterwards for the bytes. 8-bit RGB only; a size or mode
   *  mismatch throws with the engine's message. */
  psdApplyAdjusted(
    psdHandle: number,
    width: number,
    height: number,
    rgba: Uint8Array,
  ): string;
  /** Re-encode straight RGBA8 as PNG or JPEG (the NON-PSD save-back
   *  lane; JPEG rides the fixed v0 quality). */
  encode(
    rgba: Uint8Array,
    width: number,
    height: number,
    format: RasterFormat,
  ): Uint8Array;
  /** FILL the bound SELECTION (the whole image when none) with a fixed
   *  two-stop gradient, compositing through the coverage mask on the
   *  GPU. DESTRUCTIVE: returns a NEW engine-held image (the crop/resize
   *  commit pattern — the caller swaps and frees the old handle). */
  fillGradient(
    handle: number,
    kind: GradientKind,
    c0: Rgba01,
    c1: Rgba01,
  ): Promise<DecodedInfo>;
  /** FILL the bound selection with deterministic monochrome noise. */
  fillNoise(handle: number, amount: number, seed: number): Promise<DecodedInfo>;
  /** C-6 — copy a LEVEL-0 tile window `(x, y, w, h)` out of a decoded
   *  image as tightly packed RGBA8. Edge tiles are clamped to the image
   *  extent; a fully-outside window returns an empty buffer. The honest
   *  subset of the resource provider (pure windowing — no mip pyramid /
   *  Engine B window eval yet; see tile-provider.ts). */
  tile(handle: number, x: number, y: number, w: number, h: number): Uint8Array;
  freeImage(handle: number): void;
  /** SELECTION doors (spec §6.1). One selection per engine realm, bound
   *  to one image (`selectionBind`); shapes rasterize engine-side at the
   *  bound resolution (mask PREP is CPU; the mask is CONSUMED GPU-only —
   *  every masked adjust dispatch applies `mix(a, result, mask)` at the
   *  ABI's r16float `@group(2)` binding). Re-binding to a different
   *  handle/resolution drops the selection; `adjust` on the bound handle
   *  automatically masks when a non-trivial selection exists. */
  selectionBind(handle: number): void;
  /** Marquee rect `[x, x+w) × [y, y+h)` (image px, fractional = AA edge). */
  selectionSetRect(
    x: number,
    y: number,
    w: number,
    h: number,
    mode: SelectionMode,
  ): void;
  /** Marquee ellipse: center + radii (image px), AA edge. */
  selectionSetEllipse(
    cx: number,
    cy: number,
    rx: number,
    ry: number,
    mode: SelectionMode,
  ): void;
  /** Lasso: a closed polygon of `[x, y]` image-px vertices (≥ 3). */
  selectionSetPolygon(
    points: Array<[number, number]>,
    mode: SelectionMode,
  ): void;
  /** Magic wand at an integer image-px seed. `tolerance` is per-channel
   *  0–255 (Chebyshev over RGBA); `contiguous` = 4-connected flood. */
  selectionMagicWand(
    x: number,
    y: number,
    tolerance: number,
    contiguous: boolean,
    mode: SelectionMode,
  ): void;
  /** Gaussian feather of the coverage (σ px; CPU mask prep). Throws when
   *  no explicit selection exists. */
  selectionFeather(sigma: number): void;
  /** Select-all (an explicit full-extent selection). */
  selectionSelectAll(): void;
  /** Deselect: back to "no selection" (adjust runs unmasked). */
  selectionClear(): void;
  /** Invert ("everything" inverts to the explicit empty selection). */
  selectionInvert(): void;
  /** The selection readout, or `null` when no explicit selection exists. */
  selectionStats(): SelectionStats | null;
  /** The raw u8 coverage (`width·height`, row-major); empty when none. */
  selectionCoverageBytes(): Uint8Array;
  /** Re-point the selection at a NEW handle of the SAME extent, KEEPING
   *  the coverage (the destructive-fill lane: the generator registers a
   *  new engine image at identical dimensions, and the selection is
   *  still meaningful there). False when the extent changed — then it
   *  behaves like `selectionBind` and the selection drops. */
  selectionTransfer(handle: number): boolean;
  /** PAINT doors (spec §6.3). One in-flight stroke per engine realm:
   *  `brushBegin` snapshots the base pixels and freezes the params (the
   *  bound selection is frozen too, so a stroke can never half-honour a
   *  selection that changed mid-drag); every `brushExtend` stamps the
   *  sample's dabs and returns the WHOLE image as straight RGBA8 (the
   *  C-1 Stage-A preview payload); `brushCommit` registers the painted
   *  pixels as a NEW engine-held image (the crop/fill commit pattern);
   *  `brushCancel` throws them away — the source was never mutated.
   *  GPU-only: `brushBegin` rejects without a device. */
  brushBegin(handle: number, tool: StrokeTool, params: BrushParams): void;
  brushExtend(x: number, y: number, pressure: number): Promise<Uint8Array>;
  /** CLOSE the stroke. With a layer stack bound the painted pixels go
   *  into the ACTIVE LAYER (journaled — undoable), the stack is
   *  re-composited, and the returned handle is the SAME one you started
   *  with (do NOT free it). Without a stack it registers a new
   *  engine-held image, as it did before layers existed. */
  brushCommit(): Promise<DecodedInfo>;
  brushCancel(): void;
  brushActive(): boolean;
  /** The in-flight stroke's readout, or null before the first dab lands. */
  brushStats(): BrushStats | null;
  /** Every blend mode a stroke can paint through, derived from the
   *  `compose.*` registry — so the panel's picker cannot drift from the
   *  kernels that actually exist. */
  brushBlendModes(): string[];

  /** LAYER doors. One stack per engine realm, BOUND to an image handle
   *  (the selection's pattern). `layersOpen` seeds it with that image's
   *  pixels as a single "Background" layer — O(1), the pixels are
   *  shared. The bound image IS the stack's composite: `layersComposite`
   *  folds the stack and writes the result back into the SAME handle, so
   *  the adjust chain, tiles, histogram and save-back keep working
   *  against one handle and never learn what a layer is. */
  layersOpen(handle: number): void;
  /** Open the stack from a retained PSD parse — the file's own layer
   *  tree instead of its flattened composite. Returns the layer count;
   *  THROWS with the engine's stated reason for every PSD the layer
   *  model does not reproduce (groups, clipping, masks, non-8-bit-RGB,
   *  over budget), and the caller then keeps the flatten. */
  layersOpenFromPsd(imageHandle: number, psdHandle: number): number;
  layersClose(): void;
  /** The bound handle, or -1. */
  layersBound(): number;
  /** The stack, BOTTOM-first. Empty (`active: -1`) when none is open. */
  layers(): LayerStackInfo;
  /** The undo readout, or null when no stack is open. */
  layersHistory(): LayerHistory | null;
  /** Add an empty transparent layer above the active one (it becomes
   *  active); returns its index. */
  layerAdd(name: string): number;
  layerDuplicate(index: number): number;
  /** Remove a layer. THROWS on the last one (a document keeps at least
   *  one) — and it is NOT journaled, so the pixels are gone. */
  layerRemove(index: number): void;
  layerReorder(from: number, to: number): void;
  layerSetActive(index: number): void;
  layerSetVisible(index: number, visible: boolean): void;
  layerSetLocked(index: number, locked: boolean): void;
  layerSetOpacity(index: number, opacity: number): void;
  layerSetName(index: number, name: string): void;
  /** Set the blend by `compose.*` wire name; an unregistered name
   *  THROWS rather than silently becoming normal. */
  layerSetBlend(index: number, blend: string): void;
  /** Stack the chain as an ADJUSTMENT LAYER above the active one —
   *  non-destructive, unlike `layersBakeAdjust`. Returns its index.
   *  Throws at identity rather than adding a row that does nothing. */
  layersAddAdjustment(name: string, params: AdjustParams): number;
  /** Convert a pixel layer into a smart object, preserving its pixels
   *  as the source. One-way: going back would discard the source. */
  layerMakeSmart(index: number): void;
  /** Re-render a smart object at `scale` FROM ITS SOURCE, never from the
   *  current cache — which is what makes scaling down and back up
   *  lossless. GPU-only. */
  layerRenderSmart(index: number, scale: number): Promise<void>;
  /** Make the current selection this layer's mask — the natural
   *  authoring path (a selection and a mask are the same coverage). */
  layerMaskFromSelection(index: number): void;
  /** DELETE the mask; distinct from disabling it. */
  layerClearMask(index: number): void;
  /** Toggle whether the mask applies, retaining the coverage. */
  layerSetMaskEnabled(index: number, enabled: boolean): void;
  /** Fold the stack and write the result into the bound image; returns
   *  the straight RGBA8. GPU-only whenever there is anything to blend; a
   *  single plain visible layer needs no device at all. */
  layersComposite(): Promise<Uint8Array>;
  /** BAKE the adjustment chain destructively into the ACTIVE layer
   *  (journaled, so undoable). The panel's chain is otherwise a
   *  re-runnable preview that mutates nothing. Throws at identity, on a
   *  locked layer, and without a GPU. */
  layersBakeAdjust(params: AdjustParams): Promise<Uint8Array>;
  /** Undo/redo the newest journaled PIXEL edit (paint / fill / bake) and
   *  re-composite. Resolves to the edit's label, or "" when there is
   *  nothing to do. Layer STRUCTURE changes are not journaled. */
  layersUndo(): Promise<string>;
  layersRedo(): Promise<string>;
}

// ---------------------------------------------------- wasm surface shape

interface DecodedHandleWasm {
  handle: number;
  width: number;
  height: number;
  /** See `displayTreatmentOf` — absent on an older engine build, which
   *  falls back to "sRGB assumed" rather than claiming management. */
  display?: number;
  free(): void;
}

/** The snake_case wasm-bindgen surface (image-js) — a structural subset
 *  of manifest/wasm/image_js.d.ts, only the members the bundle drives. */
export interface ImageWasmModule {
  default(input?: unknown): Promise<unknown>;
  initSync(module: { module: BufferSource | WebAssembly.Module }): unknown;
  abi_version(): number;
  kernel_count(): number;
  init_gpu(): Promise<void>;
  gpu_ready(): boolean;
  decode_image(bytes: Uint8Array): DecodedHandleWasm;
  ingest_rgba8(
    width: number,
    height: number,
    bytes: Uint8Array,
  ): DecodedHandleWasm;
  adjust_image(
    handle: number,
    exposure_ev: number,
    brightness: number,
    contrast: number,
    saturation: number,
  ): Promise<Uint8Array>;
  adjust_image_full(
    handle: number,
    exposure_ev: number,
    brightness: number,
    contrast: number,
    saturation: number,
    temp: number,
    tint: number,
    in_black: number,
    in_white: number,
    gamma: number,
    out_black: number,
    out_white: number,
    curve_lut: Uint8Array,
    blur_sigma: number,
    sharpen_amount: number,
    hue_degrees: number,
    invert: boolean,
  ): Promise<Uint8Array>;
  adjust_image_ext(
    handle: number,
    exposure_ev: number,
    brightness: number,
    contrast: number,
    saturation: number,
    temp: number,
    tint: number,
    in_black: number,
    in_white: number,
    gamma: number,
    out_black: number,
    out_white: number,
    curve_lut: Uint8Array,
    blur_sigma: number,
    sharpen_amount: number,
    hue_degrees: number,
    invert: boolean,
    ext: Float32Array,
  ): Promise<Uint8Array>;
  fill_gradient(
    handle: number,
    kind: string,
    c0: Float32Array,
    c1: Float32Array,
  ): Promise<DecodedHandleWasm>;
  fill_noise(
    handle: number,
    amount: number,
    seed: number,
  ): Promise<DecodedHandleWasm>;
  encode_image(
    rgba: Uint8Array,
    width: number,
    height: number,
    format: string,
  ): Uint8Array;
  psd_apply_adjusted(
    psd_handle: number,
    width: number,
    height: number,
    rgba: Uint8Array,
  ): string;
  image_histogram(handle: number): Uint32Array;
  image_auto_enhance_params(handle: number): Float32Array;
  crop_image(
    handle: number,
    x: number,
    y: number,
    w: number,
    h: number,
  ): DecodedHandleWasm;
  straighten_crop_image(
    handle: number,
    x: number,
    y: number,
    w: number,
    h: number,
    degrees: number,
  ): Promise<DecodedHandleWasm>;
  crop_hit_handle(
    x: number,
    y: number,
    w: number,
    h: number,
    px: number,
    py: number,
    tol: number,
  ): number;
  crop_apply_drag(
    x: number,
    y: number,
    w: number,
    h: number,
    handle: number,
    sx: number,
    sy: number,
    px: number,
    py: number,
    aspect_w: number,
    aspect_h: number,
    image_w: number,
    image_h: number,
  ): Float32Array;
  crop_frame_corners(
    x: number,
    y: number,
    w: number,
    h: number,
    degrees: number,
  ): Float32Array;
  curve_lut(points: Float32Array): Uint8Array;
  image_tile_rgba8(
    handle: number,
    x: number,
    y: number,
    w: number,
    h: number,
  ): Uint8Array;
  free_image(handle: number): void;
  resize_image(
    handle: number,
    outW: number,
    outH: number,
    filter: string,
  ): Promise<DecodedHandleWasm>;
  selection_bind(handle: number): void;
  selection_set_rect(
    x: number,
    y: number,
    w: number,
    h: number,
    mode: number,
  ): void;
  selection_set_ellipse(
    cx: number,
    cy: number,
    rx: number,
    ry: number,
    mode: number,
  ): void;
  selection_set_polygon(points_flat: Float32Array, mode: number): void;
  selection_magic_wand(
    x: number,
    y: number,
    tolerance: number,
    contiguous: boolean,
    mode: number,
  ): void;
  selection_feather(sigma: number): void;
  selection_select_all(): void;
  selection_clear(): void;
  selection_invert(): void;
  selection_bounds(): Uint32Array;
  selection_stats(): Float32Array;
  selection_coverage_bytes(): Uint8Array;
  selection_transfer(handle: number): boolean;
  brush_stroke_begin(
    handle: number,
    tool: string,
    size: number,
    hardness: number,
    opacity: number,
    flow: number,
    spacing: number,
    blend: string,
    color: Float32Array,
    pressure_target: string,
  ): void;
  brush_stroke_extend(
    x: number,
    y: number,
    pressure: number,
  ): Promise<Uint8Array>;
  brush_stroke_commit(): Promise<DecodedHandleWasm>;
  brush_stroke_cancel(): void;
  brush_stroke_active(): boolean;
  brush_stroke_stats(): Float64Array;
  brush_blend_modes(): string;
  psd_open(bytes: Uint8Array): number;
  psd_layer_list(handle: number): string;
  psd_set_layer_opacity(handle: number, layer: number, opacity: number): void;
  psd_set_layer_name(handle: number, layer: number, name: string): void;
  psd_remove_layer(handle: number, layer: number): void;
  psd_save(handle: number): Uint8Array;
  psd_close(handle: number): void;
  layers_open(handle: number): void;
  layers_open_from_psd(image_handle: number, psd_handle: number): number;
  layers_close(): void;
  layers_bound(): number;
  layers_list(): string;
  layers_history(): string;
  layers_add(name: string): number;
  layers_duplicate(index: number): number;
  layers_remove(index: number): void;
  layers_reorder(from: number, to: number): void;
  layers_set_active(index: number): void;
  layers_set_visible(index: number, visible: boolean): void;
  layers_set_locked(index: number, locked: boolean): void;
  layers_set_opacity(index: number, opacity: number): void;
  layers_set_name(index: number, name: string): void;
  layers_set_blend(index: number, blend: string): void;
  layers_add_adjustment(
    name: string,
    exposureEv: number,
    brightness: number,
    contrast: number,
    saturation: number,
    temp: number,
    tint: number,
    inBlack: number,
    inWhite: number,
    gamma: number,
    outBlack: number,
    outWhite: number,
    curveLut: Uint8Array,
    blurSigma: number,
    sharpenAmount: number,
    hueDegrees: number,
    invert: boolean,
    ext: Float32Array,
  ): number;
  layers_make_smart(index: number): void;
  layers_render_smart(index: number, scale: number): Promise<void>;
  layers_mask_from_selection(index: number): void;
  layers_clear_mask(index: number): void;
  layers_set_mask_enabled(index: number, enabled: boolean): void;
  layers_composite(): Promise<Uint8Array>;
  layers_bake_adjust(
    exposure_ev: number,
    brightness: number,
    contrast: number,
    saturation: number,
    temp: number,
    tint: number,
    in_black: number,
    in_white: number,
    gamma: number,
    out_black: number,
    out_white: number,
    curve_lut: Uint8Array,
    blur_sigma: number,
    sharpen_amount: number,
    hue_degrees: number,
    invert: boolean,
    ext: Float32Array,
  ): Promise<Uint8Array>;
  layers_undo(): Promise<string>;
  layers_redo(): Promise<string>;
}

// ----------------------------------------------------------- the facade

/** Wrap a booted wasm module in the camelCase facade. Split out so the
 *  mapping is unit-testable over a fake wasm object (no real wasm). */
export function wrapEngine(wasm: ImageWasmModule): ImageEngine {
  return {
    abiVersion: () => wasm.abi_version(),
    kernelCount: () => wasm.kernel_count(),
    async initGpu() {
      if (wasm.gpu_ready()) return true;
      try {
        await wasm.init_gpu();
        return true;
      } catch {
        // The honest no-GPU state — no CPU kernel fallback ships.
        return false;
      }
    },
    gpuReady: () => wasm.gpu_ready(),
    decode(bytes) {
      const h = wasm.decode_image(bytes);
      const info = {
        handle: h.handle,
        width: h.width,
        height: h.height,
        display: displayTreatmentOf(h.display),
      };
      h.free();
      return info;
    },
    ingestRgba8(width, height, rgba) {
      const h = wasm.ingest_rgba8(width, height, rgba);
      const info = {
        handle: h.handle,
        width: h.width,
        height: h.height,
        display: displayTreatmentOf(h.display),
      };
      h.free();
      return info;
    },
    adjust: (handle, p) => {
      // Base-only params take the legacy 4-scalar fast path; anything in
      // the FULL panel set (WB / levels / curves) routes to the extended
      // surface, passing the curve LUT (empty = no curve).
      if (isBaseOnly(p)) {
        return wasm.adjust_image(
          handle,
          p.exposureEv,
          p.brightness,
          p.contrast,
          p.saturation,
        );
      }
      // One extended door for the whole panel set — the flat ext block
      // carries the kernel-breadth stages so the boundary does not grow
      // an argument per adjustment.
      return wasm.adjust_image_ext(
        handle,
        p.exposureEv,
        p.brightness,
        p.contrast,
        p.saturation,
        p.temp,
        p.tint,
        p.levels.inBlack,
        p.levels.inWhite,
        p.levels.gamma,
        p.levels.outBlack,
        p.levels.outWhite,
        p.curveLut ?? new Uint8Array(0),
        p.blurSigma,
        p.sharpenAmount,
        p.hueDegrees,
        p.invert,
        packAdjustExt(p),
      );
    },
    async fillGradient(handle, kind, c0, c1) {
      const h = await wasm.fill_gradient(
        handle,
        kind,
        Float32Array.from(c0),
        Float32Array.from(c1),
      );
      const info = {
        handle: h.handle,
        width: h.width,
        height: h.height,
        display: displayTreatmentOf(h.display),
      };
      h.free();
      return info;
    },
    async fillNoise(handle, amount, seed) {
      const h = await wasm.fill_noise(handle, amount, seed);
      const info = {
        handle: h.handle,
        width: h.width,
        height: h.height,
        display: displayTreatmentOf(h.display),
      };
      h.free();
      return info;
    },
    encode: (rgba, width, height, format) =>
      wasm.encode_image(rgba, width, height, format),
    psdApplyAdjusted: (psdHandle, width, height, rgba) =>
      wasm.psd_apply_adjusted(psdHandle, width, height, rgba),
    resize: async (handle, outW, outH, filter) => {
      const h = await wasm.resize_image(handle, outW, outH, filter);
      const info = {
        handle: h.handle,
        width: h.width,
        height: h.height,
        display: displayTreatmentOf(h.display),
      };
      h.free();
      return info;
    },
    psdOpen: (bytes) => wasm.psd_open(bytes),
    psdLayers: (handle) =>
      JSON.parse(wasm.psd_layer_list(handle)) as PsdLayerInfo[],
    psdSetLayerOpacity: (handle, layer, opacity) =>
      wasm.psd_set_layer_opacity(handle, layer, opacity),
    psdSetLayerName: (handle, layer, name) =>
      wasm.psd_set_layer_name(handle, layer, name),
    psdRemoveLayer: (handle, layer) => wasm.psd_remove_layer(handle, layer),
    psdSave: (handle) => wasm.psd_save(handle),
    psdClose: (handle) => wasm.psd_close(handle),
    histogram(handle) {
      const flat = wasm.image_histogram(handle);
      return {
        r: flat.slice(0, 256),
        g: flat.slice(256, 512),
        b: flat.slice(512, 768),
        luma: flat.slice(768, 1024),
      };
    },
    autoEnhanceParams(handle) {
      // Rust returns [in_black, in_white, temp, tint] (image-js lib.rs).
      const a = wasm.image_auto_enhance_params(handle);
      return { inBlack: a[0], inWhite: a[1], temp: a[2], tint: a[3] };
    },
    crop(handle, rect) {
      const h = wasm.crop_image(handle, rect.x, rect.y, rect.w, rect.h);
      const info = {
        handle: h.handle,
        width: h.width,
        height: h.height,
        display: displayTreatmentOf(h.display),
      };
      h.free();
      return info;
    },
    async straightenCrop(handle, rect, degrees) {
      const h = await wasm.straighten_crop_image(
        handle,
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        degrees,
      );
      const info = {
        handle: h.handle,
        width: h.width,
        height: h.height,
        display: displayTreatmentOf(h.display),
      };
      h.free();
      return info;
    },
    cropHitHandle: (rect, point, tol) =>
      wasm.crop_hit_handle(
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        point[0],
        point[1],
        tol,
      ),
    cropApplyDrag(rect, handle, start, point, aspect, imageW, imageH) {
      const out = wasm.crop_apply_drag(
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        handle,
        start[0],
        start[1],
        point[0],
        point[1],
        aspect ? aspect.w : 0,
        aspect ? aspect.h : 0,
        imageW,
        imageH,
      );
      return { x: out[0], y: out[1], w: out[2], h: out[3] };
    },
    cropFrameCorners(rect, degrees) {
      const f = wasm.crop_frame_corners(
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        degrees,
      );
      return [
        [f[0], f[1]],
        [f[2], f[3]],
        [f[4], f[5]],
        [f[6], f[7]],
      ];
    },
    curveLut(points) {
      const flat = new Float32Array(points.length * 2);
      for (let i = 0; i < points.length; i++) {
        flat[i * 2] = points[i][0];
        flat[i * 2 + 1] = points[i][1];
      }
      return wasm.curve_lut(flat);
    },
    tile: (handle, x, y, w, h) => wasm.image_tile_rgba8(handle, x, y, w, h),
    freeImage: (h) => wasm.free_image(h),
    selectionBind: (handle) => wasm.selection_bind(handle),
    selectionSetRect: (x, y, w, h, mode) =>
      wasm.selection_set_rect(x, y, w, h, selectionModeCode(mode)),
    selectionSetEllipse: (cx, cy, rx, ry, mode) =>
      wasm.selection_set_ellipse(cx, cy, rx, ry, selectionModeCode(mode)),
    selectionSetPolygon(points, mode) {
      const flat = new Float32Array(points.length * 2);
      for (let i = 0; i < points.length; i++) {
        flat[i * 2] = points[i][0];
        flat[i * 2 + 1] = points[i][1];
      }
      wasm.selection_set_polygon(flat, selectionModeCode(mode));
    },
    selectionMagicWand: (x, y, tolerance, contiguous, mode) =>
      wasm.selection_magic_wand(
        x,
        y,
        tolerance,
        contiguous,
        selectionModeCode(mode),
      ),
    selectionFeather: (sigma) => wasm.selection_feather(sigma),
    selectionSelectAll: () => wasm.selection_select_all(),
    selectionClear: () => wasm.selection_clear(),
    selectionInvert: () => wasm.selection_invert(),
    selectionStats() {
      // Rust returns [has, x, y, w, h, fraction, revision] (7 f32s).
      const s = wasm.selection_stats();
      if (s.length < 7 || s[0] === 0) return null;
      return {
        x: s[1],
        y: s[2],
        w: s[3],
        h: s[4],
        coverage: s[5],
        revision: s[6],
      };
    },
    selectionCoverageBytes: () => wasm.selection_coverage_bytes(),
    selectionTransfer: (handle) => wasm.selection_transfer(handle),
    brushBegin: (handle, tool, p) =>
      wasm.brush_stroke_begin(
        handle,
        tool,
        p.size,
        p.hardness,
        p.opacity,
        p.flow,
        p.spacing,
        p.blend,
        Float32Array.from(p.color),
        p.pressureTarget,
      ),
    brushExtend: (x, y, pressure) => wasm.brush_stroke_extend(x, y, pressure),
    async brushCommit() {
      const h = await wasm.brush_stroke_commit();
      const info = {
        handle: h.handle,
        width: h.width,
        height: h.height,
        display: displayTreatmentOf(h.display),
      };
      h.free();
      return info;
    },
    brushCancel: () => wasm.brush_stroke_cancel(),
    brushActive: () => wasm.brush_stroke_active(),
    brushStats() {
      // Rust returns [dabs, x, y, w, h], or an EMPTY vector before the
      // first dab lands on the canvas.
      const s = wasm.brush_stroke_stats();
      if (s.length < 5) return null;
      return { dabs: s[0], x: s[1], y: s[2], w: s[3], h: s[4] };
    },
    brushBlendModes: () =>
      wasm
        .brush_blend_modes()
        .split("\n")
        .filter((n) => n.length > 0),
    layersOpen: (handle) => wasm.layers_open(handle),
    layersOpenFromPsd: (imageHandle, psdHandle) =>
      wasm.layers_open_from_psd(imageHandle, psdHandle),
    layersClose: () => wasm.layers_close(),
    layersBound: () => wasm.layers_bound(),
    layers() {
      const parsed = JSON.parse(wasm.layers_list()) as LayerStackInfo;
      return parsed.layers.length > 0 ? parsed : EMPTY_LAYER_STACK;
    },
    layersHistory() {
      // The engine answers the JSON literal `null` when no stack is open.
      return JSON.parse(wasm.layers_history()) as LayerHistory | null;
    },
    layerAdd: (name) => wasm.layers_add(name),
    layerDuplicate: (index) => wasm.layers_duplicate(index),
    layerRemove: (index) => wasm.layers_remove(index),
    layerReorder: (from, to) => wasm.layers_reorder(from, to),
    layerSetActive: (index) => wasm.layers_set_active(index),
    layerSetVisible: (index, visible) =>
      wasm.layers_set_visible(index, visible),
    layerSetLocked: (index, locked) => wasm.layers_set_locked(index, locked),
    layerSetOpacity: (index, opacity) =>
      wasm.layers_set_opacity(index, opacity),
    layerSetName: (index, name) => wasm.layers_set_name(index, name),
    layerSetBlend: (index, blend) => wasm.layers_set_blend(index, blend),
    layerMakeSmart: (index) => wasm.layers_make_smart(index),
    layerRenderSmart: (index, scale) => wasm.layers_render_smart(index, scale),
    layerMaskFromSelection: (index) => wasm.layers_mask_from_selection(index),
    layerClearMask: (index) => wasm.layers_clear_mask(index),
    layerSetMaskEnabled: (index, enabled) =>
      wasm.layers_set_mask_enabled(index, enabled),
    layersComposite: () => wasm.layers_composite(),
    layersAddAdjustment: (name, p) =>
      // The SAME wire block the bake and the preview use — one decode,
      // one meaning. The only difference is where the chain LIVES.
      wasm.layers_add_adjustment(
        name,
        p.exposureEv,
        p.brightness,
        p.contrast,
        p.saturation,
        p.temp,
        p.tint,
        p.levels.inBlack,
        p.levels.inWhite,
        p.levels.gamma,
        p.levels.outBlack,
        p.levels.outWhite,
        p.curveLut ?? new Uint8Array(0),
        p.blurSigma,
        p.sharpenAmount,
        p.hueDegrees,
        p.invert,
        packAdjustExt(p),
      ),
    layersBakeAdjust: (p) =>
      // The SAME wire the preview chain uses (`adjust_image_ext` minus
      // the handle) — one decode of the block, one meaning.
      wasm.layers_bake_adjust(
        p.exposureEv,
        p.brightness,
        p.contrast,
        p.saturation,
        p.temp,
        p.tint,
        p.levels.inBlack,
        p.levels.inWhite,
        p.levels.gamma,
        p.levels.outBlack,
        p.levels.outWhite,
        p.curveLut ?? new Uint8Array(0),
        p.blurSigma,
        p.sharpenAmount,
        p.hueDegrees,
        p.invert,
        packAdjustExt(p),
      ),
    layersUndo: () => wasm.layers_undo(),
    layersRedo: () => wasm.layers_redo(),
  };
}

// ------------------------------------------------------------- the boot

export const ENGINE_NOT_BUILT =
  "paged.image engine wasm not built — run scripts/build-wasm.sh " +
  "(manifest/wasm/image_js.js missing)";

function isNode(): boolean {
  return (
    typeof process !== "undefined" &&
    process.versions?.node != null &&
    typeof (globalThis as { window?: unknown }).window === "undefined"
  );
}

/** Load + instantiate the engine wasm (the glue + the `_bg.wasm`),
 *  branching browser vs Node exactly like plugin-sheets' loadModule.
 *  Rejects with ENGINE_NOT_BUILT-flavoured detail when absent. */
async function loadModule(): Promise<ImageWasmModule> {
  let mod: ImageWasmModule;
  try {
    // @ts-ignore — the artifact (manifest/wasm/image_js.js, wasm-bindgen
    // --target web glue) is produced by scripts/build-wasm.sh and is
    // intentionally absent from the source tree; the dynamic import
    // resolves at runtime once built. Typed via ImageWasmModule.
    mod = (await import("../wasm/image_js.js")) as ImageWasmModule;
  } catch (cause) {
    throw new Error(ENGINE_NOT_BUILT, { cause });
  }

  if (isNode()) {
    const { readFile } = await import("node:fs/promises");
    const { fileURLToPath } = await import("node:url");
    const { createRequire } = await import("node:module");
    // Resolve through the manifest package's exports map (the artifact
    // lives in a SIBLING workspace package, unlike sheets' ../bin).
    const require = createRequire(import.meta.url);
    const wasmPath = require.resolve("../wasm/image_js_bg.wasm");
    const bytes = await readFile(
      wasmPath.startsWith("file:") ? fileURLToPath(wasmPath) : wasmPath,
    );
    mod.initSync({
      module: new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
    });
  } else {
    // Browser path: resolve the artifact through the bundler's explicit
    // `?url` import (the editor's wasm-loading convention; a bare
    // relative URL would resolve against the served module path and get
    // the dev server's HTML fallback — the "expected magic word" trap).
    // @ts-ignore — `?url` is a bundler affordance, untyped.
    const wasmUrl = (await import(
      // @ts-ignore — see above.
      "../wasm/image_js_bg.wasm?url"
    )) as { default: string };
    await mod.default({ module_or_path: wasmUrl.default });
  }
  return mod;
}

/** Load + boot the engine wasm, returning the facade. Rejects with
 *  ENGINE_NOT_BUILT-flavoured detail when the artifact is missing so
 *  the panel can surface the honest "not built" state. */
export async function bootEngine(): Promise<ImageEngine> {
  return wrapEngine(await loadModule());
}
