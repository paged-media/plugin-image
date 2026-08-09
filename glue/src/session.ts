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

// The M4 ingest-slice session: select a placed image frame → read its
// ORIGINAL bytes (C-5 host.assets.getPlacedImage) → decode in the
// engine wasm (codec/PSD CPU lanes) → run the committed adjustments
// (Engine A, GPU-only) → composite the RGBA8 result back IN-FRAME via
// the C-1 Stage-A image scene item (host.contribute.sceneLayer).
//
// Stage-A contract honesty: re-submission happens on COMMITTED changes
// (the panel's Apply), never per-drag — the retained-image lane is
// static quality by design (the interactive path is Stage B / M2). The
// layer clears on deselect of the composited frame and on Reset; the
// DOCUMENT is never mutated (the original placed bytes stay the truth —
// adjusted-pixel save-back is a later milestone, stated in the panel).

import type {
  BundleHost,
  Disposable,
  ElementId,
} from "@paged-media/plugin-api";

import {
  bootEngine,
  DEFAULT_BRUSH_PARAMS,
  EMPTY_LAYER_STACK,
  freshIdentityParams,
  isIdentity,
  type AdjustParams,
  type BrushLibraryInfo,
  type ChannelStatsInfo,
  type BrushParams,
  type BrushStats,
  type DecodedInfo,
  type DisplayTreatment,
  type WarpKindName,
  displayTreatmentOf,
  type GradientKind,
  type ImageEngine,
  type ImageHistogram,
  type LayerHistory,
  type LayerStackInfo,
  type PsdLayerInfo,
  type RasterFormat,
  type ResampleFilter,
  type Rgba01,
  SAMPLING_TOOLS,
  type LevelsParams,
  type SelectionStats,
  type StrokeTool,
} from "./engine";
import {
  imageToPage,
  pageToImage,
  resolveFrameFit,
} from "./frame-fit";
import { claimImageTiles } from "./tile-provider";
import { createDecodePool, type DecodePool } from "./decode-pool";
import { createCropMachine, type CropMachine } from "./crop-machine";
import {
  createSelectionMachine,
  type SelectionMachine,
} from "./selection-machine";
import {
  createBrushMachine,
  type BrushMachine,
  type BrushSample,
} from "./brush-machine";

/** The fixed v0 feather sigma (px) the `featherSelection` command uses
 *  when the caller passes none (a slider is a follow-up). */
export const FEATHER_SIGMA_DEFAULT = 4;

/** The fixed v0 noise seed the `fillSelection` command uses when the
 *  caller passes none (a seed field is a follow-up). */
export const FILL_NOISE_SEED_DEFAULT = 1;

/** What to paint into the selection (the `gen.*` family's editor
 *  reach). Colours are straight RGBA in [0,1]. */
export type FillRequest =
  | { kind: "gradient"; gradient: GradientKind; c0: Rgba01; c1: Rgba01 }
  | { kind: "noise"; amount: number; seed?: number }
  /** CONTENT-AWARE: synthesise the selection from the rest of the image.
   *  The only fill that is a SEARCH rather than a generator, and the
   *  only one that needs no parameters — the image is the parameter. */
  | { kind: "contentAware" };

/** Save-back bytes staged by `applyToFile` — what the exporters hand
 *  out and what the panel reports. */
export interface SaveBackResult {
  bytes: Uint8Array;
  fileName: string;
  mimeType: string;
  /** The HONEST one-liner (which lane ran, what it did to the layer
   *  structure). Shown in the panel verbatim. */
  note: string;
}

/** The ingested source image (engine-held pixels behind `handle`). */
export interface SourceImage {
  /** Display name — the resolved link URI or the imported file name. */
  name: string;
  width: number;
  height: number;
  handle: number;
  origin: "selection" | "import";
  /** The frame to composite into (null for an import until Apply
   *  targets the current selection). */
  elementId: string | null;
  /** CMS rung 1 — how the ingest lane treated this source's colour.
   *  Reported in the panel because "sRGB assumed" is a real state a
   *  colour-critical user needs to see, not a default to hide. */
  display: DisplayTreatment;
  /** The source was 16 bits per channel and was reduced to 8 at ingest.
   *  A stated lossy step, not a hidden one — and a file that opens,
   *  where a 16-bit PSD used to be refused outright. */
  depthReduced: boolean;
}

export type EngineStatus = "idle" | "booting" | "ready" | "unavailable";

export interface ImageSessionState {
  engine: EngineStatus;
  /** The honest boot/GPU detail when something is off. */
  engineDetail: string | null;
  /** WebGPU device acquired (kernels are GPU-only; false ⇒ only
   *  identity composites work). */
  gpu: boolean;
  source: SourceImage | null;
  params: AdjustParams;
  /** The active image's RGB + luma histogram (the levels/curves panel
   *  readout), or null until an image is ingested. Recomputed on ingest
   *  and after a crop commit (it follows the engine-held pixels). */
  histogram: ImageHistogram | null;
  /** A scene layer is currently submitted for `compositedFrame`. */
  compositedFrame: string | null;
  busy: boolean;
  /** One-line panel status (honest, never fake-progress). */
  status: string;
  /** PSD structural session (the mutatable tier): present when the
   *  imported source is a PSD/PSB. Edits accumulate on the retained
   *  parse; `psdExport` re-emits with full preservation. The canvas
   *  composite stays the import-time flatten (re-flatten after record
   *  edits is a follow-up). */
  psd: { name: string; layers: PsdLayerInfo[] } | null;
  /** The active SELECTION readout (engine-side coverage; §6.1), or null
   *  when no explicit selection exists — adjustments then apply to the
   *  whole image. When set, the committed Apply masks every GPU
   *  adjust/filter dispatch (and the CPU curves pass) by the coverage. */
  selection: SelectionStats | null;
  /** The SAVE-BACK bytes staged by `applyToFile` (null until asked
   *  for). The panel reports them; the Export Center delivers them —
   *  the host wires no save-FILE door (`shell.pickFile` reads, it does
   *  not write), so the exporter registry is the whole delivery lane. */
  saveBack: SaveBackResult | null;
  /** The brush/pencil/eraser parameters, FROZEN into each stroke at
   *  `brushBegin` (the engine refuses a stroke whose size changes
   *  mid-drag — it would not be replayable). */
  brush: BrushParams;
  /** Every blend mode a stroke can paint through, read from the engine's
   *  `compose.*` registry at boot (empty until the engine is up — the
   *  panel's picker is never a hardcoded list). */
  blendModes: string[];
  /** A stroke is in progress (the tools' pointer-down → up window). */
  strokeActive: boolean;
  /** The in-flight stroke's dab count + bounds, or null before the
   *  first dab lands on the canvas. */
  strokeStats: BrushStats | null;
  /** THE LAYER GRAPH — the engine's layer stack for the ingested image,
   *  BOTTOM-first (the panel renders it reversed, as a layers palette
   *  reads). Opened automatically on ingest, so painting always lands in
   *  a layer rather than in the one flat image. */
  layers: LayerStackInfo;
  /** The undo readout — depth, redo depth, and the BOUND the journal
   *  holds to. Null until a stack is open. */
  history: LayerHistory | null;
  /** How the stack was opened, in the panel's words: the PSD's own N
   *  layers, or the honest reason the layered import was DECLINED and
   *  the flattened composite kept. Null for non-PSD sources. */
  layersNote: string | null;
  /** RASTER TYPE settings. Session state rather than tool state: the
   *  string and face survive between clicks, which is what makes setting
   *  several runs of the same type usable. */
  type: {
    text: string;
    family: string;
    sizePx: number;
    /** Letter spacing in 1/1000 em — IDML's unit, and UNITLESS, so it
     *  crosses the px/pt boundary unchanged (unlike size and leading). */
    trackingPerMille: number;
    /** Baseline-to-baseline distance in IMAGE PX, or null for AUTO —
     *  the face's own line height. Auto is not `1.2 × size`: a face
     *  carries its own ascent/descent/gap, and honouring them is what
     *  makes two faces at one size lead correctly. */
    leadingPx: number | null;
  };
  /** POINTS PER IMAGE PIXEL for the frame this session last composited
   *  into, or null before the first composite.
   *
   *  It is CACHED here rather than computed on demand, and that is the
   *  point: `submitLayer` already derives this exact number to lay the
   *  image out (`min(box.w/width, box.h/height)`), so reading the cache
   *  guarantees a converted size agrees with the pixels actually on
   *  screen. Recomputing would let the two drift the moment a frame is
   *  resized between a composite and a panel read.
   *
   *  Null is meaningful: nothing has been laid out, so no conversion is
   *  possible and callers must say so rather than assume 1. */
  ptPerPx: number | null;
  /** The clone/heal source anchor in IMAGE px, or null before one is
   *  set. Session state rather than stroke state: the anchor SURVIVES a
   *  stroke, which is what makes repeated retouching strokes usable. */
  cloneSource: { x: number; y: number; aligned: boolean } | null;
  /** The per-channel readout for the ingested source (R/G/B/A + luma),
   *  or null before an ingest. Refreshed alongside the histogram, from
   *  the same buffer, so the two never disagree. */
  channels: ChannelStatsInfo[] | null;
  /** A loaded `.abr` brush library, or null before one is opened. The
   *  presets are PARAMETERS, not pixels — loading one changes nothing
   *  until the designer applies a row. */
  brushLibrary: BrushLibraryInfo | null;
  /** The library file's name, for the panel's header. */
  brushLibraryName: string | null;
  /** The applied preset's index, so the panel can mark the active row.
   *  Cleared the moment a slider moves — the brush no longer IS the
   *  preset once it is edited, and pretending otherwise would mislabel
   *  a stroke's provenance. */
  brushPreset: number | null;
}

export interface ImageSession {
  state(): ImageSessionState;
  onDidChange(listener: () => void): Disposable;
  /** Ingest the single selected element's placed image via C-5. */
  ingestSelection(): Promise<boolean>;
  /** Ingest opened/dropped file bytes (the K-2 importer path). */
  importBytes(name: string, bytes: Uint8Array): Promise<boolean>;
  /** RESAMPLE the engine-held source to a new pixel size (GPU-only —
   *  rejects honestly without a device). Swaps the source like a crop
   *  commit: document unchanged until Apply re-composites. */
  resizeTo(w: number, h: number, filter: ResampleFilter): Promise<boolean>;
  /** PSD layer-record edits (only when `state().psd` is set). Each
   *  applies to the retained parse and refreshes the layer list. */
  psdSetLayerOpacity(index: number, opacity: number): boolean;
  psdRenameLayer(index: number, name: string): boolean;
  psdRemoveLayer(index: number): boolean;
  /** The edited PSD, preservation-safe (zero-edit ⇒ byte-identical).
   *  Null when no PSD is loaded / the engine is gone. */
  psdExport(): { bytes: Uint8Array; fileName: string } | null;
  /** Hand the STAGED save-back bytes to the host's saver (K-10). Stages
   *  them first when nothing is staged yet, so the panel button is one
   *  action rather than two.
   *
   *  Resolves false when no saver is wired, when there is nothing to
   *  save, or when the host refused. HONEST CEILING, inherited from the
   *  door: a host backed by the browser's anchor-download cannot observe
   *  a user cancel, so `true` means "handed to the download path", not
   *  "a file exists on disk". The status text says which of those the
   *  answer is. */
  saveToFile(): Promise<boolean>;
  /** SAVE-BACK: composite the current adjustments at FULL resolution and
   *  bake them into the source file's bytes.
   *
   *  * PSD source → `replace_channel_pixels` on the single canvas-sized
   *    content layer when there is one, else the composite is FLATTENED
   *    into a new single-layer PSD (announced in `note`, never silent);
   *    then `psd_save` (full carry-through preservation).
   *  * PNG/JPEG source → re-encode through `image-codecs` (PNG by
   *    default; JPEG when the ingested source was a JPEG, fixed v0
   *    quality).
   *
   *  Stages the result on `state().saveBack` and returns it; null (with
   *  an honest status) when there is nothing to save or the lane is
   *  unsupported. The DOCUMENT is untouched either way — this produces
   *  bytes, it does not write files (the host wires no save-file door). */
  applyToFile(): Promise<SaveBackResult | null>;
  /** What the PSD exporter hands out: the ADJUSTED save-back when the
   *  panel is off identity, else the preservation-safe re-emit
   *  (zero-edit ⇒ BYTE-IDENTICAL — the §10.4 invariant survives; a plain
   *  export never rewrites the composite). Null when no PSD is loaded. */
  psdExportBytes(): Promise<{ bytes: Uint8Array; fileName: string } | null>;
  /** What the PNG / JPEG exporters hand out: the adjusted result
   *  re-encoded in the REQUESTED format (whatever the source was). */
  rasterExportBytes(
    format: RasterFormat,
  ): Promise<{ bytes: Uint8Array; fileName: string } | null>;
  /** FILL the current selection (the whole image when none) with a
   *  generator, composited through the coverage mask. DESTRUCTIVE: it
   *  swaps the engine-held source like a crop commit (the document and
   *  the placed file are still untouched; re-ingest restores). */
  fillSelection(req: FillRequest): Promise<boolean>;
  /** Run a whole-image effect kernel and land it like a fill. */
  applyEffect(
    label: string,
    run: (handle: number) => Promise<DecodedInfo>,
  ): Promise<boolean>;
  /** Luminance through a two-stop colour ramp (adjust.gradient_map). */
  applyGradientMap(
    shadow: readonly [number, number, number],
    highlight: readonly [number, number, number],
  ): Promise<boolean>;
  /** A parametric distortion (geom.warp_backward). */
  applyWarp(
    kind: WarpKindName,
    amount: number,
    frequency?: number,
  ): Promise<boolean>;
  setParams(p: Partial<AdjustParams>): void;
  /** Set the composite levels (merged into params.levels). */
  setLevels(l: Partial<LevelsParams>): void;
  /** Build + set the curves tone LUT from `(input, output)` control points
   *  in [0,1] (an empty / identity set clears the curve). */
  setCurvePoints(points: Array<[number, number]>): void;
  /** Auto-enhance: derive levels (in black/white) + white balance
   *  (temp/tint) from the ingested image's histogram and set them on the
   *  params (PREVIEW-only — the user commits with Apply, like every other
   *  edit). A no-op estimate (flat/neutral image) leaves the result
   *  identity. */
  autoEnhance(): void;
  /** The crop interaction machine for the ingested image (null until an
   *  image is ingested + the engine is ready). The crop tool's gesture and
   *  the panel's crop controls drive it. */
  cropMachine(): CropMachine | null;
  /** The SELECTION interaction machine (null until an image is
   *  ingested). The marquee/lasso/wand tools drive it; the selection
   *  itself is engine state bound to the live source handle. */
  selectionMachine(): SelectionMachine | null;
  /** Re-read the engine's selection stats into `state().selection` and
   *  notify (the machines call this after every committed change). */
  refreshSelection(): void;
  /** Select the whole image (an explicit full-extent selection). */
  selectAll(): boolean;
  /** Deselect (back to "no selection" — adjust applies everywhere). */
  deselect(): boolean;
  /** Invert the selection ("everything" inverts to the empty selection). */
  invertSelection(): boolean;
  /** Gaussian-feather the selection edge (σ px; the fixed v0 default
   *  when omitted). False when there is no selection to feather. */
  featherSelection(sigma?: number): boolean;
  /** The PAINT interaction machine (null until an image is ingested).
   *  The brush / pencil / eraser tools drive it; it holds the pointer
   *  sampling only — the stroke's pixels are engine state. */
  brushMachine(): BrushMachine | null;
  /** Merge into the brush parameters (the panel's Brush section). Takes
   *  effect on the NEXT stroke — the in-flight one is frozen. */
  /** Turn a channel into the selection — `red`|`green`|`blue`|`alpha`|
   *  `luma`. Replaces the current selection (the panel's other combine
   *  modes belong to the marquee tools, not to a channel load). */
  selectionFromChannel(channel: string): boolean;
  /** SELECTION → PATH: trace the selection and insert the result as real
   *  vector polygons on the page, in DOCUMENT coordinates. The first
   *  document mutation this bundle makes — everything else it does lands
   *  in the engine or on the scene channel — so it is undoable through
   *  the host's own history, not the image journal. */
  selectionToPath(threshold?: number, tolerance?: number): Promise<boolean>;
  /** PATH → SELECTION: read the selected vector element's anchors through
   *  the host, flatten its curves, and make it the selection. */
  selectionFromPath(): Promise<boolean>;
  /** Update the type settings the tool reads on its next click. */
  setType(
    p: Partial<{
      text: string;
      family: string;
      sizePx: number;
      trackingPerMille: number;
      leadingPx: number | null;
    }>,
  ): void;
  /** RASTER TYPE: paint shaped, rasterized text into the active layer at
   *  the BASELINE origin `point` (image px).
   *
   *  The font comes from the HOST — `assets.getFontFace` serves bytes the
   *  document already embeds, never a network fetch — so a bundle
   *  renders exactly the faces the document has and stays offline by
   *  construction. */
  paintText(
    point: [number, number],
    text: string,
    family: string,
    sizePx: number,
  ): Promise<boolean>;
  /** Set the clone/heal anchor (the alt-click). Image px. */
  setCloneSource(point: [number, number]): void;
  /** Aligned: the source keeps a fixed offset from the cursor and tracks
   *  the brush. Unaligned restarts from the anchor on every stroke,
   *  which is how a motif is stamped repeatedly. */
  setCloneAligned(aligned: boolean): void;
  setBrushParams(p: Partial<BrushParams>): void;
  /** Load a Photoshop `.abr` brush library. Parses only — nothing about
   *  the current brush changes until `applyBrushPreset`. Resolves false
   *  (with the reader's own reason in `status`) when the bytes are not
   *  an `.abr`.
   *
   *  ASYNC because it BOOTS the engine if needed: presets are
   *  parameters, so a designer can pick a brush library before there is
   *  any image to paint on, and refusing until after an ingest would be
   *  an implementation detail leaking into the workflow. */
  loadBrushLibrary(name: string, bytes: Uint8Array): Promise<boolean>;
  /** Open the host picker for a `.abr` and load what comes back. Resolves
   *  false when the user cancels OR no picker is wired — `pickFile`
   *  answers `[]` for both, so the two are reported apart by probing
   *  `shell.pickFile@1`. */
  pickBrushLibrary(): Promise<boolean>;
  /** Adopt a preset's parameters into the live brush. Applies ONLY what
   *  the engine's tip has — size, hardness, spacing — and says in
   *  `status` what the preset carried that could not be applied
   *  (roundness/angle: the tip is circular; a non-pixel diameter). */
  applyBrushPreset(index: number): boolean;
  /** Forget the loaded library (the panel's Close). */
  closeBrushLibrary(): void;
  /** OPEN a stroke on the engine-held source with the current brush
   *  params + the bound selection (both frozen for its duration).
   *  GPU-only: false with an honest status when there is no device,
   *  nothing ingested, or the engine refuses. */
  brushBegin(tool: StrokeTool): Promise<boolean>;
  /** Feed one pointer sample and re-submit the in-frame preview from the
   *  painted pixels. False when no stroke is open. */
  brushExtend(sample: BrushSample): Promise<boolean>;
  /** CLOSE the stroke: the painted pixels become a NEW engine-held
   *  source (the crop / fill commit pattern — DESTRUCTIVE into the
   *  engine's working image; the document and the source file are
   *  untouched, and re-ingesting the frame is the only restore this
   *  plugin owns). Carries the selection over (the result is the same
   *  size) and re-composites through the adjust chain. */
  brushCommit(): Promise<boolean>;
  /** ABANDON the stroke — the engine-held source was never mutated, so
   *  this restores it exactly (and re-composites the frame). */
  brushCancel(): void;
  // ── THE LAYER GRAPH ───────────────────────────────────────────────
  /** Add an empty transparent layer above the active one (it becomes
   *  active) and re-composite. */
  addLayer(name?: string): Promise<boolean>;
  duplicateLayer(index: number): Promise<boolean>;
  /** Remove a layer. Refused for the last one; NOT undoable (the
   *  journal is a pixel log — stated in the panel). */
  removeLayer(index: number): Promise<boolean>;
  /** Move a layer in stack order (0 = bottom). */
  reorderLayer(from: number, to: number): Promise<boolean>;
  /** Choose the layer that paint / fill / bake land in. */
  setActiveLayer(index: number): boolean;
  setLayerVisible(index: number, visible: boolean): Promise<boolean>;
  setLayerOpacity(index: number, opacity: number): Promise<boolean>;
  setLayerBlend(index: number, blend: string): Promise<boolean>;
  setLayerLocked(index: number, locked: boolean): boolean;
  /** Make the current selection this layer's mask, then re-composite —
   *  a mask changes what the page shows, unlike a lock. */
  layerMaskFromSelection(index: number): Promise<boolean>;
  /** Toggle whether the mask applies; the coverage is retained. */
  setLayerMaskEnabled(index: number, enabled: boolean): Promise<boolean>;
  /** Clip a layer to the one beneath it (or release it). Confines it to
   *  the base's ALPHA, multiplied with any mask it already has. */
  setLayerClipped(index: number, clipped: boolean): Promise<boolean>;
  /** Group the CONTIGUOUS run `from..=to`. */
  groupLayers(from: number, to: number, name?: string): Promise<boolean>;
  /** Dissolve a group; its layers stay. */
  ungroupLayers(id: number): Promise<boolean>;
  setGroupVisible(id: number, visible: boolean): Promise<boolean>;
  setGroupOpacity(id: number, opacity: number): Promise<boolean>;
  setGroupName(id: number, name: string): Promise<boolean>;
  setGroupBlend(id: number, blend: string): Promise<boolean>;
  /** Pass-through (members reach the stack below) vs isolated. */
  setGroupPassThrough(id: number, passThrough: boolean): Promise<boolean>;
  /** Delete the mask outright. */
  clearLayerMask(index: number): Promise<boolean>;
  /** Convert a pixel layer into a smart object (one-way — the source is
   *  preserved and going back would discard it). */
  makeLayerSmart(index: number): Promise<boolean>;
  /** Re-render a smart object at `scale` from its preserved source. */
  renderLayerSmart(index: number, scale: number): Promise<boolean>;
  setLayerName(index: number, name: string): boolean;
  /** BAKE the panel's adjustment chain destructively into the ACTIVE
   *  layer (journaled, so undoable). The chain is otherwise a
   *  re-runnable preview that never mutates a pixel. */
  bakeAdjustToLayer(): Promise<boolean>;
  /** Stack the panel's chain as an ADJUSTMENT LAYER — the
   *  non-destructive counterpart of {@link bakeAdjustToLayer}. No pixel
   *  is written, so deleting the layer restores the original exactly. */
  addAdjustmentLayer(): Promise<boolean>;
  /** Undo / redo the newest journaled PIXEL edit (paint, fill, bake).
   *  Layer STRUCTURE changes are not journaled. */
  undo(): Promise<boolean>;
  /** Walk BACK `n` journal steps — the History panel's "jump to here".
   *  The journal is a stack, not a random-access log, so this is n
   *  undos; holding every intermediate state to make it O(1) would cost
   *  exactly what the byte budget exists to bound. */
  undoSteps(n: number): Promise<boolean>;
  /** Walk FORWARD `n` journal steps. */
  redoSteps(n: number): Promise<boolean>;
  redo(): Promise<boolean>;
  /** Commit the crop: cut the machine's rect out of the source image, swap
   *  the engine-held source to the cropped result, recompute the
   *  histogram, and re-composite in-frame. Returns false when there is
   *  nothing to crop or the rect is empty. */
  commitCrop(): Promise<boolean>;
  /** COMMITTED apply: adjust on the GPU + submit the in-frame layer. */
  apply(): Promise<boolean>;
  /** C-6 — claim the ingested image's tile resource so the renderer
   *  pulls level-0 tiles for it (the v44 wire). Returns false when there
   *  is nothing ingested, no target frame, or the host wires no resource
   *  channel. Disposing the session (or re-ingesting) releases the claim. */
  claimTiles(): boolean;
  /** True while a tile resource is claimed (the panel reflects it). */
  tilesClaimed(): boolean;
  /** Clear the layer + return to identity params. */
  reset(): Promise<void>;
  dispose(): void;
}

/** The raw id string of an `ElementId`-ish value (wire ids carry a
 *  string `id`; tolerate a plain string). Structural — no wire import. */
export function elementIdOf(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (typeof value === "object" && value !== null) {
    const e = value as { id?: unknown };
    if (typeof e.id === "string") return e.id;
  }
  return null;
}

/** Subdivisions per cubic segment when flattening a host path into the
 *  pixel mask. The target is a per-pixel coverage, so more precision
 *  than this buys nothing measurable and the cost stays bounded. */
const CURVE_STEPS = 16;

/** A cubic bezier at `t`. */
function cubicAt(
  p0: readonly [number, number],
  p1: readonly [number, number],
  p2: readonly [number, number],
  p3: readonly [number, number],
  t: number,
): [number, number] {
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const c = 3 * u * t * t;
  const d = t * t * t;
  return [
    a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
  ];
}

export function createImageSession(host: BundleHost): ImageSession {
  const listeners = new Set<() => void>();
  let engine: ImageEngine | null = null;
  let bootPromise: Promise<ImageEngine | null> | null = null;
  let sceneSurface: ReturnType<typeof host.contribute.sceneLayer> | null = null;
  // C-6 — the active tile-resource claim (null when nothing is claimed).
  let tileClaim: { elementId: string; dispose(): void; bump(): void } | null =
    null;
  // K-3 — the decode worker pool (null when the host wires no workers /
  // grants none → the session decodes on the main thread instead).
  let decodePool: DecodePool | null = null;
  let decodePoolPromise: Promise<DecodePool | null> | null = null;
  // The crop interaction machine for the live source (rebuilt on ingest).
  let cropMachineRef: CropMachine | null = null;
  // The selection interaction machine (rebuilt alongside the crop
  // machine; the selection itself is ENGINE state keyed to the handle).
  let selectionMachineRef: SelectionMachine | null = null;
  // The paint interaction machine (pointer sampling only — the stroke's
  // base snapshot + coverage live engine-side).
  let brushMachineRef: BrushMachine | null = null;
  // The frame the IN-FLIGHT stroke previews into, resolved once at
  // begin: per-sample geometry reads would put an async round-trip in
  // the middle of a drag.
  let strokeTarget: string | null = null;
  let strokeBox: { w: number; h: number } | null = null;
  let disposed = false;

  const state: ImageSessionState = {
    engine: "idle",
    engineDetail: null,
    gpu: false,
    source: null,
    params: freshIdentityParams(),
    histogram: null,
    compositedFrame: null,
    busy: false,
    status: "Select a placed image frame, then ingest.",
    psd: null,
    selection: null,
    saveBack: null,
    brush: { ...DEFAULT_BRUSH_PARAMS, color: [...DEFAULT_BRUSH_PARAMS.color] },
    blendModes: [],
    strokeActive: false,
    strokeStats: null,
    layers: EMPTY_LAYER_STACK,
    history: null,
    layersNote: null,
    type: {
      text: "",
      family: "Helvetica",
      sizePx: 48,
      trackingPerMille: 0,
      leadingPx: null,
    },
    ptPerPx: null,
    cloneSource: null,
    channels: null,
    brushLibrary: null,
    brushLibraryName: null,
    brushPreset: null,
  };
  /** The retained PSD parse handle (wasm-side), when state.psd is set. */
  let psdHandle: number | null = null;
  /** The ingested source's container, for the PNG/JPEG save-back lane
   *  ("jpeg" only when the ORIGINAL bytes were a JPEG — a re-encode
   *  never invents a lossy format). */
  let sourceFormat: RasterFormat | "psd" | null = null;

  const emit = () => {
    for (const l of [...listeners]) l();
  };
  const setStatus = (s: string) => {
    state.status = s;
    emit();
  };

  // C-1 — the scene channel (lazy; warns once via supports()).
  const scene = () => {
    if (!host.supports("rendering.sceneLayer@1")) return null;
    if (!sceneSurface) sceneSurface = host.contribute.sceneLayer();
    return sceneSurface;
  };

  /** Boot the engine + GPU once, on first need. */
  const ensureEngine = async (): Promise<ImageEngine | null> => {
    if (engine) return engine;
    if (!bootPromise) {
      state.engine = "booting";
      emit();
      bootPromise = (async () => {
        try {
          const e = await bootEngine();
          const gpu = await e.initGpu();
          if (disposed) return null;
          engine = e;
          state.engine = "ready";
          state.gpu = gpu;
          try {
            // The blend picker's options come from the kernel registry,
            // never from a list this side could let drift.
            state.blendModes = e.brushBlendModes();
          } catch (err) {
            host.log.debug("blend-mode registry read failed", err);
            state.blendModes = [];
          }
          state.engineDetail = gpu
            ? null
            : "WebGPU unavailable — adjustments disabled (kernels are " +
              "GPU-only; identity composite still works)";
          emit();
          return e;
        } catch (err) {
          state.engine = "unavailable";
          state.engineDetail = err instanceof Error ? err.message : String(err);
          emit();
          return null;
        }
      })();
    }
    return bootPromise;
  };

  // K-3 — boot the decode worker pool once, on first ingest. The pool is
  // an OPTIONAL accelerator: when the host wires no workers (or grants
  // none) createDecodePool returns null and decode falls back to the
  // main-thread engine — same pixels, just on-thread (the honest
  // degradation; never fake parallelism).
  const ensureDecodePool = async (): Promise<DecodePool | null> => {
    if (decodePool) return decodePool;
    if (!decodePoolPromise) {
      decodePoolPromise = createDecodePool(host).then((pool) => {
        if (disposed) {
          pool?.dispose();
          return null;
        }
        decodePool = pool;
        return pool;
      });
    }
    return decodePoolPromise;
  };

  /** Decode bytes to an engine-held handle. K-3 fast path: when the worker
   *  pool is available, the codec/PSD CPU decode runs OFF the main thread
   *  and the raw RGBA is registered into the engine here; otherwise the
   *  engine decodes on the main thread. Both yield the same handle the
   *  adjust + tile paths consume. */
  const decodeToHandle = async (bytes: Uint8Array): Promise<DecodedInfo> => {
    if (!engine) throw new Error("engine not booted");
    const pool = await ensureDecodePool();
    if (pool) {
      // The pool transfers the input buffer — copy first so the caller's
      // bytes survive (the importer may reuse them).
      const copy = bytes.slice();
      const decoded = await pool.decode(copy);
      const info = engine.ingestRgba8(
        decoded.width,
        decoded.height,
        decoded.rgba,
      );
      // The worker already ran the display transform inside its own
      // `decode_image`; `ingestRgba8` takes raw pixels and so reports
      // "sRGB assumed". Restore the treatment the worker measured, or the
      // panel would understate what happened on the K-3 fast path.
      return { ...info, display: displayTreatmentOf(decoded.display) };
    }
    return engine.decode(bytes);
  };

  const releaseTiles = () => {
    if (tileClaim) {
      tileClaim.dispose();
      tileClaim = null;
    }
  };

  /** The pixels behind the live handle changed IN PLACE (a layer edit
   *  re-composites into the same handle), so the renderer's cached tiles
   *  are stale — bump the claim's revision to make it re-pull. */
  const bumpTiles = () => {
    tileClaim?.bump();
  };

  /** Re-read the engine's layer stack + undo readout into state. */
  const refreshLayers = () => {
    if (!engine || !state.source) {
      state.layers = EMPTY_LAYER_STACK;
      state.history = null;
      return;
    }
    try {
      state.layers = engine.layers();
      state.history = engine.layersHistory();
    } catch (err) {
      host.log.debug("layer readout failed", err);
      state.layers = EMPTY_LAYER_STACK;
      state.history = null;
    }
  };

  /** Fold the stack into the bound engine-held image, re-read the
   *  readouts, and put the adjusted composite back in-frame. Every layer
   *  mutation ends here so the canvas and the panel can never disagree.
   *  A one-layer stack short-circuits engine-side (no GPU, no dispatch). */
  const recomposite = async (): Promise<boolean> => {
    if (!engine || !state.source) return false;
    try {
      await engine.layersComposite();
    } catch (err) {
      // The stack DID change even though the fold could not run (no
      // device); show the truth rather than a stale palette.
      refreshLayers();
      setStatus(
        `Composite failed: ${err instanceof Error ? err.message : err}`,
      );
      emit();
      return false;
    }
    bumpTiles();
    refreshLayers();
    refreshHistogram();
    if (state.source.elementId) await api.apply();
    else emit();
    return true;
  };

  /** Adopt a commit's result. When the engine handed back the SAME
   *  handle it edited in place (the layer lane) — the pixels changed but
   *  the handle did not, so freeing it would destroy the live document
   *  and the tile claim only needs a revision bump. A DIFFERENT handle
   *  is the pre-layer destructive commit: release the claim, free the
   *  old pixels, adopt the new. */
  const swapSource = (
    src: SourceImage,
    next: { handle: number; width: number; height: number },
  ) => {
    if (next.handle === src.handle) {
      bumpTiles();
    } else {
      releaseTiles();
      engine?.freeImage(src.handle);
      src.handle = next.handle;
    }
    src.width = next.width;
    src.height = next.height;
  };

  /** The layer-mutation sandwich: run `mutate` engine-side, then
   *  re-composite. A refusal (locked layer, last layer, unknown blend)
   *  becomes the panel's status verbatim — the engine owns the reason. */
  const layerOp = async (
    what: string,
    mutate: () => void,
  ): Promise<boolean> => {
    if (!engine || !state.source) {
      setStatus("Nothing ingested — ingest a placed image first.");
      return false;
    }
    try {
      mutate();
    } catch (err) {
      setStatus(`${what} failed: ${err instanceof Error ? err.message : err}`);
      refreshLayers();
      emit();
      return false;
    }
    return recomposite();
  };

  /** After a mask edit: the stack changed what it SHOWS, so re-read the
   *  rows and re-composite. A mask is unlike a lock in exactly this way —
   *  it changes pixels on the page without changing any layer's pixels. */
  const finishMaskEdit = async (): Promise<boolean> => {
    refreshLayers();
    return recomposite();
  };

  /** One shape for the four group-property setters: they differ only in
   *  which door they call, and duplicating the guard and the recomposite
   *  four times is how they drift apart. */
  const groupEdit = async (run: () => void): Promise<boolean> => {
    if (!engine || !state.source) return false;
    try {
      run();
    } catch (err) {
      setStatus(`Group edit failed: ${err instanceof Error ? err.message : err}`);
      return false;
    }
    return finishMaskEdit();
  };

  /** Undo/redo one journaled PIXEL edit. An empty label means the
   *  journal had nothing — said plainly rather than as a silent no-op,
   *  since a bounded journal CAN run out earlier than the user expects. */
  const historyStep = async (undo: boolean): Promise<boolean> => {
    if (!engine || !state.source) {
      setStatus("Nothing ingested — ingest a placed image first.");
      return false;
    }
    // An in-flight stroke holds a base snapshot of the active layer;
    // undoing under it would commit into pixels that no longer exist.
    discardStroke();
    let label: string;
    try {
      label = undo ? await engine.layersUndo() : await engine.layersRedo();
    } catch (err) {
      setStatus(
        `${undo ? "Undo" : "Redo"} failed: ${err instanceof Error ? err.message : err}`,
      );
      return false;
    }
    if (!label) {
      const h = state.history;
      setStatus(
        `Nothing to ${undo ? "undo" : "redo"}.` +
          (undo && h?.dropped
            ? ` ${h.dropped} older edit${h.dropped === 1 ? " is" : "s are"} past the ` +
              "undo budget and permanent."
            : ""),
      );
      emit();
      return false;
    }
    const ok = await recomposite();
    if (ok) setStatus(`${undo ? "Undid" : "Redid"} “${label}”.`);
    return ok;
  };

  /** Drop an in-flight stroke engine-side (the base pixels were never
   *  mutated, so this is a true restore). Shared by the explicit cancel
   *  and by every path that pulls the source out from under a stroke. */
  const discardStroke = () => {
    if (engine && state.strokeActive) {
      try {
        engine.brushCancel();
      } catch (err) {
        host.log.debug("stroke cancel failed", err);
      }
    }
    state.strokeActive = false;
    state.strokeStats = null;
    strokeTarget = null;
    strokeBox = null;
    brushMachineRef?.cancel();
  };

  const freeSource = () => {
    // A stroke holds the OLD handle's base snapshot — a re-ingest under
    // one would commit into pixels that no longer exist.
    discardStroke();
    // A claim points at THIS source's handle — release it before the
    // pixels go (the renderer drops to the whole-image fallback lane).
    releaseTiles();
    // The stack composites INTO the source handle, so it goes with it
    // (the engine drops it on free_image too — this keeps the mirror).
    if (engine) engine.layersClose();
    state.layers = EMPTY_LAYER_STACK;
    state.history = null;
    state.layersNote = null;
    if (state.source && engine) engine.freeImage(state.source.handle);
    state.source = null;
    state.histogram = null;
    state.channels = null;
    cropMachineRef = null;
    selectionMachineRef = null;
    brushMachineRef = null;
    state.selection = null;
    if (psdHandle !== null && engine) engine.psdClose(psdHandle);
    psdHandle = null;
    state.psd = null;
    sourceFormat = null;
    state.saveBack = null;
  };

  /** "8BPS" — the PSD/PSB magic. */
  const isPsd = (bytes: Uint8Array): boolean =>
    bytes.length >= 4 &&
    bytes[0] === 0x38 &&
    bytes[1] === 0x42 &&
    bytes[2] === 0x50 &&
    bytes[3] === 0x53;

  /** Sniff the ORIGINAL container so the save-back lane re-encodes into
   *  the format the file already was (never inventing a lossy one).
   *  Unknown containers fall to PNG — the lossless default. */
  const sniffFormat = (bytes: Uint8Array): RasterFormat | "psd" => {
    if (isPsd(bytes)) return "psd";
    if (
      bytes.length >= 3 &&
      bytes[0] === 0xff &&
      bytes[1] === 0xd8 &&
      bytes[2] === 0xff
    )
      return "jpeg";
    return "png";
  };

  /** Strip the extension off a display name (a link URI or file name) so
   *  the save-back can stamp its own. */
  const baseName = (name: string): string => {
    const leaf = name.split(/[\\/]/).pop() || "image";
    const dot = leaf.lastIndexOf(".");
    return dot > 0 ? leaf.slice(0, dot) : leaf;
  };

  /** Open the PSD structural session for an imported PSD (the mutatable
   *  tier's editor reach). Failure is honest-null — the raster lane
   *  (composite decode) is unaffected. */
  const openPsdSession = (name: string, bytes: Uint8Array): void => {
    if (!engine || !isPsd(bytes)) return;
    try {
      psdHandle = engine.psdOpen(bytes);
      state.psd = { name, layers: engine.psdLayers(psdHandle) };
    } catch {
      psdHandle = null;
      state.psd = null;
    }
  };

  /** Open the layer stack from the retained PSD parse — the file's own
   *  layer tree instead of its merged composite.
   *
   *  The engine DECLINES for every PSD whose structure the layer model
   *  does not reproduce (groups, clipping, masks, non-8-bit-RGB, over
   *  budget). That refusal is not a failure: it is the correct answer,
   *  because a layered open replaces Photoshop's OWN composite with
   *  ours. Either way `state.layersNote` says what happened, so the
   *  flatten is never silent. */
  const openLayersFromPsd = async (): Promise<void> => {
    state.layersNote = null;
    if (!engine || !state.source || psdHandle === null) return;
    try {
      const n = engine.layersOpenFromPsd(state.source.handle, psdHandle);
      // Our composite of the layer tree, not the file's merged one.
      await engine.layersComposite();
      state.layersNote = `Opened the PSD's own ${n} layer${n === 1 ? "" : "s"}.`;
    } catch (err) {
      const why = err instanceof Error ? err.message : String(err);
      state.layersNote = `Layered PSD import declined — ${why}`;
    }
  };

  const refreshPsdLayers = (): void => {
    if (psdHandle === null || !engine || !state.psd) return;
    try {
      state.psd = { ...state.psd, layers: engine.psdLayers(psdHandle) };
    } catch {
      // A failed refresh leaves the last-known list; edits still saved.
    }
  };

  /** Recompute the histogram + the CHANNELS readout + (re)build the crop
   *  machine for the live engine-held source. Called after a decode/
   *  ingest and after a crop commit (both change the source pixels).
   *  Both readouts come from the SAME buffer in the same pass, which is
   *  why the Channels panel's means can never disagree with the
   *  histogram it sits under. A read failure is non-fatal (the panel
   *  just shows no readout). */
  const refreshHistogram = () => {
    if (!engine || !state.source) return;
    try {
      state.histogram = engine.histogram(state.source.handle);
    } catch (err) {
      host.log.debug("histogram read failed", err);
      state.histogram = null;
    }
    try {
      state.channels = engine.channelStats(state.source.handle);
    } catch (err) {
      host.log.debug("channel stats read failed", err);
      state.channels = null;
    }
  };

  const refreshSourceReadout = () => {
    if (!engine || !state.source) return;
    refreshHistogram();
    // THE LAYER GRAPH — bind a stack to the LIVE handle so painting
    // always lands in a layer. Re-opening on the same handle keeps the
    // stack (and its undo history); a crop / resize / straighten
    // registers a DIFFERENT handle, so those commits re-open over the
    // result — i.e. they FLATTEN, which the panel says.
    try {
      engine.layersOpen(state.source.handle);
    } catch (err) {
      host.log.debug("layer stack open failed", err);
    }
    refreshLayers();
    cropMachineRef = createCropMachine(
      engine,
      state.source.width,
      state.source.height,
    );
    // Bind the engine selection to the LIVE handle (a crop/resize swap
    // is a different handle, so the old selection drops engine-side —
    // honest: a selection is meaningless across resolutions) and rebuild
    // the machine + readout.
    try {
      engine.selectionBind(state.source.handle);
    } catch (err) {
      host.log.debug("selection bind failed", err);
    }
    selectionMachineRef = createSelectionMachine(engine, () =>
      api.refreshSelection(),
    );
    state.selection = engine.selectionStats();
    // The paint machine is pure pointer bookkeeping, but it is rebuilt
    // with the source so a stale sample queue can never leak across a
    // crop / resize / fill / paint handle swap.
    brushMachineRef = createBrushMachine();
  };

  /** The frame's content box in page pt (falls back to the image's own
   *  extent when the geometry read misses — an honest 1:1 rather than a
   *  throw). */
  const frameBox = async (
    target: string,
  ): Promise<{ w: number; h: number }> => {
    const src = state.source;
    const fallback = { w: src?.width ?? 1, h: src?.height ?? 1 };
    try {
      const geom = await host.document.elementGeometry([
        host.selection.get().find((i) => elementIdOf(i) === target) ??
          ({ kind: "rectangle", id: target } as never),
      ]);
      const bounds = geom[0]?.bounds;
      if (!bounds) return fallback;
      const [top, left, bottom, right] = bounds;
      return { w: Math.max(right - left, 1), h: Math.max(bottom - top, 1) };
    } catch (err) {
      host.log.debug("frame geometry read failed", err);
      return fallback;
    }
  };

  /** Submit one whole-image RGBA8 payload as the C-1 Stage-A image scene
   *  item, aspect-fit + centered in the frame's content box (the layer is
   *  clipped + transformed by core; §8.5 — the plugin never compensates).
   *  The ONE composite path: the committed Apply and the live stroke
   *  preview both land here, so they cannot lay pixels out differently. */
  const submitLayer = async (
    target: string,
    rgba: Uint8Array,
    width: number,
    height: number,
    box: { w: number; h: number },
  ): Promise<boolean> => {
    const surface = scene();
    if (!surface) return false;
    const scale = Math.min(box.w / width, box.h / height);
    // The SAME number the layout uses — see `state.ptPerPx`. Recorded
    // here so a panel converting image px to points cannot disagree
    // with what was drawn.
    state.ptPerPx = scale;
    const w = width * scale;
    const h = height * scale;
    await surface.submit(target, {
      items: [
        {
          kind: "image",
          rgba: Array.from(rgba),
          width,
          height,
          x: (box.w - w) / 2,
          y: (box.h - h) / 2,
          w,
          h,
        },
      ],
    });
    state.compositedFrame = target;
    return true;
  };

  const clearLayer = async () => {
    if (state.compositedFrame) {
      await scene()?.clear(state.compositedFrame);
      state.compositedFrame = null;
      emit();
    }
  };

  /**
   * The `x-paged:media.paged.image` marker that makes a frame
   * double-click-enterable (ADR-023 / K-13). Version 1; `data`
   * deliberately carries NO state — the marker says who owns the
   * frame's pixels and nothing else, so it can never disagree with the
   * engine about them.
   *
   * Takes the real `ElementId`, not the string id the session stores:
   * `ElementId` is a discriminated union over element KIND, so a bare
   * id cannot reconstruct one, and inventing a kind here would be a
   * guess the document would have to live with.
   */
  const stampOwnership = async (id: ElementId): Promise<void> => {
    try {
      // Already marked — do NOT write. A no-op write is still a
      // MUTATION, so re-selecting the same frame would push an undo
      // step for something the user never did.
      if (await host.document.getMetadata(id)) return;
      await host.document.setMetadata(id, { v: 1, data: { owns: "pixels" } });
    } catch (err) {
      host.log.debug("ownership marker not written", err);
    }
  };

  const decodeInto = async (
    name: string,
    bytes: Uint8Array,
    origin: "selection" | "import",
    elementId: string | null,
  ): Promise<boolean> => {
    if (!engine) return false;
    try {
      // K-3 — off-main-thread decode when the worker pool is available;
      // honest main-thread fallback otherwise.
      const info = await decodeToHandle(bytes);
      freeSource();
      state.source = {
        name,
        width: info.width,
        height: info.height,
        handle: info.handle,
        origin,
        elementId,
        display: info.display,
        depthReduced: info.depthReduced,
      };
      sourceFormat = sniffFormat(bytes);
      state.saveBack = null;
      // The PSD structural session rides EVERY PSD ingest (the K-2
      // import AND the C-5 selection lane) — it is what the PSD
      // save-back writes into.
      openPsdSession(name, bytes);
      // A PSD carries a LAYER TREE; open the stack from it rather than
      // from the flattened composite when — and only when — the layer
      // model reproduces it faithfully. The engine refuses (with its
      // reason) otherwise and we keep the flatten, saying so.
      await openLayersFromPsd();
      // Compute the levels/curves histogram, bind the stack + selection,
      // and build the crop machine.
      refreshSourceReadout();
      const lane = decodePool ? " (off-thread)" : "";
      setStatus(
        `${name} — ${info.width}×${info.height} decoded${lane}.` +
          (state.layersNote ? ` ${state.layersNote}` : ""),
      );
      return true;
    } catch (err) {
      // The engine's honest unsupported/decode message (16-bit, CMYK, …).
      setStatus(`Decode failed: ${err instanceof Error ? err.message : err}`);
      return false;
    }
  };

  // Clear-on-deselect (the M4 contract): when the composited frame
  // leaves the selection, the in-frame layer clears — the preview is
  // session-scoped, the document untouched.
  const selectionSub = host.selection.onDidChange((ids) => {
    if (!state.compositedFrame) return;
    const still = ids.some((id) => elementIdOf(id) === state.compositedFrame);
    if (!still) {
      void clearLayer();
      setStatus("Frame deselected — in-frame preview cleared.");
    }
  });

  const api: ImageSession = {
    state: () => state,

    onDidChange(listener) {
      listeners.add(listener);
      return {
        dispose() {
          listeners.delete(listener);
        },
      };
    },

    async ingestSelection() {
      const ids = host.selection.get();
      if (ids.length !== 1) {
        setStatus("Select exactly one placed image frame.");
        return false;
      }
      const id = elementIdOf(ids[0]);
      if (!id) {
        setStatus("Selection carries no element id.");
        return false;
      }
      if (!host.supports("assets.images@1")) {
        setStatus(
          "Host serves no placed-image bytes (assets.images@1 is false).",
        );
        return false;
      }
      if (!(await ensureEngine())) {
        setStatus(`Engine unavailable: ${state.engineDetail ?? "unknown"}`);
        return false;
      }
      state.busy = true;
      setStatus("Reading placed bytes…");
      try {
        const asset = await host.assets.getPlacedImage(id);
        if (!asset) {
          setStatus(
            "No placed image on this frame (or the link is unresolved).",
          );
          return false;
        }
        const ok = await decodeInto(
          asset.uri || "placed image",
          asset.bytes,
          "selection",
          id,
        );
        if (ok) {
          // ADR-023 / K-13 — MARK THE FRAME AS OURS, so double-clicking
          // it enters the `rasterImage` context and the host Layers /
          // Character panels retarget here.
          //
          // The context matcher is a PURE PREDICATE over a snapshot
          // carrying only id / kind / groupChain / this plugin's own
          // metadata. It cannot ask the document "does this rectangle
          // hold placed raster bytes?" — which is the condition we
          // actually claim on — and matching by KIND instead would
          // claim every rectangle in the document and hijack a gesture
          // that belongs to the host. So we do what the contract
          // intends and paged.web already does: a bundle matches on the
          // namespace it owns.
          //
          // FIRE-AND-FORGET, and after the `ok` gate rather than before
          // it: a marker on a frame we failed to read would claim a
          // context that then has nothing to show. This is a
          // convenience marker and not part of the ingest contract — a
          // host with no metadata door, or an engine that refuses the
          // write, leaves the image fully editable through the panel
          // exactly as before. It only ever costs the double-click.
          void stampOwnership(ids[0]);
        }
        return ok;
      } finally {
        state.busy = false;
        emit();
      }
    },

    async importBytes(name, bytes) {
      if (!(await ensureEngine())) {
        setStatus(`Engine unavailable: ${state.engineDetail ?? "unknown"}`);
        return false;
      }
      state.busy = true;
      emit();
      try {
        // BIND THE SELECTED FRAME when there is exactly one, the same way
        // `ingestSelection` does. Without an `elementId` the session holds
        // pixels that no page element owns, and every frame-fit tool —
        // brush, pencil, eraser, crop — is dead on arrival:
        // `resolveFrameFit` returns null without the id, so the gesture's
        // `onPointerDown` returns before opening a stroke. That failed
        // SILENTLY (no error, no dab, an unchanged composite), which is
        // how it survived: import an image, reach for the brush, and
        // nothing happens for a reason nothing reports.
        //
        // Binding is a strict improvement and never a lie: with no
        // selection, or an ambiguous multi-selection, the id stays null
        // and the status line keeps asking for a frame — exactly as
        // before.
        const selected = host.selection.get();
        const boundTo = selected.length === 1 ? elementIdOf(selected[0]) : null;
        const ok = await decodeInto(name, bytes, "import", boundTo);
        if (ok) {
          setStatus(
            boundTo
              ? `${name} — ${state.source?.width}×${state.source?.height} decoded ` +
                  "into the selected frame. Apply to composite, or paint on it."
              : `${name} — ${state.source?.width}×${state.source?.height} decoded. ` +
                  "Select an image frame and Apply to composite.",
          );
        }
        return ok;
      } finally {
        state.busy = false;
        emit();
      }
    },

    setParams(p) {
      state.params = { ...state.params, ...p };
      emit();
    },

    psdSetLayerOpacity(index, opacity) {
      if (psdHandle === null || !engine) return false;
      try {
        engine.psdSetLayerOpacity(
          psdHandle,
          index,
          Math.max(0, Math.min(255, Math.round(opacity))),
        );
        refreshPsdLayers();
        emit();
        return true;
      } catch (err) {
        setStatus(
          `PSD edit failed: ${err instanceof Error ? err.message : err}`,
        );
        emit();
        return false;
      }
    },

    psdRenameLayer(index, name) {
      if (psdHandle === null || !engine) return false;
      try {
        engine.psdSetLayerName(psdHandle, index, name);
        refreshPsdLayers();
        emit();
        return true;
      } catch (err) {
        setStatus(
          `PSD rename failed: ${err instanceof Error ? err.message : err}`,
        );
        emit();
        return false;
      }
    },

    psdRemoveLayer(index) {
      if (psdHandle === null || !engine) return false;
      try {
        engine.psdRemoveLayer(psdHandle, index);
        refreshPsdLayers();
        emit();
        return true;
      } catch (err) {
        setStatus(
          `PSD remove failed: ${err instanceof Error ? err.message : err}`,
        );
        emit();
        return false;
      }
    },

    psdExport() {
      if (psdHandle === null || !engine || !state.psd) return null;
      try {
        return { bytes: engine.psdSave(psdHandle), fileName: state.psd.name };
      } catch (err) {
        setStatus(
          `PSD save failed: ${err instanceof Error ? err.message : err}`,
        );
        emit();
        return null;
      }
    },

    async applyToFile() {
      const src = state.source;
      if (!src || !engine) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return null;
      }
      if (!state.gpu && !isIdentity(state.params)) {
        setStatus(
          "WebGPU unavailable — the adjusted save-back needs a device.",
        );
        return null;
      }
      state.busy = true;
      setStatus("Compositing the full-resolution result…");
      try {
        // The SAME chain the canvas preview runs, at full resolution and
        // through the same selection mask — the file gets exactly what
        // the user saw.
        const rgba = await engine.adjust(src.handle, state.params);
        const stem = baseName(src.name);
        let result: SaveBackResult;
        if (sourceFormat === "psd" && psdHandle !== null) {
          const shape = engine.psdApplyAdjusted(
            psdHandle,
            src.width,
            src.height,
            rgba,
          );
          refreshPsdLayers();
          result = {
            bytes: engine.psdSave(psdHandle),
            fileName: `${stem}.psd`,
            mimeType: "image/vnd.adobe.photoshop",
            note: `PSD save-back — ${shape}.`,
          };
        } else if (sourceFormat === "psd") {
          // A PSD whose structural parse failed (the raster lane still
          // decoded it) — say so instead of silently writing a PNG.
          setStatus(
            "PSD save-back unavailable — the structural parse failed on this file " +
              "(the adjusted PNG lane is still available via Export).",
          );
          return null;
        } else {
          const fmt: RasterFormat = sourceFormat === "jpeg" ? "jpeg" : "png";
          result = {
            bytes: engine.encode(rgba, src.width, src.height, fmt),
            fileName: `${stem}${fmt === "jpeg" ? ".jpg" : ".png"}`,
            mimeType: fmt === "jpeg" ? "image/jpeg" : "image/png",
            note:
              fmt === "jpeg"
                ? "JPEG save-back — re-encoded at the fixed v0 quality (the source was a JPEG)."
                : "PNG save-back — lossless re-encode of the adjusted pixels.",
          };
        }
        state.saveBack = result;
        setStatus(
          `${result.note} ${result.fileName}, ${result.bytes.length} bytes — ready. ` +
            (host.supports("shell.saveFile@1")
              ? "Save it from the panel, or deliver it from the Export Center."
              : "This host wires no save-file door — deliver it from the Export Center."),
        );
        return result;
      } catch (err) {
        setStatus(
          `Save-back failed: ${err instanceof Error ? err.message : err}`,
        );
        return null;
      } finally {
        state.busy = false;
        emit();
      }
    },

    async saveToFile() {
      if (!host.supports("shell.saveFile@1")) {
        // Probed, not assumed. Without this the door's `false` would be
        // indistinguishable from a user cancel.
        setStatus(
          "This host wires no save-file door (shell.saveFile@1) — use the " +
            "Export Center.",
        );
        return false;
      }
      const staged = state.saveBack ?? (await api.applyToFile());
      if (!staged) return false;
      const accepted = await host.shell.saveFile({
        suggestedName: staged.fileName,
        bytes: staged.bytes,
        mimeType: staged.mimeType,
      });
      setStatus(
        accepted
          ? `${staged.fileName} handed to the host's save path ` +
            "(a browser download cannot report a cancel, so this is " +
            "delivery, not proof of a file on disk)."
          : `The host declined to save ${staged.fileName}.`,
      );
      return accepted;
    },

    async psdExportBytes() {
      if (psdHandle === null || !engine || !state.psd) return null;
      // PRESERVATION FIRST: an unadjusted export must stay byte-identical,
      // so the save-back only runs when the panel is actually off
      // identity.
      if (isIdentity(state.params)) return api.psdExport();
      const back = await api.applyToFile();
      if (back && back.mimeType === "image/vnd.adobe.photoshop") {
        return { bytes: back.bytes, fileName: back.fileName };
      }
      // The save-back lane declined (unsupported mode/size) — fall back
      // to the record-edit re-emit rather than exporting nothing.
      return api.psdExport();
    },

    async rasterExportBytes(format) {
      const src = state.source;
      if (!src || !engine) return null;
      if (!state.gpu && !isIdentity(state.params)) {
        setStatus("WebGPU unavailable — the adjusted export needs a device.");
        return null;
      }
      try {
        const rgba = await engine.adjust(src.handle, state.params);
        const bytes = engine.encode(rgba, src.width, src.height, format);
        return {
          bytes,
          fileName: `${baseName(src.name)}${format === "jpeg" ? ".jpg" : ".png"}`,
        };
      } catch (err) {
        setStatus(`Export failed: ${err instanceof Error ? err.message : err}`);
        emit();
        return null;
      }
    },

    /** Apply a whole-image EFFECT kernel and land it exactly the way a
     *  fill lands: into the active layer when a stack is bound (journaled
     *  and re-composited into the same handle), destructively otherwise.
     *  Sharing this is what stops a new effect from quietly acquiring
     *  different semantics from the fills. */
    async applyEffect(label, run) {
      const src = state.source;
      if (!src || !engine) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      if (!state.gpu) {
        setStatus(`WebGPU unavailable — ${label} is a GPU-only kernel.`);
        return false;
      }
      state.busy = true;
      emit();
      let out: DecodedInfo;
      try {
        out = await run(src.handle);
      } catch (err) {
        setStatus(
          `${label} failed: ${err instanceof Error ? err.message : err}`,
        );
        state.busy = false;
        emit();
        return false;
      }
      const layered = out.handle === src.handle;
      swapSource(src, out);
      state.saveBack = null;
      if (!layered) {
        try {
          engine.selectionTransfer(out.handle);
        } catch (err) {
          host.log.debug("selection transfer failed", err);
        }
      }
      state.busy = false;
      const ok = await recomposite();
      if (ok) setStatus(`${label} applied.`);
      return ok;
    },

    async applyGradientMap(shadow, highlight) {
      return this.applyEffect("Gradient map", (h) =>
        engine!.applyGradientMap(h, shadow, highlight),
      );
    },

    async applyWarp(kind, amount, frequency = 1) {
      return this.applyEffect(`Distort (${kind})`, (h) =>
        engine!.applyWarp(h, kind, amount, frequency),
      );
    },

    async fillSelection(req) {
      const src = state.source;
      if (!src || !engine) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      // Content-aware fill is a CPU search, so it works with no device —
      // the generators do not. Gating it on WebGPU would refuse a fill
      // that has no GPU work in it.
      if (req.kind !== "contentAware" && !state.gpu) {
        setStatus("WebGPU unavailable — the generators are GPU-only kernels.");
        return false;
      }
      if (req.kind === "contentAware" && !state.selection) {
        setStatus(
          "Select the area to fill first — content-aware fill synthesises " +
            "the SELECTION from the rest of the image.",
        );
        return false;
      }
      state.busy = true;
      emit();
      let filled: { handle: number; width: number; height: number };
      try {
        filled =
          req.kind === "gradient"
            ? await engine.fillGradient(
                src.handle,
                req.gradient,
                req.c0,
                req.c1,
              )
            : req.kind === "contentAware"
              ? await engine.fillContentAware(src.handle)
              : await engine.fillNoise(
                  src.handle,
                  req.amount,
                  req.seed ?? FILL_NOISE_SEED_DEFAULT,
                );
      } catch (err) {
        setStatus(`Fill failed: ${err instanceof Error ? err.message : err}`);
        state.busy = false;
        emit();
        return false;
      }
      // With a layer stack bound the fill went INTO the active layer
      // (journaled) and re-composited into the SAME handle; without one
      // it is the pre-layer destructive commit (see fill.rs).
      const layerName = state.layers.layers[state.layers.active]?.name ?? null;
      const layered = filled.handle === src.handle;
      swapSource(src, filled);
      state.saveBack = null;
      if (!layered) {
        // The fill is same-size, so the SELECTION is still meaningful on
        // the result — carry it over instead of making the user
        // reselect (a crop/resize swap still drops it).
        try {
          engine.selectionTransfer(filled.handle);
        } catch (err) {
          host.log.debug("selection transfer failed", err);
        }
      }
      refreshSourceReadout();
      const where = state.selection ? "the selection" : "the whole image";
      setStatus(
        `Filled ${where} with ${
          req.kind === "gradient" ? `a ${req.gradient} gradient` : "noise"
        }` +
          (layered && layerName
            ? ` on layer “${layerName}” — undoable.`
            : " (destructive, no layer stack bound).") +
          " Document unchanged; Apply to recomposite.",
      );
      state.busy = false;
      emit();
      return true;
    },

    setLevels(l) {
      state.params = {
        ...state.params,
        levels: { ...state.params.levels, ...l },
      };
      emit();
    },

    setCurvePoints(points) {
      if (!engine) return;
      // The identity curve [(0,0),(1,1)] clears the LUT (no curve pass).
      const isIdentityCurve =
        points.length === 2 &&
        points[0][0] === 0 &&
        points[0][1] === 0 &&
        points[1][0] === 1 &&
        points[1][1] === 1;
      state.params = {
        ...state.params,
        curveLut: isIdentityCurve ? null : engine.curveLut(points),
      };
      emit();
    },

    autoEnhance() {
      const src = state.source;
      if (!src || !engine) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return;
      }
      try {
        // Pure CPU readout (no GPU needed) — derive auto-levels + gray-world
        // WB from the histogram and merge into the SAME params the sliders
        // drive. The composite still waits for the explicit Apply.
        const a = engine.autoEnhanceParams(src.handle);
        state.params = {
          ...state.params,
          temp: a.temp,
          tint: a.tint,
          levels: {
            ...state.params.levels,
            inBlack: a.inBlack,
            inWhite: a.inWhite,
          },
        };
        emit();
        setStatus(
          "Auto-enhance set levels + white balance — click Apply to composite " +
            "(document unchanged).",
        );
      } catch (err) {
        setStatus(
          `Auto-enhance failed: ${err instanceof Error ? err.message : err}`,
        );
      }
    },

    cropMachine() {
      return cropMachineRef;
    },

    selectionMachine() {
      return selectionMachineRef;
    },

    refreshSelection() {
      if (!engine || !state.source) return;
      state.selection = engine.selectionStats();
      emit();
    },

    selectAll() {
      if (!engine || !state.source) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      try {
        engine.selectionSelectAll();
      } catch (err) {
        setStatus(
          `Select all failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      api.refreshSelection();
      setStatus("Selected all — adjustments apply to the whole image.");
      return true;
    },

    deselect() {
      if (!engine || !state.source) return false;
      engine.selectionClear();
      api.refreshSelection();
      setStatus("Deselected — adjustments apply to the whole image.");
      return true;
    },

    invertSelection() {
      if (!engine || !state.source) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      try {
        engine.selectionInvert();
      } catch (err) {
        setStatus(
          `Invert selection failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      api.refreshSelection();
      setStatus("Selection inverted.");
      return true;
    },

    featherSelection(sigma = FEATHER_SIGMA_DEFAULT) {
      if (!engine || !state.source) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      try {
        engine.selectionFeather(sigma);
      } catch (err) {
        // The honest miss: feather needs an explicit selection.
        setStatus(
          `Feather failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      api.refreshSelection();
      setStatus(`Selection feathered (σ ${sigma}px).`);
      return true;
    },

    async resizeTo(w, h, filter) {
      const src = state.source;
      if (!src || !engine) {
        setStatus("Nothing to resize — ingest an image first.");
        return false;
      }
      let resized: { handle: number; width: number; height: number };
      try {
        resized = await engine.resize(src.handle, w, h, filter);
      } catch (err) {
        setStatus(`Resize failed: ${err instanceof Error ? err.message : err}`);
        emit();
        return false;
      }
      // Swap the engine-held source (the crop-commit pattern): release
      // the tile claim on the old handle, free it, adopt the new.
      releaseTiles();
      engine.freeImage(src.handle);
      src.handle = resized.handle;
      src.width = resized.width;
      src.height = resized.height;
      refreshSourceReadout();
      setStatus(
        `Resampled to ${resized.width}×${resized.height} (${filter}) — ` +
          "document unchanged; Apply to recomposite.",
      );
      emit();
      return true;
    },

    brushMachine() {
      return brushMachineRef;
    },

    selectionFromChannel(channel) {
      const src = state.source;
      if (!engine || !src) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      try {
        engine.selectionFromChannel(src.handle, channel, "replace");
      } catch (err) {
        // The engine names the channel it refused; a generic failure
        // would hide a typo'd channel behind "could not select".
        setStatus(
          `Channel selection failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      api.refreshSelection();
      setStatus(
        `Selection loaded from the ${channel} channel — its values ARE the ` +
          "coverage, so mid-tones select partially.",
      );
      return true;
    },

    async selectionToPath(threshold = 128, tolerance = 0.5) {
      const src = state.source;
      if (!engine || !src) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      if (!state.selection) {
        setStatus("Nothing selected — select a region first.");
        return false;
      }
      const contours = engine.selectionToPaths(threshold, tolerance);
      if (contours.length === 0) {
        // A selection that exists but covers nothing at this threshold is
        // a real state, and raising the threshold is the fix — so say
        // which threshold produced nothing rather than "no paths".
        setStatus(
          `Nothing traced at coverage ≥ ${threshold} — the selection is ` +
            "empty at that cut.",
        );
        return false;
      }
      // Image px → page pt, through the SAME fit the paint tools use, so
      // a traced path lands exactly on top of what it traced.
      const fit = await resolveFrameFit(host, src, "selectionToPath");
      if (!fit) {
        setStatus(
          "The frame's geometry could not be read, so the path would land " +
            "in the wrong place. Nothing was inserted.",
        );
        return false;
      }
      let inserted = 0;
      let lastError: string | null = null;
      for (const contour of contours) {
        // Straight segments: each anchor's handles sit ON the anchor,
        // which is how a bezier path expresses a corner. Fitting curves
        // to a traced staircase would invent smoothness the mask does
        // not have — so `smooth: false`, and the Pen is right there for
        // anyone who wants curves.
        const anchors = contour.points.map((pt) => {
          const [x, y] = imageToPage(fit, pt);
          return {
            anchor: [x, y] as [number, number],
            left: [x, y] as [number, number],
            right: [x, y] as [number, number],
          };
        });
        try {
          // `MutationOutcome` is a discriminated union and a REFUSAL is a
          // resolved value, not a throw — so truthiness proves nothing.
          // Counting on it once produced "Traced 1 path" over a document
          // that had gained no path at all.
          const outcome = await host.document.mutate({
            op: "insertPath",
            args: {
              pageId: fit.pageId,
              anchors,
              open: false,
              smooth: false,
            },
          });
          if (outcome.applied) {
            inserted += 1;
          } else {
            lastError = String(outcome.error ?? "the host refused the insert");
            host.log.debug("path insert refused", outcome.error);
          }
        } catch (err) {
          lastError = err instanceof Error ? err.message : String(err);
          host.log.debug("path insert failed", err);
        }
      }
      if (inserted === 0) {
        setStatus(
          `The host refused the path insert${lastError ? `: ${lastError}` : "."}`,
        );
        return false;
      }
      const holes = contours.filter((c) => !c.outer).length;
      setStatus(
        `Traced ${inserted} path${inserted === 1 ? "" : "s"} from the ` +
          `selection` +
          (holes > 0
            ? ` (${holes} of them hole${holes === 1 ? "" : "s"}, inserted ` +
              "separately — this engine emits one polygon per contour)"
            : ""),
      );
      return true;
    },

    async selectionFromPath() {
      const src = state.source;
      if (!engine || !src) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      const picked = host.selection
        .get()
        .filter((i) => elementIdOf(i) !== src.elementId);
      if (picked.length !== 1) {
        setStatus(
          "Select exactly ONE vector element (other than the image frame) " +
            "to load its path as the selection.",
        );
        return false;
      }
      let result;
      try {
        result = await host.document.pathAnchors(picked[0]);
      } catch (err) {
        setStatus(
          `Path read failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      if (!result || result.anchors.length < 3) {
        // A rectangle can legitimately carry NO PathGeometry, which the
        // contract documents as "nothing to draw" — reporting that is
        // more useful than an error the designer cannot act on.
        setStatus(
          "That element exposes no path with at least three anchors " +
            "(a plain rectangle often carries none).",
        );
        return false;
      }
      const fit = await resolveFrameFit(host, src, "selectionFromPath");
      if (!fit) {
        setStatus("The image frame's geometry could not be read.");
        return false;
      }
      // Flatten each cubic segment. A fixed subdivision is honest here:
      // the target is a PIXEL mask, so more than a pixel's worth of
      // precision buys nothing and the cost is bounded.
      const flat: Array<[number, number]> = [];
      const a = result.anchors;
      for (let i = 0; i < a.length; i += 1) {
        const cur = a[i];
        const nxt = a[(i + 1) % a.length];
        flat.push(pageToImage(fit, cur.anchor as [number, number]));
        const straight =
          cur.anchor[0] === cur.right[0] &&
          cur.anchor[1] === cur.right[1] &&
          nxt.anchor[0] === nxt.left[0] &&
          nxt.anchor[1] === nxt.left[1];
        if (straight) continue;
        for (let step = 1; step < CURVE_STEPS; step += 1) {
          const t = step / CURVE_STEPS;
          flat.push(
            pageToImage(
              fit,
              cubicAt(cur.anchor, cur.right, nxt.left, nxt.anchor, t),
            ),
          );
        }
      }
      try {
        engine.selectionSetPolygon(flat, "replace");
      } catch (err) {
        setStatus(
          `Path selection failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      api.refreshSelection();
      setStatus(
        `Selection loaded from a ${a.length}-anchor path ` +
          `(${flat.length} points after flattening).`,
      );
      return true;
    },

    setType(p) {
      state.type = { ...state.type, ...p };
      emit();
    },

    async paintText([x, y], text, family, sizePx) {
      const src = state.source;
      if (!engine || !src) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      if (!host.supports("assets.fonts@1")) {
        // PROBED, not assumed. Without the probe an unwired asset door
        // and a font the document does not carry look identical, and
        // they need different sentences.
        setStatus("This host serves no fonts (assets.fonts@1).");
        return false;
      }
      const face = await host.assets.getFontFace(family);
      if (!face) {
        setStatus(
          `The document carries no bytes for "${family}" — type renders ` +
            "the faces the document already embeds, never a web fetch.",
        );
        return false;
      }
      state.busy = true;
      emit();
      let missing = 0;
      try {
        missing = await engine.textPaint(
          src.handle,
          face.bytes,
          text,
          sizePx,
          x,
          y,
          state.brush.color,
          state.type.trackingPerMille,
          // The wasm reads non-positive as AUTO, so null passes through
          // as 0 rather than needing a second parameter to mean "unset".
          state.type.leadingPx ?? 0,
        );
      } catch (err) {
        setStatus(`Type failed: ${err instanceof Error ? err.message : err}`);
        state.busy = false;
        emit();
        return false;
      }
      refreshSourceReadout();
      state.busy = false;
      setStatus(
        missing > 0
          ? `Type set in ${family} — ${missing} character${
              missing === 1 ? " has" : "s have"
            } no glyph in this font and ${missing === 1 ? "was" : "were"} not drawn.`
          : `Type set in ${family}.`,
      );
      return true;
    },

    setCloneSource([x, y]) {
      const aligned = state.cloneSource?.aligned ?? true;
      state.cloneSource = { x, y, aligned };
      setStatus(
        `Clone source set at ${Math.round(x)}, ${Math.round(y)} — ` +
          "paint to copy from there.",
      );
    },

    setCloneAligned(aligned) {
      state.cloneSource = state.cloneSource
        ? { ...state.cloneSource, aligned }
        : null;
      emit();
    },

    setBrushParams(p) {
      state.brush = { ...state.brush, ...p };
      // Editing any parameter means the brush is no longer the preset.
      // Keeping the row highlighted would claim a provenance the stroke
      // does not have.
      state.brushPreset = null;
      emit();
    },

    async loadBrushLibrary(name, bytes) {
      const eng = await ensureEngine();
      if (!eng) {
        setStatus("The image engine could not start.");
        return false;
      }
      let library: BrushLibraryInfo;
      try {
        library = eng.abrPresets(bytes);
      } catch (e) {
        // The reader's message names WHAT it refused (legacy version,
        // bad signature); a generic "could not load" would throw that
        // away. "Not an .abr" and "an .abr with no presets" are
        // different answers and the panel shows both.
        setStatus(`${name}: ${e instanceof Error ? e.message : String(e)}`);
        return false;
      }
      state.brushLibrary = library;
      state.brushLibraryName = name;
      state.brushPreset = null;
      const warned = library.warnings.length;
      setStatus(
        `${name}: ${library.presetCount} brush preset${
          library.presetCount === 1 ? "" : "s"
        }` +
          (library.sampleCount > 0
            ? `, ${library.sampleCount} sampled tip${
                library.sampleCount === 1 ? "" : "s"
              } (bitmap tips are not painted — the engine's tip is parametric)`
            : "") +
          (warned > 0 ? `, ${warned} reader warning${warned === 1 ? "" : "s"}` : ""),
      );
      return true;
    },

    async pickBrushLibrary() {
      if (!host.supports("shell.pickFile@1")) {
        // `pickFile` resolves `[]` both when the user cancels and when no
        // picker is wired at all. Probing keeps "you cancelled" from
        // masquerading as "this host cannot open files".
        setStatus("This host wires no file picker (shell.pickFile@1).");
        return false;
      }
      const picked = await host.shell.pickFile({ accept: [".abr"] });
      const file = picked[0];
      if (!file) {
        setStatus("No brush library chosen.");
        return false;
      }
      return await this.loadBrushLibrary(file.name, file.bytes);
    },

    applyBrushPreset(index) {
      const preset = state.brushLibrary?.presets.find((p) => p.index === index);
      if (!preset) {
        setStatus("No such brush preset.");
        return false;
      }
      // Apply only what the engine's tip HAS. Everything else is
      // reported rather than silently dropped — a designer who picks a
      // 45°-angled elliptical brush and gets a circle should be told.
      const patch: Partial<BrushParams> = {};
      const unapplied: string[] = [];
      if (preset.diameter !== null && preset.diameterUnit === "#Pxl") {
        patch.size = Math.max(1, Math.round(preset.diameter));
      } else if (preset.diameter !== null) {
        unapplied.push(`diameter in ${preset.diameterUnit ?? "an unknown unit"}`);
      }
      if (preset.hardness !== null) {
        patch.hardness = Math.min(1, Math.max(0, preset.hardness));
      } else {
        // A sampled tip's softness lives in its BITMAP, which this
        // engine does not stamp; saying so beats leaving the previous
        // brush's hardness silently in force.
        unapplied.push("hardness (the file carries none for this tip kind)");
      }
      if (preset.spacing !== null && preset.spacingEnabled !== false) {
        patch.spacing = Math.max(0.01, preset.spacing);
      } else if (preset.spacing !== null) {
        unapplied.push("spacing (disabled in the preset)");
      }
      if (preset.roundness !== null && Math.abs(preset.roundness - 1) > 0.001) {
        unapplied.push("roundness — the tip is circular");
      }
      if (preset.angle !== null && Math.abs(preset.angle) > 0.001) {
        unapplied.push("angle — the tip is circular");
      }
      state.brush = { ...state.brush, ...patch };
      state.brushPreset = index;
      setStatus(
        `Brush: ${preset.name}` +
          (unapplied.length > 0 ? ` — not applied: ${unapplied.join("; ")}` : ""),
      );
      return true;
    },

    closeBrushLibrary() {
      state.brushLibrary = null;
      state.brushLibraryName = null;
      state.brushPreset = null;
      emit();
    },

    async brushBegin(tool) {
      const src = state.source;
      if (!src || !engine) {
        setStatus("Nothing to paint on — ingest a placed image first.");
        return false;
      }
      if (!state.gpu) {
        // No CPU blend path ships (spec §6) — the dab composite IS a
        // registered WGSL dispatch.
        setStatus("WebGPU unavailable — painting is GPU-only.");
        return false;
      }
      if (state.strokeActive) return false;
      // A sampling tool with no anchor would paint NOTHING and look
      // broken, so it declines with the instruction instead. Photoshop
      // says the same thing in a modal; this says it in the status line.
      if (SAMPLING_TOOLS.includes(tool) && !state.cloneSource) {
        setStatus(
          "Alt-click to set the clone source first — a clone with no " +
            "source has nothing to copy.",
        );
        return false;
      }
      try {
        engine.brushBegin(src.handle, tool, state.brush);
        if (SAMPLING_TOOLS.includes(tool) && state.cloneSource) {
          // BEFORE the first extend: the offset is fixed at the first
          // dab, and an anchor arriving later would be ignored silently.
          engine.brushSetSource(
            state.cloneSource.x,
            state.cloneSource.y,
            state.cloneSource.aligned,
          );
        }
      } catch (err) {
        setStatus(`Paint failed: ${err instanceof Error ? err.message : err}`);
        return false;
      }
      state.strokeActive = true;
      state.strokeStats = null;
      // Resolve the preview frame ONCE — a geometry read per pointer
      // sample would put an async round-trip inside the drag.
      strokeTarget = src.elementId;
      if (!strokeTarget) {
        const ids = host.selection.get();
        strokeTarget = ids.length === 1 ? elementIdOf(ids[0]) : null;
        if (strokeTarget) src.elementId = strokeTarget;
      }
      strokeBox = strokeTarget ? await frameBox(strokeTarget) : null;
      if (!strokeTarget) {
        // The stroke still paints into the engine image — say why nothing
        // shows on the canvas rather than letting it look broken.
        setStatus(
          "Painting into the engine image — select the target frame to see " +
            "the stroke in-frame.",
        );
      }
      emit();
      return true;
    },

    async brushExtend(sample) {
      const src = state.source;
      if (!src || !engine || !state.strokeActive) return false;
      let rgba: Uint8Array;
      try {
        rgba = await engine.brushExtend(sample.x, sample.y, sample.pressure);
      } catch (err) {
        setStatus(`Paint failed: ${err instanceof Error ? err.message : err}`);
        discardStroke();
        emit();
        return false;
      }
      state.strokeStats = engine.brushStats();
      // The LIVE preview is the painted pixels themselves — NOT the
      // adjust chain's output. The chain is a function of the source, so
      // re-running it per sample would double the GPU work in the middle
      // of a drag; the committed Apply (which brushCommit triggers) puts
      // the adjusted composite back. Stated in the panel.
      if (strokeTarget && strokeBox) {
        await submitLayer(strokeTarget, rgba, src.width, src.height, strokeBox);
      }
      emit();
      return true;
    },

    async brushCommit() {
      const src = state.source;
      if (!src || !engine || !state.strokeActive) return false;
      const dabs = state.strokeStats?.dabs ?? 0;
      const layerName = state.layers.layers[state.layers.active]?.name ?? null;
      let painted: { handle: number; width: number; height: number };
      try {
        painted = await engine.brushCommit();
      } catch (err) {
        setStatus(
          `Paint commit failed: ${err instanceof Error ? err.message : err}`,
        );
        discardStroke();
        emit();
        return false;
      }
      state.strokeActive = false;
      state.strokeStats = null;
      strokeTarget = null;
      strokeBox = null;
      brushMachineRef?.cancel();
      // With a layer stack bound the engine wrote the stroke INTO the
      // active layer (journaled) and re-composited into the SAME handle
      // — so there is nothing to swap and the old handle must NOT be
      // freed. Without one it is the pre-layer destructive commit.
      const layered = painted.handle === src.handle;
      swapSource(src, painted);
      state.saveBack = null;
      if (!layered) {
        // Same size, so the SELECTION is still meaningful on the result.
        try {
          engine.selectionTransfer(painted.handle);
        } catch (err) {
          host.log.debug("selection transfer failed", err);
        }
      }
      refreshSourceReadout();
      // Put the ADJUSTED composite back over the raw stroke preview.
      if (src.elementId) await api.apply();
      const where = layerName ? `layer “${layerName}”` : "the engine image";
      setStatus(
        `Painted ${dabs} dab${dabs === 1 ? "" : "s"} into ${where}` +
          (layered
            ? " — undoable (the stroke's tiles are journaled). The document " +
              "and the source file are unchanged."
            : " (destructive, no layer stack bound — re-ingest to restore).") +
          (state.history?.dropped
            ? ` History window: ${state.history.depth} step${
                state.history.depth === 1 ? "" : "s"
              }, ${state.history.dropped} older edit${
                state.history.dropped === 1 ? "" : "s"
              } past the undo budget and now permanent.`
            : ""),
      );
      return true;
    },

    // ── THE LAYER GRAPH ─────────────────────────────────────────────

    async addLayer(name = "") {
      const ok = await layerOp("Add layer", () => engine!.layerAdd(name));
      if (ok) {
        const l = state.layers.layers[state.layers.active];
        setStatus(
          `Added layer “${l?.name ?? name}” — paint, fills and bakes land ` +
            "there until you pick another.",
        );
      }
      return ok;
    },

    duplicateLayer(index) {
      return layerOp("Duplicate layer", () => engine!.layerDuplicate(index));
    },

    async removeLayer(index) {
      const name = state.layers.layers[index]?.name ?? `layer ${index}`;
      const ok = await layerOp("Remove layer", () =>
        engine!.layerRemove(index),
      );
      if (ok) {
        setStatus(
          `Removed “${name}”. Layer structure is not journaled, and a removed ` +
            "layer's history could never be replayed — so the undo history was " +
            "CLEARED. Nothing before this point can be undone.",
        );
      }
      return ok;
    },

    reorderLayer(from, to) {
      return layerOp("Reorder layers", () => engine!.layerReorder(from, to));
    },

    setActiveLayer(index) {
      if (!engine || !state.source) return false;
      try {
        engine.layerSetActive(index);
      } catch (err) {
        setStatus(
          `Select layer failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      refreshLayers();
      emit();
      return true;
    },

    setLayerVisible(index, visible) {
      return layerOp("Layer visibility", () =>
        engine!.layerSetVisible(index, visible),
      );
    },

    setLayerOpacity(index, opacity) {
      return layerOp("Layer opacity", () =>
        engine!.layerSetOpacity(index, opacity),
      );
    },

    setLayerBlend(index, blend) {
      return layerOp("Layer blend", () => engine!.layerSetBlend(index, blend));
    },

    async layerMaskFromSelection(index) {
      if (!engine || !state.source) return false;
      try {
        engine.layerMaskFromSelection(index);
      } catch (err) {
        // The engine refuses with no selection rather than attaching an
        // all-one mask that would look like success and mask nothing.
        setStatus(
          `Layer mask failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      return finishMaskEdit();
    },

    async setLayerMaskEnabled(index, enabled) {
      if (!engine || !state.source) return false;
      try {
        engine.layerSetMaskEnabled(index, enabled);
      } catch (err) {
        setStatus(
          `Layer mask toggle failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      return finishMaskEdit();
    },

    async setLayerClipped(index, clipped) {
      if (!engine || !state.source) return false;
      try {
        engine.layerSetClipped(index, clipped);
      } catch (err) {
        setStatus(
          `Clip failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      return finishMaskEdit();
    },

    async groupLayers(from, to, name = "Group") {
      if (!engine || !state.source) return false;
      try {
        engine.layersGroup(from, to, name);
      } catch (err) {
        // The engine's refusal names its reason (out of range, already
        // grouped) — a generic message would hide which.
        setStatus(`Group failed: ${err instanceof Error ? err.message : err}`);
        return false;
      }
      return finishMaskEdit();
    },

    async ungroupLayers(id) {
      if (!engine || !state.source) return false;
      try {
        engine.layersUngroup(id);
      } catch (err) {
        setStatus(`Ungroup failed: ${err instanceof Error ? err.message : err}`);
        return false;
      }
      return finishMaskEdit();
    },

    async setGroupVisible(id, visible) {
      return groupEdit(() => engine!.layerSetGroupVisible(id, visible));
    },
    async setGroupOpacity(id, opacity) {
      return groupEdit(() => engine!.layerSetGroupOpacity(id, opacity));
    },
    async setGroupName(id, name) {
      return groupEdit(() => engine!.layerSetGroupName(id, name));
    },
    async setGroupBlend(id, blend) {
      return groupEdit(() => engine!.layerSetGroupBlend(id, blend));
    },
    async setGroupPassThrough(id, passThrough) {
      return groupEdit(() => engine!.layerSetGroupPassThrough(id, passThrough));
    },

    async makeLayerSmart(index) {
      if (!engine || !state.source) return false;
      try {
        engine.layerMakeSmart(index);
      } catch (err) {
        setStatus(
          `Convert to smart object failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      refreshLayers();
      emit();
      // Converting alone changes no pixel, so there is nothing to
      // re-composite — unlike a mask edit.
      return true;
    },

    async renderLayerSmart(index, scale) {
      if (!engine || !state.source) return false;
      state.busy = true;
      emit();
      try {
        await engine.layerRenderSmart(index, scale);
      } catch (err) {
        setStatus(
          `Smart re-render failed: ${err instanceof Error ? err.message : err}`,
        );
        state.busy = false;
        emit();
        return false;
      }
      state.busy = false;
      const ok = await finishMaskEdit();
      if (ok) {
        setStatus(
          `Re-rendered from the preserved source at ${Math.round(scale * 100)}% — ` +
            "scaling a smart object never loses the original.",
        );
      }
      return ok;
    },

    async clearLayerMask(index) {
      if (!engine || !state.source) return false;
      try {
        engine.layerClearMask(index);
      } catch (err) {
        setStatus(
          `Layer mask clear failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      return finishMaskEdit();
    },

    setLayerLocked(index, locked) {
      // A lock changes no pixel, so it does not re-composite.
      if (!engine || !state.source) return false;
      try {
        engine.layerSetLocked(index, locked);
      } catch (err) {
        setStatus(
          `Layer lock failed: ${err instanceof Error ? err.message : err}`,
        );
        return false;
      }
      refreshLayers();
      emit();
      return true;
    },

    setLayerName(index, name) {
      if (!engine || !state.source) return false;
      try {
        engine.layerSetName(index, name);
      } catch (err) {
        setStatus(`Rename failed: ${err instanceof Error ? err.message : err}`);
        return false;
      }
      refreshLayers();
      emit();
      return true;
    },

    async addAdjustmentLayer() {
      if (!engine || !state.source) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      state.busy = true;
      emit();
      try {
        engine.layersAddAdjustment("", state.params);
      } catch (err) {
        // Identity is the common refusal, and it is honest: a row that
        // adjusts nothing would just be clutter that looks like work.
        setStatus(
          `Adjustment layer failed: ${err instanceof Error ? err.message : err}`,
        );
        state.busy = false;
        emit();
        return false;
      }
      state.busy = false;
      const ok = await finishMaskEdit();
      if (ok) {
        setStatus(
          "Adjustment layer added — it transforms everything beneath it, " +
            "and deleting it restores the original exactly.",
        );
      }
      return ok;
    },

    async bakeAdjustToLayer() {
      if (!engine || !state.source) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      state.busy = true;
      emit();
      try {
        await engine.layersBakeAdjust(state.params);
      } catch (err) {
        // Identity, a locked layer, or no device — the engine's reason.
        setStatus(`Bake failed: ${err instanceof Error ? err.message : err}`);
        state.busy = false;
        emit();
        return false;
      }
      const name =
        state.layers.layers[state.layers.active]?.name ?? "the active layer";
      state.busy = false;
      // The chain is now IN the pixels, so the panel returns to identity
      // — leaving the sliders up would apply it a second time on Apply.
      state.params = freshIdentityParams();
      state.saveBack = null;
      const ok = await recomposite();
      if (ok) {
        setStatus(
          `Baked the adjustments into layer “${name}” (undoable) and reset ` +
            "the panel to identity — the chain now lives in those pixels.",
        );
      }
      return ok;
    },

    undo() {
      return historyStep(true);
    },

    async undoSteps(n) {
      let ok = true;
      for (let i = 0; i < n && ok; i++) ok = await historyStep(true);
      return ok;
    },

    async redoSteps(n) {
      let ok = true;
      for (let i = 0; i < n && ok; i++) ok = await historyStep(false);
      return ok;
    },

    redo() {
      return historyStep(false);
    },

    brushCancel() {
      if (!state.strokeActive) return;
      discardStroke();
      setStatus(
        "Stroke cancelled — the engine pixels are exactly as they were.",
      );
      // The frame is still showing the abandoned preview; put the
      // committed composite back.
      if (state.compositedFrame && state.source?.elementId) void api.apply();
      else emit();
    },

    async commitCrop() {
      const src = state.source;
      if (!src || !engine || !cropMachineRef) {
        setStatus("Nothing to crop — ingest a placed image first.");
        return false;
      }
      let cropped: { handle: number; width: number; height: number };
      try {
        // The commit carries the STRAIGHTEN angle: at 0° it is the pure
        // axis-aligned cut, otherwise the engine rotates first
        // (geom.rotate_bilinear) so the rotated frame lands upright.
        cropped = await cropMachineRef.commit(engine, src.handle);
      } catch (err) {
        setStatus(`Crop failed: ${err instanceof Error ? err.message : err}`);
        return false;
      }
      // Swap the engine-held source to the cropped image (free the old
      // pixels; a tile claim points at the old handle, so release it).
      releaseTiles();
      engine.freeImage(src.handle);
      src.handle = cropped.handle;
      src.width = cropped.width;
      src.height = cropped.height;
      refreshSourceReadout();
      const straightened = cropMachineRef.state().angle !== 0;
      setStatus(
        `Cropped to ${cropped.width}×${cropped.height}` +
          (straightened
            ? ` (straightened ${cropMachineRef.state().angle}° — bilinear resample)`
            : "") +
          " (document unchanged — engine source only; Apply to recomposite).",
      );
      // Re-composite the cropped pixels in-frame when a frame is targeted.
      if (src.elementId) await api.apply();
      else emit();
      return true;
    },

    async apply() {
      const src = state.source;
      if (!src || !engine) {
        setStatus("Nothing ingested — select an image frame and ingest first.");
        return false;
      }
      // An import targets the currently selected frame at Apply time.
      let target = src.elementId;
      if (!target) {
        const ids = host.selection.get();
        target = ids.length === 1 ? elementIdOf(ids[0]) : null;
        if (!target) {
          setStatus("Select the target frame to composite the import into.");
          return false;
        }
        src.elementId = target;
      }
      const surface = scene();
      if (!surface) {
        setStatus("No scene channel (rendering.sceneLayer@1 is false).");
        return false;
      }
      if (!state.gpu && !isIdentity(state.params)) {
        setStatus("WebGPU unavailable — only the identity composite works.");
        return false;
      }

      state.busy = true;
      setStatus("Adjusting…");
      try {
        const rgba = await engine.adjust(src.handle, state.params);
        await submitLayer(
          target,
          rgba,
          src.width,
          src.height,
          await frameBox(target),
        );
        // The engine masked the chain by the bound selection (if any) —
        // say so, honestly.
        const sel = state.selection
          ? " (adjustments masked to the selection)"
          : "";
        setStatus(
          `Composited ${src.width}×${src.height} into the frame` +
            `${sel} (document unchanged — preview layer only).`,
        );
        return true;
      } catch (err) {
        setStatus(`Adjust failed: ${err instanceof Error ? err.message : err}`);
        return false;
      } finally {
        state.busy = false;
        emit();
      }
    },

    claimTiles() {
      const src = state.source;
      if (!src || !engine) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      // An import claims the frame it was (or will be) composited into; a
      // selection ingest already carries its frame.
      let target = src.elementId;
      if (!target) {
        const ids = host.selection.get();
        target = ids.length === 1 ? elementIdOf(ids[0]) : null;
        if (!target) {
          setStatus("Select the frame to claim tiles for.");
          return false;
        }
        src.elementId = target;
      }
      if (!host.supports("rendering.resourceProvider@1")) {
        setStatus(
          "Host serves no tile resource (rendering.resourceProvider@1 is false).",
        );
        return false;
      }
      releaseTiles();
      // The provider reads the LIVE handle on each pull — a re-ingest of
      // the same frame is picked up without re-claiming.
      const claim = claimImageTiles(
        host,
        {
          elementId: target,
          handle: src.handle,
          width: src.width,
          height: src.height,
        },
        engine,
        () => (state.source ? state.source.handle : null),
      );
      tileClaim = {
        elementId: target,
        dispose: () => claim.dispose(),
        bump: () => claim.bump(),
      };
      setStatus(
        `Claimed tile resource for the frame (level-0 lane; ${src.width}×${src.height}). ` +
          "The renderer pulls tiles at its current scale.",
      );
      return true;
    },

    tilesClaimed() {
      return tileClaim !== null;
    },

    async reset() {
      state.params = freshIdentityParams();
      state.saveBack = null;
      cropMachineRef?.reset(
        state.source?.width ?? 0,
        state.source?.height ?? 0,
      );
      await clearLayer();
      setStatus("Reset — in-frame preview cleared.");
    },

    dispose() {
      disposed = true;
      selectionSub.dispose();
      discardStroke();
      releaseTiles();
      // K-3 — terminate the decode pool's workers (the host ALSO
      // auto-terminates them on bundle dispose; this is the explicit,
      // earlier teardown, and terminate is idempotent).
      decodePool?.dispose();
      decodePool = null;
      freeSource();
      // The host tears the scene surface down (contribute-tracked); its
      // dispose clears every submitted layer.
      listeners.clear();
    },
  };
  return api;
}
