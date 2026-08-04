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

import type { BundleHost, Disposable } from "@paged-media/plugin-api";

import {
  bootEngine,
  DEFAULT_BRUSH_PARAMS,
  freshIdentityParams,
  isIdentity,
  type AdjustParams,
  type BrushParams,
  type BrushStats,
  type GradientKind,
  type ImageEngine,
  type ImageHistogram,
  type PsdLayerInfo,
  type RasterFormat,
  type ResampleFilter,
  type Rgba01,
  type LevelsParams,
  type SelectionStats,
  type StrokeTool,
} from "./engine";
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
  | { kind: "noise"; amount: number; seed?: number };

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
  setBrushParams(p: Partial<BrushParams>): void;
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

export function createImageSession(host: BundleHost): ImageSession {
  const listeners = new Set<() => void>();
  let engine: ImageEngine | null = null;
  let bootPromise: Promise<ImageEngine | null> | null = null;
  let sceneSurface: ReturnType<typeof host.contribute.sceneLayer> | null = null;
  // C-6 — the active tile-resource claim (null when nothing is claimed).
  let tileClaim: { elementId: string; dispose(): void } | null = null;
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
  const decodeToHandle = async (
    bytes: Uint8Array,
  ): Promise<{ handle: number; width: number; height: number }> => {
    if (!engine) throw new Error("engine not booted");
    const pool = await ensureDecodePool();
    if (pool) {
      // The pool transfers the input buffer — copy first so the caller's
      // bytes survive (the importer may reuse them).
      const copy = bytes.slice();
      const decoded = await pool.decode(copy);
      return engine.ingestRgba8(decoded.width, decoded.height, decoded.rgba);
    }
    return engine.decode(bytes);
  };

  const releaseTiles = () => {
    if (tileClaim) {
      tileClaim.dispose();
      tileClaim = null;
    }
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
    if (state.source && engine) engine.freeImage(state.source.handle);
    state.source = null;
    state.histogram = null;
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
    if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff)
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

  const refreshPsdLayers = (): void => {
    if (psdHandle === null || !engine || !state.psd) return;
    try {
      state.psd = { ...state.psd, layers: engine.psdLayers(psdHandle) };
    } catch {
      // A failed refresh leaves the last-known list; edits still saved.
    }
  };

  /** Recompute the histogram + (re)build the crop machine for the live
   *  engine-held source. Called after a decode/ingest and after a crop
   *  commit (both change the source pixels). The histogram is the panel's
   *  levels/curves readout. A histogram-read failure is non-fatal (the
   *  panel just shows no histogram). */
  const refreshSourceReadout = () => {
    if (!engine || !state.source) return;
    try {
      state.histogram = engine.histogram(state.source.handle);
    } catch (err) {
      host.log.debug("histogram read failed", err);
      state.histogram = null;
    }
    cropMachineRef = createCropMachine(engine, state.source.width, state.source.height);
    // Bind the engine selection to the LIVE handle (a crop/resize swap
    // is a different handle, so the old selection drops engine-side —
    // honest: a selection is meaningless across resolutions) and rebuild
    // the machine + readout.
    try {
      engine.selectionBind(state.source.handle);
    } catch (err) {
      host.log.debug("selection bind failed", err);
    }
    selectionMachineRef = createSelectionMachine(engine, () => api.refreshSelection());
    state.selection = engine.selectionStats();
    // The paint machine is pure pointer bookkeeping, but it is rebuilt
    // with the source so a stale sample queue can never leak across a
    // crop / resize / fill / paint handle swap.
    brushMachineRef = createBrushMachine();
  };

  /** The frame's content box in page pt (falls back to the image's own
   *  extent when the geometry read misses — an honest 1:1 rather than a
   *  throw). */
  const frameBox = async (target: string): Promise<{ w: number; h: number }> => {
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
      };
      sourceFormat = sniffFormat(bytes);
      state.saveBack = null;
      // The PSD structural session rides EVERY PSD ingest (the K-2
      // import AND the C-5 selection lane) — it is what the PSD
      // save-back writes into.
      openPsdSession(name, bytes);
      // Compute the levels/curves histogram + build the crop machine.
      refreshSourceReadout();
      const lane = decodePool ? " (off-thread)" : "";
      setStatus(`${name} — ${info.width}×${info.height} decoded${lane}.`);
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
        setStatus("Host serves no placed-image bytes (assets.images@1 is false).");
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
          setStatus("No placed image on this frame (or the link is unresolved).");
          return false;
        }
        return await decodeInto(asset.uri || "placed image", asset.bytes, "selection", id);
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
        const ok = await decodeInto(name, bytes, "import", null);
        if (ok) {
          setStatus(
            `${name} — ${state.source?.width}×${state.source?.height} decoded. ` +
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
        engine.psdSetLayerOpacity(psdHandle, index, Math.max(0, Math.min(255, Math.round(opacity))));
        refreshPsdLayers();
        emit();
        return true;
      } catch (err) {
        setStatus(`PSD edit failed: ${err instanceof Error ? err.message : err}`);
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
        setStatus(`PSD rename failed: ${err instanceof Error ? err.message : err}`);
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
        setStatus(`PSD remove failed: ${err instanceof Error ? err.message : err}`);
        emit();
        return false;
      }
    },

    psdExport() {
      if (psdHandle === null || !engine || !state.psd) return null;
      try {
        return { bytes: engine.psdSave(psdHandle), fileName: state.psd.name };
      } catch (err) {
        setStatus(`PSD save failed: ${err instanceof Error ? err.message : err}`);
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
        setStatus("WebGPU unavailable — the adjusted save-back needs a device.");
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
          const shape = engine.psdApplyAdjusted(psdHandle, src.width, src.height, rgba);
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
            "Deliver it from the Export Center (the host wires no save-file door).",
        );
        return result;
      } catch (err) {
        setStatus(`Save-back failed: ${err instanceof Error ? err.message : err}`);
        return null;
      } finally {
        state.busy = false;
        emit();
      }
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

    async fillSelection(req) {
      const src = state.source;
      if (!src || !engine) {
        setStatus("Nothing ingested — ingest a placed image first.");
        return false;
      }
      if (!state.gpu) {
        setStatus("WebGPU unavailable — the generators are GPU-only kernels.");
        return false;
      }
      state.busy = true;
      emit();
      let filled: { handle: number; width: number; height: number };
      try {
        filled =
          req.kind === "gradient"
            ? await engine.fillGradient(src.handle, req.gradient, req.c0, req.c1)
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
      // Swap the engine-held source (the crop-commit pattern): the fill
      // is DESTRUCTIVE into the working image by design (see fill.rs).
      releaseTiles();
      engine.freeImage(src.handle);
      src.handle = filled.handle;
      src.width = filled.width;
      src.height = filled.height;
      state.saveBack = null;
      // The fill is same-size, so the SELECTION is still meaningful on
      // the result — carry it over instead of making the user reselect
      // (a crop/resize swap still drops it; the extents differ there).
      try {
        engine.selectionTransfer(filled.handle);
      } catch (err) {
        host.log.debug("selection transfer failed", err);
      }
      refreshSourceReadout();
      const where = state.selection ? "the selection" : "the whole image";
      setStatus(
        `Filled ${where} with ${
          req.kind === "gradient" ? `a ${req.gradient} gradient` : "noise"
        } (engine source only — document unchanged; Apply to recomposite).`,
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
          levels: { ...state.params.levels, inBlack: a.inBlack, inWhite: a.inWhite },
        };
        emit();
        setStatus(
          "Auto-enhance set levels + white balance — click Apply to composite " +
            "(document unchanged).",
        );
      } catch (err) {
        setStatus(`Auto-enhance failed: ${err instanceof Error ? err.message : err}`);
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
        setStatus(`Select all failed: ${err instanceof Error ? err.message : err}`);
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
        setStatus(`Invert selection failed: ${err instanceof Error ? err.message : err}`);
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
        setStatus(`Feather failed: ${err instanceof Error ? err.message : err}`);
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

    setBrushParams(p) {
      state.brush = { ...state.brush, ...p };
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
      try {
        engine.brushBegin(src.handle, tool, state.brush);
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
      let painted: { handle: number; width: number; height: number };
      try {
        painted = engine.brushCommit();
      } catch (err) {
        setStatus(`Paint commit failed: ${err instanceof Error ? err.message : err}`);
        discardStroke();
        emit();
        return false;
      }
      state.strokeActive = false;
      state.strokeStats = null;
      strokeTarget = null;
      strokeBox = null;
      brushMachineRef?.cancel();
      // Swap the engine-held source (the crop / fill commit pattern):
      // DESTRUCTIVE into the working pixels, document untouched.
      releaseTiles();
      engine.freeImage(src.handle);
      src.handle = painted.handle;
      src.width = painted.width;
      src.height = painted.height;
      state.saveBack = null;
      // Same size, so the SELECTION is still meaningful on the result —
      // carry it over instead of making the user reselect (the fill
      // lane's rule; a crop/resize swap still drops it).
      try {
        engine.selectionTransfer(painted.handle);
      } catch (err) {
        host.log.debug("selection transfer failed", err);
      }
      refreshSourceReadout();
      // Put the ADJUSTED composite back over the raw stroke preview.
      if (src.elementId) await api.apply();
      setStatus(
        `Painted ${dabs} dab${dabs === 1 ? "" : "s"} into the engine image ` +
          "(destructive — there is no layer graph; the document and the " +
          "source file are unchanged, re-ingest to restore).",
      );
      return true;
    },

    brushCancel() {
      if (!state.strokeActive) return;
      discardStroke();
      setStatus("Stroke cancelled — the engine pixels are exactly as they were.");
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
        await submitLayer(target, rgba, src.width, src.height, await frameBox(target));
        // The engine masked the chain by the bound selection (if any) —
        // say so, honestly.
        const sel = state.selection ? " (adjustments masked to the selection)" : "";
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
        { elementId: target, handle: src.handle, width: src.width, height: src.height },
        engine,
        () => (state.source ? state.source.handle : null),
      );
      tileClaim = { elementId: target, dispose: () => claim.dispose() };
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
      cropMachineRef?.reset(state.source?.width ?? 0, state.source?.height ?? 0);
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
