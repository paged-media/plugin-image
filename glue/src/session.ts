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
  isIdentity,
  IDENTITY_PARAMS,
  type AdjustParams,
  type ImageEngine,
} from "./engine";
import { claimImageTiles } from "./tile-provider";
import { createDecodePool, type DecodePool } from "./decode-pool";

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
  /** A scene layer is currently submitted for `compositedFrame`. */
  compositedFrame: string | null;
  busy: boolean;
  /** One-line panel status (honest, never fake-progress). */
  status: string;
}

export interface ImageSession {
  state(): ImageSessionState;
  onDidChange(listener: () => void): Disposable;
  /** Ingest the single selected element's placed image via C-5. */
  ingestSelection(): Promise<boolean>;
  /** Ingest opened/dropped file bytes (the K-2 importer path). */
  importBytes(name: string, bytes: Uint8Array): Promise<boolean>;
  setParams(p: Partial<AdjustParams>): void;
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
  let disposed = false;

  const state: ImageSessionState = {
    engine: "idle",
    engineDetail: null,
    gpu: false,
    source: null,
    params: { ...IDENTITY_PARAMS },
    compositedFrame: null,
    busy: false,
    status: "Select a placed image frame, then ingest.",
  };

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

  const freeSource = () => {
    // A claim points at THIS source's handle — release it before the
    // pixels go (the renderer drops to the whole-image fallback lane).
    releaseTiles();
    if (state.source && engine) engine.freeImage(state.source.handle);
    state.source = null;
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

  return {
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

        // Frame content box (the layer is clipped + transformed by core;
        // §8.5 — the plugin never compensates). Aspect-fit, centered.
        let boxW = src.width;
        let boxH = src.height;
        try {
          const geom = await host.document.elementGeometry([
            host.selection.get().find((i) => elementIdOf(i) === target) ??
              ({ kind: "rectangle", id: target } as never),
          ]);
          const bounds = geom[0]?.bounds;
          if (bounds) {
            const [top, left, bottom, right] = bounds;
            boxW = Math.max(right - left, 1);
            boxH = Math.max(bottom - top, 1);
          }
        } catch (err) {
          host.log.debug("apply: frame geometry read failed", err);
        }
        const scale = Math.min(boxW / src.width, boxH / src.height);
        const w = src.width * scale;
        const h = src.height * scale;
        const x = (boxW - w) / 2;
        const y = (boxH - h) / 2;

        await surface.submit(target, {
          items: [
            {
              kind: "image",
              rgba: Array.from(rgba),
              width: src.width,
              height: src.height,
              x,
              y,
              w,
              h,
            },
          ],
        });
        state.compositedFrame = target;
        setStatus(
          `Composited ${src.width}×${src.height} into the frame ` +
            "(document unchanged — preview layer only).",
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
      state.params = { ...IDENTITY_PARAMS };
      await clearLayer();
      setStatus("Reset — in-frame preview cleared.");
    },

    dispose() {
      disposed = true;
      selectionSub.dispose();
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
}
