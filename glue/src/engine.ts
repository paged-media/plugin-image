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

/** The committed adjustment parameters. Identity = every field neutral:
 *  exposure 0 / brightness 0 / contrast 1 / saturation 1, white balance
 *  0/0, levels identity, no curve LUT. */
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
}

export const IDENTITY_PARAMS: AdjustParams = {
  exposureEv: 0,
  brightness: 0,
  contrast: 1,
  saturation: 1,
  temp: 0,
  tint: 0,
  levels: { ...IDENTITY_LEVELS },
  curveLut: null,
};

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
    p.curveLut === null
  );
}

/** True when ONLY the base exposure/brightness/contrast/saturation are set
 *  (no WB / levels / curves) — the legacy `adjust_image` fast path. */
function isBaseOnly(p: AdjustParams): boolean {
  return (
    p.temp === 0 &&
    p.tint === 0 &&
    levelsIdentity(p.levels) &&
    p.curveLut === null
  );
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
}

/** The stable engine contract the bundle codes against. Every method
 *  forwards to the wasm surface; the facade only renames + shapes. */
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
  /** Hit-test the crop chrome (the nearest grip within `tol`, else Move
   *  inside the body, else -1). Pure geometry from `image_core::crop`. */
  cropHitHandle(rect: CropRect, point: [number, number], tol: number): CropHandle;
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
  /** C-6 — copy a LEVEL-0 tile window `(x, y, w, h)` out of a decoded
   *  image as tightly packed RGBA8. Edge tiles are clamped to the image
   *  extent; a fully-outside window returns an empty buffer. The honest
   *  subset of the resource provider (pure windowing — no mip pyramid /
   *  Engine B window eval yet; see tile-provider.ts). */
  tile(handle: number, x: number, y: number, w: number, h: number): Uint8Array;
  freeImage(handle: number): void;
}

// ---------------------------------------------------- wasm surface shape

interface DecodedHandleWasm {
  handle: number;
  width: number;
  height: number;
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
  ingest_rgba8(width: number, height: number, bytes: Uint8Array): DecodedHandleWasm;
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
  ): Promise<Uint8Array>;
  image_histogram(handle: number): Uint32Array;
  image_auto_enhance_params(handle: number): Float32Array;
  crop_image(
    handle: number,
    x: number,
    y: number,
    w: number,
    h: number,
  ): DecodedHandleWasm;
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
      const info = { handle: h.handle, width: h.width, height: h.height };
      h.free();
      return info;
    },
    ingestRgba8(width, height, rgba) {
      const h = wasm.ingest_rgba8(width, height, rgba);
      const info = { handle: h.handle, width: h.width, height: h.height };
      h.free();
      return info;
    },
    adjust: (handle, p) => {
      // Base-only params take the legacy 4-scalar fast path; anything in
      // the FULL panel set (WB / levels / curves) routes to the extended
      // surface, passing the curve LUT (empty = no curve).
      if (isBaseOnly(p)) {
        return wasm.adjust_image(handle, p.exposureEv, p.brightness, p.contrast, p.saturation);
      }
      return wasm.adjust_image_full(
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
      );
    },
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
      const info = { handle: h.handle, width: h.width, height: h.height };
      h.free();
      return info;
    },
    cropHitHandle: (rect, point, tol) =>
      wasm.crop_hit_handle(rect.x, rect.y, rect.w, rect.h, point[0], point[1], tol),
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
      const f = wasm.crop_frame_corners(rect.x, rect.y, rect.w, rect.h, degrees);
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
    const wasmPath = require.resolve(
      "../wasm/image_js_bg.wasm",
    );
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
