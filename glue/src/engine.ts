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

/** The committed adjustment parameters (identity: 0 / 0 / 1 / 1). */
export interface AdjustParams {
  exposureEv: number;
  brightness: number;
  contrast: number;
  saturation: number;
}

export const IDENTITY_PARAMS: AdjustParams = {
  exposureEv: 0,
  brightness: 0,
  contrast: 1,
  saturation: 1,
};

export function isIdentity(p: AdjustParams): boolean {
  return (
    p.exposureEv === 0 &&
    p.brightness === 0 &&
    p.contrast === 1 &&
    p.saturation === 1
  );
}

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
  /** Run the adjustments chain (GPU) and return straight RGBA8 — the
   *  C-1 Stage-A scene-item payload. Identity params return the decode
   *  verbatim without touching the GPU. */
  adjust(handle: number, params: AdjustParams): Promise<Uint8Array>;
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
  adjust_image(
    handle: number,
    exposure_ev: number,
    brightness: number,
    contrast: number,
    saturation: number,
  ): Promise<Uint8Array>;
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
    adjust: (handle, p) =>
      wasm.adjust_image(handle, p.exposureEv, p.brightness, p.contrast, p.saturation),
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
    mod = (await import("@paged-media/image-manifest/wasm/image_js.js")) as ImageWasmModule;
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
      "@paged-media/image-manifest/wasm/image_js_bg.wasm",
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
      "@paged-media/image-manifest/wasm/image_js_bg.wasm?url"
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
