/* tslint:disable */
/* eslint-disable */

/**
 * A decoded image's identity on the surface: the handle keys the
 * engine-held pixels; width/height are the natural extent.
 */
export class DecodedHandle {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    handle: number;
    height: number;
    width: number;
}

export function abi_version(): number;

/**
 * Run the M4 adjustments chain on a decoded image and return the
 * straight-RGBA8 result — the C-1 Stage-A scene-item payload.
 * Identity params return the decode verbatim (no dispatch to run);
 * anything else requires `init_gpu` to have succeeded.
 */
export function adjust_image(handle: number, exposure_ev: number, brightness: number, contrast: number, saturation: number): Promise<Uint8Array>;

/**
 * The FULL adjustments pass — the levels/curves/white-balance panel's
 * committed values. The 9 scalars are exposure/brightness/contrast/
 * saturation (as `adjust_image`), white balance (temp/tint), and the
 * composite levels in/gamma/out window; `curve_lut` is an OPTIONAL
 * 256-byte tone LUT (the panel builds it from its curve control points
 * via `image_core::curve_lut`; pass an empty array for no curve). The
 * curves stage is a CPU LUT pass (no GPU LUT kernel yet — the honest
 * deferral); everything else is the GPU adjust chain. Returns straight
 * RGBA8 (the C-1 Stage-A scene payload).
 */
export function adjust_image_full(handle: number, exposure_ev: number, brightness: number, contrast: number, saturation: number, temp: number, tint: number, in_black: number, in_white: number, gamma: number, out_black: number, out_white: number, curve_lut: Uint8Array, blur_sigma: number, sharpen_amount: number, hue_degrees: number, invert: boolean): Promise<Uint8Array>;

/**
 * Apply a pointer drag from `(sx, sy)` to `(px, py)` (image-px) to the
 * rect `[x, y, w, h]` at `handle` (the [`crop_hit_handle`]
 * discriminant), with the aspect lock + image-extent clamp. Returns
 * the new rect as `[x, y, w, h]`. An unknown handle returns the rect
 * unchanged (defensive).
 */
export function crop_apply_drag(x: number, y: number, w: number, h: number, handle: number, sx: number, sy: number, px: number, py: number, aspect_w: number, aspect_h: number, image_w: number, image_h: number): Float32Array;

/**
 * The four corners of the crop FRAME rotated by the straighten
 * `degrees`, as a flat `[x0,y0, x1,y1, x2,y2, x3,y3]` (TL, TR, BR, BL)
 * the overlay draws as a closed polyline.
 */
export function crop_frame_corners(x: number, y: number, w: number, h: number, degrees: number): Float32Array;

/**
 * Hit-test the crop chrome at `(px, py)` (image-px) against the rect
 * `[x, y, w, h]` with grab radius `tol`. Returns the [`image_core::
 * Handle`] discriminant (0..=7 grips, 8 = body Move) or `-1` for a
 * miss — the TS machine maps it to a cursor + the active grip.
 */
export function crop_hit_handle(x: number, y: number, w: number, h: number, px: number, py: number, tol: number): number;

/**
 * Commit a CROP: cut the integer pixel rectangle `(x, y, w, h)`
 * (clamped to the image extent) out of an engine-held image and
 * register the result as a NEW engine-held image, returning its
 * handle. The source handle is left intact (the caller frees it). An
 * out-of-bounds / empty rectangle is a clean error (never a torn
 * image). The straighten-angle resample is a separate stage (not in
 * this axis-aligned cut — see the crop interaction machine).
 */
export function crop_image(handle: number, x: number, y: number, w: number, h: number): DecodedHandle;

/**
 * Build a 256-byte tone LUT from flat `[i0,o0, i1,o1, …]` curve
 * control points in `[0,1]` (the CURVES editor's points) — the LUT
 * `adjust_image_full` consumes. Wraps `image_core::curve_lut`.
 */
export function curve_lut(points: Float32Array): Uint8Array;

/**
 * Decode PSD/PNG/JPEG bytes (sniffed by magic) into an engine-held
 * RGBA8 image. Free with `free_image`.
 */
export function decode_image(bytes: Uint8Array): DecodedHandle;

/**
 * Release an engine-held decoded image (and its mip pyramid cache).
 */
export function free_image(handle: number): void;

/**
 * Whether `init_gpu` succeeded (the glue probes this to gate the
 * adjust controls honestly).
 */
export function gpu_ready(): boolean;

/**
 * Compute the AUTO-ENHANCE adjustment parameters for an engine-held
 * image and return them as `[in_black, in_white, temp, tint]` (4
 * `f32`). A single "auto" estimate composing the EXISTING levels +
 * white-balance kernels: it builds the RGB+luma histogram (the same
 * `histogram_rgba8` reduction the panel reads), derives a percentile-
 * clipped auto-levels black/white range (0.5%/99.5% of luma) and a
 * gray-world white-balance `temp`/`tint`, and emits the params the
 * LEVELS/WB panel commits through `adjust_image_full` (levels
 * `in_black`/`in_white`, white-balance `temp`/`tint`; gamma/output
 * range stay identity). Pure CPU readout/orchestration (spec §6) —
 * deterministic, no GPU, no kernel dispatch row. A flat or already-
 * neutral image yields the identity `[0, 1, 0, 0]` (a guaranteed
 * no-op), never a wrong-looking auto-correction.
 */
export function image_auto_enhance_params(handle: number): Float32Array;

/**
 * Compute the RGB + luma 256-bin histogram of an engine-held image as
 * a flat `[r…, g…, b…, luma…]` 1024-`u32` array (the LEVELS / CURVES
 * panel slices it into four channels). Pure CPU reduction over the
 * straight-RGBA8 buffer (no GPU); deterministic.
 */
export function image_histogram(handle: number): Uint32Array;

/**
 * C-6 (I-06) — copy a LEVEL-0 tile window `(x, y, w, h)` out of a
 * decoded image as tightly packed RGBA8 (`w*h*4` bytes, row-major).
 * Edge tiles are clamped to the image extent (the caller passes the
 * requested grid origin + size; the returned buffer is the clipped
 * intersection). This is the HONEST SUBSET of the resource provider:
 * pure windowing of the already-decoded buffer (no resampling kernel,
 * no GPU dispatch — orchestration, spec §6). The mip pyramid + the
 * Engine B `(node, region, level)` window evaluation
 * (`image_graph::BufferGraph::request`, rgba16float) are NOT yet
 * wired across this wasm boundary — see the gap note in
 * glue/src/tile-provider.ts. Returns an empty buffer when the window
 * lies fully outside the image (a transparent miss the provider skips).
 */
export function image_tile_rgba8(handle: number, x: number, y: number, w: number, h: number): Uint8Array;

/**
 * C-6 (I-06) — copy a tile window `(x, y, w, h)` out of a decoded
 * image at mip `level` as tightly packed RGBA8 (`w'*h'*4` bytes,
 * clipped to the level extent, row-major). `level == 0` is the fast
 * level-0 path ([`image_tile_rgba8`]'s pure windowing); `level > 0`
 * routes through Engine B's tiled buffer graph
 * (`image_graph::BufferGraph`): a 2×-box mip pyramid of rgba16float
 * source tiles is built once per handle (cached) and the requested
 * window is gathered from `(level, coord)` source reads and
 * downconverted back to RGBA8. The coordinates are in the LEVEL's
 * pixel space (already halved per level — the caller scales). No GPU
 * is required (a source read carries no kernel dispatch). Returns an
 * empty buffer when the window lies fully outside the level, or when
 * `level` exceeds the pyramid top (a transparent miss the provider
 * skips). `max_level` bounds the pyramid height built on first touch.
 */
export function image_tile_rgba8_level(handle: number, level: number, x: number, y: number, size: number): Uint8Array;

/**
 * K-3 (S-07 / I-02) — register a PRE-DECODED straight-RGBA8 buffer
 * (from the decode worker pool, which ran the codec/PSD CPU lanes
 * off-thread) as an engine-held image, returning a handle for the GPU
 * adjust + tile paths. `bytes` must be exactly `width*height*4` RGBA8;
 * a length mismatch is a clean error. Free with `free_image`.
 */
export function ingest_rgba8(width: number, height: number, bytes: Uint8Array): DecodedHandle;

export function init(): void;

/**
 * Request the WebGPU adapter/device for kernel execution.
 * Idempotent. Rejects when the environment has no WebGPU — the
 * honest no-GPU state (no CPU kernel path ships, spec §6).
 */
export function init_gpu(): Promise<void>;

export function kernel_count(): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_decodedhandle_free: (a: number, b: number) => void;
    readonly __wbg_get_decodedhandle_handle: (a: number) => number;
    readonly __wbg_get_decodedhandle_height: (a: number) => number;
    readonly __wbg_get_decodedhandle_width: (a: number) => number;
    readonly __wbg_set_decodedhandle_handle: (a: number, b: number) => void;
    readonly __wbg_set_decodedhandle_height: (a: number, b: number) => void;
    readonly __wbg_set_decodedhandle_width: (a: number, b: number) => void;
    readonly abi_version: () => number;
    readonly adjust_image: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly adjust_image_full: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => any;
    readonly crop_apply_drag: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly crop_frame_corners: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly crop_hit_handle: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly crop_image: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly curve_lut: (a: number, b: number) => [number, number];
    readonly decode_image: (a: number, b: number) => [number, number, number];
    readonly free_image: (a: number) => void;
    readonly gpu_ready: () => number;
    readonly image_auto_enhance_params: (a: number) => [number, number, number];
    readonly image_histogram: (a: number) => [number, number, number];
    readonly image_tile_rgba8: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly image_tile_rgba8_level: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly ingest_rgba8: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly init: () => void;
    readonly init_gpu: () => any;
    readonly kernel_count: () => number;
    readonly qcms_enable_iccv4: () => void;
    readonly qcms_profile_precache_output_transform: (a: number) => void;
    readonly qcms_transform_data_bgra_out_lut: (a: number, b: number, c: number, d: number) => void;
    readonly qcms_transform_data_bgra_out_lut_precache: (a: number, b: number, c: number, d: number) => void;
    readonly qcms_transform_data_rgb_out_lut: (a: number, b: number, c: number, d: number) => void;
    readonly qcms_transform_data_rgb_out_lut_precache: (a: number, b: number, c: number, d: number) => void;
    readonly qcms_transform_data_rgba_out_lut: (a: number, b: number, c: number, d: number) => void;
    readonly qcms_transform_data_rgba_out_lut_precache: (a: number, b: number, c: number, d: number) => void;
    readonly qcms_transform_release: (a: number) => void;
    readonly qcms_profile_is_bogus: (a: number) => number;
    readonly qcms_white_point_sRGB: (a: number) => void;
    readonly lut_inverse_interp16: (a: number, b: number, c: number) => number;
    readonly lut_interp_linear16: (a: number, b: number, c: number) => number;
    readonly wasm_bindgen__convert__closures_____invoke__hb3a19924738e3ab7: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h9344d28c99dc1e49: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h03919243d83d1356: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
