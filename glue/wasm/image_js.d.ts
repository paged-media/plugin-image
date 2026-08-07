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
    /**
     * CMS rung 1 — what the RGB display transform did at decode, as a
     * discriminant the bundle maps to a label: 0 = ICC managed,
     * 1 = sRGB assumed (no embedded profile), 2 = sRGB assumed
     * because an embedded profile was rejected. Surfaced so the panel
     * can STATE the colour treatment instead of leaving the user to
     * guess which numbers they are looking at.
     */
    display: number;
    handle: number;
    height: number;
    width: number;
}

export function abi_version(): number;

/**
 * Read a Photoshop `.abr` brush library and return its presets as
 * JSON — the door that makes the `.abr` reader REACHABLE.
 *
 * Without a caller a wasm32 release build eliminates the whole
 * parser, so this is the difference between a capability that
 * exists in the repository and one that exists in the artifact.
 * The projection (which parameters, and why the absent ones stay
 * absent) lives in [`crate::brushes`], which is host-testable —
 * `mod wasm` is `#[cfg(target_arch = "wasm32")]` and never is.
 */
export function abr_presets(bytes: Uint8Array): string;

/**
 * Run the M4 adjustments chain on a decoded image and return the
 * straight-RGBA8 result — the C-1 Stage-A scene-item payload.
 * Identity params return the decode verbatim (no dispatch to run);
 * anything else requires `init_gpu` to have succeeded.
 */
export function adjust_image(handle: number, exposure_ev: number, brightness: number, contrast: number, saturation: number): Promise<Uint8Array>;

/**
 * [`adjust_image_full`] PLUS the EXTENDED (kernel-breadth) stages —
 * vibrance, color balance, black & white, posterize, threshold,
 * photo filter, channel mixer and per-channel levels — carried in
 * ONE flat `f32` block so the boundary does not grow an argument
 * per stage. `ext` is either EMPTY (every extended stage at
 * identity — what `adjust_image_full` passes) or exactly
 * `ingest::ADJUST_EXT_LEN` floats in the layout documented on that
 * constant. The chain order is documented on `ingest::adjust_rgba8`;
 * every stage is mask-aware (the bound selection rides `@group(2)`).
 */
export function adjust_image_ext(handle: number, exposure_ev: number, brightness: number, contrast: number, saturation: number, temp: number, tint: number, in_black: number, in_white: number, gamma: number, out_black: number, out_white: number, curve_lut: Uint8Array, blur_sigma: number, sharpen_amount: number, hue_degrees: number, invert: boolean, ext: Float32Array): Promise<Uint8Array>;

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
 * APPLY a gradient map — luminance through a two-stop colour ramp.
 * A pixel edit into the active layer, journaled and selection-masked
 * exactly like a fill, because that is what it is.
 */
export function apply_gradient_map(handle: number, shadow: Float32Array, highlight: Float32Array): Promise<DecodedHandle>;

/**
 * APPLY a parametric distortion (`geom.warp_backward`). `kind` is
 * 0 pinch / 1 spherize / 2 twirl / 3 wave; `amount == 0` is the
 * identity for every kind, so a UI slider needs no special cases.
 */
export function apply_warp(handle: number, kind: number, amount: number, frequency: number): Promise<DecodedHandle>;

/**
 * Every blend mode a stroke can paint through, newline-separated —
 * derived from the `compose.*` registry so the panel's picker can
 * never drift from the kernels that actually exist.
 */
export function brush_blend_modes(): string;

/**
 * Is a stroke in progress?
 */
export function brush_stroke_active(): boolean;

/**
 * BEGIN a stroke on the engine-held image `handle`.
 *
 * `tool` ∈ `brush | pencil | eraser`; `blend` is a `compose.*`
 * kernel name with the prefix optional (`"multiply"` or
 * `"compose.multiply"`); `pressure_target` ∈
 * `none | size | opacity | both` selects what the pen's pressure
 * drives (default `both` — size AND opacity, the Photoshop pen
 * preset). `color` is 4 straight RGBA floats in `[0, 1]`.
 *
 * PRESSURE, honestly: `PointerEvent.pressure` is a constant `0.5`
 * for a mouse and a real reading only for a pen. The CALLER
 * normalizes — the glue passes `1.0` for a mouse so a mouse stroke
 * is not permanently half-size — and this door takes whatever it is
 * given verbatim so a recorded stroke replays identically.
 *
 * Parameters are FROZEN for the stroke's duration: a stroke whose
 * size changed halfway through would not be replayable.
 * GPU-only — rejects without `init_gpu`.
 */
export function brush_stroke_begin(handle: number, tool: string, size: number, hardness: number, opacity: number, flow: number, spacing: number, blend: string, color: Float32Array, pressure_target: string): void;

/**
 * CANCEL the stroke: throw the painted pixels away. The engine-held
 * source was never mutated, so this restores it exactly.
 */
export function brush_stroke_cancel(): void;

export function brush_stroke_commit(): Promise<DecodedHandle>;

/**
 * EXTEND the stroke with one pointer sample (image px + normalized
 * pressure) and return the resulting straight RGBA8 for the WHOLE
 * image — the C-1 Stage-A preview payload.
 *
 * Dabs are interpolated from the previous sample at
 * `spacing · diameter` px of arc length (the residual carries across
 * samples), so a fast drag paints a continuous stroke rather than
 * one dot per pointer event. Only the dirty rectangle is
 * re-composited, always FROM the base pixels, so extending is
 * idempotent and the incremental result equals a from-scratch
 * composite of the same samples.
 */
export function brush_stroke_extend(x: number, y: number, pressure: number): Promise<Uint8Array>;

/**
 * COMMIT the stroke.
 *
 * * **With a layer stack bound** (the normal case): the painted
 *   pixels are written into the ACTIVE LAYER, the tiles the stroke's
 *   bounding box covers are journaled first (so the stroke is
 *   undoable, tile-granularly, within the journal's stated bound),
 *   and the stack is re-composited into the SAME engine-held image.
 *   The returned handle is therefore the handle you started with —
 *   the caller must NOT free it.
 * * **Without one**: the pre-layer behaviour — the painted pixels
 *   are registered as a NEW engine-held image and the caller swaps
 *   handles and frees the old one.
 *
 * Either way the result is the same size, so the caller may carry
 * the selection over with `selection_transfer`.
 * Point the IN-FLIGHT clone/heal stroke at its source (the
 * alt-click anchor), in image px.
 *
 * Must be called between `brush_stroke_begin` and the first
 * `brush_stroke_extend`: the source offset is fixed at the first
 * dab, because an offset that moved mid-stroke would smear the copy
 * instead of translating it. Calling it for a non-sampling tool is
 * an ERROR rather than a no-op — silently accepting it would let a
 * caller believe the brush was cloning.
 */
export function brush_stroke_set_source(x: number, y: number, aligned: boolean): void;

/**
 * The in-flight stroke's readout for the panel:
 * `[dabs, x, y, w, h]` — the dab count and the stroke's bounding
 * box in image px. Empty when no stroke is in progress or nothing
 * has landed on the canvas yet.
 */
export function brush_stroke_stats(): Float64Array;

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
 * image). This door is the AXIS-ALIGNED cut only — the straighten
 * angle rides `straighten_crop_image`, which rotates first.
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
 * Re-encode straight RGBA8 as PNG or JPEG (`format` ∈ `png | jpeg`)
 * — the NON-PSD save-back lane. Codec entropy coding is inherently
 * CPU work (spec §1); JPEG rides the fixed v0 quality documented on
 * `saveback::JPEG_QUALITY_DEFAULT`.
 */
export function encode_image(rgba: Uint8Array, width: number, height: number, format: string): Uint8Array;

/**
 * CONTENT-AWARE FILL: synthesise the selection from the rest of the
 * image (exemplar-based inpainting).
 *
 * Unlike every other fill here it is CPU: it is a search, not a
 * dispatch, and there is no kernel that could express "find the
 * patch elsewhere in this image that best continues this one". The
 * GPU-only rule (spec §6) is about the KERNEL path, and this adds
 * none — it produces pixels that land through the same journaled
 * layer write as any other fill.
 *
 * Requires a SELECTION: with nothing selected there is no hole, and
 * with everything selected there is no source. Both are errors
 * rather than a silent no-op, because a fill that quietly did
 * nothing would read as a broken button.
 */
export function fill_content_aware(handle: number): Promise<DecodedHandle>;

/**
 * FILL the current selection (the whole image when none) with a
 * fixed TWO-STOP gradient. `kind` ∈ `linear | radial | angular |
 * reflected | diamond`; `c0`/`c1` are straight RGBA in `[0, 1]`
 * (4 floats each). The gradient GEOMETRY is derived from the
 * selection's bounding box — there is no on-canvas drag handle in
 * v0 (`crate::fill` documents the derivation). Returns the NEW
 * engine-held image's handle.
 */
export function fill_gradient(handle: number, kind: string, c0: Float32Array, c1: Float32Array): Promise<DecodedHandle>;

/**
 * FILL the current selection (the whole image when none) with
 * deterministic monochrome noise — `amount` scales the hash
 * amplitude, `seed` makes a repeat reproducible. Returns the NEW
 * engine-held image's handle.
 */
export function fill_noise(handle: number, amount: number, seed: number): Promise<DecodedHandle>;

/**
 * Release an engine-held decoded image (its mip pyramid cache, and
 * the layer stack bound to it — a stack whose composite target is
 * gone has nowhere to land).
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
 * The CHANNELS readout for an engine-held image: `[{name, min, max,
 * mean}]` for red/green/blue/alpha and the derived Rec.709 luma.
 * Pure CPU reduction over the same straight-RGBA8 buffer
 * `image_histogram` reads, so the two agree by construction.
 */
export function image_channel_stats(handle: number): string;

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

/**
 * Add an empty transparent layer above the active one (it becomes
 * active). Returns its index.
 */
export function layers_add(name: string): number;

/**
 * Insert an ADJUSTMENT LAYER carrying the panel's current chain.
 *
 * The non-destructive counterpart of `layers_bake_adjust` below: the
 * bake writes the chain into the active layer's pixels and journals
 * it; this stacks the chain ABOVE and touches no pixel at all, so
 * deleting the layer restores the original exactly. Same wire block
 * so the two can never disagree about what the panel meant.
 *
 * Refuses at identity — an adjustment layer that adjusts nothing is
 * a row that does nothing, and adding one silently is worse than
 * saying so.
 */
export function layers_add_adjustment(name: string, exposure_ev: number, brightness: number, contrast: number, saturation: number, temp: number, tint: number, in_black: number, in_white: number, gamma: number, out_black: number, out_white: number, curve_lut: Uint8Array, blur_sigma: number, sharpen_amount: number, hue_degrees: number, invert: boolean, ext: Float32Array): number;

/**
 * BAKE the adjustment chain into the ACTIVE layer — the DESTRUCTIVE
 * per-layer adjustment (the panel's chain is otherwise a re-runnable
 * PREVIEW of the composite and mutates nothing). Journaled over the
 * whole canvas, so it is undoable; refuses on a locked layer and at
 * identity. Arguments mirror `adjust_image_ext` minus the handle.
 */
export function layers_bake_adjust(exposure_ev: number, brightness: number, contrast: number, saturation: number, temp: number, tint: number, in_black: number, in_white: number, gamma: number, out_black: number, out_white: number, curve_lut: Uint8Array, blur_sigma: number, sharpen_amount: number, hue_degrees: number, invert: boolean, ext: Float32Array): Promise<Uint8Array>;

/**
 * The handle the stack is bound to, or `-1` when none is open.
 */
export function layers_bound(): number;

/**
 * DELETE the mask (the coverage is gone), as distinct from
 * disabling it.
 */
export function layers_clear_mask(index: number): void;

/**
 * Drop the bound stack (and its undo history).
 */
export function layers_close(): void;

/**
 * COMPOSITE the stack bottom-up and write the result back into the
 * bound engine-held image, returning the straight RGBA8 (the C-1
 * Stage-A payload). GPU-only whenever there is anything to blend; a
 * single plain visible layer short-circuits to its own pixels with
 * no dispatch at all, so a one-layer document needs no device.
 */
export function layers_composite(): Promise<Uint8Array>;

/**
 * Duplicate `index` above itself (the copy becomes active).
 */
export function layers_duplicate(index: number): number;

/**
 * Toggle whether the attached mask applies, RETAINING it either way
 * — losing painted coverage to a toggle would be a real loss.
 * Clip a layer to the one beneath it — the mechanism "smart
 * filters" wanted: an adjustment layer clipped to a smart object IS
 * a smart filter. Confines the layer to its base's ALPHA, and
 * multiplies with any mask it already has rather than replacing it.
 * Group the CONTIGUOUS run `from..=to`. Returns the new group id.
 * Refuses a range that is out of bounds or already grouped —
 * nesting needs a tree and this stack is a list.
 */
export function layers_group(from: number, to: number, name: string): number;

/**
 * The undo/redo readout as JSON — including the BOUND and how much
 * of it is used, so "history is a window" is stated rather than
 * discovered. `null` when no stack is open.
 */
export function layers_history(): string;

/**
 * The stack as JSON, BOTTOM-first:
 * `{"active":i,"layers":[{index,id,name,visible,locked,opacity,blend}]}`.
 * `opacity` is 0–1; `blend` is the `compose.*` wire name.
 */
export function layers_list(): string;

/**
 * CONVERT a pixel layer into a smart object, preserving its pixels
 * as the source. One-way by design: going back would discard the
 * source, which is the destructive move this exists to prevent.
 */
export function layers_make_smart(index: number): void;

/**
 * Make the CURRENT SELECTION this layer's mask. The natural
 * authoring path, and the reason layer masks needed no new
 * authoring engine: the marquee / lasso / wand already produce
 * exactly the coverage a mask is. Errors when nothing is selected —
 * silently attaching an all-one mask would look like success and
 * mask nothing.
 */
export function layers_mask_from_selection(index: number): void;

/**
 * OPEN a layer stack over an engine-held image: one full-canvas
 * "Background" layer sharing that image's pixels. Re-opening on the
 * SAME handle is a no-op (the stack survives); opening on a
 * different handle replaces it, which is what a crop / resize /
 * straighten commit does (it flattens).
 */
export function layers_open(handle: number): void;

/**
 * OPEN a layer stack from a retained PSD parse instead of from the
 * flattened composite — the PSD's own layer tree, bottom-first, with
 * its names, blend modes, opacities and visibility. Returns the
 * layer count.
 *
 * This DECLINES (with the engine's stated reason) for every PSD
 * whose structure the layer model does not reproduce — groups,
 * clipping layers, layer masks, non-8-bit-RGB, or an over-budget
 * canvas — because swapping Photoshop's own composite for a
 * different-looking one of ours would be worse than flattening. On a
 * refusal the caller keeps `layers_open` (the flatten) and shows the
 * reason.
 *
 * `image_handle` must be the composite already ingested from the
 * same file (same extent); `psd_handle` is a `psd_open` handle.
 */
export function layers_open_from_psd(image_handle: number, psd_handle: number): number;

/**
 * REDO the newest undone pixel edit.
 */
export function layers_redo(): Promise<string>;

/**
 * Remove `index`. Removing the ONLY layer is refused (a document
 * keeps at least one). NOT journaled — see the section docs.
 */
export function layers_remove(index: number): void;

/**
 * RE-RENDER a smart object at `scale` — from its preserved SOURCE,
 * never from the current cache, which is the whole point: scaling
 * down and back up loses nothing.
 *
 * GPU-only (the resample is a kernel dispatch). The rendered result
 * is letterboxed into the canvas extent, so the layer keeps its
 * place in a stack whose layers are all canvas-sized.
 */
export function layers_render_smart(index: number, scale: number): Promise<void>;

/**
 * Move a layer in stack order (0 = bottom).
 */
export function layers_reorder(from: number, to: number): void;

export function layers_set_active(index: number): void;

/**
 * Set a layer's blend by `compose.*` wire name (prefix optional).
 * An unregistered name is a clean error, never a silent normal.
 */
export function layers_set_blend(index: number, blend: string): void;

export function layers_set_clipped(index: number, clipped: boolean): void;

export function layers_set_group_blend(id: number, blend: string): void;

export function layers_set_group_name(id: number, name: string): void;

export function layers_set_group_opacity(id: number, opacity: number): void;

export function layers_set_group_visible(id: number, visible: boolean): void;

/**
 * Lock a layer's PIXELS: paint / fill / bake refuse on it. Its
 * properties stay editable — that is what the lock means.
 */
export function layers_set_locked(index: number, locked: boolean): void;

export function layers_set_mask_enabled(index: number, enabled: boolean): void;

export function layers_set_name(index: number, name: string): void;

/**
 * Set a layer's opacity (0–1, clamped).
 */
export function layers_set_opacity(index: number, opacity: number): void;

export function layers_set_visible(index: number, visible: boolean): void;

/**
 * UNDO the newest journaled pixel edit (paint / fill / bake),
 * re-composite, and answer the reverted edit's label — an EMPTY
 * string when there is nothing to undo. Layer STRUCTURE changes are
 * not journaled (see the section docs).
 */
export function layers_undo(): Promise<string>;

/**
 * Dissolve a group. Its layers stay, in place and unchanged.
 */
export function layers_ungroup(id: number): void;

/**
 * PSD SAVE-BACK: write the ADJUSTED full-resolution `rgba` into the
 * retained parse behind `psd_handle` (the merged composite is always
 * rewritten; the layer structure is handled per the returned shape)
 * and answer the honest description the panel shows —
 * `"layer-replaced: …"` when the file's single canvas-sized content
 * layer was updated in place via `replace_channel_pixels`, or
 * `"flattened: …"` when a multi-layer file was flattened into a NEW
 * single-layer PSD. Call `psd_save` afterwards for the bytes.
 *
 * 8-bit RGB only, and the size must match the parsed header —
 * anything else is a clean error, never a wrong-looking file.
 */
export function psd_apply_adjusted(psd_handle: number, width: number, height: number, rgba: Uint8Array): string;

export function psd_close(handle: number): void;

/**
 * The layer list as JSON, in record order:
 * `[{index, name, opacity, hidden, top, left, bottom, right}]`.
 * `hidden` is PSD flags bit 1 (0x02).
 */
export function psd_layer_list(handle: number): string;

/**
 * Parse a `.psd`/`.psb` and retain the structural model behind a
 * handle (independent of `decode_image`'s composite lane). Free with
 * `psd_close`.
 */
export function psd_open(bytes: Uint8Array): number;

/**
 * Remove a layer (balanced `lsct` group-divider bookkeeping engine-side).
 */
export function psd_remove_layer(handle: number, layer: number): void;

/**
 * Save the (possibly edited) PSD with full preservation: unmodeled
 * blocks verbatim; a zero-edit save is byte-identical.
 */
export function psd_save(handle: number): Uint8Array;

/**
 * Rename a layer (updates the legacy Pascal name AND the canonical
 * `luni` block).
 */
export function psd_set_layer_name(handle: number, layer: number, name: string): void;

/**
 * Set a layer's opacity (0–255) through the mutatable tier.
 */
export function psd_set_layer_opacity(handle: number, layer: number, opacity: number): void;

/**
 * RESAMPLE an engine-held image to `out_w`×`out_h` and register the
 * result as a NEW engine-held image (the source stays intact — the
 * crop precedent). `filter` ∈ nearest | mitchell | lanczos3 (the T1
 * resample kernels, GPU-only per spec §6 — requires `init_gpu`;
 * there is no CPU fallback and this rejects honestly without one).
 * Rides the async windowed dispatch (a blocking readback cannot
 * pump the map callback on wasm).
 */
export function resize_image(handle: number, out_w: number, out_h: number, filter: string): Promise<DecodedHandle>;

/**
 * Bind the session selection to an engine-held image (the selection
 * field takes ITS resolution; the magic wand floods ITS pixels; the
 * adjust doors mask only when adjusting THIS handle). Re-binding to
 * the same handle keeps the selection; a different handle (a crop /
 * resize swap) or resolution drops it.
 */
export function selection_bind(handle: number): void;

/**
 * The bounding box of the selection's non-zero coverage as
 * `[x, y, w, h]`; an EMPTY array when there is no explicit selection
 * OR the selection is empty (distinguish via `selection_stats`).
 */
export function selection_bounds(): Uint32Array;

/**
 * Deselect: back to "no selection" (adjustments run unmasked).
 */
export function selection_clear(): void;

/**
 * The raw u8 coverage bytes (`width·height`, row-major) — the
 * overlay/debug readout. Empty when no explicit selection exists.
 */
export function selection_coverage_bytes(): Uint8Array;

/**
 * Feather the selection: a Gaussian of `sigma` px on the COVERAGE
 * (mask prep — CPU on the u8 mask by design, not image processing;
 * the softened mask is still consumed GPU-side). Errors when no
 * explicit selection exists.
 */
export function selection_feather(sigma: number): void;

/**
 * LOAD A CHANNEL AS THE SELECTION — the operation a channels list
 * exists to enable (luminosity masks, a PSD's alpha as a selection).
 *
 * The channel's bytes ARE the coverage representation, so this is a
 * COPY and not a threshold: a 50%-grey channel yields a 50%-selected
 * region, which is exactly what a luminosity mask means and what the
 * masked kernel pipeline already honours at `@group(2)`.
 *
 * `channel` is one of `red`/`green`/`blue`/`alpha`/`luma`; an
 * unknown name is an ERROR rather than a fallback, because masking
 * on the wrong channel is a silent wrong answer.
 */
export function selection_from_channel(handle: number, channel: string, mode: number): void;

/**
 * Invert the selection ("everything" inverts to the explicit EMPTY
 * selection — adjust applies nowhere until reselected).
 */
export function selection_invert(): void;

/**
 * MAGIC WAND at `(x, y)`: color-distance flood over the BOUND
 * image's straight-RGBA8 pixels — `contiguous` = 4-connected BFS
 * from the seed; otherwise a global threshold. `tolerance` is the
 * per-channel (Chebyshev) distance 0–255. Binary coverage (hard
 * edges; `selection_feather` softens), folded under `mode`.
 */
export function selection_magic_wand(x: number, y: number, tolerance: number, contiguous: boolean, mode: number): void;

/**
 * Select ALL explicitly (a full-extent selection in the readouts;
 * the adjust chain still takes the trivial-mask fast path).
 */
export function selection_select_all(): void;

/**
 * Marquee ELLIPSE: center `(cx, cy)`, radii `(rx, ry)` (image px),
 * anti-aliased (4×4 supersampled edge), folded under `mode`.
 */
export function selection_set_ellipse(cx: number, cy: number, rx: number, ry: number, mode: number): void;

/**
 * LASSO polygon: `points_flat` is `[x0, y0, x1, y1, …]` image-px
 * vertices of a closed polygon (the closing edge is implicit),
 * scanline-rasterized with AA coverage, folded under `mode`. Fewer
 * than 3 vertices is a clean error (nothing to select).
 */
export function selection_set_polygon(points_flat: Float32Array, mode: number): void;

/**
 * Marquee RECT: fold the anti-aliased rectangle `[x, x+w) × [y, y+h)`
 * (image px; fractional coords carry the AA edge) into the selection
 * under `mode`.
 */
export function selection_set_rect(x: number, y: number, w: number, h: number, mode: number): void;

/**
 * Selection readout for the panel/tools, as 7 `f32`s:
 * `[has_selection (0|1), x, y, w, h, coverage_fraction, revision]`.
 * `has_selection == 0` ⇒ no explicit selection (everything, the
 * unmasked default) and the box/fraction are 0. An explicit-but-
 * empty selection reads `has == 1, w == h == 0, fraction == 0`.
 */
export function selection_stats(): Float32Array;

/**
 * SELECTION → PATH: trace the live selection's coverage into closed
 * polygons, as `[{outer, points: [[x, y], …]}]` in IMAGE pixel
 * coordinates on pixel EDGES.
 *
 * `threshold` (0–255) is the cut at which partial coverage counts as
 * selected, and it is a PARAMETER because it is a decision: a
 * feathered or luminosity selection has no single right answer, and
 * picking one silently would discard the anti-aliased boundary the
 * selection tools produced. `tolerance` (image px) collapses
 * near-collinear runs; `0` keeps every staircase step.
 *
 * An EMPTY array means "nothing selected" — a caller that wants to
 * distinguish that from "no selection at all" reads
 * `selection_stats`.
 */
export function selection_to_paths(threshold: number, tolerance: number): string;

/**
 * Re-point the selection at a NEW image handle that holds the SAME
 * extent, KEEPING the coverage — the door a destructive in-place
 * edit (the generator FILL) uses so the selection survives its own
 * result. Answers `true` when the coverage carried over, `false`
 * when the extent changed (then it behaves exactly like
 * `selection_bind`: the selection drops).
 */
export function selection_transfer(handle: number): boolean;

/**
 * STRAIGHTEN + CROP commit: rotate the image by `−degrees` about
 * the crop rectangle's centre (`geom.rotate_bilinear`, backward
 * mapped, bilinear, clamp-to-edge) so the rotated FRAME the overlay
 * previewed lands upright, then cut `(x, y, w, h)` out of the
 * result and register it as a NEW engine-held image. The source
 * handle is left intact.
 *
 * `degrees == 0` takes the pure-windowing [`crop_image`] path — no
 * GPU, no resample, no interpolation blur for an axis-aligned crop.
 * A non-zero angle IS a resample and so is GPU-only (`init_gpu`
 * first); it rejects honestly without a device.
 */
export function straighten_crop_image(handle: number, x: number, y: number, w: number, h: number, degrees: number): Promise<DecodedHandle>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_decodedhandle_free: (a: number, b: number) => void;
    readonly __wbg_get_decodedhandle_display: (a: number) => number;
    readonly __wbg_get_decodedhandle_handle: (a: number) => number;
    readonly __wbg_get_decodedhandle_height: (a: number) => number;
    readonly __wbg_get_decodedhandle_width: (a: number) => number;
    readonly __wbg_set_decodedhandle_display: (a: number, b: number) => void;
    readonly __wbg_set_decodedhandle_handle: (a: number, b: number) => void;
    readonly __wbg_set_decodedhandle_height: (a: number, b: number) => void;
    readonly __wbg_set_decodedhandle_width: (a: number, b: number) => void;
    readonly abi_version: () => number;
    readonly abr_presets: (a: number, b: number) => [number, number, number, number];
    readonly adjust_image: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly adjust_image_ext: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number) => any;
    readonly adjust_image_full: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => any;
    readonly apply_gradient_map: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly apply_warp: (a: number, b: number, c: number, d: number) => any;
    readonly brush_blend_modes: () => [number, number];
    readonly brush_stroke_active: () => number;
    readonly brush_stroke_begin: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number];
    readonly brush_stroke_cancel: () => void;
    readonly brush_stroke_commit: () => any;
    readonly brush_stroke_extend: (a: number, b: number, c: number) => any;
    readonly brush_stroke_set_source: (a: number, b: number, c: number) => [number, number];
    readonly brush_stroke_stats: () => [number, number];
    readonly crop_apply_drag: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly crop_frame_corners: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly crop_hit_handle: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly crop_image: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly curve_lut: (a: number, b: number) => [number, number];
    readonly decode_image: (a: number, b: number) => [number, number, number];
    readonly encode_image: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly fill_content_aware: (a: number) => any;
    readonly fill_gradient: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => any;
    readonly fill_noise: (a: number, b: number, c: number) => any;
    readonly free_image: (a: number) => void;
    readonly gpu_ready: () => number;
    readonly image_auto_enhance_params: (a: number) => [number, number, number];
    readonly image_channel_stats: (a: number) => [number, number, number, number];
    readonly image_histogram: (a: number) => [number, number, number];
    readonly image_tile_rgba8: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly image_tile_rgba8_level: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly ingest_rgba8: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly init: () => void;
    readonly init_gpu: () => any;
    readonly kernel_count: () => number;
    readonly layers_add: (a: number, b: number) => [number, number, number];
    readonly layers_add_adjustment: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number) => [number, number, number];
    readonly layers_bake_adjust: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number) => any;
    readonly layers_bound: () => number;
    readonly layers_clear_mask: (a: number) => [number, number];
    readonly layers_close: () => void;
    readonly layers_composite: () => any;
    readonly layers_duplicate: (a: number) => [number, number, number];
    readonly layers_group: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly layers_history: () => [number, number];
    readonly layers_list: () => [number, number];
    readonly layers_make_smart: (a: number) => [number, number];
    readonly layers_mask_from_selection: (a: number) => [number, number];
    readonly layers_open: (a: number) => [number, number];
    readonly layers_open_from_psd: (a: number, b: number) => [number, number, number];
    readonly layers_redo: () => any;
    readonly layers_remove: (a: number) => [number, number];
    readonly layers_render_smart: (a: number, b: number) => any;
    readonly layers_reorder: (a: number, b: number) => [number, number];
    readonly layers_set_active: (a: number) => [number, number];
    readonly layers_set_blend: (a: number, b: number, c: number) => [number, number];
    readonly layers_set_clipped: (a: number, b: number) => [number, number];
    readonly layers_set_group_blend: (a: number, b: number, c: number) => [number, number];
    readonly layers_set_group_name: (a: number, b: number, c: number) => [number, number];
    readonly layers_set_group_opacity: (a: number, b: number) => [number, number];
    readonly layers_set_group_visible: (a: number, b: number) => [number, number];
    readonly layers_set_locked: (a: number, b: number) => [number, number];
    readonly layers_set_mask_enabled: (a: number, b: number) => [number, number];
    readonly layers_set_name: (a: number, b: number, c: number) => [number, number];
    readonly layers_set_opacity: (a: number, b: number) => [number, number];
    readonly layers_set_visible: (a: number, b: number) => [number, number];
    readonly layers_undo: () => any;
    readonly layers_ungroup: (a: number) => [number, number];
    readonly psd_apply_adjusted: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly psd_close: (a: number) => void;
    readonly psd_layer_list: (a: number) => [number, number, number, number];
    readonly psd_open: (a: number, b: number) => [number, number, number];
    readonly psd_remove_layer: (a: number, b: number) => [number, number];
    readonly psd_save: (a: number) => [number, number, number];
    readonly psd_set_layer_name: (a: number, b: number, c: number, d: number) => [number, number];
    readonly psd_set_layer_opacity: (a: number, b: number, c: number) => [number, number];
    readonly resize_image: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly selection_bind: (a: number) => [number, number];
    readonly selection_bounds: () => [number, number];
    readonly selection_clear: () => void;
    readonly selection_coverage_bytes: () => any;
    readonly selection_feather: (a: number) => [number, number];
    readonly selection_from_channel: (a: number, b: number, c: number, d: number) => [number, number];
    readonly selection_invert: () => [number, number];
    readonly selection_magic_wand: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly selection_select_all: () => [number, number];
    readonly selection_set_ellipse: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly selection_set_polygon: (a: number, b: number, c: number) => [number, number];
    readonly selection_set_rect: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly selection_stats: () => [number, number];
    readonly selection_to_paths: (a: number, b: number) => [number, number, number, number];
    readonly selection_transfer: (a: number) => [number, number, number];
    readonly straighten_crop_image: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
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
