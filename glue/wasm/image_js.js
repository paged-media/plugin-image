/* @ts-self-types="./image_js.d.ts" */

/**
 * A decoded image's identity on the surface: the handle keys the
 * engine-held pixels; width/height are the natural extent.
 */
export class DecodedHandle {
    static __wrap(ptr) {
        const obj = Object.create(DecodedHandle.prototype);
        obj.__wbg_ptr = ptr;
        DecodedHandleFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecodedHandleFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decodedhandle_free(ptr, 0);
    }
    /**
     * CMS rung 1 — what the RGB display transform did at decode, as a
     * discriminant the bundle maps to a label: 0 = ICC managed,
     * 1 = sRGB assumed (no embedded profile), 2 = sRGB assumed
     * because an embedded profile was rejected. Surfaced so the panel
     * can STATE the colour treatment instead of leaving the user to
     * guess which numbers they are looking at.
     * @returns {number}
     */
    get display() {
        const ret = wasm.__wbg_get_decodedhandle_display(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get handle() {
        const ret = wasm.__wbg_get_decodedhandle_handle(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get height() {
        const ret = wasm.__wbg_get_decodedhandle_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get width() {
        const ret = wasm.__wbg_get_decodedhandle_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * CMS rung 1 — what the RGB display transform did at decode, as a
     * discriminant the bundle maps to a label: 0 = ICC managed,
     * 1 = sRGB assumed (no embedded profile), 2 = sRGB assumed
     * because an embedded profile was rejected. Surfaced so the panel
     * can STATE the colour treatment instead of leaving the user to
     * guess which numbers they are looking at.
     * @param {number} arg0
     */
    set display(arg0) {
        wasm.__wbg_set_decodedhandle_display(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set handle(arg0) {
        wasm.__wbg_set_decodedhandle_handle(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set height(arg0) {
        wasm.__wbg_set_decodedhandle_height(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set width(arg0) {
        wasm.__wbg_set_decodedhandle_width(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) DecodedHandle.prototype[Symbol.dispose] = DecodedHandle.prototype.free;

/**
 * @returns {number}
 */
export function abi_version() {
    const ret = wasm.abi_version();
    return ret >>> 0;
}

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
 * @param {Uint8Array} bytes
 * @returns {string}
 */
export function abr_presets(bytes) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.abr_presets(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Run the M4 adjustments chain on a decoded image and return the
 * straight-RGBA8 result — the C-1 Stage-A scene-item payload.
 * Identity params return the decode verbatim (no dispatch to run);
 * anything else requires `init_gpu` to have succeeded.
 * @param {number} handle
 * @param {number} exposure_ev
 * @param {number} brightness
 * @param {number} contrast
 * @param {number} saturation
 * @returns {Promise<Uint8Array>}
 */
export function adjust_image(handle, exposure_ev, brightness, contrast, saturation) {
    const ret = wasm.adjust_image(handle, exposure_ev, brightness, contrast, saturation);
    return ret;
}

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
 * @param {number} handle
 * @param {number} exposure_ev
 * @param {number} brightness
 * @param {number} contrast
 * @param {number} saturation
 * @param {number} temp
 * @param {number} tint
 * @param {number} in_black
 * @param {number} in_white
 * @param {number} gamma
 * @param {number} out_black
 * @param {number} out_white
 * @param {Uint8Array} curve_lut
 * @param {number} blur_sigma
 * @param {number} sharpen_amount
 * @param {number} hue_degrees
 * @param {boolean} invert
 * @param {Float32Array} ext
 * @returns {Promise<Uint8Array>}
 */
export function adjust_image_ext(handle, exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, curve_lut, blur_sigma, sharpen_amount, hue_degrees, invert, ext) {
    const ptr0 = passArray8ToWasm0(curve_lut, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(ext, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.adjust_image_ext(handle, exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, ptr0, len0, blur_sigma, sharpen_amount, hue_degrees, invert, ptr1, len1);
    return ret;
}

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
 * @param {number} handle
 * @param {number} exposure_ev
 * @param {number} brightness
 * @param {number} contrast
 * @param {number} saturation
 * @param {number} temp
 * @param {number} tint
 * @param {number} in_black
 * @param {number} in_white
 * @param {number} gamma
 * @param {number} out_black
 * @param {number} out_white
 * @param {Uint8Array} curve_lut
 * @param {number} blur_sigma
 * @param {number} sharpen_amount
 * @param {number} hue_degrees
 * @param {boolean} invert
 * @returns {Promise<Uint8Array>}
 */
export function adjust_image_full(handle, exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, curve_lut, blur_sigma, sharpen_amount, hue_degrees, invert) {
    const ptr0 = passArray8ToWasm0(curve_lut, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.adjust_image_full(handle, exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, ptr0, len0, blur_sigma, sharpen_amount, hue_degrees, invert);
    return ret;
}

/**
 * APPLY a gradient map — luminance through a two-stop colour ramp.
 * A pixel edit into the active layer, journaled and selection-masked
 * exactly like a fill, because that is what it is.
 * @param {number} handle
 * @param {Float32Array} shadow
 * @param {Float32Array} highlight
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gradient_map(handle, shadow, highlight) {
    const ptr0 = passArrayF32ToWasm0(shadow, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(highlight, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.apply_gradient_map(handle, ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * APPLY a parametric distortion (`geom.warp_backward`). `kind` is
 * 0 pinch / 1 spherize / 2 twirl / 3 wave; `amount == 0` is the
 * identity for every kind, so a UI slider needs no special cases.
 * @param {number} handle
 * @param {number} kind
 * @param {number} amount
 * @param {number} frequency
 * @returns {Promise<DecodedHandle>}
 */
export function apply_warp(handle, kind, amount, frequency) {
    const ret = wasm.apply_warp(handle, kind, amount, frequency);
    return ret;
}

/**
 * Every blend mode a stroke can paint through, newline-separated —
 * derived from the `compose.*` registry so the panel's picker can
 * never drift from the kernels that actually exist.
 * @returns {string}
 */
export function brush_blend_modes() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.brush_blend_modes();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Is a stroke in progress?
 * @returns {boolean}
 */
export function brush_stroke_active() {
    const ret = wasm.brush_stroke_active();
    return ret !== 0;
}

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
 * @param {number} handle
 * @param {string} tool
 * @param {number} size
 * @param {number} hardness
 * @param {number} opacity
 * @param {number} flow
 * @param {number} spacing
 * @param {string} blend
 * @param {Float32Array} color
 * @param {string} pressure_target
 */
export function brush_stroke_begin(handle, tool, size, hardness, opacity, flow, spacing, blend, color, pressure_target) {
    const ptr0 = passStringToWasm0(tool, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(blend, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(color, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(pressure_target, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.brush_stroke_begin(handle, ptr0, len0, size, hardness, opacity, flow, spacing, ptr1, len1, ptr2, len2, ptr3, len3);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * CANCEL the stroke: throw the painted pixels away. The engine-held
 * source was never mutated, so this restores it exactly.
 */
export function brush_stroke_cancel() {
    wasm.brush_stroke_cancel();
}

/**
 * @returns {Promise<DecodedHandle>}
 */
export function brush_stroke_commit() {
    const ret = wasm.brush_stroke_commit();
    return ret;
}

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
 * @param {number} x
 * @param {number} y
 * @param {number} pressure
 * @returns {Promise<Uint8Array>}
 */
export function brush_stroke_extend(x, y, pressure) {
    const ret = wasm.brush_stroke_extend(x, y, pressure);
    return ret;
}

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
 * @param {number} x
 * @param {number} y
 * @param {boolean} aligned
 */
export function brush_stroke_set_source(x, y, aligned) {
    const ret = wasm.brush_stroke_set_source(x, y, aligned);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * The in-flight stroke's readout for the panel:
 * `[dabs, x, y, w, h]` — the dab count and the stroke's bounding
 * box in image px. Empty when no stroke is in progress or nothing
 * has landed on the canvas yet.
 * @returns {Float64Array}
 */
export function brush_stroke_stats() {
    const ret = wasm.brush_stroke_stats();
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Apply a pointer drag from `(sx, sy)` to `(px, py)` (image-px) to the
 * rect `[x, y, w, h]` at `handle` (the [`crop_hit_handle`]
 * discriminant), with the aspect lock + image-extent clamp. Returns
 * the new rect as `[x, y, w, h]`. An unknown handle returns the rect
 * unchanged (defensive).
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} handle
 * @param {number} sx
 * @param {number} sy
 * @param {number} px
 * @param {number} py
 * @param {number} aspect_w
 * @param {number} aspect_h
 * @param {number} image_w
 * @param {number} image_h
 * @returns {Float32Array}
 */
export function crop_apply_drag(x, y, w, h, handle, sx, sy, px, py, aspect_w, aspect_h, image_w, image_h) {
    const ret = wasm.crop_apply_drag(x, y, w, h, handle, sx, sy, px, py, aspect_w, aspect_h, image_w, image_h);
    var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * The four corners of the crop FRAME rotated by the straighten
 * `degrees`, as a flat `[x0,y0, x1,y1, x2,y2, x3,y3]` (TL, TR, BR, BL)
 * the overlay draws as a closed polyline.
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} degrees
 * @returns {Float32Array}
 */
export function crop_frame_corners(x, y, w, h, degrees) {
    const ret = wasm.crop_frame_corners(x, y, w, h, degrees);
    var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * Hit-test the crop chrome at `(px, py)` (image-px) against the rect
 * `[x, y, w, h]` with grab radius `tol`. Returns the [`image_core::
 * Handle`] discriminant (0..=7 grips, 8 = body Move) or `-1` for a
 * miss — the TS machine maps it to a cursor + the active grip.
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} px
 * @param {number} py
 * @param {number} tol
 * @returns {number}
 */
export function crop_hit_handle(x, y, w, h, px, py, tol) {
    const ret = wasm.crop_hit_handle(x, y, w, h, px, py, tol);
    return ret;
}

/**
 * Commit a CROP: cut the integer pixel rectangle `(x, y, w, h)`
 * (clamped to the image extent) out of an engine-held image and
 * register the result as a NEW engine-held image, returning its
 * handle. The source handle is left intact (the caller frees it). An
 * out-of-bounds / empty rectangle is a clean error (never a torn
 * image). This door is the AXIS-ALIGNED cut only — the straighten
 * angle rides `straighten_crop_image`, which rotates first.
 * @param {number} handle
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @returns {DecodedHandle}
 */
export function crop_image(handle, x, y, w, h) {
    const ret = wasm.crop_image(handle, x, y, w, h);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return DecodedHandle.__wrap(ret[0]);
}

/**
 * Build a 256-byte tone LUT from flat `[i0,o0, i1,o1, …]` curve
 * control points in `[0,1]` (the CURVES editor's points) — the LUT
 * `adjust_image_full` consumes. Wraps `image_core::curve_lut`.
 * @param {Float32Array} points
 * @returns {Uint8Array}
 */
export function curve_lut(points) {
    const ptr0 = passArrayF32ToWasm0(points, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.curve_lut(ptr0, len0);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Decode PSD/PNG/JPEG bytes (sniffed by magic) into an engine-held
 * RGBA8 image. Free with `free_image`.
 * @param {Uint8Array} bytes
 * @returns {DecodedHandle}
 */
export function decode_image(bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.decode_image(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return DecodedHandle.__wrap(ret[0]);
}

/**
 * Re-encode straight RGBA8 as PNG or JPEG (`format` ∈ `png | jpeg`)
 * — the NON-PSD save-back lane. Codec entropy coding is inherently
 * CPU work (spec §1); JPEG rides the fixed v0 quality documented on
 * `saveback::JPEG_QUALITY_DEFAULT`.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {string} format
 * @returns {Uint8Array}
 */
export function encode_image(rgba, width, height, format) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(format, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.encode_image(ptr0, len0, width, height, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

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
 * @param {number} handle
 * @returns {Promise<DecodedHandle>}
 */
export function fill_content_aware(handle) {
    const ret = wasm.fill_content_aware(handle);
    return ret;
}

/**
 * FILL the current selection (the whole image when none) with a
 * fixed TWO-STOP gradient. `kind` ∈ `linear | radial | angular |
 * reflected | diamond`; `c0`/`c1` are straight RGBA in `[0, 1]`
 * (4 floats each). The gradient GEOMETRY is derived from the
 * selection's bounding box — there is no on-canvas drag handle in
 * v0 (`crate::fill` documents the derivation). Returns the NEW
 * engine-held image's handle.
 * @param {number} handle
 * @param {string} kind
 * @param {Float32Array} c0
 * @param {Float32Array} c1
 * @returns {Promise<DecodedHandle>}
 */
export function fill_gradient(handle, kind, c0, c1) {
    const ptr0 = passStringToWasm0(kind, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(c0, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(c1, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.fill_gradient(handle, ptr0, len0, ptr1, len1, ptr2, len2);
    return ret;
}

/**
 * FILL the current selection (the whole image when none) with
 * deterministic monochrome noise — `amount` scales the hash
 * amplitude, `seed` makes a repeat reproducible. Returns the NEW
 * engine-held image's handle.
 * @param {number} handle
 * @param {number} amount
 * @param {number} seed
 * @returns {Promise<DecodedHandle>}
 */
export function fill_noise(handle, amount, seed) {
    const ret = wasm.fill_noise(handle, amount, seed);
    return ret;
}

/**
 * Release an engine-held decoded image (its mip pyramid cache, and
 * the layer stack bound to it — a stack whose composite target is
 * gone has nowhere to land).
 * @param {number} handle
 */
export function free_image(handle) {
    wasm.free_image(handle);
}

/**
 * Whether `init_gpu` succeeded (the glue probes this to gate the
 * adjust controls honestly).
 * @returns {boolean}
 */
export function gpu_ready() {
    const ret = wasm.gpu_ready();
    return ret !== 0;
}

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
 * @param {number} handle
 * @returns {Float32Array}
 */
export function image_auto_enhance_params(handle) {
    const ret = wasm.image_auto_enhance_params(handle);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * The CHANNELS readout for an engine-held image: `[{name, min, max,
 * mean}]` for red/green/blue/alpha and the derived Rec.709 luma.
 * Pure CPU reduction over the same straight-RGBA8 buffer
 * `image_histogram` reads, so the two agree by construction.
 * @param {number} handle
 * @returns {string}
 */
export function image_channel_stats(handle) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.image_channel_stats(handle);
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Compute the RGB + luma 256-bin histogram of an engine-held image as
 * a flat `[r…, g…, b…, luma…]` 1024-`u32` array (the LEVELS / CURVES
 * panel slices it into four channels). Pure CPU reduction over the
 * straight-RGBA8 buffer (no GPU); deterministic.
 * @param {number} handle
 * @returns {Uint32Array}
 */
export function image_histogram(handle) {
    const ret = wasm.image_histogram(handle);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

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
 * @param {number} handle
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @returns {Uint8Array}
 */
export function image_tile_rgba8(handle, x, y, w, h) {
    const ret = wasm.image_tile_rgba8(handle, x, y, w, h);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

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
 * @param {number} handle
 * @param {number} level
 * @param {number} x
 * @param {number} y
 * @param {number} size
 * @returns {Uint8Array}
 */
export function image_tile_rgba8_level(handle, level, x, y, size) {
    const ret = wasm.image_tile_rgba8_level(handle, level, x, y, size);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * K-3 (S-07 / I-02) — register a PRE-DECODED straight-RGBA8 buffer
 * (from the decode worker pool, which ran the codec/PSD CPU lanes
 * off-thread) as an engine-held image, returning a handle for the GPU
 * adjust + tile paths. `bytes` must be exactly `width*height*4` RGBA8;
 * a length mismatch is a clean error. Free with `free_image`.
 * @param {number} width
 * @param {number} height
 * @param {Uint8Array} bytes
 * @returns {DecodedHandle}
 */
export function ingest_rgba8(width, height, bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ingest_rgba8(width, height, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return DecodedHandle.__wrap(ret[0]);
}

export function init() {
    wasm.init();
}

/**
 * Request the WebGPU adapter/device for kernel execution.
 * Idempotent. Rejects when the environment has no WebGPU — the
 * honest no-GPU state (no CPU kernel path ships, spec §6).
 * @returns {Promise<void>}
 */
export function init_gpu() {
    const ret = wasm.init_gpu();
    return ret;
}

/**
 * @returns {number}
 */
export function kernel_count() {
    const ret = wasm.kernel_count();
    return ret >>> 0;
}

/**
 * Add an empty transparent layer above the active one (it becomes
 * active). Returns its index.
 * @param {string} name
 * @returns {number}
 */
export function layers_add(name) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.layers_add(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
}

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
 * @param {string} name
 * @param {number} exposure_ev
 * @param {number} brightness
 * @param {number} contrast
 * @param {number} saturation
 * @param {number} temp
 * @param {number} tint
 * @param {number} in_black
 * @param {number} in_white
 * @param {number} gamma
 * @param {number} out_black
 * @param {number} out_white
 * @param {Uint8Array} curve_lut
 * @param {number} blur_sigma
 * @param {number} sharpen_amount
 * @param {number} hue_degrees
 * @param {boolean} invert
 * @param {Float32Array} ext
 * @returns {number}
 */
export function layers_add_adjustment(name, exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, curve_lut, blur_sigma, sharpen_amount, hue_degrees, invert, ext) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(curve_lut, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(ext, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.layers_add_adjustment(ptr0, len0, exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, ptr1, len1, blur_sigma, sharpen_amount, hue_degrees, invert, ptr2, len2);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
}

/**
 * BAKE the adjustment chain into the ACTIVE layer — the DESTRUCTIVE
 * per-layer adjustment (the panel's chain is otherwise a re-runnable
 * PREVIEW of the composite and mutates nothing). Journaled over the
 * whole canvas, so it is undoable; refuses on a locked layer and at
 * identity. Arguments mirror `adjust_image_ext` minus the handle.
 * @param {number} exposure_ev
 * @param {number} brightness
 * @param {number} contrast
 * @param {number} saturation
 * @param {number} temp
 * @param {number} tint
 * @param {number} in_black
 * @param {number} in_white
 * @param {number} gamma
 * @param {number} out_black
 * @param {number} out_white
 * @param {Uint8Array} curve_lut
 * @param {number} blur_sigma
 * @param {number} sharpen_amount
 * @param {number} hue_degrees
 * @param {boolean} invert
 * @param {Float32Array} ext
 * @returns {Promise<Uint8Array>}
 */
export function layers_bake_adjust(exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, curve_lut, blur_sigma, sharpen_amount, hue_degrees, invert, ext) {
    const ptr0 = passArray8ToWasm0(curve_lut, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(ext, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.layers_bake_adjust(exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, ptr0, len0, blur_sigma, sharpen_amount, hue_degrees, invert, ptr1, len1);
    return ret;
}

/**
 * The handle the stack is bound to, or `-1` when none is open.
 * @returns {number}
 */
export function layers_bound() {
    const ret = wasm.layers_bound();
    return ret;
}

/**
 * DELETE the mask (the coverage is gone), as distinct from
 * disabling it.
 * @param {number} index
 */
export function layers_clear_mask(index) {
    const ret = wasm.layers_clear_mask(index);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Drop the bound stack (and its undo history).
 */
export function layers_close() {
    wasm.layers_close();
}

/**
 * COMPOSITE the stack bottom-up and write the result back into the
 * bound engine-held image, returning the straight RGBA8 (the C-1
 * Stage-A payload). GPU-only whenever there is anything to blend; a
 * single plain visible layer short-circuits to its own pixels with
 * no dispatch at all, so a one-layer document needs no device.
 * @returns {Promise<Uint8Array>}
 */
export function layers_composite() {
    const ret = wasm.layers_composite();
    return ret;
}

/**
 * Duplicate `index` above itself (the copy becomes active).
 * @param {number} index
 * @returns {number}
 */
export function layers_duplicate(index) {
    const ret = wasm.layers_duplicate(index);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
}

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
 * @param {number} from
 * @param {number} to
 * @param {string} name
 * @returns {number}
 */
export function layers_group(from, to, name) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.layers_group(from, to, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
}

/**
 * The undo/redo readout as JSON — including the BOUND and how much
 * of it is used, so "history is a window" is stated rather than
 * discovered. `null` when no stack is open.
 * @returns {string}
 */
export function layers_history() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.layers_history();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * The stack as JSON, BOTTOM-first:
 * `{"active":i,"layers":[{index,id,name,visible,locked,opacity,blend}]}`.
 * `opacity` is 0–1; `blend` is the `compose.*` wire name.
 * @returns {string}
 */
export function layers_list() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.layers_list();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * CONVERT a pixel layer into a smart object, preserving its pixels
 * as the source. One-way by design: going back would discard the
 * source, which is the destructive move this exists to prevent.
 * @param {number} index
 */
export function layers_make_smart(index) {
    const ret = wasm.layers_make_smart(index);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Make the CURRENT SELECTION this layer's mask. The natural
 * authoring path, and the reason layer masks needed no new
 * authoring engine: the marquee / lasso / wand already produce
 * exactly the coverage a mask is. Errors when nothing is selected —
 * silently attaching an all-one mask would look like success and
 * mask nothing.
 * @param {number} index
 */
export function layers_mask_from_selection(index) {
    const ret = wasm.layers_mask_from_selection(index);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * OPEN a layer stack over an engine-held image: one full-canvas
 * "Background" layer sharing that image's pixels. Re-opening on the
 * SAME handle is a no-op (the stack survives); opening on a
 * different handle replaces it, which is what a crop / resize /
 * straighten commit does (it flattens).
 * @param {number} handle
 */
export function layers_open(handle) {
    const ret = wasm.layers_open(handle);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

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
 * @param {number} image_handle
 * @param {number} psd_handle
 * @returns {number}
 */
export function layers_open_from_psd(image_handle, psd_handle) {
    const ret = wasm.layers_open_from_psd(image_handle, psd_handle);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
}

/**
 * REDO the newest undone pixel edit.
 * @returns {Promise<string>}
 */
export function layers_redo() {
    const ret = wasm.layers_redo();
    return ret;
}

/**
 * Remove `index`. Removing the ONLY layer is refused (a document
 * keeps at least one). NOT journaled — see the section docs.
 * @param {number} index
 */
export function layers_remove(index) {
    const ret = wasm.layers_remove(index);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * RE-RENDER a smart object at `scale` — from its preserved SOURCE,
 * never from the current cache, which is the whole point: scaling
 * down and back up loses nothing.
 *
 * GPU-only (the resample is a kernel dispatch). The rendered result
 * is letterboxed into the canvas extent, so the layer keeps its
 * place in a stack whose layers are all canvas-sized.
 * @param {number} index
 * @param {number} scale
 * @returns {Promise<void>}
 */
export function layers_render_smart(index, scale) {
    const ret = wasm.layers_render_smart(index, scale);
    return ret;
}

/**
 * Move a layer in stack order (0 = bottom).
 * @param {number} from
 * @param {number} to
 */
export function layers_reorder(from, to) {
    const ret = wasm.layers_reorder(from, to);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * @param {number} index
 */
export function layers_set_active(index) {
    const ret = wasm.layers_set_active(index);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Set a layer's blend by `compose.*` wire name (prefix optional).
 * An unregistered name is a clean error, never a silent normal.
 * @param {number} index
 * @param {string} blend
 */
export function layers_set_blend(index, blend) {
    const ptr0 = passStringToWasm0(blend, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.layers_set_blend(index, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * @param {number} index
 * @param {boolean} clipped
 */
export function layers_set_clipped(index, clipped) {
    const ret = wasm.layers_set_clipped(index, clipped);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * @param {number} id
 * @param {string} blend
 */
export function layers_set_group_blend(id, blend) {
    const ptr0 = passStringToWasm0(blend, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.layers_set_group_blend(id, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * @param {number} id
 * @param {string} name
 */
export function layers_set_group_name(id, name) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.layers_set_group_name(id, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * @param {number} id
 * @param {number} opacity
 */
export function layers_set_group_opacity(id, opacity) {
    const ret = wasm.layers_set_group_opacity(id, opacity);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * @param {number} id
 * @param {boolean} visible
 */
export function layers_set_group_visible(id, visible) {
    const ret = wasm.layers_set_group_visible(id, visible);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Lock a layer's PIXELS: paint / fill / bake refuse on it. Its
 * properties stay editable — that is what the lock means.
 * @param {number} index
 * @param {boolean} locked
 */
export function layers_set_locked(index, locked) {
    const ret = wasm.layers_set_locked(index, locked);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * @param {number} index
 * @param {boolean} enabled
 */
export function layers_set_mask_enabled(index, enabled) {
    const ret = wasm.layers_set_mask_enabled(index, enabled);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * @param {number} index
 * @param {string} name
 */
export function layers_set_name(index, name) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.layers_set_name(index, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Set a layer's opacity (0–1, clamped).
 * @param {number} index
 * @param {number} opacity
 */
export function layers_set_opacity(index, opacity) {
    const ret = wasm.layers_set_opacity(index, opacity);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * @param {number} index
 * @param {boolean} visible
 */
export function layers_set_visible(index, visible) {
    const ret = wasm.layers_set_visible(index, visible);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * UNDO the newest journaled pixel edit (paint / fill / bake),
 * re-composite, and answer the reverted edit's label — an EMPTY
 * string when there is nothing to undo. Layer STRUCTURE changes are
 * not journaled (see the section docs).
 * @returns {Promise<string>}
 */
export function layers_undo() {
    const ret = wasm.layers_undo();
    return ret;
}

/**
 * Dissolve a group. Its layers stay, in place and unchanged.
 * @param {number} id
 */
export function layers_ungroup(id) {
    const ret = wasm.layers_ungroup(id);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

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
 * @param {number} psd_handle
 * @param {number} width
 * @param {number} height
 * @param {Uint8Array} rgba
 * @returns {string}
 */
export function psd_apply_adjusted(psd_handle, width, height, rgba) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.psd_apply_adjusted(psd_handle, width, height, ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * @param {number} handle
 */
export function psd_close(handle) {
    wasm.psd_close(handle);
}

/**
 * The layer list as JSON, in record order:
 * `[{index, name, opacity, hidden, top, left, bottom, right}]`.
 * `hidden` is PSD flags bit 1 (0x02).
 * @param {number} handle
 * @returns {string}
 */
export function psd_layer_list(handle) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.psd_layer_list(handle);
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Parse a `.psd`/`.psb` and retain the structural model behind a
 * handle (independent of `decode_image`'s composite lane). Free with
 * `psd_close`.
 * @param {Uint8Array} bytes
 * @returns {number}
 */
export function psd_open(bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.psd_open(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
}

/**
 * Remove a layer (balanced `lsct` group-divider bookkeeping engine-side).
 * @param {number} handle
 * @param {number} layer
 */
export function psd_remove_layer(handle, layer) {
    const ret = wasm.psd_remove_layer(handle, layer);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Save the (possibly edited) PSD with full preservation: unmodeled
 * blocks verbatim; a zero-edit save is byte-identical.
 * @param {number} handle
 * @returns {Uint8Array}
 */
export function psd_save(handle) {
    const ret = wasm.psd_save(handle);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Rename a layer (updates the legacy Pascal name AND the canonical
 * `luni` block).
 * @param {number} handle
 * @param {number} layer
 * @param {string} name
 */
export function psd_set_layer_name(handle, layer, name) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.psd_set_layer_name(handle, layer, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Set a layer's opacity (0–255) through the mutatable tier.
 * @param {number} handle
 * @param {number} layer
 * @param {number} opacity
 */
export function psd_set_layer_opacity(handle, layer, opacity) {
    const ret = wasm.psd_set_layer_opacity(handle, layer, opacity);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * RESAMPLE an engine-held image to `out_w`×`out_h` and register the
 * result as a NEW engine-held image (the source stays intact — the
 * crop precedent). `filter` ∈ nearest | mitchell | lanczos3 (the T1
 * resample kernels, GPU-only per spec §6 — requires `init_gpu`;
 * there is no CPU fallback and this rejects honestly without one).
 * Rides the async windowed dispatch (a blocking readback cannot
 * pump the map callback on wasm).
 * @param {number} handle
 * @param {number} out_w
 * @param {number} out_h
 * @param {string} filter
 * @returns {Promise<DecodedHandle>}
 */
export function resize_image(handle, out_w, out_h, filter) {
    const ptr0 = passStringToWasm0(filter, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.resize_image(handle, out_w, out_h, ptr0, len0);
    return ret;
}

/**
 * Bind the session selection to an engine-held image (the selection
 * field takes ITS resolution; the magic wand floods ITS pixels; the
 * adjust doors mask only when adjusting THIS handle). Re-binding to
 * the same handle keeps the selection; a different handle (a crop /
 * resize swap) or resolution drops it.
 * @param {number} handle
 */
export function selection_bind(handle) {
    const ret = wasm.selection_bind(handle);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * The bounding box of the selection's non-zero coverage as
 * `[x, y, w, h]`; an EMPTY array when there is no explicit selection
 * OR the selection is empty (distinguish via `selection_stats`).
 * @returns {Uint32Array}
 */
export function selection_bounds() {
    const ret = wasm.selection_bounds();
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * Deselect: back to "no selection" (adjustments run unmasked).
 */
export function selection_clear() {
    wasm.selection_clear();
}

/**
 * The raw u8 coverage bytes (`width·height`, row-major) — the
 * overlay/debug readout. Empty when no explicit selection exists.
 * @returns {Uint8Array}
 */
export function selection_coverage_bytes() {
    const ret = wasm.selection_coverage_bytes();
    return ret;
}

/**
 * Feather the selection: a Gaussian of `sigma` px on the COVERAGE
 * (mask prep — CPU on the u8 mask by design, not image processing;
 * the softened mask is still consumed GPU-side). Errors when no
 * explicit selection exists.
 * @param {number} sigma
 */
export function selection_feather(sigma) {
    const ret = wasm.selection_feather(sigma);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

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
 * @param {number} handle
 * @param {string} channel
 * @param {number} mode
 */
export function selection_from_channel(handle, channel, mode) {
    const ptr0 = passStringToWasm0(channel, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.selection_from_channel(handle, ptr0, len0, mode);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Invert the selection ("everything" inverts to the explicit EMPTY
 * selection — adjust applies nowhere until reselected).
 */
export function selection_invert() {
    const ret = wasm.selection_invert();
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * MAGIC WAND at `(x, y)`: color-distance flood over the BOUND
 * image's straight-RGBA8 pixels — `contiguous` = 4-connected BFS
 * from the seed; otherwise a global threshold. `tolerance` is the
 * per-channel (Chebyshev) distance 0–255. Binary coverage (hard
 * edges; `selection_feather` softens), folded under `mode`.
 * @param {number} x
 * @param {number} y
 * @param {number} tolerance
 * @param {boolean} contiguous
 * @param {number} mode
 */
export function selection_magic_wand(x, y, tolerance, contiguous, mode) {
    const ret = wasm.selection_magic_wand(x, y, tolerance, contiguous, mode);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Select ALL explicitly (a full-extent selection in the readouts;
 * the adjust chain still takes the trivial-mask fast path).
 */
export function selection_select_all() {
    const ret = wasm.selection_select_all();
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Marquee ELLIPSE: center `(cx, cy)`, radii `(rx, ry)` (image px),
 * anti-aliased (4×4 supersampled edge), folded under `mode`.
 * @param {number} cx
 * @param {number} cy
 * @param {number} rx
 * @param {number} ry
 * @param {number} mode
 */
export function selection_set_ellipse(cx, cy, rx, ry, mode) {
    const ret = wasm.selection_set_ellipse(cx, cy, rx, ry, mode);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * LASSO polygon: `points_flat` is `[x0, y0, x1, y1, …]` image-px
 * vertices of a closed polygon (the closing edge is implicit),
 * scanline-rasterized with AA coverage, folded under `mode`. Fewer
 * than 3 vertices is a clean error (nothing to select).
 * @param {Float32Array} points_flat
 * @param {number} mode
 */
export function selection_set_polygon(points_flat, mode) {
    const ptr0 = passArrayF32ToWasm0(points_flat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.selection_set_polygon(ptr0, len0, mode);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Marquee RECT: fold the anti-aliased rectangle `[x, x+w) × [y, y+h)`
 * (image px; fractional coords carry the AA edge) into the selection
 * under `mode`.
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} mode
 */
export function selection_set_rect(x, y, w, h, mode) {
    const ret = wasm.selection_set_rect(x, y, w, h, mode);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Selection readout for the panel/tools, as 7 `f32`s:
 * `[has_selection (0|1), x, y, w, h, coverage_fraction, revision]`.
 * `has_selection == 0` ⇒ no explicit selection (everything, the
 * unmasked default) and the box/fraction are 0. An explicit-but-
 * empty selection reads `has == 1, w == h == 0, fraction == 0`.
 * @returns {Float32Array}
 */
export function selection_stats() {
    const ret = wasm.selection_stats();
    var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

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
 * @param {number} threshold
 * @param {number} tolerance
 * @returns {string}
 */
export function selection_to_paths(threshold, tolerance) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.selection_to_paths(threshold, tolerance);
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Re-point the selection at a NEW image handle that holds the SAME
 * extent, KEEPING the coverage — the door a destructive in-place
 * edit (the generator FILL) uses so the selection survives its own
 * result. Answers `true` when the coverage carried over, `false`
 * when the extent changed (then it behaves exactly like
 * `selection_bind`: the selection drops).
 * @param {number} handle
 * @returns {boolean}
 */
export function selection_transfer(handle) {
    const ret = wasm.selection_transfer(handle);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] !== 0;
}

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
 * @param {number} handle
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} degrees
 * @returns {Promise<DecodedHandle>}
 */
export function straighten_crop_image(handle, x, y, w, h, degrees) {
    const ret = wasm.straighten_crop_image(handle, x, y, w, h, degrees);
    return ret;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Window_65ef42d29dc8174d: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_d272430d4a323303: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_debug_string_0accd80f45e5faa2: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_754e9f305ff6029e: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_87c3bfe968c6a5ad: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_undefined_67b456be8673d3d7: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_1506f2235d1bdba0: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_61db23ac97f16c31: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_beginComputePass_43b0c6751d870fcf: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_call_9c758de292015997: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_copyTextureToBuffer_a9b82ac765521aab: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            arg0.copyTextureToBuffer(arg1, arg2, arg3);
        }, arguments); },
        __wbg_createBindGroupLayout_59891d473ac8665d: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBindGroupLayout(arg1);
            return ret;
        }, arguments); },
        __wbg_createBindGroup_4cb86ff853df5c69: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_3fa0256cba655273: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        }, arguments); },
        __wbg_createCommandEncoder_98e3b731629054b4: function(arg0, arg1) {
            const ret = arg0.createCommandEncoder(arg1);
            return ret;
        },
        __wbg_createComputePipeline_9d101515d504e110: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createPipelineLayout_270b4fd0b4230373: function(arg0, arg1) {
            const ret = arg0.createPipelineLayout(arg1);
            return ret;
        },
        __wbg_createShaderModule_f0aa469466c7bdaa: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_createTexture_28341edbcc7d129e: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createTexture(arg1);
            return ret;
        }, arguments); },
        __wbg_createView_d04a0f9bdd723238: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createView(arg1);
            return ret;
        }, arguments); },
        __wbg_decodedhandle_new: function(arg0) {
            const ret = DecodedHandle.__wrap(arg0);
            return ret;
        },
        __wbg_description_f6ebcdce701b056b: function(arg0, arg1) {
            const ret = arg1.description;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_dispatchWorkgroups_26f6198195c36ca4: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_end_8437a975bbfe0297: function(arg0) {
            arg0.end();
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_finish_6c7bba424ffe1bbc: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_finish_c40b67ff2af88e0c: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_getMappedRange_59829576da3edd39: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getMappedRange(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_get_de6a0f7d4d18a304: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_gpu_cbd27ad0589bc0b3: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_info_91a8fcd51fd17fff: function(arg0) {
            const ret = arg0.info;
            return ret;
        },
        __wbg_instanceof_GpuAdapter_1297a3a5ce0db3ff: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUAdapter;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_label_9a8583e3a20fafc7: function(arg0, arg1) {
            const ret = arg1.label;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_length_4a591ecaa01354d9: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_limits_25f7265ea0cad6c5: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_mapAsync_e3cfbd141919d03c: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.mapAsync(arg1 >>> 0, arg2, arg3);
            return ret;
        },
        __wbg_maxBindGroups_7e4965b5daa53b23: function(arg0) {
            const ret = arg0.maxBindGroups;
            return ret;
        },
        __wbg_maxBindingsPerBindGroup_5d11588150650215: function(arg0) {
            const ret = arg0.maxBindingsPerBindGroup;
            return ret;
        },
        __wbg_maxBufferSize_b59f147488bf047a: function(arg0) {
            const ret = arg0.maxBufferSize;
            return ret;
        },
        __wbg_maxColorAttachmentBytesPerSample_726ea37aedfb839a: function(arg0) {
            const ret = arg0.maxColorAttachmentBytesPerSample;
            return ret;
        },
        __wbg_maxColorAttachments_62ecca7ef94d78e4: function(arg0) {
            const ret = arg0.maxColorAttachments;
            return ret;
        },
        __wbg_maxComputeInvocationsPerWorkgroup_a14458d75e0b90ac: function(arg0) {
            const ret = arg0.maxComputeInvocationsPerWorkgroup;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeX_6b8c17d5e4738e77: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeX;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeY_13b5de41c6e0bc2a: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeY;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeZ_b12d7f3e670aa0a2: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeZ;
            return ret;
        },
        __wbg_maxComputeWorkgroupStorageSize_886498bc3b0baa23: function(arg0) {
            const ret = arg0.maxComputeWorkgroupStorageSize;
            return ret;
        },
        __wbg_maxComputeWorkgroupsPerDimension_144b6bbf6ac24451: function(arg0) {
            const ret = arg0.maxComputeWorkgroupsPerDimension;
            return ret;
        },
        __wbg_maxDynamicStorageBuffersPerPipelineLayout_d81239ef90f4f920: function(arg0) {
            const ret = arg0.maxDynamicStorageBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxDynamicUniformBuffersPerPipelineLayout_0cca7d1cb9e5adf7: function(arg0) {
            const ret = arg0.maxDynamicUniformBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxInterStageShaderVariables_4504147f810dd43d: function(arg0) {
            const ret = arg0.maxInterStageShaderVariables;
            return ret;
        },
        __wbg_maxSampledTexturesPerShaderStage_54e5ed0537676c83: function(arg0) {
            const ret = arg0.maxSampledTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxSamplersPerShaderStage_71315fab0d7f34b1: function(arg0) {
            const ret = arg0.maxSamplersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageBufferBindingSize_779fd522aaaa6f90: function(arg0) {
            const ret = arg0.maxStorageBufferBindingSize;
            return ret;
        },
        __wbg_maxStorageBuffersPerShaderStage_c99b4f72aaf19e34: function(arg0) {
            const ret = arg0.maxStorageBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageTexturesPerShaderStage_5403c17d11da5280: function(arg0) {
            const ret = arg0.maxStorageTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxTextureArrayLayers_eca9fa36b3d46099: function(arg0) {
            const ret = arg0.maxTextureArrayLayers;
            return ret;
        },
        __wbg_maxTextureDimension1D_a7d9d7ecd19aae9b: function(arg0) {
            const ret = arg0.maxTextureDimension1D;
            return ret;
        },
        __wbg_maxTextureDimension2D_c6a3937eb3ab18df: function(arg0) {
            const ret = arg0.maxTextureDimension2D;
            return ret;
        },
        __wbg_maxTextureDimension3D_d941aa547d9e0801: function(arg0) {
            const ret = arg0.maxTextureDimension3D;
            return ret;
        },
        __wbg_maxUniformBufferBindingSize_1e8c92a2094b7ce7: function(arg0) {
            const ret = arg0.maxUniformBufferBindingSize;
            return ret;
        },
        __wbg_maxUniformBuffersPerShaderStage_83cde6650612f178: function(arg0) {
            const ret = arg0.maxUniformBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxVertexAttributes_dd313a3540d56e88: function(arg0) {
            const ret = arg0.maxVertexAttributes;
            return ret;
        },
        __wbg_maxVertexBufferArrayStride_6fd082d9954d1f4a: function(arg0) {
            const ret = arg0.maxVertexBufferArrayStride;
            return ret;
        },
        __wbg_maxVertexBuffers_bbd14712ac158c6f: function(arg0) {
            const ret = arg0.maxVertexBuffers;
            return ret;
        },
        __wbg_minStorageBufferOffsetAlignment_726c386298254510: function(arg0) {
            const ret = arg0.minStorageBufferOffsetAlignment;
            return ret;
        },
        __wbg_minUniformBufferOffsetAlignment_6df1f95f5974788e: function(arg0) {
            const ret = arg0.minUniformBufferOffsetAlignment;
            return ret;
        },
        __wbg_navigator_3833ecdbc19d2757: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_391291470f58c650: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_ce1ab61c1c2b300d: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_d90091b82fdf5b91: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_18fa1f71286d66b8: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_from_slice_47be4219028de35d: function(arg0, arg1) {
            const ret = new Uint32Array(getArrayU32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_from_slice_956df4f769fb782c: function(arg0, arg1) {
            const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_typed_bf31d18f92484486: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen__convert__closures_____invoke__h9344d28c99dc1e49(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_with_byte_offset_and_length_d836f26d916dd9ad: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_length_36a4998e27b014c5: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return ret;
        },
        __wbg_onSubmittedWorkDone_5f36409816d68e04: function(arg0) {
            const ret = arg0.onSubmittedWorkDone();
            return ret;
        },
        __wbg_prototypesetcall_3249fc62a0fafa30: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_a6822215aa43e71c: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_queueMicrotask_35c611f4a14830b2: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queueMicrotask_404ed0a58e0b63cc: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queue_7bbf92178b06da19: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_requestAdapter_0049683abd339828: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_921f0a221b4492fa: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolve_25a7e548d5881dca: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_setBindGroup_0500d49bcf971ad6: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        }, arguments); },
        __wbg_setBindGroup_863d2daeb3c4fa01: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setPipeline_c6aca1c13ec27120: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_set_6e30c9374c26414c: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_access_08d6bdbda9aaa266: function(arg0, arg1) {
            arg0.access = __wbindgen_enum_GpuStorageTextureAccess[arg1];
        },
        __wbg_set_array_layer_count_01e36293bee85e02: function(arg0, arg1) {
            arg0.arrayLayerCount = arg1 >>> 0;
        },
        __wbg_set_aspect_0675b2844dd12eb1: function(arg0, arg1) {
            arg0.aspect = __wbindgen_enum_GpuTextureAspect[arg1];
        },
        __wbg_set_aspect_e09cb246c2df6f46: function(arg0, arg1) {
            arg0.aspect = __wbindgen_enum_GpuTextureAspect[arg1];
        },
        __wbg_set_base_array_layer_ff3450be9aa7d232: function(arg0, arg1) {
            arg0.baseArrayLayer = arg1 >>> 0;
        },
        __wbg_set_base_mip_level_43e77e5d237ede24: function(arg0, arg1) {
            arg0.baseMipLevel = arg1 >>> 0;
        },
        __wbg_set_beginning_of_pass_write_index_ebe753eeeade6f6c: function(arg0, arg1) {
            arg0.beginningOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_bind_group_layouts_078241cf2822c39e: function(arg0, arg1) {
            arg0.bindGroupLayouts = arg1;
        },
        __wbg_set_binding_d683cd9c1d4bcfed: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_binding_e9ba14423117de0a: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_buffer_598ab98a251b8f91: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_73d9f6fea9c41867: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_88dfc353992be57b: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_bytes_per_row_0bdd54b7fc03c765: function(arg0, arg1) {
            arg0.bytesPerRow = arg1 >>> 0;
        },
        __wbg_set_bytes_per_row_4d62ead4cbf1cd75: function(arg0, arg1) {
            arg0.bytesPerRow = arg1 >>> 0;
        },
        __wbg_set_c775d84916be79ea: function(arg0, arg1, arg2) {
            arg0.set(arg1, arg2 >>> 0);
        },
        __wbg_set_code_6a0d763da082dcfb: function(arg0, arg1, arg2) {
            arg0.code = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_compute_5dd7704ee8a825c6: function(arg0, arg1) {
            arg0.compute = arg1;
        },
        __wbg_set_depth_or_array_layers_f8981011496f12e7: function(arg0, arg1) {
            arg0.depthOrArrayLayers = arg1 >>> 0;
        },
        __wbg_set_dimension_b4da3979dc699ef8: function(arg0, arg1) {
            arg0.dimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_dimension_d4f0c50e75083b7f: function(arg0, arg1) {
            arg0.dimension = __wbindgen_enum_GpuTextureDimension[arg1];
        },
        __wbg_set_end_of_pass_write_index_49de5f6017fb9a1f: function(arg0, arg1) {
            arg0.endOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_entries_070b048e4bea0c29: function(arg0, arg1) {
            arg0.entries = arg1;
        },
        __wbg_set_entries_f9b7f3d4e9faccf4: function(arg0, arg1) {
            arg0.entries = arg1;
        },
        __wbg_set_entry_point_52a2481a52f9799d: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_external_texture_cf122b1392d58f37: function(arg0, arg1) {
            arg0.externalTexture = arg1;
        },
        __wbg_set_format_27c63de9b0ec1cb3: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_b08d87d5f33bcd89: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_c1a342a37ced3e12: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_has_dynamic_offset_69725fed837748fe: function(arg0, arg1) {
            arg0.hasDynamicOffset = arg1 !== 0;
        },
        __wbg_set_height_975770494a218d52: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_label_26577513096f145b: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_2a41a6f671383447: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_37d0faa0c9b7dee4: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_3e306b2e8f9db666: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_570d3dee0e80279e: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_58fbc9fcc6363f16: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_5a4dbb42c3b27bf7: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_5c952448f9d59f36: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_5fadf65a1f0f4714: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_782e33de78d86641: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_837a3b8ff99c2db3: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_8df6673e1e141fcc: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_layout_cd5d951ba305620a: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_d701bf37a1e489c6: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_mapped_at_creation_7f0aad21612f3e22: function(arg0, arg1) {
            arg0.mappedAtCreation = arg1 !== 0;
        },
        __wbg_set_min_binding_size_d70e460d165d9144: function(arg0, arg1) {
            arg0.minBindingSize = arg1;
        },
        __wbg_set_mip_level_8d4dfc5d506cb37f: function(arg0, arg1) {
            arg0.mipLevel = arg1 >>> 0;
        },
        __wbg_set_mip_level_count_04af0d33c4905fac: function(arg0, arg1) {
            arg0.mipLevelCount = arg1 >>> 0;
        },
        __wbg_set_mip_level_count_dcb2ad32716506a5: function(arg0, arg1) {
            arg0.mipLevelCount = arg1 >>> 0;
        },
        __wbg_set_module_22d452288cef846d: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_multisampled_4ce4c32144215354: function(arg0, arg1) {
            arg0.multisampled = arg1 !== 0;
        },
        __wbg_set_offset_0e56098d94f81ccd: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_offset_baf6780761c43b24: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_offset_e316586bb85f0bd6: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_origin_24a61b4427e330e9: function(arg0, arg1) {
            arg0.origin = arg1;
        },
        __wbg_set_power_preference_7d669fb9b41f7bf2: function(arg0, arg1) {
            arg0.powerPreference = __wbindgen_enum_GpuPowerPreference[arg1];
        },
        __wbg_set_query_set_604a8ae10429942b: function(arg0, arg1) {
            arg0.querySet = arg1;
        },
        __wbg_set_required_features_3d00070d09235d7d: function(arg0, arg1) {
            arg0.requiredFeatures = arg1;
        },
        __wbg_set_required_limits_e0de55a49a48e3dc: function(arg0, arg1) {
            arg0.requiredLimits = arg1;
        },
        __wbg_set_resource_fe1f979fce4afee2: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_rows_per_image_1f4a56a3c5d57e93: function(arg0, arg1) {
            arg0.rowsPerImage = arg1 >>> 0;
        },
        __wbg_set_rows_per_image_c616c70e60a35618: function(arg0, arg1) {
            arg0.rowsPerImage = arg1 >>> 0;
        },
        __wbg_set_sample_count_2b8ac49e1626ac13: function(arg0, arg1) {
            arg0.sampleCount = arg1 >>> 0;
        },
        __wbg_set_sample_type_3cecbd4699e2e5fb: function(arg0, arg1) {
            arg0.sampleType = __wbindgen_enum_GpuTextureSampleType[arg1];
        },
        __wbg_set_sampler_12544c21977075c1: function(arg0, arg1) {
            arg0.sampler = arg1;
        },
        __wbg_set_size_0c20f73abce8f1ce: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_cf04b4174c30722b: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_f1207de283144c72: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_storage_texture_36be4834c501acab: function(arg0, arg1) {
            arg0.storageTexture = arg1;
        },
        __wbg_set_texture_64823aa8aca790b5: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_texture_738e6f6215515de3: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_timestamp_writes_6854d9d17bf5b0b4: function(arg0, arg1) {
            arg0.timestampWrites = arg1;
        },
        __wbg_set_type_17a1387b620bc902: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuBufferBindingType[arg1];
        },
        __wbg_set_type_d4edb621ec2051e0: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuSamplerBindingType[arg1];
        },
        __wbg_set_usage_41b7d18f3f220e6c: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_usage_e167dd772123f679: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_usage_f084cd416060ceee: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_view_dimension_4a840560a13b4860: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_dimension_9ae69db849267b1a: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_formats_cba8520bf0d83d62: function(arg0, arg1) {
            arg0.viewFormats = arg1;
        },
        __wbg_set_visibility_bbbf3d2b70571950: function(arg0, arg1) {
            arg0.visibility = arg1 >>> 0;
        },
        __wbg_set_width_0f26635b289b3c67: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_x_15a4c893b3366fab: function(arg0, arg1) {
            arg0.x = arg1 >>> 0;
        },
        __wbg_set_y_c631920a1c51a694: function(arg0, arg1) {
            arg0.y = arg1 >>> 0;
        },
        __wbg_set_z_7c526101c55ea2ae: function(arg0, arg1) {
            arg0.z = arg1 >>> 0;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_9d53f2689e622ca1: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_a1a35cec07001a8a: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_4c59f6c7ea29a144: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_e70ae9f2eb052253: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_submit_b3bbead76cbf7627: function(arg0, arg1) {
            arg0.submit(arg1);
        },
        __wbg_then_18f476d590e58992: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_529ea37d9bdbf95d: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_ac7b025999b52837: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_unmap_817a2e3248a553fb: function(arg0) {
            arg0.unmap();
        },
        __wbg_writeBuffer_24a10bfd5a8a57f7: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.writeBuffer(arg1, arg2, getArrayU8FromWasm0(arg3, arg4), arg5, arg6);
        }, arguments); },
        __wbg_writeTexture_acb28796746826c8: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.writeTexture(arg1, getArrayU8FromWasm0(arg2, arg3), arg4, arg5);
        }, arguments); },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 344, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h03919243d83d1356);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 379, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__hb3a19924738e3ab7);
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000006: function(arg0, arg1) {
            var v0 = getArrayU8FromWasm0(arg0, arg1).slice();
            wasm.__wbindgen_free(arg0, arg1 * 1, 1);
            // Cast intrinsic for `Vector(U8) -> Externref`.
            const ret = v0;
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./image_js_bg.js": import0,
    };
}

function wasm_bindgen__convert__closures_____invoke__h03919243d83d1356(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h03919243d83d1356(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__hb3a19924738e3ab7(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen__convert__closures_____invoke__hb3a19924738e3ab7(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen__convert__closures_____invoke__h9344d28c99dc1e49(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen__convert__closures_____invoke__h9344d28c99dc1e49(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuBufferBindingType = ["uniform", "storage", "read-only-storage"];


const __wbindgen_enum_GpuPowerPreference = ["low-power", "high-performance"];


const __wbindgen_enum_GpuSamplerBindingType = ["filtering", "non-filtering", "comparison"];


const __wbindgen_enum_GpuStorageTextureAccess = ["write-only", "read-only", "read-write"];


const __wbindgen_enum_GpuTextureAspect = ["all", "stencil-only", "depth-only"];


const __wbindgen_enum_GpuTextureDimension = ["1d", "2d", "3d"];


const __wbindgen_enum_GpuTextureFormat = ["r8unorm", "r8snorm", "r8uint", "r8sint", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint", "r32float", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint", "rg32float", "rgba16uint", "rgba16sint", "rgba16float", "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"];


const __wbindgen_enum_GpuTextureSampleType = ["float", "unfilterable-float", "depth", "sint", "uint"];


const __wbindgen_enum_GpuTextureViewDimension = ["1d", "2d", "2d-array", "cube", "cube-array", "3d"];
const DecodedHandleFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decodedhandle_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('image_js_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
