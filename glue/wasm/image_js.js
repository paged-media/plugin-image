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
     * The source was 16 bits per channel and was reduced to 8 at
     * ingest. Same reason `display` is here: a lossy step the user
     * can see stated is a different thing from one they cannot.
     * @returns {boolean}
     */
    get depth_reduced() {
        const ret = wasm.__wbg_get_decodedhandle_depth_reduced(this.__wbg_ptr);
        return ret !== 0;
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
     * The source was 16 bits per channel and was reduced to 8 at
     * ingest. Same reason `display` is here: a lossy step the user
     * can see stated is a different thing from one they cannot.
     * @param {boolean} arg0
     */
    set depth_reduced(arg0) {
        wasm.__wbg_set_decodedhandle_depth_reduced(this.__wbg_ptr, arg0);
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
 * NOISE — despeckle. `amount` 0 is the identity. The edge gate
 * means this smooths speckle WITHOUT softening an edge, which a
 * plain median cannot do.
 * @param {number} handle
 * @param {number} edge_threshold
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_despeckle(handle, edge_threshold, amount) {
    const ret = wasm.apply_despeckle(handle, edge_threshold, amount);
    return ret;
}

/**
 * NOISE — dust & scratches. `threshold` 1.0 is the identity for
 * display-referred data, and `radius` 0 is the unconditional one.
 * The radius is CLAMPED to the kernel's declared ROI rather than
 * silently reading outside it.
 * @param {number} handle
 * @param {number} radius
 * @param {number} threshold
 * @returns {Promise<DecodedHandle>}
 */
export function apply_dust_scratches(handle, radius, threshold) {
    const ret = wasm.apply_dust_scratches(handle, radius, threshold);
    return ret;
}

/**
 * STYLIZE — emboss. `angle_deg` is the light direction, `height`
 * the relief strength; height 0 is the identity (flat mid-grey
 * plus nothing).
 * @param {number} handle
 * @param {number} angle_deg
 * @param {number} height
 * @returns {Promise<DecodedHandle>}
 */
export function apply_emboss(handle, angle_deg, height) {
    const ret = wasm.apply_emboss(handle, angle_deg, height);
    return ret;
}

/**
 * STYLIZE — find edges. `strength` scales the gradient before the
 * inversion; the result is dark lines on white.
 * @param {number} handle
 * @param {number} strength
 * @returns {Promise<DecodedHandle>}
 */
export function apply_find_edges(handle, strength) {
    const ret = wasm.apply_find_edges(handle, strength);
    return ret;
}

/**
 * Directional lighting over a luminance heightfield. Chrome and Plastic
 * Wrap are APPROXIMATED at extreme settings, not genuinely modelled.
 * @param {number} handle
 * @param {number} angle_deg
 * @param {number} elevation_deg
 * @param {number} height
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_bas_relief(handle, angle_deg, elevation_deg, height, amount) {
    const ret = wasm.apply_gallery_bas_relief(handle, angle_deg, elevation_deg, height, amount);
    return ret;
}

/**
 * Directional hatching modulated by luminance. `sets` alone separates
 * three named gallery filters: 1 = Graphic Pen, 2 = Crosshatch,
 * 3 = Sumi-e — which is why they are one kernel and not three.
 * @param {number} handle
 * @param {number} angle_deg
 * @param {number} spacing_px
 * @param {number} strength
 * @param {number} sets
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_crosshatch(handle, angle_deg, spacing_px, strength, sets, amount) {
    const ret = wasm.apply_gallery_crosshatch(handle, angle_deg, spacing_px, strength, sets, amount);
    return ret;
}

/**
 * Random local displacement. `anisotropy` is the whole difference
 * between Spatter (isotropic) and Sprayed Strokes (directional).
 * @param {number} handle
 * @param {number} seed
 * @param {number} radius_px
 * @param {number} angle_deg
 * @param {number} anisotropy
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_diffuse(handle, seed, radius_px, angle_deg, anisotropy, amount) {
    const ret = wasm.apply_gallery_diffuse(handle, seed, radius_px, angle_deg, anisotropy, amount);
    return ret;
}

/**
 * Refraction-style displacement through a procedural normal field.
 * @param {number} handle
 * @param {number} seed
 * @param {number} scale_px
 * @param {number} distortion
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_glass(handle, seed, scale_px, distortion, amount) {
    const ret = wasm.apply_gallery_glass(handle, seed, scale_px, distortion, amount);
    return ret;
}

/**
 * Edge magnitude colourised and boosted — keeps the source hue rather
 * than going grey, which is the difference from an inverted find-edges.
 * @param {number} handle
 * @param {number} intensity
 * @param {number} smoothness
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_glowing_edges(handle, intensity, smoothness, amount) {
    const ret = wasm.apply_gallery_glowing_edges(handle, intensity, smoothness, amount);
    return ret;
}

/**
 * Film grain. The noise is a deterministic hash of the coordinate and
 * the seed, never host randomness, so undo/redo cannot shimmer.
 * @param {number} handle
 * @param {number} seed
 * @param {number} size_px
 * @param {boolean} mono
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_grain(handle, seed, size_px, mono, amount) {
    const ret = wasm.apply_gallery_grain(handle, seed, size_px, mono, amount);
    return ret;
}

/**
 * Screen-angle halftone. Dot AREA carries the tone, not dot darkness —
 * the property that lets a halftone survive a 1-bit output.
 * @param {number} handle
 * @param {number} cell_px
 * @param {number} angle_deg
 * @param {number} contrast
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_halftone(handle, cell_px, angle_deg, contrast, amount) {
    const ret = wasm.apply_gallery_halftone(handle, cell_px, angle_deg, contrast, amount);
    return ret;
}

/**
 * Painterly flattening (Kuwahara): the lowest-variance quadrant's mean,
 * so a region moves toward paint while its boundary stays put. Serves
 * Paint Daubs, Palette Knife, Watercolor, Underpainting, Dry Brush.
 * @param {number} handle
 * @param {number} radius_px
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_kuwahara(handle, radius_px, amount) {
    const ret = wasm.apply_gallery_kuwahara(handle, radius_px, amount);
    return ret;
}

/**
 * Posterised colour plus a darkened outline. The two halves are
 * independent, which is what separates Cutout (flat, no ink) from Ink
 * Outlines (ink, little flattening).
 * @param {number} handle
 * @param {number} levels
 * @param {number} edge_amount
 * @param {number} edge_threshold
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_posterize_edges(handle, levels, edge_amount, edge_threshold, amount) {
    const ret = wasm.apply_gallery_posterize_edges(handle, levels, edge_amount, edge_threshold, amount);
    return ret;
}

/**
 * Cellular tiling with a border term. The border weight alone separates
 * Crystallize (none) from Stained Glass (heavy).
 * @param {number} handle
 * @param {number} seed
 * @param {number} cell_px
 * @param {number} border
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_stained_glass(handle, seed, cell_px, border, amount) {
    const ret = wasm.apply_gallery_stained_glass(handle, seed, cell_px, border, amount);
    return ret;
}

/**
 * Procedural surface texture modulating shading; `kind` picks the
 * surface (Canvas / Burlap / Brick) rather than a kernel per material.
 * @param {number} handle
 * @param {number} seed
 * @param {number} kind
 * @param {number} scale_px
 * @param {number} relief
 * @param {number} angle_deg
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_texturizer(handle, seed, kind, scale_px, relief, angle_deg, amount) {
    const ret = wasm.apply_gallery_texturizer(handle, seed, kind, scale_px, relief, angle_deg, amount);
    return ret;
}

/**
 * Luminance threshold with a soft ramp; the ramp width is what makes
 * Stamp hard and Charcoal soft.
 * @param {number} handle
 * @param {number} threshold
 * @param {number} softness
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_gallery_threshold_ink(handle, threshold, softness, amount) {
    const ret = wasm.apply_gallery_threshold_ink(handle, threshold, softness, amount);
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
 * BLUR — lens / bokeh. `radius_px` below 0.5 is the identity.
 * `threshold` is the luminance above which a pixel counts as a
 * highlight and gets weighted up; `boost` is how hard.
 * @param {number} handle
 * @param {number} radius_px
 * @param {number} threshold
 * @param {number} boost
 * @returns {Promise<DecodedHandle>}
 */
export function apply_lens_blur(handle, radius_px, threshold, boost) {
    const ret = wasm.apply_lens_blur(handle, radius_px, threshold, boost);
    return ret;
}

/**
 * ADJUST — Color Lookup: a 9×9×9 RGB cube (`cube` = 729 rgb triples,
 * red fastest, values 0–1), applied trilinearly. The panel resamples
 * a .cube file of any size to this edge first.
 * @param {number} handle
 * @param {Float32Array} cube
 * @returns {Promise<DecodedHandle>}
 */
export function apply_lut3d(handle, cube) {
    const ptr0 = passArrayF32ToWasm0(cube, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.apply_lut3d(handle, ptr0, len0);
    return ret;
}

/**
 * NOISE — Median, 3×3 (the kernel's fixed comparator network; a
 * larger radius needs a histogram method, which is a different
 * kernel). Every output texel is one of the input samples.
 * @param {number} handle
 * @returns {Promise<DecodedHandle>}
 */
export function apply_median(handle) {
    const ret = wasm.apply_median(handle);
    return ret;
}

/**
 * OTHER — Maximum (`kind` "max", grey dilation) or Minimum ("min",
 * grey erosion), 3×3.
 * @param {number} handle
 * @param {string} kind
 * @returns {Promise<DecodedHandle>}
 */
export function apply_morph(handle, kind) {
    const ptr0 = passStringToWasm0(kind, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.apply_morph(handle, ptr0, len0);
    return ret;
}

/**
 * PIXELATE — mosaic. `cell_px` <= 1 is the identity.
 * @param {number} handle
 * @param {number} cell_px
 * @returns {Promise<DecodedHandle>}
 */
export function apply_mosaic(handle, cell_px) {
    const ret = wasm.apply_mosaic(handle, cell_px);
    return ret;
}

/**
 * BLUR GALLERY — motion. `length_px` 0 is the identity.
 * @param {number} handle
 * @param {number} angle_deg
 * @param {number} length_px
 * @returns {Promise<DecodedHandle>}
 */
export function apply_motion_blur(handle, angle_deg, length_px) {
    const ret = wasm.apply_motion_blur(handle, angle_deg, length_px);
    return ret;
}

/**
 * MOVE the SELECTED pixels — the Move tool with a live selection.
 *
 * Distinct from [`apply_offset`], which moves the whole layer. With
 * a selection this is what a user means by "move": the selected
 * pixels relocate and the region they came from is vacated.
 * `vacate` is 0 transparent (plain drag) / 1 copy (alt-drag).
 * dx=dy=0 is the identity under either.
 * @param {number} handle
 * @param {number} dx
 * @param {number} dy
 * @param {number} vacate
 * @returns {Promise<DecodedHandle>}
 */
export function apply_move_selection(handle, dx, dy, vacate) {
    const ret = wasm.apply_move_selection(handle, dx, dy, vacate);
    return ret;
}

/**
 * MOVE the pixels of the active layer by (dx, dy). `edge` is
 * 0 transparent / 1 clamp / 2 wrap; dx=dy=0 is the identity for
 * every policy. At edge=2 this is Photoshop's Filter > Other >
 * Offset — one kernel, two surfaces.
 * @param {number} handle
 * @param {number} dx
 * @param {number} dy
 * @param {number} edge
 * @returns {Promise<DecodedHandle>}
 */
export function apply_offset(handle, dx, dy, edge) {
    const ret = wasm.apply_offset(handle, dx, dy, edge);
    return ret;
}

/**
 * BLUR GALLERY — radial. `spin` picks the mode; `amount` 0 is the
 * identity for both. The centre is NORMALISED (0..1) so it survives
 * a resize, unlike a pixel centre which would drift.
 * @param {number} handle
 * @param {number} cx
 * @param {number} cy
 * @param {number} amount
 * @param {boolean} spin
 * @returns {Promise<DecodedHandle>}
 */
export function apply_radial_blur(handle, cx, cy, amount, spin) {
    const ret = wasm.apply_radial_blur(handle, cx, cy, amount, spin);
    return ret;
}

/**
 * RED-EYE removal inside the ellipse `(cx, cy, rx, ry)` (image px):
 * only RED pixels change — the mask is the ellipse times each
 * pixel's redness — and they become the average of their green and
 * blue, darkened by `darken` (0–1). The channel mixer does the
 * colour work under that mask; no new kernel.
 * @param {number} handle
 * @param {number} cx
 * @param {number} cy
 * @param {number} rx
 * @param {number} ry
 * @param {number} darken
 * @returns {Promise<DecodedHandle>}
 */
export function apply_red_eye(handle, cx, cy, rx, ry, darken) {
    const ret = wasm.apply_red_eye(handle, cx, cy, rx, ry, darken);
    return ret;
}

/**
 * NOISE — reduce noise (bilateral). `amount` 0 is the identity, and
 * so is a `sigma_range` small enough that only the centre tap
 * carries weight.
 * @param {number} handle
 * @param {number} radius_px
 * @param {number} sigma_range
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_reduce_noise(handle, radius_px, sigma_range, amount) {
    const ret = wasm.apply_reduce_noise(handle, radius_px, sigma_range, amount);
    return ret;
}

/**
 * ADJUST — selective colour. `range` 0..8; all-zero deltas are the
 * identity for every range.
 * @param {number} handle
 * @param {number} range
 * @param {number} cyan
 * @param {number} magenta
 * @param {number} yellow
 * @param {number} black
 * @param {boolean} absolute
 * @returns {Promise<DecodedHandle>}
 */
export function apply_selective_color(handle, range, cyan, magenta, yellow, black, absolute) {
    const ret = wasm.apply_selective_color(handle, range, cyan, magenta, yellow, black, absolute);
    return ret;
}

/**
 * BLUR — SHAPE. Filter > Blur > Shape Blur: convolve with an
 * ARBITRARY shape rather than a formula.
 *
 * `shape_handle` is an ordinary decoded-image handle, so any image
 * the plugin can open is a blur shape. Coverage is read from RED,
 * not alpha — a shape library ships white-on-black (where alpha is
 * identically 1, and reading it would make every shape a box) or
 * white-on-transparent (where premultiplied red equals alpha), and
 * red is correct for both conventions.
 *
 * `radius_px` is the HALF-extent the shape is scaled to; below 0.5
 * it is the identity, as is `amount == 0`.
 * @param {number} handle
 * @param {number} shape_handle
 * @param {number} radius_px
 * @param {number} amount
 * @returns {Promise<DecodedHandle>}
 */
export function apply_shape_blur(handle, shape_handle, radius_px, amount) {
    const ret = wasm.apply_shape_blur(handle, shape_handle, radius_px, amount);
    return ret;
}

/**
 * SHARPEN — smart sharpen. `amount` 0 is the identity; regions
 * whose local contrast is below `threshold` are left untouched
 * whatever the amount, which is the point of the "smart".
 * @param {number} handle
 * @param {number} radius_px
 * @param {number} amount
 * @param {number} threshold
 * @param {number} clamp_hi
 * @returns {Promise<DecodedHandle>}
 */
export function apply_smart_sharpen(handle, radius_px, amount, threshold, clamp_hi) {
    const ret = wasm.apply_smart_sharpen(handle, radius_px, amount, threshold, clamp_hi);
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
 * The rectangle the last `brush_stroke_extend` changed, as
 * `[x, y, w, h]` in image px (empty when it painted nothing) — what a
 * host that takes tiles needs to resend instead of the whole preview.
 * @returns {Uint32Array}
 */
export function brush_stroke_dirty_rect() {
    const ret = wasm.brush_stroke_dirty_rect();
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
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
 * Set the IN-FLIGHT dodge / burn / sponge stroke's options:
 * `range` ∈ `shadows | midtones | highlights` and `exposure` 0–1 for
 * dodge and burn; `saturate` picks the sponge's direction (false =
 * desaturate). Call between `brush_stroke_begin` and the first
 * extend — the options are frozen with the stroke. An error for any
 * other tool, so a caller cannot believe a brush is dodging.
 * @param {string} range
 * @param {number} exposure
 * @param {boolean} saturate
 */
export function brush_stroke_set_tone(range, exposure, saturate) {
    const ptr0 = passStringToWasm0(range, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.brush_stroke_set_tone(ptr0, len0, exposure, saturate);
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
 * The PAINT BUCKET: flood from image pixel `(x, y)` over pixels within
 * `tolerance` of it (connected ones only when `contiguous`), within
 * the selection when there is one, and fill that with `color`
 * (straight RGBA in `[0, 1]`). The flood samples the COMPOSITE —
 * Photoshop's "sample all layers" — and paints the active layer.
 * @param {number} handle
 * @param {number} x
 * @param {number} y
 * @param {number} tolerance
 * @param {boolean} contiguous
 * @param {Float32Array} color
 * @returns {Promise<DecodedHandle>}
 */
export function bucket_fill(handle, x, y, tolerance, contiguous, color) {
    const ptr0 = passArrayF32ToWasm0(color, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bucket_fill(handle, x, y, tolerance, contiguous, ptr0, len0);
    return ret;
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
 * [`encode_image`] with the two knobs it has not got: a JPEG
 * `quality` (the plain door rides a fixed 90), and a LOSSLESS
 * channel reduction for PNG.
 *
 * `reduce` is not a quality setting. It re-expresses a buffer that
 * was ALREADY greyscale in a layout that says so — one byte per
 * pixel instead of four when the alpha is constant-opaque, two
 * when it is not — so the decoded pixels come back identical byte
 * for byte. It pays on exactly the images that tend to be large
 * and greyscale: scans, masks, line art, alpha mattes. On a colour
 * photograph the classifier exits within a few pixels and nothing
 * changes.
 *
 * What it deliberately does NOT do is drop a constant-opaque alpha
 * to three channels, which would save a quarter of the raw bytes
 * on every screenshot. `ChannelLayout` has no RGB arm and the type
 * is frozen — RFI E-6.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {string} format
 * @param {number} quality
 * @param {boolean} reduce
 * @returns {Uint8Array}
 */
export function encode_image_opt(rgba, width, height, format, quality, reduce) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(format, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.encode_image_opt(ptr0, len0, width, height, ptr1, len1, quality, reduce);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @returns {string}
 */
export function engine_source_hash() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.engine_source_hash();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
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
 * The GRADIENT TOOL: fill the selection (the whole image when none)
 * with a two-stop gradient along the dragged line `(x0, y0)` →
 * `(x1, y1)` in image pixels (`FillSpec::GradientLine`).
 * @param {number} handle
 * @param {string} kind
 * @param {Float32Array} c0
 * @param {Float32Array} c1
 * @param {number} x0
 * @param {number} y0
 * @param {number} x1
 * @param {number} y1
 * @returns {Promise<DecodedHandle>}
 */
export function fill_gradient_line(handle, kind, c0, c1, x0, y0, x1, y1) {
    const ptr0 = passStringToWasm0(kind, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(c0, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(c1, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.fill_gradient_line(handle, ptr0, len0, ptr1, len1, ptr2, len2, x0, y0, x1, y1);
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
 * FILL with a PATTERN — Edit > Fill > Pattern, and the pattern
 * half of the paint bucket.
 *
 * `tile_handle` is an ordinary decoded-image handle, so any image
 * the plugin can open is a pattern; there is no separate pattern
 * asset type and deliberately no SWATCH. The vector pattern-paint
 * this is often confused with is RFI C-31 — core model + both
 * renderers + an IDML round-trip decision — and none of it is
 * needed to tile pixels into a raster layer.
 *
 * Selection-scoped like every other fill, so "fill the selection
 * with a pattern" comes free from the ABI mask.
 * `opacity == 0` is the identity.
 * @param {number} handle
 * @param {number} tile_handle
 * @param {number} scale
 * @param {number} angle_deg
 * @param {number} offset_x
 * @param {number} offset_y
 * @param {number} opacity
 * @returns {Promise<DecodedHandle>}
 */
export function fill_pattern(handle, tile_handle, scale, angle_deg, offset_x, offset_y, opacity) {
    const ret = wasm.fill_pattern(handle, tile_handle, scale, angle_deg, offset_x, offset_y, opacity);
    return ret;
}

/**
 * FILL the current selection (the whole image when none) with ONE
 * colour — Edit ▸ Fill ▸ Foreground colour. `color` is straight RGBA
 * in `[0, 1]`. Returns the engine-held image's handle.
 * @param {number} handle
 * @param {Float32Array} color
 * @returns {Promise<DecodedHandle>}
 */
export function fill_solid(handle, color) {
    const ptr0 = passArrayF32ToWasm0(color, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.fill_solid(handle, ptr0, len0);
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
 * ADD LAYER MASK — REVEAL ALL (`reveal_all`, an all-white mask that
 * changes nothing until painted) or HIDE ALL (all-black, the layer
 * vanishes until painted back in). One undo step. The new mask
 * becomes the EDIT TARGET, as Photoshop does, so the next stroke
 * paints it. Refused when the layer already has a mask.
 * @param {number} index
 * @param {boolean} reveal_all
 */
export function layers_add_mask(index, reveal_all) {
    const ret = wasm.layers_add_mask(index, reveal_all);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
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
 * IMAGE ▸ Rotate / Flip / Canvas Size, over the WHOLE stack (every
 * layer, mask and smart source moves together; see
 * `LayerStack::transform_canvas`). `op` is "rotate-cw", "rotate-ccw",
 * "rotate-180", "flip-h", "flip-v" or "canvas" (then `width`,
 * `height` and the anchors apply; they are ignored otherwise).
 *
 * The bound image takes the new extent and the composite; its mip
 * pyramid goes, and the selection is re-bound (dropped), because
 * both address the old extent. The undo history is cleared.
 * Returns `[width, height]`.
 * @param {string} op
 * @param {number} width
 * @param {number} height
 * @param {number} anchor_x
 * @param {number} anchor_y
 * @returns {Promise<Uint32Array>}
 */
export function layers_canvas_op(op, width, height, anchor_x, anchor_y) {
    const ptr0 = passStringToWasm0(op, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.layers_canvas_op(ptr0, len0, width, height, anchor_x, anchor_y);
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
 * The open stack as bytes for storing in the document: element 0
 * is the manifest, the rest are the buffers it refers to by slot
 * (`LayerStack::export`). The history is not included.
 * @returns {Array<any>}
 */
export function layers_export() {
    const ret = wasm.layers_export();
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
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
 * Replace the open stack with one stored by `layers_export`, then
 * composite it into the bound image (GPU-only unless the stored stack
 * is a single plain layer). The stored extent must equal the bound
 * image's; the history starts empty.
 * @param {Uint8Array} manifest
 * @param {Array<any>} buffers
 * @returns {Promise<void>}
 */
export function layers_import(manifest, buffers) {
    const ptr0 = passArray8ToWasm0(manifest, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.layers_import(ptr0, len0, buffers);
    return ret;
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
 * whose structure the layer model does not reproduce, because
 * swapping Photoshop's own composite for a different-looking one of
 * ours would be worse than flattening. On a refusal the caller keeps
 * `layers_open` (the flatten) and shows the reason.
 *
 * The refusal list SHRANK on 2026-08-06: CLIPPING is imported now
 * that the model has it. Still refused: groups (this stack has
 * them, but PSD groups nest and default to pass-through and this
 * one does neither), layer masks, non-8-bit-RGB and an over-budget
 * canvas. A refusal for a capability we since gained is a lie about
 * ourselves, so the list is worth re-reading whenever the model
 * grows.
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
 * keeps at least one). One undo step (`LayerStack::remove`).
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
 * EDIT an adjustment layer's chain in place (same arguments as
 * `layers_add_adjustment`, after the layer index). Identity is
 * allowed here — an edited layer may be dialled back to nothing.
 * @param {number} index
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
 */
export function layers_set_adjustment(index, exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, curve_lut, blur_sigma, sharpen_amount, hue_degrees, invert, ext) {
    const ptr0 = passArray8ToWasm0(curve_lut, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(ext, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.layers_set_adjustment(index, exposure_ev, brightness, contrast, saturation, temp, tint, in_black, in_white, gamma, out_black, out_white, ptr0, len0, blur_sigma, sharpen_amount, hue_degrees, invert, ptr1, len1);
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
 * Make `index` active and choose what the paint tools write on it:
 * its pixels, or (`mask`) its layer MASK. Selecting the mask of a
 * layer with none is an error. Not an undo step — choosing a target
 * changes nothing in the document.
 * @param {number} index
 * @param {boolean} mask
 */
export function layers_set_edit_target(index, mask) {
    const ret = wasm.layers_set_edit_target(index, mask);
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
 * Switch a group between PASS THROUGH (the default: members reach
 * the stack below) and ISOLATED (members composite into their own
 * buffer first). Note that an opacity below 1 forces isolation
 * regardless — `layers_list`'s `isolates` reports the effective
 * mode, `passThrough` the declared one.
 * @param {number} id
 * @param {boolean} pass_through
 */
export function layers_set_group_pass_through(id, pass_through) {
    const ret = wasm.layers_set_group_pass_through(id, pass_through);
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
 * PATCH: replace the SELECTION with the region `(dx, dy)` image px
 * away from it, healed so it blends (`retouch::patch_rgba8`: the
 * shifted source plus the membrane tone correction, composited
 * through the selection's coverage on the GPU). Lands in the active
 * layer as one journaled undo step ("Patch"). Needs a selection; a
 * zero offset is refused rather than spending an undo step on the
 * identity.
 * @param {number} handle
 * @param {number} dx
 * @param {number} dy
 * @returns {Promise<DecodedHandle>}
 */
export function patch_selection(handle, dx, dy) {
    const ret = wasm.patch_selection(handle, dx, dy);
    return ret;
}

/**
 * The hash of the sources this wasm was built from
 * (`scripts/source-hash.mjs`, stamped by `scripts/build-wasm.sh`;
 * "unstamped" for any other build). `glue/test/wasm-fresh.spec.ts`
 * compares it with the checkout, so a stale committed wasm fails
 * the suite instead of being tested in place of the code.
 * Engine + GPU work counters since the last reset, as JSON
 * (`{engine:{…}, gpu:{…}}`, see `counters::to_json`). What the
 * bundle's performance budgets count.
 * @returns {string}
 */
export function perf_counters() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.perf_counters();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Zero the engine and GPU work counters.
 */
export function perf_counters_reset() {
    wasm.perf_counters_reset();
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
 * Write the open LAYER STACK into the retained PSD as its layers
 * (`saveback::psd_write_stack`), with the bound image (the stack's
 * composite) as the merged image. Refuses stacks with adjustment
 * layers and non-8-bit-RGB files, so the caller can fall back to the
 * flattened save. Returns the user-facing description.
 * @param {number} psd_handle
 * @returns {string}
 */
export function psd_save_layers(psd_handle) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.psd_save_layers(psd_handle);
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
 * Select ▸ Modify: `op` ∈ expand | contract | border | smooth, by
 * `radius` px. Errors when there is no explicit selection.
 * @param {string} op
 * @param {number} radius
 */
export function selection_modify(op, radius) {
    const ptr0 = passStringToWasm0(op, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.selection_modify(ptr0, len0, radius);
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

/**
 * RASTER TYPE: shape `text` with `font_bytes`, rasterize it, and
 * paint it into the image at `(x, y)` in `rgba`.
 *
 * The glyph run becomes a `SelectionCoverage` and the paint is the
 * ORDINARY masked solid fill — no new kernel and no new compositing
 * path, because coverage is coverage. `(x, y)` is the run's
 * BASELINE ORIGIN, which is where type is positioned from; the
 * rasterizer reports its own ink offset and this places it.
 *
 * Returns the number of glyphs the font had no coverage for
 * (`.notdef`), so a caller can say "3 characters are missing from
 * this font" instead of silently dropping them.
 * @param {number} handle
 * @param {Uint8Array} font_bytes
 * @param {string} text
 * @param {number} size_px
 * @param {number} x
 * @param {number} y
 * @param {Float32Array} color
 * @param {number} tracking_per_mille
 * @param {number} leading_px
 * @param {number} baseline_shift_px
 * @param {number} h_scale
 * @param {number} v_scale
 * @param {number} skew_deg
 * @param {number} underline
 * @param {number} strikethrough
 * @param {number} ligatures
 * @param {number} align
 * @returns {Promise<number>}
 */
export function text_paint(handle, font_bytes, text, size_px, x, y, color, tracking_per_mille, leading_px, baseline_shift_px, h_scale, v_scale, skew_deg, underline, strikethrough, ligatures, align) {
    const ptr0 = passArray8ToWasm0(font_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(color, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.text_paint(handle, ptr0, len0, ptr1, len1, size_px, x, y, ptr2, len2, tracking_per_mille, leading_px, baseline_shift_px, h_scale, v_scale, skew_deg, underline, strikethrough, ligatures, align);
    return ret;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Window_a07901001eb4269f: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_d1b9459d53a39f3d: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_debug_string_0e68cf47c9cbd9b0: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_fcda5e3902d732fe: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_5160b3e381865372: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_undefined_8c687d0b90d5b524: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_5d9e815e6fdf150f: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_997e73d32238e655: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_beginComputePass_705eb14eefc2b94e: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_call_6bcf8d3e20937e46: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_copyTextureToBuffer_4186c16aef1922a5: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            arg0.copyTextureToBuffer(arg1, arg2, arg3);
        }, arguments); },
        __wbg_copyTextureToTexture_1be188df1e535c0a: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            arg0.copyTextureToTexture(arg1, arg2, arg3);
        }, arguments); },
        __wbg_createBindGroupLayout_9ea1a44942aaf13e: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBindGroupLayout(arg1);
            return ret;
        }, arguments); },
        __wbg_createBindGroup_2320df4db188406c: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_2f08c0205e04efca: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        }, arguments); },
        __wbg_createCommandEncoder_cd88faca35d9ed68: function(arg0, arg1) {
            const ret = arg0.createCommandEncoder(arg1);
            return ret;
        },
        __wbg_createComputePipeline_3e135ff73c8fc483: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createPipelineLayout_7a186f2e9bf0d605: function(arg0, arg1) {
            const ret = arg0.createPipelineLayout(arg1);
            return ret;
        },
        __wbg_createShaderModule_53701de4fb271c90: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_createTexture_9e76b80a2dc0d12e: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createTexture(arg1);
            return ret;
        }, arguments); },
        __wbg_createView_cc96b5bdd3d5bf5e: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createView(arg1);
            return ret;
        }, arguments); },
        __wbg_decodedhandle_new: function(arg0) {
            const ret = DecodedHandle.__wrap(arg0);
            return ret;
        },
        __wbg_description_18d0a6d4077fec8e: function(arg0, arg1) {
            const ret = arg1.description;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_dispatchWorkgroups_0cf298d736b85a78: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_end_fb560a3ae8e3624e: function(arg0) {
            arg0.end();
        },
        __wbg_error_757e9472f8410341: function(arg0, arg1) {
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
        __wbg_finish_087cb89c65c06eb1: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_finish_cfaeede3baf55be1: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_getMappedRange_5ed22727c9679168: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getMappedRange(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_get_989d0a1309644f2b: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_get_unchecked_363572bdd397d473: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_gpu_a7c12045c25d009a: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_info_22dcf1fd1b12bc7d: function(arg0) {
            const ret = arg0.info;
            return ret;
        },
        __wbg_instanceof_GpuAdapter_fc7b89fc546de0bc: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUAdapter;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_label_47480289cc2bce71: function(arg0, arg1) {
            const ret = arg1.label;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_length_31bdaf014f5fbde2: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_4e1adc0d42e23620: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_limits_50a8c5e629dbfe40: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_mapAsync_bb0029907dd91181: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.mapAsync(arg1 >>> 0, arg2, arg3);
            return ret;
        },
        __wbg_maxBindGroups_14611ac9ed1c6b56: function(arg0) {
            const ret = arg0.maxBindGroups;
            return ret;
        },
        __wbg_maxBindingsPerBindGroup_dd3f66044d2a9bfb: function(arg0) {
            const ret = arg0.maxBindingsPerBindGroup;
            return ret;
        },
        __wbg_maxBufferSize_f7ce3e1856349d2f: function(arg0) {
            const ret = arg0.maxBufferSize;
            return ret;
        },
        __wbg_maxColorAttachmentBytesPerSample_55e64194645ea041: function(arg0) {
            const ret = arg0.maxColorAttachmentBytesPerSample;
            return ret;
        },
        __wbg_maxColorAttachments_fd9187f9f786da18: function(arg0) {
            const ret = arg0.maxColorAttachments;
            return ret;
        },
        __wbg_maxComputeInvocationsPerWorkgroup_9b3b1fc261129782: function(arg0) {
            const ret = arg0.maxComputeInvocationsPerWorkgroup;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeX_c55bbbcc02b75241: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeX;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeY_96f40b1ec3102a3a: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeY;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeZ_c2b1061d521561bb: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeZ;
            return ret;
        },
        __wbg_maxComputeWorkgroupStorageSize_fac26e89d99e08f9: function(arg0) {
            const ret = arg0.maxComputeWorkgroupStorageSize;
            return ret;
        },
        __wbg_maxComputeWorkgroupsPerDimension_cd001f910e9b4d70: function(arg0) {
            const ret = arg0.maxComputeWorkgroupsPerDimension;
            return ret;
        },
        __wbg_maxDynamicStorageBuffersPerPipelineLayout_29399b82af020d86: function(arg0) {
            const ret = arg0.maxDynamicStorageBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxDynamicUniformBuffersPerPipelineLayout_6d6cf80f3bd08e52: function(arg0) {
            const ret = arg0.maxDynamicUniformBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxInterStageShaderVariables_8b000f47a166b1d5: function(arg0) {
            const ret = arg0.maxInterStageShaderVariables;
            return ret;
        },
        __wbg_maxSampledTexturesPerShaderStage_618a49f33217dde2: function(arg0) {
            const ret = arg0.maxSampledTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxSamplersPerShaderStage_aa09fa0311712a1a: function(arg0) {
            const ret = arg0.maxSamplersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageBufferBindingSize_0ec83ae10ad73180: function(arg0) {
            const ret = arg0.maxStorageBufferBindingSize;
            return ret;
        },
        __wbg_maxStorageBuffersPerShaderStage_0cca5b468fcf10b6: function(arg0) {
            const ret = arg0.maxStorageBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageTexturesPerShaderStage_9d6c35770f37866c: function(arg0) {
            const ret = arg0.maxStorageTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxTextureArrayLayers_c2bf9c85285832d4: function(arg0) {
            const ret = arg0.maxTextureArrayLayers;
            return ret;
        },
        __wbg_maxTextureDimension1D_e09f86e22ea6bac9: function(arg0) {
            const ret = arg0.maxTextureDimension1D;
            return ret;
        },
        __wbg_maxTextureDimension2D_2631916ef9a3efa8: function(arg0) {
            const ret = arg0.maxTextureDimension2D;
            return ret;
        },
        __wbg_maxTextureDimension3D_06ee54121b37d431: function(arg0) {
            const ret = arg0.maxTextureDimension3D;
            return ret;
        },
        __wbg_maxUniformBufferBindingSize_af9e8a077907ed64: function(arg0) {
            const ret = arg0.maxUniformBufferBindingSize;
            return ret;
        },
        __wbg_maxUniformBuffersPerShaderStage_f871b70865df8c11: function(arg0) {
            const ret = arg0.maxUniformBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxVertexAttributes_e72dabb2714f5cf5: function(arg0) {
            const ret = arg0.maxVertexAttributes;
            return ret;
        },
        __wbg_maxVertexBufferArrayStride_6a1cd814386082ce: function(arg0) {
            const ret = arg0.maxVertexBufferArrayStride;
            return ret;
        },
        __wbg_maxVertexBuffers_9c61c5fd286ebcc6: function(arg0) {
            const ret = arg0.maxVertexBuffers;
            return ret;
        },
        __wbg_minStorageBufferOffsetAlignment_e214f59628fb3558: function(arg0) {
            const ret = arg0.minStorageBufferOffsetAlignment;
            return ret;
        },
        __wbg_minUniformBufferOffsetAlignment_58b69e1c3924f6a4: function(arg0) {
            const ret = arg0.minUniformBufferOffsetAlignment;
            return ret;
        },
        __wbg_navigator_d217ca64c4bbff48: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_d25c0f071226f233: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_1da3429bc3c4541c: function(arg0) {
            const ret = new Uint8Array(arg0);
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_bebc3f4757acf305: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_ffa92086ea89f79c: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_2221cabb71753908: function(arg0, arg1) {
            const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_from_slice_4ee02165f9de919e: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_from_slice_a500ec81601be48f: function(arg0, arg1) {
            const ret = new Uint32Array(getArrayU32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_typed_6f8b0d724fe26c07: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen__convert__closures_____invoke__h4104b19d0e1a1b9a(a, state0.b, arg0, arg1);
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
        __wbg_new_with_byte_offset_and_length_492c969e8b5da8a4: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_length_5ffeddb9d9fbb96f: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return ret;
        },
        __wbg_onSubmittedWorkDone_1460145eecea40ef: function(arg0) {
            const ret = arg0.onSubmittedWorkDone();
            return ret;
        },
        __wbg_prototypesetcall_ae9f5e7459250748: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_bfdf956ba476f65b: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_queueMicrotask_85c90f6987555d65: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_f6a1fa10b81d1fc0: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queue_65d985f3e6d786a6: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_requestAdapter_9ff5c9d1ff271165: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_c1c34f88a477e509: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolve_35ec7e0c6af4c82c: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_setBindGroup_79afcff8b9db8be3: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        }, arguments); },
        __wbg_setBindGroup_84eb639ac393a9f4: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setPipeline_95c76ab8da697fcf: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_set_9cfc0f17d60ff0af: function(arg0, arg1, arg2) {
            arg0.set(arg1, arg2 >>> 0);
        },
        __wbg_set_a377297433dfea63: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_access_091f317905cd76a5: function(arg0, arg1) {
            arg0.access = __wbindgen_enum_GpuStorageTextureAccess[arg1];
        },
        __wbg_set_array_layer_count_daec613068108a9d: function(arg0, arg1) {
            arg0.arrayLayerCount = arg1 >>> 0;
        },
        __wbg_set_aspect_77332ac136ee94eb: function(arg0, arg1) {
            arg0.aspect = __wbindgen_enum_GpuTextureAspect[arg1];
        },
        __wbg_set_aspect_a823a14d00d42d37: function(arg0, arg1) {
            arg0.aspect = __wbindgen_enum_GpuTextureAspect[arg1];
        },
        __wbg_set_base_array_layer_cc6c68d233489c4b: function(arg0, arg1) {
            arg0.baseArrayLayer = arg1 >>> 0;
        },
        __wbg_set_base_mip_level_e07a3efe9006d5ea: function(arg0, arg1) {
            arg0.baseMipLevel = arg1 >>> 0;
        },
        __wbg_set_beginning_of_pass_write_index_c12e7856ee670800: function(arg0, arg1) {
            arg0.beginningOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_bind_group_layouts_5325d038771af328: function(arg0, arg1) {
            arg0.bindGroupLayouts = arg1;
        },
        __wbg_set_binding_b6b0fe5c281b8c69: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_binding_f3c188a8cd21455b: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_buffer_55f096330c8912b4: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_aa7bf4ad8f17b2bd: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_e89095a9f0cafad3: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_bytes_per_row_68a1ea90d4710bc9: function(arg0, arg1) {
            arg0.bytesPerRow = arg1 >>> 0;
        },
        __wbg_set_bytes_per_row_91681ca78d744888: function(arg0, arg1) {
            arg0.bytesPerRow = arg1 >>> 0;
        },
        __wbg_set_code_56e2d45ec1ff6c2d: function(arg0, arg1, arg2) {
            arg0.code = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_compute_5a859e405c9eb6c6: function(arg0, arg1) {
            arg0.compute = arg1;
        },
        __wbg_set_depth_or_array_layers_4bbbeadacb393f02: function(arg0, arg1) {
            arg0.depthOrArrayLayers = arg1 >>> 0;
        },
        __wbg_set_dimension_174ad7e2fb67fb4e: function(arg0, arg1) {
            arg0.dimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_dimension_36e13ccecae5af4b: function(arg0, arg1) {
            arg0.dimension = __wbindgen_enum_GpuTextureDimension[arg1];
        },
        __wbg_set_end_of_pass_write_index_f4ab90c5743df805: function(arg0, arg1) {
            arg0.endOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_entries_3017e6132f938c6e: function(arg0, arg1) {
            arg0.entries = arg1;
        },
        __wbg_set_entries_fc76ca4d7da6a709: function(arg0, arg1) {
            arg0.entries = arg1;
        },
        __wbg_set_entry_point_4443daff87d82ef1: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_external_texture_825fe2bc7a0c0603: function(arg0, arg1) {
            arg0.externalTexture = arg1;
        },
        __wbg_set_format_1786adb7bc74c7c9: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_90860b0321868db4: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_e9d4b1475bb3bd3b: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_has_dynamic_offset_7d30014fdbfe90c5: function(arg0, arg1) {
            arg0.hasDynamicOffset = arg1 !== 0;
        },
        __wbg_set_height_e8b5483b8c117d5e: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_label_03d2396d4655a3e1: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_0c1bd0e976cf0a9a: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_1175a3329a06e52b: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_2d2227f4d5991e50: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_2f592bd1be3db6b3: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_4a1dd4244f80abc9: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_8b0da33fd11b2572: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_bae57fb9f24fde5c: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_be45aed56e4b9fee: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_cd567b7b35838e4c: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_d1c24b5a7a3ac31d: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_f92ae11c77d74198: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_layout_19e558a0fa724e95: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_eeef59714f5bf48b: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_mapped_at_creation_48de4735fab51e78: function(arg0, arg1) {
            arg0.mappedAtCreation = arg1 !== 0;
        },
        __wbg_set_min_binding_size_689661b9ed25e083: function(arg0, arg1) {
            arg0.minBindingSize = arg1;
        },
        __wbg_set_mip_level_246db61be15bdd69: function(arg0, arg1) {
            arg0.mipLevel = arg1 >>> 0;
        },
        __wbg_set_mip_level_count_72f8bc1f80f7539b: function(arg0, arg1) {
            arg0.mipLevelCount = arg1 >>> 0;
        },
        __wbg_set_mip_level_count_b19a0d9192e62d5d: function(arg0, arg1) {
            arg0.mipLevelCount = arg1 >>> 0;
        },
        __wbg_set_module_9b938909233aed50: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_multisampled_40505c1381e1c32c: function(arg0, arg1) {
            arg0.multisampled = arg1 !== 0;
        },
        __wbg_set_offset_2c374e604504e0b2: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_offset_73156b0e0b41d79a: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_offset_a097a8050a3a9a33: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_origin_9b3b0fbe0a5dc469: function(arg0, arg1) {
            arg0.origin = arg1;
        },
        __wbg_set_power_preference_c0d3fa7ce46b1a2e: function(arg0, arg1) {
            arg0.powerPreference = __wbindgen_enum_GpuPowerPreference[arg1];
        },
        __wbg_set_query_set_f1314b06c84c4b00: function(arg0, arg1) {
            arg0.querySet = arg1;
        },
        __wbg_set_required_features_54918de8185c5fab: function(arg0, arg1) {
            arg0.requiredFeatures = arg1;
        },
        __wbg_set_required_limits_3b031f66f838f4e3: function(arg0, arg1) {
            arg0.requiredLimits = arg1;
        },
        __wbg_set_resource_fe385d2e3dadaf63: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_rows_per_image_d198b7e73a38978b: function(arg0, arg1) {
            arg0.rowsPerImage = arg1 >>> 0;
        },
        __wbg_set_rows_per_image_f9878f4b10f4fd7f: function(arg0, arg1) {
            arg0.rowsPerImage = arg1 >>> 0;
        },
        __wbg_set_sample_count_865e1d19b84e27e6: function(arg0, arg1) {
            arg0.sampleCount = arg1 >>> 0;
        },
        __wbg_set_sample_type_7088b1efddce6a69: function(arg0, arg1) {
            arg0.sampleType = __wbindgen_enum_GpuTextureSampleType[arg1];
        },
        __wbg_set_sampler_8c5d7fb1b02058c6: function(arg0, arg1) {
            arg0.sampler = arg1;
        },
        __wbg_set_size_1e6281b07cd39177: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_41cd9255ca1e4242: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_a61ff22205255d61: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_storage_texture_ab9eed9786337ef0: function(arg0, arg1) {
            arg0.storageTexture = arg1;
        },
        __wbg_set_texture_16d2be474ce6ad0c: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_texture_e25a73da75cf5808: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_timestamp_writes_26336a2ad72cdcaf: function(arg0, arg1) {
            arg0.timestampWrites = arg1;
        },
        __wbg_set_type_38961e08504ca674: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuBufferBindingType[arg1];
        },
        __wbg_set_type_c1eebc19f8a6aeb9: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuSamplerBindingType[arg1];
        },
        __wbg_set_usage_7f0dda8309469b1c: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_usage_7fa9cd18d1104aca: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_usage_908213a4d4bb8bde: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_view_dimension_263387976511ebc9: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_dimension_3ed01b237e85826f: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_formats_bab284fc81b40e70: function(arg0, arg1) {
            arg0.viewFormats = arg1;
        },
        __wbg_set_visibility_1bca121a89accba5: function(arg0, arg1) {
            arg0.visibility = arg1 >>> 0;
        },
        __wbg_set_width_1a5e2e86fa5bdcd8: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_x_56f0c2c08a62725c: function(arg0, arg1) {
            arg0.x = arg1 >>> 0;
        },
        __wbg_set_y_04fb8ce84735b4e1: function(arg0, arg1) {
            arg0.y = arg1 >>> 0;
        },
        __wbg_set_z_a51316db27a4941e: function(arg0, arg1) {
            arg0.z = arg1 >>> 0;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_8eb4cd83130a11a0: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_1e7044f654e934db: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_d8b50611246a6d92: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_fd0bc376bf0f8b42: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_submit_1290d44bb76ecef4: function(arg0, arg1) {
            arg0.submit(arg1);
        },
        __wbg_then_114b14e3854c2390: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_7a850dae4493f353: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_b830475380919203: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_unmap_8f06698a75b8331a: function(arg0) {
            arg0.unmap();
        },
        __wbg_writeBuffer_b4bdd36178348ca5: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.writeBuffer(arg1, arg2, getArrayU8FromWasm0(arg3, arg4), arg5, arg6);
        }, arguments); },
        __wbg_writeTexture_b45b69132e46a227: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.writeTexture(arg1, getArrayU8FromWasm0(arg2, arg3), arg4, arg5);
        }, arguments); },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 794, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h29982c8643b1dde7);
            return ret;
        },
        __wbindgen_generic_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 810, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h04599f72514a41ad);
            return ret;
        },
        __wbindgen_generic_0000000000000003: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_0000000000000006: function(arg0, arg1) {
            var v0 = getArrayU32FromWasm0(arg0, arg1).slice();
            wasm.__wbindgen_free(arg0, arg1 * 4, 4);
            // Cast intrinsic for `Vector(U32) -> Externref`.
            const ret = v0;
            return ret;
        },
        __wbindgen_generic_0000000000000007: function(arg0, arg1) {
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

function wasm_bindgen__convert__closures_____invoke__h29982c8643b1dde7(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h29982c8643b1dde7(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h04599f72514a41ad(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen__convert__closures_____invoke__h04599f72514a41ad(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen__convert__closures_____invoke__h4104b19d0e1a1b9a(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen__convert__closures_____invoke__h4104b19d0e1a1b9a(arg0, arg1, arg2, arg3);
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
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

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
