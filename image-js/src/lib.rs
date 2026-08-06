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

/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This file is part of paged (https://paged.media) and is additionally
 * available under the Paged Media Enterprise License (PMEL). Full
 * copyright and license information is available in LICENSE.md which is
 * distributed with this source code.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    MPL-2.0 OR Paged Media Enterprise License (PMEL)
 */

//! The wasm-bindgen surface consumed by `glue/` — the bundle's compute
//! artifact (manifest `capabilities.wasm[0]`).
//!
//! ARCHITECTURE NOTE (BREAKAGE I-07): a module loaded via
//! `loadBundleWasm` has no ambient authority — no `navigator.gpu`. So
//! this crate is loaded through its wasm-bindgen JS glue in the bundle
//! realm (the `@paged-media/sdk` pattern), where WebGPU IS reachable;
//! the engines' GPU device lives behind this surface (`init_gpu`).
//!
//! M4 ingest slice: `decode_image` (PSD/PNG/JPEG → an engine-held RGBA8
//! handle) + `adjust_image` (Engine A adjustments via the ASYNC GPU
//! sink → RGBA8 bytes for the C-1 Stage-A image scene item). Pixels
//! held between calls stay engine-side behind handles (spec §2.1.3);
//! the one RGBA buffer `adjust_image` returns is the Stage-A render
//! payload destined for the HOST scene channel — the narrowed §2.1.3
//! contract the C-1 spike records.
//!
//! THE LAYER GRAPH (`layers_*`) is bound the same way the selection is:
//! one stack per realm, tied to an engine-held image handle, whose
//! COMPOSITE is written back into that same handle. So paint, fills and
//! bakes land in the ACTIVE LAYER (journaled — `layers_undo`), while
//! every other lane here keeps addressing one handle and never has to
//! learn what a layer is.
//!
//! The release-build guarantee proven by CI: NO reference code
//! (image-conformance / `image-kernels` feature `reference`) is
//! reachable from this crate (cargo-tree guard, spec §4 dep rule 2).

pub mod cmyk;
pub mod display;
pub mod fill;
pub mod ingest;
pub mod layers;
pub mod mip;
pub mod saveback;
pub mod selection;
pub mod stroke;

/// The frozen kernel ABI version this artifact was built against.
pub fn abi_version() -> u32 {
    image_kernels::ABI_VERSION
}

/// Registered kernel count (dispatch-table probe).
pub fn kernel_count() -> usize {
    image_kernels::registry().len()
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::rc::Rc;
    use std::sync::Arc;

    use image_gpu::{CombineMode, GpuContext, SelectionCoverage};
    use wasm_bindgen::prelude::*;

    use crate::fill::{fill_rgba8, FillSpec, GradientKind};
    use crate::ingest::{
        adjust_rgba8, crop_rgba8, decode_rgba8, straighten_crop_rgba8, AdjustParams, DecodedImage,
        IngestError, LevelsParams,
    };
    use crate::layers::LayerStack;
    use crate::mip::MipPyramid;
    use crate::saveback::{encode_rgba8, psd_write_adjusted, RasterFormat};
    use crate::selection::SessionSelection;
    use crate::stroke::{blend_kernel, blend_names, StrokeParams, StrokeSession, StrokeTool};
    use image_gpu::dab::{PressureTarget, StrokeSample};

    thread_local! {
        /// The bundle-realm GPU device (I-07: created HERE, where
        /// `navigator.gpu` is reachable; the wasm sandbox has none).
        static GPU: RefCell<Option<Rc<GpuContext>>> = const { RefCell::new(None) };
        /// Decoded images held engine-side behind handles (§2.1.3).
        static IMAGES: RefCell<HashMap<u32, DecodedImage>> =
            RefCell::new(HashMap::new());
        static NEXT_HANDLE: Cell<u32> = const { Cell::new(1) };
        /// Engine-B mip pyramids built lazily for the `level > 0` tile
        /// window path, cached per image handle (build once, sample many).
        /// Dropped when the image is freed.
        static PYRAMIDS: RefCell<HashMap<u32, Rc<MipPyramid>>> =
            RefCell::new(HashMap::new());
        /// The per-session SELECTION (one per wasm realm — the session's
        /// active source image binds it via `selection_bind`). The adjust
        /// chain reads it through `SessionSelection::mask_for`.
        static SELECTION: RefCell<SessionSelection> =
            RefCell::new(SessionSelection::new());
    }

    #[wasm_bindgen(start)]
    pub fn init() {
        console_error_panic_hook::set_once();
    }

    #[wasm_bindgen]
    pub fn abi_version() -> u32 {
        super::abi_version()
    }

    #[wasm_bindgen]
    pub fn kernel_count() -> usize {
        super::kernel_count()
    }

    /// Does the embedding realm expose WebGPU (`navigator.gpu`)? Probed
    /// BEFORE touching wgpu so a GPU-less realm (Node tests, an old
    /// browser) gets a clean rejection instead of a wasm panic that
    /// would poison the instance for the still-valid decode lanes.
    fn has_webgpu() -> bool {
        let global = js_sys::global();
        let Ok(navigator) = js_sys::Reflect::get(&global, &JsValue::from_str("navigator")) else {
            return false;
        };
        if navigator.is_undefined() || navigator.is_null() {
            return false;
        }
        js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"))
            .map(|gpu| !gpu.is_undefined() && !gpu.is_null())
            .unwrap_or(false)
    }

    /// Request the WebGPU adapter/device for kernel execution.
    /// Idempotent. Rejects when the environment has no WebGPU — the
    /// honest no-GPU state (no CPU kernel path ships, spec §6).
    #[wasm_bindgen]
    pub async fn init_gpu() -> Result<(), JsValue> {
        if GPU.with(|g| g.borrow().is_some()) {
            return Ok(());
        }
        if !has_webgpu() {
            return Err(JsValue::from_str(
                "WebGPU unavailable in this realm (no navigator.gpu)",
            ));
        }
        let ctx = GpuContext::new()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        GPU.with(|g| *g.borrow_mut() = Some(Rc::new(ctx)));
        Ok(())
    }

    /// Whether `init_gpu` succeeded (the glue probes this to gate the
    /// adjust controls honestly).
    #[wasm_bindgen]
    pub fn gpu_ready() -> bool {
        GPU.with(|g| g.borrow().is_some())
    }

    /// A decoded image's identity on the surface: the handle keys the
    /// engine-held pixels; width/height are the natural extent.
    #[wasm_bindgen]
    #[derive(Clone, Copy)]
    pub struct DecodedHandle {
        pub handle: u32,
        pub width: u32,
        pub height: u32,
        /// CMS rung 1 — what the RGB display transform did at decode, as a
        /// discriminant the bundle maps to a label: 0 = ICC managed,
        /// 1 = sRGB assumed (no embedded profile), 2 = sRGB assumed
        /// because an embedded profile was rejected. Surfaced so the panel
        /// can STATE the colour treatment instead of leaving the user to
        /// guess which numbers they are looking at.
        pub display: u8,
    }

    /// Map the Rust treatment onto the wire discriminant above.
    fn display_code(t: crate::display::DisplayTreatment) -> u8 {
        use crate::display::DisplayTreatment as D;
        match t {
            D::Managed => 0,
            D::AssumedSrgb => 1,
            D::ProfileRejected => 2,
        }
    }

    /// The stored treatment for an engine-held image, defaulting to
    /// "sRGB assumed" for an unknown handle — a missing image is already
    /// an error on every path that calls this, and guessing "managed"
    /// would be the one dishonest answer.
    fn read_display(handle: u32) -> u8 {
        IMAGES
            .with(|m| m.borrow().get(&handle).map(|i| i.display))
            .map(display_code)
            .unwrap_or(1)
    }

    /// Decode PSD/PNG/JPEG bytes (sniffed by magic) into an engine-held
    /// RGBA8 image. Free with `free_image`.
    #[wasm_bindgen]
    pub fn decode_image(bytes: &[u8]) -> Result<DecodedHandle, JsValue> {
        let img = decode_rgba8(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let handle = NEXT_HANDLE.with(|n| {
            let h = n.get();
            n.set(h + 1);
            h
        });
        let (width, height) = (img.width, img.height);
        let display = display_code(img.display);
        IMAGES.with(|m| m.borrow_mut().insert(handle, img));
        Ok(DecodedHandle {
            handle,
            width,
            height,
            display,
        })
    }

    /// K-3 (S-07 / I-02) — register a PRE-DECODED straight-RGBA8 buffer
    /// (from the decode worker pool, which ran the codec/PSD CPU lanes
    /// off-thread) as an engine-held image, returning a handle for the GPU
    /// adjust + tile paths. `bytes` must be exactly `width*height*4` RGBA8;
    /// a length mismatch is a clean error. Free with `free_image`.
    #[wasm_bindgen]
    pub fn ingest_rgba8(width: u32, height: u32, bytes: Vec<u8>) -> Result<DecodedHandle, JsValue> {
        let img = DecodedImage::from_rgba8(width, height, bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let handle = NEXT_HANDLE.with(|n| {
            let h = n.get();
            n.set(h + 1);
            h
        });
        let display = display_code(img.display);
        IMAGES.with(|m| m.borrow_mut().insert(handle, img));
        Ok(DecodedHandle {
            handle,
            width,
            height,
            display,
        })
    }

    /// Run the M4 adjustments chain on a decoded image and return the
    /// straight-RGBA8 result — the C-1 Stage-A scene-item payload.
    /// Identity params return the decode verbatim (no dispatch to run);
    /// anything else requires `init_gpu` to have succeeded.
    #[wasm_bindgen]
    pub async fn adjust_image(
        handle: u32,
        exposure_ev: f32,
        brightness: f32,
        contrast: f32,
        saturation: f32,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let params = AdjustParams {
            exposure_ev,
            brightness,
            contrast,
            saturation,
            ..Default::default()
        };
        run_adjust(handle, &img, params).await
    }

    /// The FULL adjustments pass — the levels/curves/white-balance panel's
    /// committed values. The 9 scalars are exposure/brightness/contrast/
    /// saturation (as `adjust_image`), white balance (temp/tint), and the
    /// composite levels in/gamma/out window; `curve_lut` is an OPTIONAL
    /// 256-byte tone LUT (the panel builds it from its curve control points
    /// via `image_core::curve_lut`; pass an empty array for no curve). The
    /// curves stage is a CPU LUT pass (no GPU LUT kernel yet — the honest
    /// deferral); everything else is the GPU adjust chain. Returns straight
    /// RGBA8 (the C-1 Stage-A scene payload).
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub async fn adjust_image_full(
        handle: u32,
        exposure_ev: f32,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        temp: f32,
        tint: f32,
        in_black: f32,
        in_white: f32,
        gamma: f32,
        out_black: f32,
        out_white: f32,
        curve_lut: &[u8],
        blur_sigma: f32,
        sharpen_amount: f32,
        hue_degrees: f32,
        invert: bool,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        adjust_image_ext(
            handle,
            exposure_ev,
            brightness,
            contrast,
            saturation,
            temp,
            tint,
            in_black,
            in_white,
            gamma,
            out_black,
            out_white,
            curve_lut,
            blur_sigma,
            sharpen_amount,
            hue_degrees,
            invert,
            &[],
        )
        .await
    }

    /// [`adjust_image_full`] PLUS the EXTENDED (kernel-breadth) stages —
    /// vibrance, color balance, black & white, posterize, threshold,
    /// photo filter, channel mixer and per-channel levels — carried in
    /// ONE flat `f32` block so the boundary does not grow an argument
    /// per stage. `ext` is either EMPTY (every extended stage at
    /// identity — what `adjust_image_full` passes) or exactly
    /// `ingest::ADJUST_EXT_LEN` floats in the layout documented on that
    /// constant. The chain order is documented on `ingest::adjust_rgba8`;
    /// every stage is mask-aware (the bound selection rides `@group(2)`).
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub async fn adjust_image_ext(
        handle: u32,
        exposure_ev: f32,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        temp: f32,
        tint: f32,
        in_black: f32,
        in_white: f32,
        gamma: f32,
        out_black: f32,
        out_white: f32,
        curve_lut: &[u8],
        blur_sigma: f32,
        sharpen_amount: f32,
        hue_degrees: f32,
        invert: bool,
        ext: &[f32],
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let params = build_adjust_params(
            exposure_ev,
            brightness,
            contrast,
            saturation,
            temp,
            tint,
            in_black,
            in_white,
            gamma,
            out_black,
            out_white,
            curve_lut,
            blur_sigma,
            sharpen_amount,
            hue_degrees,
            invert,
            ext,
        )?;
        run_adjust(handle, &img, params).await
    }

    /// Decode the flat adjust wire block into [`AdjustParams`]. Shared by
    /// `adjust_image_ext` (the re-runnable PREVIEW chain) and
    /// `layers_bake_adjust` (the DESTRUCTIVE per-layer bake) so the two
    /// can never interpret the same wire differently.
    #[allow(clippy::too_many_arguments)]
    fn build_adjust_params(
        exposure_ev: f32,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        temp: f32,
        tint: f32,
        in_black: f32,
        in_white: f32,
        gamma: f32,
        out_black: f32,
        out_white: f32,
        curve_lut: &[u8],
        blur_sigma: f32,
        sharpen_amount: f32,
        hue_degrees: f32,
        invert: bool,
        ext: &[f32],
    ) -> Result<AdjustParams, JsValue> {
        let lut = if curve_lut.len() == 256 {
            let mut a = [0u8; 256];
            a.copy_from_slice(curve_lut);
            Some(a)
        } else if curve_lut.is_empty() {
            None
        } else {
            return Err(JsValue::from_str(&format!(
                "curve_lut must be 256 bytes or empty (got {})",
                curve_lut.len()
            )));
        };
        let mut params = AdjustParams {
            exposure_ev,
            brightness,
            contrast,
            saturation,
            temp,
            tint,
            levels: LevelsParams {
                in_black,
                in_white,
                gamma,
                out_black,
                out_white,
            },
            curve_lut: lut,
            blur_sigma,
            sharpen_amount,
            hue_degrees,
            invert,
            ..Default::default()
        };
        params
            .apply_extended(ext)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(params)
    }

    /// Shared adjust runner: identity → the decode verbatim; otherwise the
    /// GPU chain (requires `init_gpu`) plus any CPU curve LUT. When the
    /// session SELECTION is bound to `handle` and non-trivial, its
    /// coverage masks EVERY dispatch (the §6.1 mask ABI) and the curve
    /// LUT pass — the adjustment lands only inside the selection.
    async fn run_adjust(
        handle: u32,
        img: &DecodedImage,
        params: AdjustParams,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        if params.is_identity() {
            return Ok(js_sys::Uint8Array::from(&img.rgba[..]));
        }
        let ctx = GPU.with(|g| g.borrow().clone()).ok_or_else(|| {
            JsValue::from_str(
                "GPU not initialized — await init_gpu() first (kernels are \
                 GPU-only; no CPU fallback ships)",
            )
        })?;
        let selection = SELECTION.with(|s| s.borrow().mask_for(handle));
        let out = adjust_rgba8(&ctx, img, &params, selection)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(js_sys::Uint8Array::from(&out[..]))
    }

    /// Compute the RGB + luma 256-bin histogram of an engine-held image as
    /// a flat `[r…, g…, b…, luma…]` 1024-`u32` array (the LEVELS / CURVES
    /// panel slices it into four channels). Pure CPU reduction over the
    /// straight-RGBA8 buffer (no GPU); deterministic.
    #[wasm_bindgen]
    pub fn image_histogram(handle: u32) -> Result<js_sys::Uint32Array, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let hist = image_gpu::histogram_rgba8(&img.rgba);
        Ok(js_sys::Uint32Array::from(&hist.to_flat()[..]))
    }

    /// Compute the AUTO-ENHANCE adjustment parameters for an engine-held
    /// image and return them as `[in_black, in_white, temp, tint]` (4
    /// `f32`). A single "auto" estimate composing the EXISTING levels +
    /// white-balance kernels: it builds the RGB+luma histogram (the same
    /// `histogram_rgba8` reduction the panel reads), derives a percentile-
    /// clipped auto-levels black/white range (0.5%/99.5% of luma) and a
    /// gray-world white-balance `temp`/`tint`, and emits the params the
    /// LEVELS/WB panel commits through `adjust_image_full` (levels
    /// `in_black`/`in_white`, white-balance `temp`/`tint`; gamma/output
    /// range stay identity). Pure CPU readout/orchestration (spec §6) —
    /// deterministic, no GPU, no kernel dispatch row. A flat or already-
    /// neutral image yields the identity `[0, 1, 0, 0]` (a guaranteed
    /// no-op), never a wrong-looking auto-correction.
    #[wasm_bindgen]
    pub fn image_auto_enhance_params(handle: u32) -> Result<js_sys::Float32Array, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let hist = image_gpu::histogram_rgba8(&img.rgba);
        let auto = image_gpu::auto_enhance(&hist);
        let out = [auto.in_black, auto.in_white, auto.temp, auto.tint];
        Ok(js_sys::Float32Array::from(&out[..]))
    }

    /// Commit a CROP: cut the integer pixel rectangle `(x, y, w, h)`
    /// (clamped to the image extent) out of an engine-held image and
    /// register the result as a NEW engine-held image, returning its
    /// handle. The source handle is left intact (the caller frees it). An
    /// out-of-bounds / empty rectangle is a clean error (never a torn
    /// image). This door is the AXIS-ALIGNED cut only — the straighten
    /// angle rides `straighten_crop_image`, which rotates first.
    #[wasm_bindgen]
    pub fn crop_image(
        handle: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<DecodedHandle, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let cropped =
            crop_rgba8(&img, x, y, w, h).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let new_handle = NEXT_HANDLE.with(|n| {
            let h = n.get();
            n.set(h + 1);
            h
        });
        let (width, height) = (cropped.width, cropped.height);
        let cropped_display = display_code(cropped.display);
        IMAGES.with(|m| m.borrow_mut().insert(new_handle, cropped));
        Ok(DecodedHandle {
            handle: new_handle,
            width,
            height,
            // A crop inherits the source image's treatment.
            display: cropped_display,
        })
    }

    /// STRAIGHTEN + CROP commit: rotate the image by `−degrees` about
    /// the crop rectangle's centre (`geom.rotate_bilinear`, backward
    /// mapped, bilinear, clamp-to-edge) so the rotated FRAME the overlay
    /// previewed lands upright, then cut `(x, y, w, h)` out of the
    /// result and register it as a NEW engine-held image. The source
    /// handle is left intact.
    ///
    /// `degrees == 0` takes the pure-windowing [`crop_image`] path — no
    /// GPU, no resample, no interpolation blur for an axis-aligned crop.
    /// A non-zero angle IS a resample and so is GPU-only (`init_gpu`
    /// first); it rejects honestly without a device.
    #[wasm_bindgen]
    pub async fn straighten_crop_image(
        handle: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        degrees: f32,
    ) -> Result<DecodedHandle, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let out = if degrees == 0.0 {
            crop_rgba8(&img, x, y, w, h).map_err(|e| JsValue::from_str(&e.to_string()))?
        } else {
            let ctx = GPU.with(|g| g.borrow().clone()).ok_or_else(|| {
                JsValue::from_str(
                    "the straighten resample is GPU-only — call init_gpu first \
                     (an axis-aligned crop at 0° needs no device)",
                )
            })?;
            straighten_crop_rgba8(&ctx, &img, x, y, w, h, degrees)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?
        };
        let (width, height) = (out.width, out.height);
        register(width, height, out.rgba.to_vec())
    }

    // ── crop interaction GEOMETRY (pure view math; the TS crop machine
    // forwards pointer points + renders the frame the overlay draws) ──
    //
    // These wrap `image_core::crop` so the deterministic, Rust-tested
    // geometry is the ONE source of truth (the TS stays thin). A crop rect
    // crosses the boundary as `[x, y, w, h]`; the aspect lock is encoded as
    // `aspect_w`/`aspect_h` (0/0 = free, equal = square, else the ratio).

    /// Decode an aspect lock from the `(aspect_w, aspect_h)` wire pair:
    /// `(0, _)`/`(_, 0)` → free; otherwise the `w:h` ratio.
    fn decode_aspect(aspect_w: f32, aspect_h: f32) -> image_core::AspectLock {
        if aspect_w <= 0.0 || aspect_h <= 0.0 {
            image_core::AspectLock::Free
        } else {
            image_core::AspectLock::Ratio(aspect_w, aspect_h)
        }
    }

    /// Hit-test the crop chrome at `(px, py)` (image-px) against the rect
    /// `[x, y, w, h]` with grab radius `tol`. Returns the [`image_core::
    /// Handle`] discriminant (0..=7 grips, 8 = body Move) or `-1` for a
    /// miss — the TS machine maps it to a cursor + the active grip.
    #[wasm_bindgen]
    pub fn crop_hit_handle(x: f32, y: f32, w: f32, h: f32, px: f32, py: f32, tol: f32) -> i32 {
        let rect = image_core::CropRect { x, y, w, h };
        match image_core::hit_handle(&rect, (px, py), tol) {
            Some(handle) => handle as i32,
            None => -1,
        }
    }

    /// Apply a pointer drag from `(sx, sy)` to `(px, py)` (image-px) to the
    /// rect `[x, y, w, h]` at `handle` (the [`crop_hit_handle`]
    /// discriminant), with the aspect lock + image-extent clamp. Returns
    /// the new rect as `[x, y, w, h]`. An unknown handle returns the rect
    /// unchanged (defensive).
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn crop_apply_drag(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        handle: i32,
        sx: f32,
        sy: f32,
        px: f32,
        py: f32,
        aspect_w: f32,
        aspect_h: f32,
        image_w: u32,
        image_h: u32,
    ) -> Vec<f32> {
        let rect = image_core::CropRect { x, y, w, h };
        let handle = match handle {
            0 => image_core::Handle::TopLeft,
            1 => image_core::Handle::Top,
            2 => image_core::Handle::TopRight,
            3 => image_core::Handle::Right,
            4 => image_core::Handle::BottomRight,
            5 => image_core::Handle::Bottom,
            6 => image_core::Handle::BottomLeft,
            7 => image_core::Handle::Left,
            8 => image_core::Handle::Move,
            _ => return vec![rect.x, rect.y, rect.w, rect.h],
        };
        let out = image_core::apply_drag(
            &rect,
            handle,
            (sx, sy),
            (px, py),
            decode_aspect(aspect_w, aspect_h),
            image_w,
            image_h,
        );
        vec![out.x, out.y, out.w, out.h]
    }

    /// The four corners of the crop FRAME rotated by the straighten
    /// `degrees`, as a flat `[x0,y0, x1,y1, x2,y2, x3,y3]` (TL, TR, BR, BL)
    /// the overlay draws as a closed polyline.
    #[wasm_bindgen]
    pub fn crop_frame_corners(x: f32, y: f32, w: f32, h: f32, degrees: f32) -> Vec<f32> {
        let rect = image_core::CropRect { x, y, w, h };
        let c = image_core::frame_corners(&rect, degrees);
        vec![
            c[0].0, c[0].1, c[1].0, c[1].1, c[2].0, c[2].1, c[3].0, c[3].1,
        ]
    }

    /// Build a 256-byte tone LUT from flat `[i0,o0, i1,o1, …]` curve
    /// control points in `[0,1]` (the CURVES editor's points) — the LUT
    /// `adjust_image_full` consumes. Wraps `image_core::curve_lut`.
    #[wasm_bindgen]
    pub fn curve_lut(points: &[f32]) -> Vec<u8> {
        let pts: Vec<(f32, f32)> = points.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        image_core::curve_lut(&pts).to_vec()
    }

    /// C-6 (I-06) — copy a LEVEL-0 tile window `(x, y, w, h)` out of a
    /// decoded image as tightly packed RGBA8 (`w*h*4` bytes, row-major).
    /// Edge tiles are clamped to the image extent (the caller passes the
    /// requested grid origin + size; the returned buffer is the clipped
    /// intersection). This is the HONEST SUBSET of the resource provider:
    /// pure windowing of the already-decoded buffer (no resampling kernel,
    /// no GPU dispatch — orchestration, spec §6). The mip pyramid + the
    /// Engine B `(node, region, level)` window evaluation
    /// (`image_graph::BufferGraph::request`, rgba16float) are NOT yet
    /// wired across this wasm boundary — see the gap note in
    /// glue/src/tile-provider.ts. Returns an empty buffer when the window
    /// lies fully outside the image (a transparent miss the provider skips).
    #[wasm_bindgen]
    pub fn image_tile_rgba8(
        handle: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let (bytes, _tw, _th) = img.tile_window_rgba8(x, y, w, h);
        Ok(js_sys::Uint8Array::from(&bytes[..]))
    }

    /// C-6 (I-06) — copy a tile window `(x, y, w, h)` out of a decoded
    /// image at mip `level` as tightly packed RGBA8 (`w'*h'*4` bytes,
    /// clipped to the level extent, row-major). `level == 0` is the fast
    /// level-0 path ([`image_tile_rgba8`]'s pure windowing); `level > 0`
    /// routes through Engine B's tiled buffer graph
    /// (`image_graph::BufferGraph`): a 2×-box mip pyramid of rgba16float
    /// source tiles is built once per handle (cached) and the requested
    /// window is gathered from `(level, coord)` source reads and
    /// downconverted back to RGBA8. The coordinates are in the LEVEL's
    /// pixel space (already halved per level — the caller scales). No GPU
    /// is required (a source read carries no kernel dispatch). Returns an
    /// empty buffer when the window lies fully outside the level, or when
    /// `level` exceeds the pyramid top (a transparent miss the provider
    /// skips). `max_level` bounds the pyramid height built on first touch.
    #[wasm_bindgen]
    pub fn image_tile_rgba8_level(
        handle: u32,
        level: u32,
        x: u32,
        y: u32,
        size: u32,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        // Fast path: level 0 needs no pyramid (pure windowing of the
        // already-decoded buffer).
        if level == 0 {
            let img = IMAGES
                .with(|m| m.borrow().get(&handle).cloned())
                .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
            let (bytes, _tw, _th) = img.tile_window_rgba8(x, y, size, size);
            return Ok(js_sys::Uint8Array::from(&bytes[..]));
        }

        let level_u8: u8 = level
            .try_into()
            .map_err(|_| JsValue::from_str(&format!("mip level {level} out of range (max 255)")))?;

        // Build (or reuse the cached) pyramid for this handle. The pyramid
        // height is bounded by the requested level (built up to it).
        let pyramid = PYRAMIDS.with(|p| p.borrow().get(&handle).cloned());
        let pyramid = match pyramid {
            Some(p) if p.max_level() >= level_u8 => p,
            _ => {
                let img = IMAGES
                    .with(|m| m.borrow().get(&handle).cloned())
                    .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
                let built = Rc::new(MipPyramid::build(
                    img.width, img.height, &img.rgba, level_u8,
                ));
                PYRAMIDS.with(|p| p.borrow_mut().insert(handle, Rc::clone(&built)));
                built
            }
        };

        let (bytes, _tw, _th) = pyramid.window_rgba8(level_u8, x, y, size, size);
        Ok(js_sys::Uint8Array::from(&bytes[..]))
    }

    /// Release an engine-held decoded image (its mip pyramid cache, and
    /// the layer stack bound to it — a stack whose composite target is
    /// gone has nowhere to land).
    #[wasm_bindgen]
    pub fn free_image(handle: u32) {
        IMAGES.with(|m| {
            m.borrow_mut().remove(&handle);
        });
        PYRAMIDS.with(|p| {
            p.borrow_mut().remove(&handle);
        });
        LAYERS.with(|l| {
            let drop_it = l.borrow().as_ref().map(|d| d.handle) == Some(handle);
            if drop_it {
                *l.borrow_mut() = None;
            }
        });
    }

    // ─────────────────────── SELECTION doors (§6.1) ───────────────────────
    //
    // The per-session selection: a u8 coverage field at the bound image's
    // resolution (image_gpu::SelectionCoverage — all mask PREP, inherently
    // CPU; consumption is GPU-only via the adjust chain's @group(2) r16float
    // mask, `out = mix(a, result, mask)` per dispatch). `mode` on the shape
    // doors is 0 = replace, 1 = add, 2 = subtract, 3 = intersect.
    // Semantics (selection.rs): no selection = everything (adjust runs
    // unmasked); subtract/intersect against no selection start from FULL;
    // `selection_clear` returns to "no selection"; re-binding to a
    // different handle/resolution drops the coverage.

    fn decode_mode(mode: u32) -> Result<CombineMode, JsValue> {
        CombineMode::from_u32(mode).ok_or_else(|| {
            JsValue::from_str(&format!(
                "unknown selection mode {mode} (0 replace | 1 add | 2 subtract | 3 intersect)"
            ))
        })
    }

    fn with_selection<T>(
        f: impl FnOnce(&mut SessionSelection) -> Result<T, String>,
    ) -> Result<T, JsValue> {
        SELECTION
            .with(|s| f(&mut s.borrow_mut()))
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Bind the session selection to an engine-held image (the selection
    /// field takes ITS resolution; the magic wand floods ITS pixels; the
    /// adjust doors mask only when adjusting THIS handle). Re-binding to
    /// the same handle keeps the selection; a different handle (a crop /
    /// resize swap) or resolution drops it.
    #[wasm_bindgen]
    pub fn selection_bind(handle: u32) -> Result<(), JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        SELECTION.with(|s| s.borrow_mut().bind(handle, img.width, img.height));
        Ok(())
    }

    /// Re-point the selection at a NEW image handle that holds the SAME
    /// extent, KEEPING the coverage — the door a destructive in-place
    /// edit (the generator FILL) uses so the selection survives its own
    /// result. Answers `true` when the coverage carried over, `false`
    /// when the extent changed (then it behaves exactly like
    /// `selection_bind`: the selection drops).
    #[wasm_bindgen]
    pub fn selection_transfer(handle: u32) -> Result<bool, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        Ok(SELECTION.with(|s| s.borrow_mut().transfer(handle, img.width, img.height)))
    }

    /// Marquee RECT: fold the anti-aliased rectangle `[x, x+w) × [y, y+h)`
    /// (image px; fractional coords carry the AA edge) into the selection
    /// under `mode`.
    #[wasm_bindgen]
    pub fn selection_set_rect(x: f32, y: f32, w: f32, h: f32, mode: u32) -> Result<(), JsValue> {
        let mode = decode_mode(mode)?;
        with_selection(|s| {
            let b = s.bound().ok_or("no image bound (selection_bind first)")?;
            let shape = SelectionCoverage::rasterize_rect(b.width, b.height, x, y, w, h);
            s.apply_shape(shape, mode)
        })
    }

    /// Marquee ELLIPSE: center `(cx, cy)`, radii `(rx, ry)` (image px),
    /// anti-aliased (4×4 supersampled edge), folded under `mode`.
    #[wasm_bindgen]
    pub fn selection_set_ellipse(
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        mode: u32,
    ) -> Result<(), JsValue> {
        let mode = decode_mode(mode)?;
        with_selection(|s| {
            let b = s.bound().ok_or("no image bound (selection_bind first)")?;
            let shape = SelectionCoverage::rasterize_ellipse(b.width, b.height, cx, cy, rx, ry);
            s.apply_shape(shape, mode)
        })
    }

    /// LASSO polygon: `points_flat` is `[x0, y0, x1, y1, …]` image-px
    /// vertices of a closed polygon (the closing edge is implicit),
    /// scanline-rasterized with AA coverage, folded under `mode`. Fewer
    /// than 3 vertices is a clean error (nothing to select).
    #[wasm_bindgen]
    pub fn selection_set_polygon(points_flat: &[f32], mode: u32) -> Result<(), JsValue> {
        let mode = decode_mode(mode)?;
        if points_flat.len() < 6 || !points_flat.len().is_multiple_of(2) {
            return Err(JsValue::from_str(
                "polygon needs ≥ 3 (x, y) vertex pairs (flat, even length)",
            ));
        }
        let pts: Vec<(f32, f32)> = points_flat.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        with_selection(|s| {
            let b = s.bound().ok_or("no image bound (selection_bind first)")?;
            let shape = SelectionCoverage::rasterize_polygon(b.width, b.height, &pts);
            s.apply_shape(shape, mode)
        })
    }

    /// MAGIC WAND at `(x, y)`: color-distance flood over the BOUND
    /// image's straight-RGBA8 pixels — `contiguous` = 4-connected BFS
    /// from the seed; otherwise a global threshold. `tolerance` is the
    /// per-channel (Chebyshev) distance 0–255. Binary coverage (hard
    /// edges; `selection_feather` softens), folded under `mode`.
    #[wasm_bindgen]
    pub fn selection_magic_wand(
        x: u32,
        y: u32,
        tolerance: u8,
        contiguous: bool,
        mode: u32,
    ) -> Result<(), JsValue> {
        let mode = decode_mode(mode)?;
        let bound = SELECTION
            .with(|s| s.borrow().bound())
            .ok_or_else(|| JsValue::from_str("no image bound (selection_bind first)"))?;
        let img = IMAGES
            .with(|m| m.borrow().get(&bound.handle).cloned())
            .ok_or_else(|| {
                JsValue::from_str(&format!("bound image handle {} is gone", bound.handle))
            })?;
        if x >= img.width || y >= img.height {
            return Err(JsValue::from_str(&format!(
                "wand seed ({x},{y}) outside {}x{}",
                img.width, img.height
            )));
        }
        let shape = SelectionCoverage::magic_wand(
            img.width, img.height, &img.rgba, x, y, tolerance, contiguous,
        );
        with_selection(|s| s.apply_shape(shape, mode))
    }

    /// Feather the selection: a Gaussian of `sigma` px on the COVERAGE
    /// (mask prep — CPU on the u8 mask by design, not image processing;
    /// the softened mask is still consumed GPU-side). Errors when no
    /// explicit selection exists.
    #[wasm_bindgen]
    pub fn selection_feather(sigma: f32) -> Result<(), JsValue> {
        with_selection(|s| s.feather(sigma))
    }

    /// Select ALL explicitly (a full-extent selection in the readouts;
    /// the adjust chain still takes the trivial-mask fast path).
    #[wasm_bindgen]
    pub fn selection_select_all() -> Result<(), JsValue> {
        with_selection(|s| s.select_all())
    }

    /// Deselect: back to "no selection" (adjustments run unmasked).
    #[wasm_bindgen]
    pub fn selection_clear() {
        SELECTION.with(|s| s.borrow_mut().clear());
    }

    /// Invert the selection ("everything" inverts to the explicit EMPTY
    /// selection — adjust applies nowhere until reselected).
    #[wasm_bindgen]
    pub fn selection_invert() -> Result<(), JsValue> {
        with_selection(|s| s.invert())
    }

    /// The bounding box of the selection's non-zero coverage as
    /// `[x, y, w, h]`; an EMPTY array when there is no explicit selection
    /// OR the selection is empty (distinguish via `selection_stats`).
    #[wasm_bindgen]
    pub fn selection_bounds() -> Vec<u32> {
        SELECTION.with(|s| match s.borrow().coverage().and_then(|c| c.bounds()) {
            Some(r) => vec![r.x as u32, r.y as u32, r.w, r.h],
            None => Vec::new(),
        })
    }

    /// Selection readout for the panel/tools, as 7 `f32`s:
    /// `[has_selection (0|1), x, y, w, h, coverage_fraction, revision]`.
    /// `has_selection == 0` ⇒ no explicit selection (everything, the
    /// unmasked default) and the box/fraction are 0. An explicit-but-
    /// empty selection reads `has == 1, w == h == 0, fraction == 0`.
    #[wasm_bindgen]
    pub fn selection_stats() -> Vec<f32> {
        SELECTION.with(|s| {
            let sel = s.borrow();
            let rev = sel.revision() as f32;
            match sel.coverage() {
                None => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rev],
                Some(c) => {
                    let (x, y, w, h) = match c.bounds() {
                        Some(r) => (r.x as f32, r.y as f32, r.w as f32, r.h as f32),
                        None => (0.0, 0.0, 0.0, 0.0),
                    };
                    vec![1.0, x, y, w, h, c.selected_fraction() as f32, rev]
                }
            }
        })
    }

    /// The raw u8 coverage bytes (`width·height`, row-major) — the
    /// overlay/debug readout. Empty when no explicit selection exists.
    #[wasm_bindgen]
    pub fn selection_coverage_bytes() -> js_sys::Uint8Array {
        SELECTION.with(|s| match s.borrow().coverage() {
            Some(c) => js_sys::Uint8Array::from(c.data()),
            None => js_sys::Uint8Array::new_with_length(0),
        })
    }

    /// RESAMPLE an engine-held image to `out_w`×`out_h` and register the
    /// result as a NEW engine-held image (the source stays intact — the
    /// crop precedent). `filter` ∈ nearest | mitchell | lanczos3 (the T1
    /// resample kernels, GPU-only per spec §6 — requires `init_gpu`;
    /// there is no CPU fallback and this rejects honestly without one).
    /// Rides the async windowed dispatch (a blocking readback cannot
    /// pump the map callback on wasm).
    #[wasm_bindgen]
    pub async fn resize_image(
        handle: u32,
        out_w: u32,
        out_h: u32,
        filter: &str,
    ) -> Result<DecodedHandle, JsValue> {
        use half::f16;
        use image_kernels::families::resample::{
            ResampleParams, RESAMPLE_LANCZOS3, RESAMPLE_MITCHELL, RESAMPLE_NEAREST,
        };

        if out_w == 0 || out_h == 0 {
            return Err(JsValue::from_str("resize: target size must be non-zero"));
        }
        let ctx = GPU
            .with(|g| g.borrow().clone())
            .ok_or_else(|| JsValue::from_str("resize is GPU-only — call init_gpu first"))?;
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let def = match filter {
            "nearest" => &RESAMPLE_NEAREST,
            "mitchell" => &RESAMPLE_MITCHELL,
            "lanczos3" => &RESAMPLE_LANCZOS3,
            other => {
                return Err(JsValue::from_str(&format!(
                    "unknown resample filter \"{other}\" (nearest | mitchell | lanczos3)"
                )))
            }
        };
        let params = ResampleParams::new(
            img.width as f32 / out_w as f32,
            img.height as f32 / out_h as f32,
            0.0,
            0.0,
        );
        // RGBA8 → rgba16float window (straight /255 — the I-02 working
        // values, same as the decode bridge).
        let mut win = Vec::with_capacity(img.rgba.len() * 2);
        for &b in img.rgba.iter() {
            win.extend_from_slice(&f16::from_f32(b as f32 / 255.0).to_le_bytes());
        }
        let out = image_gpu::execute_windowed_once_async(
            &ctx,
            def,
            &win,
            img.width,
            img.height,
            params.as_bytes(),
            None,
            out_w,
            out_h,
        )
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut rgba = Vec::with_capacity((out_w as usize) * (out_h as usize) * 4);
        for pair in out.chunks_exact(2) {
            let v = f16::from_le_bytes([pair[0], pair[1]]).to_f32();
            rgba.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        let resized = DecodedImage::from_rgba8(out_w, out_h, rgba)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let new_handle = NEXT_HANDLE.with(|n| {
            let h = n.get();
            n.set(h + 1);
            h
        });
        let resized_display = display_code(resized.display);
        IMAGES.with(|m| m.borrow_mut().insert(new_handle, resized));
        Ok(DecodedHandle {
            handle: new_handle,
            width: out_w,
            height: out_h,
            // A resize inherits the source image's treatment.
            display: resized_display,
        })
    }

    // ─────────────────────── GENERATE / FILL doors ───────────────────
    //
    // The generator family's editor reach (`crate::fill`). Both doors
    // are DESTRUCTIVE by design: they paint into the working image and
    // register the result as a NEW engine-held image (the crop/resize
    // commit pattern — the source handle is left for the caller to
    // free). The paint is composited through the bound SELECTION
    // (`mix(original, generated, coverage)` on the GPU); with no
    // selection the whole image is filled. GPU-only — both passes are
    // registered WGSL kernels, so this rejects honestly without
    // `init_gpu`.

    /// Register `pixels` as a new engine-held image, returning its handle.
    fn register(width: u32, height: u32, pixels: Vec<u8>) -> Result<DecodedHandle, JsValue> {
        let img = DecodedImage::from_rgba8(width, height, pixels)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let handle = NEXT_HANDLE.with(|n| {
            let h = n.get();
            n.set(h + 1);
            h
        });
        let img_display = display_code(img.display);
        IMAGES.with(|m| m.borrow_mut().insert(handle, img));
        Ok(DecodedHandle {
            handle,
            width,
            height,
            // Raw registered pixels carry no container, so no profile.
            display: img_display,
        })
    }

    /// Shared head of the fill doors: the BACKDROP the generator paints
    /// over, the GPU context, and the mask bound to THIS handle.
    ///
    /// With a layer stack bound, the backdrop is the ACTIVE LAYER — a
    /// gradient laid on an empty layer above the photo covers nothing
    /// below it, which is the whole point of the stack. Without one it is
    /// the engine-held image (the pre-layer behaviour).
    #[allow(clippy::type_complexity)]
    fn fill_prelude(
        handle: u32,
    ) -> Result<
        (
            DecodedImage,
            Rc<GpuContext>,
            Option<Arc<SelectionCoverage>>,
            bool,
        ),
        JsValue,
    > {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let ctx = GPU.with(|g| g.borrow().clone()).ok_or_else(|| {
            JsValue::from_str(
                "fill is GPU-only — call init_gpu first (the generator and the \
                 composite are both WGSL kernels; no CPU fallback ships)",
            )
        })?;
        let sel = SELECTION.with(|s| s.borrow().mask_for(handle));
        let layered = LAYERS.with(|l| {
            let b = l.borrow();
            match b.as_ref() {
                Some(d) if d.handle == handle => {
                    d.stack.active_is_editable()?;
                    Ok(Some(DecodedImage {
                        width: d.stack.width(),
                        height: d.stack.height(),
                        rgba: Arc::clone(&d.stack.active().rgba),
                        // A layer view of an already-ingested image; the
                        // transform ran at decode, not per layer.
                        display: crate::display::DisplayTreatment::AssumedSrgb,
                    }))
                }
                _ => Ok(None),
            }
        });
        match layered.map_err(ingest_err)? {
            Some(layer) => Ok((layer, ctx, sel, true)),
            None => Ok((img, ctx, sel, false)),
        }
    }

    /// Land a fill result: into the ACTIVE LAYER (journaled, then
    /// re-composited into the same handle) when a stack is bound, else
    /// as a NEW engine-held image (the pre-layer destructive commit).
    async fn land_fill(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        layered: bool,
    ) -> Result<DecodedHandle, JsValue> {
        if !layered {
            return register(width, height, pixels);
        }
        // The damage is the SELECTION's box when there is one (the fill
        // cannot land outside it), else the whole canvas.
        let damage = SELECTION
            .with(|s| s.borrow().coverage().and_then(|c| c.bounds()))
            .unwrap_or(image_core::Region::new(0, 0, width, height));
        let ctx = GPU.with(|g| g.borrow().clone());
        let pixels: Arc<[u8]> = Arc::from(pixels.into_boxed_slice());
        with_stack_async(|mut doc| async move {
            let result = async {
                doc.stack
                    .edit_active("Fill", damage, pixels)
                    .map_err(ingest_err)?;
                let rgba = doc
                    .stack
                    .composite(ctx.as_deref(), None)
                    .await
                    .map_err(ingest_err)?;
                set_image_pixels(doc.handle, rgba)?;
                Ok(DecodedHandle {
                    handle: doc.handle,
                    width,
                    height,
                    // The layer composite inherits the document image's treatment.
                    display: read_display(doc.handle),
                })
            }
            .await;
            (doc, result)
        })
        .await
    }

    /// FILL the current selection (the whole image when none) with a
    /// fixed TWO-STOP gradient. `kind` ∈ `linear | radial | angular |
    /// reflected | diamond`; `c0`/`c1` are straight RGBA in `[0, 1]`
    /// (4 floats each). The gradient GEOMETRY is derived from the
    /// selection's bounding box — there is no on-canvas drag handle in
    /// v0 (`crate::fill` documents the derivation). Returns the NEW
    /// engine-held image's handle.
    #[wasm_bindgen]
    pub async fn fill_gradient(
        handle: u32,
        kind: String,
        c0: Vec<f32>,
        c1: Vec<f32>,
    ) -> Result<DecodedHandle, JsValue> {
        let kind = GradientKind::from_wire(&kind).ok_or_else(|| {
            JsValue::from_str(&format!(
                "unknown gradient kind \"{kind}\" (linear | radial | angular | \
                 reflected | diamond)"
            ))
        })?;
        if c0.len() != 4 || c1.len() != 4 {
            return Err(JsValue::from_str(
                "gradient stops must be 4 floats each (straight RGBA in [0,1])",
            ));
        }
        let (img, ctx, sel, layered) = fill_prelude(handle)?;
        let spec = FillSpec::Gradient {
            kind,
            c0: [c0[0], c0[1], c0[2], c0[3]],
            c1: [c1[0], c1[1], c1[2], c1[3]],
        };
        let out = fill_rgba8(&ctx, &img, &spec, sel)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        land_fill(img.width, img.height, out, layered).await
    }

    /// FILL the current selection (the whole image when none) with
    /// deterministic monochrome noise — `amount` scales the hash
    /// amplitude, `seed` makes a repeat reproducible. Returns the NEW
    /// engine-held image's handle.
    #[wasm_bindgen]
    pub async fn fill_noise(handle: u32, amount: f32, seed: u32) -> Result<DecodedHandle, JsValue> {
        let (img, ctx, sel, layered) = fill_prelude(handle)?;
        let out = fill_rgba8(&ctx, &img, &FillSpec::Noise { amount, seed }, sel)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        land_fill(img.width, img.height, out, layered).await
    }

    // ─────────────────────── LAYER doors (§6.2) ──────────────────────
    //
    // THE LAYER GRAPH. One stack per wasm realm, BOUND to an engine-held
    // image handle exactly like the selection is — `layers_open` seeds it
    // with that image's pixels as a single "Background" layer (an `Arc`
    // clone, so opening is O(1) and costs no extra memory).
    //
    // The bound image is the stack's COMPOSITE: `layers_composite` folds
    // the stack and writes the result back into the SAME handle, so every
    // downstream lane (the adjust chain, tiles, histogram, save-back,
    // export) keeps working against one handle and never has to learn
    // what a layer is. Pixel edits — paint, fill, bake — go into the
    // ACTIVE layer and are journaled tile-granularly (`layers_undo`).
    //
    // HONEST SCOPE, in the code as well as in the panel:
    //   * Layers are canvas-extent PIXEL layers. No groups, no clipping,
    //     no adjustment layers, no per-layer masks.
    //   * The journal is a PIXEL log: add / remove / reorder / rename /
    //     opacity / blend / visibility are NOT undoable.
    //   * A crop, resize or straighten changes the EXTENT, so it
    //     registers a new handle and the stack is re-opened over the
    //     result — i.e. those commits FLATTEN the stack.

    struct LayerDoc {
        /// The engine-held image this stack composites into.
        handle: u32,
        stack: LayerStack,
    }

    thread_local! {
        static LAYERS: RefCell<Option<LayerDoc>> = const { RefCell::new(None) };
    }

    /// Replace an engine-held image's pixels in place (same handle, same
    /// extent) and drop its mip pyramid, which the new pixels invalidate.
    fn set_image_pixels(handle: u32, rgba: Arc<[u8]>) -> Result<(), JsValue> {
        IMAGES.with(|m| {
            let mut map = m.borrow_mut();
            let img = map
                .get_mut(&handle)
                .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
            if rgba.len() != img.rgba.len() {
                return Err(JsValue::from_str(
                    "internal: composite extent does not match the bound image",
                ));
            }
            img.rgba = rgba;
            Ok(())
        })?;
        PYRAMIDS.with(|p| {
            p.borrow_mut().remove(&handle);
        });
        Ok(())
    }

    /// Take the bound stack out of the `RefCell` (no borrow across an
    /// `await`), run `f`, put it back — restored even when `f` fails.
    async fn with_stack_async<T, F, Fut>(f: F) -> Result<T, JsValue>
    where
        F: FnOnce(LayerDoc) -> Fut,
        Fut: std::future::Future<Output = (LayerDoc, Result<T, JsValue>)>,
    {
        let doc = LAYERS
            .with(|l| l.borrow_mut().take())
            .ok_or_else(|| JsValue::from_str("no layer stack open (layers_open first)"))?;
        let (doc, out) = f(doc).await;
        LAYERS.with(|l| *l.borrow_mut() = Some(doc));
        out
    }

    /// Synchronous access to the bound stack.
    fn with_stack<T>(f: impl FnOnce(&mut LayerDoc) -> Result<T, JsValue>) -> Result<T, JsValue> {
        LAYERS.with(|l| {
            let mut b = l.borrow_mut();
            let doc = b
                .as_mut()
                .ok_or_else(|| JsValue::from_str("no layer stack open (layers_open first)"))?;
            f(doc)
        })
    }

    fn ingest_err(e: IngestError) -> JsValue {
        JsValue::from_str(&e.to_string())
    }

    /// OPEN a layer stack over an engine-held image: one full-canvas
    /// "Background" layer sharing that image's pixels. Re-opening on the
    /// SAME handle is a no-op (the stack survives); opening on a
    /// different handle replaces it, which is what a crop / resize /
    /// straighten commit does (it flattens).
    #[wasm_bindgen]
    pub fn layers_open(handle: u32) -> Result<(), JsValue> {
        if LAYERS.with(|l| l.borrow().as_ref().map(|d| d.handle)) == Some(handle) {
            return Ok(());
        }
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        let stack = LayerStack::from_image(img.width, img.height, Arc::clone(&img.rgba))
            .map_err(ingest_err)?;
        LAYERS.with(|l| *l.borrow_mut() = Some(LayerDoc { handle, stack }));
        Ok(())
    }

    /// OPEN a layer stack from a retained PSD parse instead of from the
    /// flattened composite — the PSD's own layer tree, bottom-first, with
    /// its names, blend modes, opacities and visibility. Returns the
    /// layer count.
    ///
    /// This DECLINES (with the engine's stated reason) for every PSD
    /// whose structure the layer model does not reproduce — groups,
    /// clipping layers, layer masks, non-8-bit-RGB, or an over-budget
    /// canvas — because swapping Photoshop's own composite for a
    /// different-looking one of ours would be worse than flattening. On a
    /// refusal the caller keeps `layers_open` (the flatten) and shows the
    /// reason.
    ///
    /// `image_handle` must be the composite already ingested from the
    /// same file (same extent); `psd_handle` is a `psd_open` handle.
    #[wasm_bindgen]
    pub fn layers_open_from_psd(image_handle: u32, psd_handle: u32) -> Result<usize, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&image_handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {image_handle}")))?;
        let import = PSDS.with(|m| {
            let map = m.borrow();
            let file = map
                .get(&psd_handle)
                .ok_or_else(|| JsValue::from_str(&format!("unknown psd handle {psd_handle}")))?;
            file.layer_plates_rgba8()
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })?;
        if import.width != img.width || import.height != img.height {
            return Err(JsValue::from_str(&format!(
                "PSD layer import is {}×{} but the ingested composite is {}×{}",
                import.width, import.height, img.width, img.height
            )));
        }
        let stack = LayerStack::from_psd_plates(&import).map_err(ingest_err)?;
        let n = stack.len();
        LAYERS.with(|l| {
            *l.borrow_mut() = Some(LayerDoc {
                handle: image_handle,
                stack,
            })
        });
        Ok(n)
    }

    /// Drop the bound stack (and its undo history).
    #[wasm_bindgen]
    pub fn layers_close() {
        LAYERS.with(|l| *l.borrow_mut() = None);
    }

    /// The handle the stack is bound to, or `-1` when none is open.
    #[wasm_bindgen]
    pub fn layers_bound() -> i32 {
        LAYERS.with(|l| l.borrow().as_ref().map_or(-1, |d| d.handle as i32))
    }

    /// The stack as JSON, BOTTOM-first:
    /// `{"active":i,"layers":[{index,id,name,visible,locked,opacity,blend}]}`.
    /// `opacity` is 0–1; `blend` is the `compose.*` wire name.
    #[wasm_bindgen]
    pub fn layers_list() -> String {
        LAYERS.with(|l| {
            let b = l.borrow();
            let Some(doc) = b.as_ref() else {
                return "{\"active\":-1,\"layers\":[]}".to_string();
            };
            let rows: Vec<String> = doc
                .stack
                .layers()
                .iter()
                .enumerate()
                .map(|(index, layer)| {
                    format!(
                        "{{\"index\":{index},\"id\":{},\"name\":{},\"visible\":{},\"locked\":{},\"opacity\":{},\"blend\":\"{}\",\"hasMask\":{},\"maskEnabled\":{},\"kind\":\"{}\"}}",
                        layer.id,
                        json_escape(&layer.name),
                        layer.visible,
                        layer.locked,
                        layer.opacity,
                        layer.blend_name(),
                        layer.mask.is_some(),
                        layer.mask_enabled,
                        if layer.adjust_params().is_some() {
                            "adjustment"
                        } else if layer.smart_source().is_some() {
                            "smart"
                        } else {
                            "pixels"
                        },
                    )
                })
                .collect();
            format!(
                "{{\"active\":{},\"layers\":[{}]}}",
                doc.stack.active_index(),
                rows.join(",")
            )
        })
    }

    /// The undo/redo readout as JSON — including the BOUND and how much
    /// of it is used, so "history is a window" is stated rather than
    /// discovered. `null` when no stack is open.
    #[wasm_bindgen]
    pub fn layers_history() -> String {
        LAYERS.with(|l| {
            let b = l.borrow();
            let Some(doc) = b.as_ref() else {
                return "null".to_string();
            };
            let h = doc.stack.history();
            format!(
                "{{\"canUndo\":{},\"canRedo\":{},\"depth\":{},\"redoDepth\":{},\"bytes\":{},\"maxBytes\":{},\"maxEntries\":{},\"dropped\":{},\"generation\":{},\"undoLabel\":{},\"redoLabel\":{},\"undoSteps\":[{}],\"redoSteps\":[{}]}}",
                h.can_undo,
                h.can_redo,
                h.depth,
                h.redo_depth,
                h.bytes,
                h.max_bytes,
                h.max_entries,
                h.dropped,
                h.generation,
                doc.stack
                    .undo_label()
                    .map_or_else(|| "null".to_string(), json_escape),
                doc.stack
                    .redo_label()
                    .map_or_else(|| "null".to_string(), json_escape),
                doc.stack
                    .undo_labels()
                    .iter()
                    .map(|l| json_escape(l))
                    .collect::<Vec<_>>()
                    .join(","),
                doc.stack
                    .redo_labels()
                    .iter()
                    .map(|l| json_escape(l))
                    .collect::<Vec<_>>()
                    .join(","),
            )
        })
    }

    /// Add an empty transparent layer above the active one (it becomes
    /// active). Returns its index.
    #[wasm_bindgen]
    pub fn layers_add(name: &str) -> Result<usize, JsValue> {
        with_stack(|d| Ok(d.stack.add(name)))
    }

    /// Duplicate `index` above itself (the copy becomes active).
    #[wasm_bindgen]
    pub fn layers_duplicate(index: usize) -> Result<usize, JsValue> {
        with_stack(|d| {
            d.stack
                .duplicate(index)
                .ok_or_else(|| JsValue::from_str(&format!("no layer {index}")))
        })
    }

    /// Remove `index`. Removing the ONLY layer is refused (a document
    /// keeps at least one). NOT journaled — see the section docs.
    #[wasm_bindgen]
    pub fn layers_remove(index: usize) -> Result<(), JsValue> {
        with_stack(|d| d.stack.remove(index).map_err(ingest_err))
    }

    /// Move a layer in stack order (0 = bottom).
    #[wasm_bindgen]
    pub fn layers_reorder(from: usize, to: usize) -> Result<(), JsValue> {
        with_stack(|d| d.stack.reorder(from, to).map_err(ingest_err))
    }

    #[wasm_bindgen]
    pub fn layers_set_active(index: usize) -> Result<(), JsValue> {
        with_stack(|d| d.stack.set_active(index).map_err(ingest_err))
    }

    #[wasm_bindgen]
    pub fn layers_set_visible(index: usize, visible: bool) -> Result<(), JsValue> {
        with_stack(|d| d.stack.set_visible(index, visible).map_err(ingest_err))
    }

    /// Lock a layer's PIXELS: paint / fill / bake refuse on it. Its
    /// properties stay editable — that is what the lock means.
    #[wasm_bindgen]
    pub fn layers_set_locked(index: usize, locked: bool) -> Result<(), JsValue> {
        with_stack(|d| d.stack.set_locked(index, locked).map_err(ingest_err))
    }

    /// Set a layer's opacity (0–1, clamped).
    #[wasm_bindgen]
    pub fn layers_set_opacity(index: usize, opacity: f32) -> Result<(), JsValue> {
        with_stack(|d| d.stack.set_opacity(index, opacity).map_err(ingest_err))
    }

    #[wasm_bindgen]
    pub fn layers_set_name(index: usize, name: &str) -> Result<(), JsValue> {
        with_stack(|d| d.stack.set_name(index, name).map_err(ingest_err))
    }

    /// Set a layer's blend by `compose.*` wire name (prefix optional).
    /// An unregistered name is a clean error, never a silent normal.
    #[wasm_bindgen]
    pub fn layers_set_blend(index: usize, blend: &str) -> Result<(), JsValue> {
        with_stack(|d| d.stack.set_blend(index, blend).map_err(ingest_err))
    }

    /// Make the CURRENT SELECTION this layer's mask. The natural
    /// authoring path, and the reason layer masks needed no new
    /// authoring engine: the marquee / lasso / wand already produce
    /// exactly the coverage a mask is. Errors when nothing is selected —
    /// silently attaching an all-one mask would look like success and
    /// mask nothing.
    #[wasm_bindgen]
    pub fn layers_mask_from_selection(index: usize) -> Result<(), JsValue> {
        let coverage = SELECTION
            .with(|s| s.borrow().coverage().cloned())
            .ok_or_else(|| {
                JsValue::from_str("no selection to make a mask from — select an area first")
            })?;
        with_stack(|d| d.stack.set_mask(index, coverage).map_err(ingest_err))
    }

    /// DELETE the mask (the coverage is gone), as distinct from
    /// disabling it.
    #[wasm_bindgen]
    pub fn layers_clear_mask(index: usize) -> Result<(), JsValue> {
        with_stack(|d| d.stack.clear_mask(index).map_err(ingest_err))
    }

    /// Toggle whether the attached mask applies, RETAINING it either way
    /// — losing painted coverage to a toggle would be a real loss.
    #[wasm_bindgen]
    pub fn layers_set_mask_enabled(index: usize, enabled: bool) -> Result<(), JsValue> {
        with_stack(|d| d.stack.set_mask_enabled(index, enabled).map_err(ingest_err))
    }

    /// COMPOSITE the stack bottom-up and write the result back into the
    /// bound engine-held image, returning the straight RGBA8 (the C-1
    /// Stage-A payload). GPU-only whenever there is anything to blend; a
    /// single plain visible layer short-circuits to its own pixels with
    /// no dispatch at all, so a one-layer document needs no device.
    #[wasm_bindgen]
    pub async fn layers_composite() -> Result<js_sys::Uint8Array, JsValue> {
        let ctx = GPU.with(|g| g.borrow().clone());
        with_stack_async(|doc| async move {
            let result = match doc.stack.composite(ctx.as_deref(), None).await {
                Ok(rgba) => set_image_pixels(doc.handle, Arc::clone(&rgba))
                    .map(|()| js_sys::Uint8Array::from(&rgba[..])),
                Err(e) => Err(ingest_err(e)),
            };
            (doc, result)
        })
        .await
    }

    /// CONVERT a pixel layer into a smart object, preserving its pixels
    /// as the source. One-way by design: going back would discard the
    /// source, which is the destructive move this exists to prevent.
    #[wasm_bindgen]
    pub fn layers_make_smart(index: usize) -> Result<(), JsValue> {
        with_stack(|d| d.stack.make_smart(index).map_err(ingest_err))
    }

    /// RE-RENDER a smart object at `scale` — from its preserved SOURCE,
    /// never from the current cache, which is the whole point: scaling
    /// down and back up loses nothing.
    ///
    /// GPU-only (the resample is a kernel dispatch). The rendered result
    /// is letterboxed into the canvas extent, so the layer keeps its
    /// place in a stack whose layers are all canvas-sized.
    #[wasm_bindgen]
    pub async fn layers_render_smart(index: usize, scale: f32) -> Result<(), JsValue> {
        use half::f16;
        use image_kernels::families::resample::{ResampleParams, RESAMPLE_MITCHELL};

        if !(scale > 0.0) || scale > 16.0 {
            return Err(JsValue::from_str(
                "smart re-render scale must be in (0, 16]",
            ));
        }
        let ctx = GPU.with(|g| g.borrow().clone()).ok_or_else(|| {
            JsValue::from_str("smart re-render is GPU-only — call init_gpu first")
        })?;

        // Read the SOURCE (not the cached render) and the canvas extent.
        let (src, cw, ch) = LAYERS
            .with(|l| {
                let b = l.borrow();
                let d = b.as_ref()?;
                let layer = d.stack.layers().get(index)?;
                let s = layer.smart_source()?;
                Some((s.clone(), d.stack.width(), d.stack.height()))
            })
            .ok_or_else(|| JsValue::from_str(&format!("layer {index} is not a smart object")))?;

        let out_w = ((src.width as f32) * scale).round().max(1.0) as u32;
        let out_h = ((src.height as f32) * scale).round().max(1.0) as u32;

        let mut win = Vec::with_capacity(src.rgba.len() * 2);
        for &b in src.rgba.iter() {
            win.extend_from_slice(&f16::from_f32(b as f32 / 255.0).to_le_bytes());
        }
        let params = ResampleParams::new(
            src.width as f32 / out_w as f32,
            src.height as f32 / out_h as f32,
            0.0,
            0.0,
        );
        let rendered = image_gpu::execute_windowed_once_async(
            &ctx,
            &RESAMPLE_MITCHELL,
            &win,
            src.width,
            src.height,
            params.as_bytes(),
            None,
            out_w,
            out_h,
        )
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // f16 window -> straight RGBA8, then letterbox into the canvas.
        let small = crate::fill::f16_to_rgba8(&rendered);
        let mut canvas = vec![0u8; (cw as usize) * (ch as usize) * 4];
        for y in 0..out_h.min(ch) {
            let srow = (y as usize) * (out_w as usize) * 4;
            let drow = (y as usize) * (cw as usize) * 4;
            let n = (out_w.min(cw) as usize) * 4;
            canvas[drow..drow + n].copy_from_slice(&small[srow..srow + n]);
        }

        with_stack(|d| {
            d.stack
                .set_smart_render(index, Arc::from(canvas.into_boxed_slice()), scale)
                .map_err(ingest_err)
        })
    }

    /// Insert an ADJUSTMENT LAYER carrying the panel's current chain.
    ///
    /// The non-destructive counterpart of `layers_bake_adjust` below: the
    /// bake writes the chain into the active layer's pixels and journals
    /// it; this stacks the chain ABOVE and touches no pixel at all, so
    /// deleting the layer restores the original exactly. Same wire block
    /// so the two can never disagree about what the panel meant.
    ///
    /// Refuses at identity — an adjustment layer that adjusts nothing is
    /// a row that does nothing, and adding one silently is worse than
    /// saying so.
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn layers_add_adjustment(
        name: &str,
        exposure_ev: f32,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        temp: f32,
        tint: f32,
        in_black: f32,
        in_white: f32,
        gamma: f32,
        out_black: f32,
        out_white: f32,
        curve_lut: &[u8],
        blur_sigma: f32,
        sharpen_amount: f32,
        hue_degrees: f32,
        invert: bool,
        ext: &[f32],
    ) -> Result<usize, JsValue> {
        let params = build_adjust_params(
            exposure_ev,
            brightness,
            contrast,
            saturation,
            temp,
            tint,
            in_black,
            in_white,
            gamma,
            out_black,
            out_white,
            curve_lut,
            blur_sigma,
            sharpen_amount,
            hue_degrees,
            invert,
            ext,
        )?;
        if params.is_identity() {
            return Err(JsValue::from_str(
                "nothing to stack — the adjustment chain is at identity",
            ));
        }
        with_stack(|d| Ok(d.stack.add_adjustment(name, params)))
    }

    /// BAKE the adjustment chain into the ACTIVE layer — the DESTRUCTIVE
    /// per-layer adjustment (the panel's chain is otherwise a re-runnable
    /// PREVIEW of the composite and mutates nothing). Journaled over the
    /// whole canvas, so it is undoable; refuses on a locked layer and at
    /// identity. Arguments mirror `adjust_image_ext` minus the handle.
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub async fn layers_bake_adjust(
        exposure_ev: f32,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        temp: f32,
        tint: f32,
        in_black: f32,
        in_white: f32,
        gamma: f32,
        out_black: f32,
        out_white: f32,
        curve_lut: &[u8],
        blur_sigma: f32,
        sharpen_amount: f32,
        hue_degrees: f32,
        invert: bool,
        ext: &[f32],
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let params = build_adjust_params(
            exposure_ev,
            brightness,
            contrast,
            saturation,
            temp,
            tint,
            in_black,
            in_white,
            gamma,
            out_black,
            out_white,
            curve_lut,
            blur_sigma,
            sharpen_amount,
            hue_degrees,
            invert,
            ext,
        )?;
        let ctx = GPU.with(|g| g.borrow().clone());
        let sel = LAYERS.with(|l| {
            l.borrow()
                .as_ref()
                .and_then(|d| SELECTION.with(|s| s.borrow().mask_for(d.handle)))
        });
        with_stack_async(|mut doc| async move {
            let result = bake_into_active(&mut doc, ctx.as_deref(), params, sel).await;
            (doc, result)
        })
        .await
    }

    /// The bake's body (split out so the stack is owned, not borrowed,
    /// across the awaits).
    async fn bake_into_active(
        doc: &mut LayerDoc,
        ctx: Option<&GpuContext>,
        params: AdjustParams,
        selection: Option<Arc<SelectionCoverage>>,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        doc.stack.active_is_editable().map_err(ingest_err)?;
        if params.is_identity() {
            return Err(JsValue::from_str(
                "nothing to bake — the adjustment chain is at identity",
            ));
        }
        let ctx = ctx.ok_or_else(|| {
            JsValue::from_str("baking an adjustment is GPU-only — call init_gpu first")
        })?;
        let (w, h) = (doc.stack.width(), doc.stack.height());
        let src = DecodedImage {
            width: w,
            height: h,
            rgba: Arc::clone(&doc.stack.active().rgba),
            // Same: the active layer's pixels are post-ingest.
            display: crate::display::DisplayTreatment::AssumedSrgb,
        };
        let out = adjust_rgba8(ctx, &src, &params, selection)
            .await
            .map_err(ingest_err)?;
        doc.stack
            .edit_active(
                "Bake adjustments",
                image_core::Region::new(0, 0, w, h),
                Arc::from(out.into_boxed_slice()),
            )
            .map_err(ingest_err)?;
        let rgba = doc
            .stack
            .composite(Some(ctx), None)
            .await
            .map_err(ingest_err)?;
        set_image_pixels(doc.handle, Arc::clone(&rgba))?;
        Ok(js_sys::Uint8Array::from(&rgba[..]))
    }

    /// UNDO the newest journaled pixel edit (paint / fill / bake),
    /// re-composite, and answer the reverted edit's label — an EMPTY
    /// string when there is nothing to undo. Layer STRUCTURE changes are
    /// not journaled (see the section docs).
    #[wasm_bindgen]
    pub async fn layers_undo() -> Result<String, JsValue> {
        history_step(true).await
    }

    /// REDO the newest undone pixel edit.
    #[wasm_bindgen]
    pub async fn layers_redo() -> Result<String, JsValue> {
        history_step(false).await
    }

    async fn history_step(undo: bool) -> Result<String, JsValue> {
        let ctx = GPU.with(|g| g.borrow().clone());
        with_stack_async(|mut doc| async move {
            let label = if undo {
                doc.stack.undo()
            } else {
                doc.stack.redo()
            };
            let Some(label) = label else {
                return (doc, Ok(String::new()));
            };
            let result = match doc.stack.composite(ctx.as_deref(), None).await {
                Ok(rgba) => set_image_pixels(doc.handle, rgba).map(|()| label),
                Err(e) => Err(ingest_err(e)),
            };
            (doc, result)
        })
        .await
    }

    // ───────────────────────── PAINT doors ───────────────────────────
    //
    // One in-flight stroke per wasm realm, mirroring the crop tool's
    // shape: begin → extend* → commit | cancel. The engine holds the
    // base snapshot and the coverage accumulator; the glue holds only
    // pointer samples, so a stroke is replayable from a recorded action.
    //
    // HONEST SCOPE — no layer graph. A stroke paints into the SINGLE
    // engine-held image behind `handle`. There is no paint layer and
    // nothing above the photo; `brush_stroke_commit` registers a NEW
    // engine-held image (the crop / fill commit pattern) and the caller
    // swaps handles. The DOCUMENT and the source file are untouched —
    // re-ingesting the frame is the only restore this plugin owns.
    //
    // LATENCY — every `brush_stroke_extend` reads the composited window
    // back to the CPU and returns the WHOLE image as RGBA8, because the
    // C-1 Stage-A scene item takes bytes (Stage B, the resident GPU
    // texture, is deferred by ADR-018 / RFI I-01). The dirty-rectangle
    // composite keeps the GPU work proportional to the dabs, but the
    // per-extend byte copy is proportional to the IMAGE. `crate::stroke`
    // records exactly what Engine B would have to expose to remove it.

    thread_local! {
        // The single in-flight stroke (one per realm, like the selection).
        static STROKE: RefCell<Option<StrokeSession>> = const { RefCell::new(None) };
    }

    /// BEGIN a stroke on the engine-held image `handle`.
    ///
    /// `tool` ∈ `brush | pencil | eraser`; `blend` is a `compose.*`
    /// kernel name with the prefix optional (`"multiply"` or
    /// `"compose.multiply"`); `pressure_target` ∈
    /// `none | size | opacity | both` selects what the pen's pressure
    /// drives (default `both` — size AND opacity, the Photoshop pen
    /// preset). `color` is 4 straight RGBA floats in `[0, 1]`.
    ///
    /// PRESSURE, honestly: `PointerEvent.pressure` is a constant `0.5`
    /// for a mouse and a real reading only for a pen. The CALLER
    /// normalizes — the glue passes `1.0` for a mouse so a mouse stroke
    /// is not permanently half-size — and this door takes whatever it is
    /// given verbatim so a recorded stroke replays identically.
    ///
    /// Parameters are FROZEN for the stroke's duration: a stroke whose
    /// size changed halfway through would not be replayable.
    /// GPU-only — rejects without `init_gpu`.
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn brush_stroke_begin(
        handle: u32,
        tool: &str,
        size: f32,
        hardness: f32,
        opacity: f32,
        flow: f32,
        spacing: f32,
        blend: &str,
        color: Vec<f32>,
        pressure_target: &str,
    ) -> Result<(), JsValue> {
        let tool = StrokeTool::from_wire(tool).ok_or_else(|| {
            JsValue::from_str(&format!(
                "unknown paint tool \"{tool}\" (brush | pencil | eraser)"
            ))
        })?;
        let blend_kernel = blend_kernel(blend).ok_or_else(|| {
            JsValue::from_str(&format!(
                "unknown blend mode \"{blend}\" (a compose.* kernel name)"
            ))
        })?;
        let pressure = PressureTarget::from_wire(pressure_target).ok_or_else(|| {
            JsValue::from_str(&format!(
                "unknown pressure target \"{pressure_target}\" \
                 (none | size | opacity | both)"
            ))
        })?;
        if color.len() != 4 {
            return Err(JsValue::from_str(
                "brush colour must be 4 floats (straight RGBA in [0,1])",
            ));
        }
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
        if !gpu_ready() {
            return Err(JsValue::from_str(
                "painting is GPU-only — call init_gpu first (the dab composite \
                 is a registered WGSL kernel dispatch; no CPU blend path ships)",
            ));
        }
        let params = StrokeParams {
            tool,
            size,
            hardness,
            opacity,
            flow,
            spacing,
            blend: blend_kernel,
            color: [color[0], color[1], color[2], color[3]],
            pressure,
        };
        // The selection is frozen at begin — a stroke never half-honours
        // a selection that changed mid-drag.
        let sel = SELECTION.with(|s| s.borrow().mask_for(handle));
        // THE LAYER GRAPH: when a stack is bound to this handle the
        // stroke opens on its ACTIVE LAYER, so paint lands there and the
        // layers below and above are untouched. Without a stack it opens
        // on the engine-held image (the pre-layer behaviour, kept so the
        // doors stay usable on their own).
        let base = LAYERS.with(|l| {
            let b = l.borrow();
            match b.as_ref() {
                Some(d) if d.handle == handle => {
                    d.stack.active_is_editable()?;
                    Ok(Some(Arc::clone(&d.stack.active().rgba)))
                }
                _ => Ok(None),
            }
        });
        let base: Option<Arc<[u8]>> = base.map_err(ingest_err)?;
        let session = match base {
            Some(px) => StrokeSession::begin_on(handle, img.width, img.height, px, params, sel),
            None => StrokeSession::begin(handle, &img, params, sel),
        }
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        STROKE.with(|s| *s.borrow_mut() = Some(session));
        Ok(())
    }

    /// EXTEND the stroke with one pointer sample (image px + normalized
    /// pressure) and return the resulting straight RGBA8 for the WHOLE
    /// image — the C-1 Stage-A preview payload.
    ///
    /// Dabs are interpolated from the previous sample at
    /// `spacing · diameter` px of arc length (the residual carries across
    /// samples), so a fast drag paints a continuous stroke rather than
    /// one dot per pointer event. Only the dirty rectangle is
    /// re-composited, always FROM the base pixels, so extending is
    /// idempotent and the incremental result equals a from-scratch
    /// composite of the same samples.
    #[wasm_bindgen]
    pub async fn brush_stroke_extend(x: f32, y: f32, pressure: f32) -> Result<Vec<u8>, JsValue> {
        let ctx = GPU
            .with(|g| g.borrow().clone())
            .ok_or_else(|| JsValue::from_str("painting is GPU-only — call init_gpu first"))?;
        // The session is taken out for the await (no RefCell across a
        // suspension point) and put back after.
        let mut session = STROKE
            .with(|s| s.borrow_mut().take())
            .ok_or_else(|| JsValue::from_str("no stroke in progress (brush_stroke_begin first)"))?;
        let result = session
            .extend(&ctx, StrokeSample::new(x, y, pressure))
            .await;
        let handle = session.handle();
        let painted: Arc<[u8]> = Arc::from(session.pixels().to_vec().into_boxed_slice());
        STROKE.with(|s| *s.borrow_mut() = Some(session));
        result.map_err(|e| JsValue::from_str(&e.to_string()))?;
        // The preview is the WHOLE stack with the in-flight stroke
        // standing in for the active layer — otherwise painting on a
        // layer above the photo would preview as that layer alone. The
        // stack is NOT mutated; nothing is committed until release.
        // A one-layer document takes the trivial fold, which returns the
        // painted pixels themselves — the pre-layer latency, unchanged.
        let preview = preview_through_stack(handle, painted).await?;
        Ok(preview.to_vec())
    }

    /// Fold the bound stack with `painted` overriding the active layer.
    /// No stack (or one bound to another handle) ⇒ the painted pixels.
    async fn preview_through_stack(handle: u32, painted: Arc<[u8]>) -> Result<Arc<[u8]>, JsValue> {
        let bound = LAYERS.with(|l| l.borrow().as_ref().map(|d| d.handle)) == Some(handle);
        if !bound {
            return Ok(painted);
        }
        let ctx = GPU.with(|g| g.borrow().clone());
        with_stack_async(|doc| async move {
            let out = doc
                .stack
                .composite(ctx.as_deref(), Some(&painted))
                .await
                .map_err(ingest_err);
            (doc, out)
        })
        .await
    }

    /// COMMIT the stroke.
    ///
    /// * **With a layer stack bound** (the normal case): the painted
    ///   pixels are written into the ACTIVE LAYER, the tiles the stroke's
    ///   bounding box covers are journaled first (so the stroke is
    ///   undoable, tile-granularly, within the journal's stated bound),
    ///   and the stack is re-composited into the SAME engine-held image.
    ///   The returned handle is therefore the handle you started with —
    ///   the caller must NOT free it.
    /// * **Without one**: the pre-layer behaviour — the painted pixels
    ///   are registered as a NEW engine-held image and the caller swaps
    ///   handles and frees the old one.
    ///
    /// Either way the result is the same size, so the caller may carry
    /// the selection over with `selection_transfer`.
    #[wasm_bindgen]
    pub async fn brush_stroke_commit() -> Result<DecodedHandle, JsValue> {
        let session = STROKE
            .with(|s| s.borrow_mut().take())
            .ok_or_else(|| JsValue::from_str("no stroke in progress"))?;
        let (w, h) = (session.width(), session.height());
        let handle = session.handle();
        let bounds = session.stroke_bounds();
        let painted = session.commit();

        let bound = LAYERS.with(|l| l.borrow().as_ref().map(|d| d.handle)) == Some(handle);
        if !bound {
            return register(w, h, painted);
        }
        // Nothing landed on the canvas: no journal entry, no composite,
        // no change (an empty entry would spend an undo step on nothing).
        let Some(damage) = bounds else {
            return Ok(DecodedHandle {
                handle,
                width: w,
                height: h,
                // A no-op stroke commit leaves the treatment as it was.
                display: read_display(handle),
            });
        };
        let ctx = GPU.with(|g| g.borrow().clone());
        let pixels: Arc<[u8]> = Arc::from(painted.into_boxed_slice());
        with_stack_async(|mut doc| async move {
            let result = commit_stroke_into_active(&mut doc, ctx.as_deref(), damage, pixels).await;
            (doc, result)
        })
        .await
    }

    async fn commit_stroke_into_active(
        doc: &mut LayerDoc,
        ctx: Option<&GpuContext>,
        damage: image_core::Region,
        pixels: Arc<[u8]>,
    ) -> Result<DecodedHandle, JsValue> {
        let (w, h) = (doc.stack.width(), doc.stack.height());
        doc.stack
            .edit_active("Paint", damage, pixels)
            .map_err(ingest_err)?;
        let rgba = doc.stack.composite(ctx, None).await.map_err(ingest_err)?;
        set_image_pixels(doc.handle, rgba)?;
        Ok(DecodedHandle {
            handle: doc.handle,
            width: w,
            height: h,
            // The stroke composite writes back into the same image.
            display: read_display(doc.handle),
        })
    }

    /// CANCEL the stroke: throw the painted pixels away. The engine-held
    /// source was never mutated, so this restores it exactly.
    #[wasm_bindgen]
    pub fn brush_stroke_cancel() {
        STROKE.with(|s| *s.borrow_mut() = None);
    }

    /// Is a stroke in progress?
    #[wasm_bindgen]
    pub fn brush_stroke_active() -> bool {
        STROKE.with(|s| s.borrow().is_some())
    }

    /// The in-flight stroke's readout for the panel:
    /// `[dabs, x, y, w, h]` — the dab count and the stroke's bounding
    /// box in image px. Empty when no stroke is in progress or nothing
    /// has landed on the canvas yet.
    #[wasm_bindgen]
    pub fn brush_stroke_stats() -> Vec<f64> {
        STROKE.with(|s| {
            let b = s.borrow();
            let Some(session) = b.as_ref() else {
                return Vec::new();
            };
            let Some(r) = session.stroke_bounds() else {
                return Vec::new();
            };
            vec![
                session.dab_count() as f64,
                r.x as f64,
                r.y as f64,
                r.w as f64,
                r.h as f64,
            ]
        })
    }

    /// Every blend mode a stroke can paint through, newline-separated —
    /// derived from the `compose.*` registry so the panel's picker can
    /// never drift from the kernels that actually exist.
    #[wasm_bindgen]
    pub fn brush_blend_modes() -> String {
        blend_names().join("\n")
    }

    // ──────────────────────── SAVE-BACK doors ────────────────────────

    /// Re-encode straight RGBA8 as PNG or JPEG (`format` ∈ `png | jpeg`)
    /// — the NON-PSD save-back lane. Codec entropy coding is inherently
    /// CPU work (spec §1); JPEG rides the fixed v0 quality documented on
    /// `saveback::JPEG_QUALITY_DEFAULT`.
    #[wasm_bindgen]
    pub fn encode_image(
        rgba: &[u8],
        width: u32,
        height: u32,
        format: &str,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let fmt = RasterFormat::from_wire(format)
            .ok_or_else(|| JsValue::from_str(&format!("unknown encode format \"{format}\"")))?;
        let bytes = encode_rgba8(rgba, width, height, fmt)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(js_sys::Uint8Array::from(&bytes[..]))
    }

    /// PSD SAVE-BACK: write the ADJUSTED full-resolution `rgba` into the
    /// retained parse behind `psd_handle` (the merged composite is always
    /// rewritten; the layer structure is handled per the returned shape)
    /// and answer the honest description the panel shows —
    /// `"layer-replaced: …"` when the file's single canvas-sized content
    /// layer was updated in place via `replace_channel_pixels`, or
    /// `"flattened: …"` when a multi-layer file was flattened into a NEW
    /// single-layer PSD. Call `psd_save` afterwards for the bytes.
    ///
    /// 8-bit RGB only, and the size must match the parsed header —
    /// anything else is a clean error, never a wrong-looking file.
    #[wasm_bindgen]
    pub fn psd_apply_adjusted(
        psd_handle: u32,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<String, JsValue> {
        PSDS.with(|m| {
            let mut map = m.borrow_mut();
            let file = map
                .get_mut(&psd_handle)
                .ok_or_else(|| JsValue::from_str(&format!("unknown psd handle {psd_handle}")))?;
            let shape = psd_write_adjusted(file, width, height, rgba)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(shape.describe().to_string())
        })
    }

    // ─────────────────────────── PSD doors ───────────────────────────
    //
    // The mutatable tier (image-psd edit.rs: opacity / rename / remove +
    // the preservation writer) was Rust-only — "Paged never destroys a
    // PSD" was a test-only property with no wasm reach (the coverage
    // spec's finding). These doors retain the PARSED PsdFile behind a
    // handle so the panel can list layers, apply record edits, and save
    // with full carry-through preservation. Pixel-level channel
    // replacement stays engine-side (its payload contract is a follow-up).

    thread_local! {
        static PSDS: RefCell<HashMap<u32, image_psd::PsdFile>> =
            RefCell::new(HashMap::new());
        static NEXT_PSD: Cell<u32> = const { Cell::new(1) };
    }

    /// Parse a `.psd`/`.psb` and retain the structural model behind a
    /// handle (independent of `decode_image`'s composite lane). Free with
    /// `psd_close`.
    #[wasm_bindgen]
    pub fn psd_open(bytes: &[u8]) -> Result<u32, JsValue> {
        let file =
            image_psd::PsdFile::parse(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let handle = NEXT_PSD.with(|n| {
            let h = n.get();
            n.set(h + 1);
            h
        });
        PSDS.with(|m| m.borrow_mut().insert(handle, file));
        Ok(handle)
    }

    /// The layer list as JSON, in record order:
    /// `[{index, name, opacity, hidden, top, left, bottom, right}]`.
    /// `hidden` is PSD flags bit 1 (0x02).
    #[wasm_bindgen]
    pub fn psd_layer_list(handle: u32) -> Result<String, JsValue> {
        PSDS.with(|m| {
            let map = m.borrow();
            let file = map
                .get(&handle)
                .ok_or_else(|| JsValue::from_str(&format!("unknown psd handle {handle}")))?;
            let rows: Vec<String> = file
                .layer_mask
                .layers
                .iter()
                .enumerate()
                .map(|(index, l)| {
                    format!(
                        "{{\"index\":{index},\"name\":{},\"opacity\":{},\"hidden\":{},\"top\":{},\"left\":{},\"bottom\":{},\"right\":{}}}",
                        json_escape(&l.name()),
                        l.opacity,
                        (l.flags & 0x02) != 0,
                        l.top,
                        l.left,
                        l.bottom,
                        l.right,
                    )
                })
                .collect();
            Ok(format!("[{}]", rows.join(",")))
        })
    }

    /// Set a layer's opacity (0–255) through the mutatable tier.
    #[wasm_bindgen]
    pub fn psd_set_layer_opacity(handle: u32, layer: usize, opacity: u8) -> Result<(), JsValue> {
        with_psd_mut(handle, |file| {
            image_psd::edit::set_layer_opacity(file, layer, opacity)
        })
    }

    /// Rename a layer (updates the legacy Pascal name AND the canonical
    /// `luni` block).
    #[wasm_bindgen]
    pub fn psd_set_layer_name(handle: u32, layer: usize, name: &str) -> Result<(), JsValue> {
        with_psd_mut(handle, |file| {
            image_psd::edit::set_layer_name(file, layer, name)
        })
    }

    /// Remove a layer (balanced `lsct` group-divider bookkeeping engine-side).
    #[wasm_bindgen]
    pub fn psd_remove_layer(handle: u32, layer: usize) -> Result<(), JsValue> {
        with_psd_mut(handle, |file| image_psd::edit::remove_layer(file, layer))
    }

    /// Save the (possibly edited) PSD with full preservation: unmodeled
    /// blocks verbatim; a zero-edit save is byte-identical.
    #[wasm_bindgen]
    pub fn psd_save(handle: u32) -> Result<js_sys::Uint8Array, JsValue> {
        PSDS.with(|m| {
            let map = m.borrow();
            let file = map
                .get(&handle)
                .ok_or_else(|| JsValue::from_str(&format!("unknown psd handle {handle}")))?;
            let bytes = file
                .write()
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(js_sys::Uint8Array::from(&bytes[..]))
        })
    }

    #[wasm_bindgen]
    pub fn psd_close(handle: u32) {
        PSDS.with(|m| {
            m.borrow_mut().remove(&handle);
        });
    }

    /// Minimal JSON string escape (quotes, backslash, control chars) —
    /// avoids a serde_json dependency for one field.
    fn json_escape(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    fn with_psd_mut<F>(handle: u32, f: F) -> Result<(), JsValue>
    where
        F: FnOnce(&mut image_psd::PsdFile) -> image_psd::Result<()>,
    {
        PSDS.with(|m| {
            let mut map = m.borrow_mut();
            let file = map
                .get_mut(&handle)
                .ok_or_else(|| JsValue::from_str(&format!("unknown psd handle {handle}")))?;
            f(file).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn abi_and_registry_reachable() {
        assert_eq!(super::abi_version(), 1);
        assert!(super::kernel_count() >= 2);
    }
}
