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
//! The release-build guarantee proven by CI: NO reference code
//! (image-conformance / `image-kernels` feature `reference`) is
//! reachable from this crate (cargo-tree guard, spec §4 dep rule 2).

pub mod ingest;

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

    use image_gpu::GpuContext;
    use wasm_bindgen::prelude::*;

    use crate::ingest::{
        adjust_rgba8, crop_rgba8, decode_rgba8, AdjustParams, DecodedImage, LevelsParams,
    };

    thread_local! {
        /// The bundle-realm GPU device (I-07: created HERE, where
        /// `navigator.gpu` is reachable; the wasm sandbox has none).
        static GPU: RefCell<Option<Rc<GpuContext>>> = const { RefCell::new(None) };
        /// Decoded images held engine-side behind handles (§2.1.3).
        static IMAGES: RefCell<HashMap<u32, DecodedImage>> =
            RefCell::new(HashMap::new());
        static NEXT_HANDLE: Cell<u32> = const { Cell::new(1) };
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
        IMAGES.with(|m| m.borrow_mut().insert(handle, img));
        Ok(DecodedHandle {
            handle,
            width,
            height,
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
        IMAGES.with(|m| m.borrow_mut().insert(handle, img));
        Ok(DecodedHandle {
            handle,
            width,
            height,
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
        run_adjust(&img, params).await
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
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let img = IMAGES
            .with(|m| m.borrow().get(&handle).cloned())
            .ok_or_else(|| JsValue::from_str(&format!("unknown image handle {handle}")))?;
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
        let params = AdjustParams {
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
        };
        run_adjust(&img, params).await
    }

    /// Shared adjust runner: identity → the decode verbatim; otherwise the
    /// GPU chain (requires `init_gpu`) plus any CPU curve LUT.
    async fn run_adjust(
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
        let out = adjust_rgba8(&ctx, img, &params)
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

    /// Commit a CROP: cut the integer pixel rectangle `(x, y, w, h)`
    /// (clamped to the image extent) out of an engine-held image and
    /// register the result as a NEW engine-held image, returning its
    /// handle. The source handle is left intact (the caller frees it). An
    /// out-of-bounds / empty rectangle is a clean error (never a torn
    /// image). The straighten-angle resample is a separate stage (not in
    /// this axis-aligned cut — see the crop interaction machine).
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
        IMAGES.with(|m| m.borrow_mut().insert(new_handle, cropped));
        Ok(DecodedHandle {
            handle: new_handle,
            width,
            height,
        })
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

    /// Release an engine-held decoded image.
    #[wasm_bindgen]
    pub fn free_image(handle: u32) {
        IMAGES.with(|m| {
            m.borrow_mut().remove(&handle);
        });
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
