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

//! The M4 ingest slice (spec §2.1.3 amended by C-1 Stage A): decode a
//! placed image's ORIGINAL bytes (C-5 serves PSD/PNG/JPEG) to RGBA8,
//! run the adjustments chain through Engine A, and hand the result back
//! for the v41 `SceneItem::Image` composite. The RGBA that crosses to
//! JS here is the Stage-A render payload destined for the HOST's scene
//! channel — never pixels for plugin-side processing (the §2.1.3 rule
//! survives in that narrowed form; the spike doc records the contract).
//!
//! Decode is inherently-CPU codec work (spec §1); the adjustment
//! kernels run GPU-only through the pipeline's async sink — there is NO
//! CPU kernel fallback (an absent adapter is an honest error). The M0
//! decode bridge maps U8 verbatim (`/255`, no premultiply, no
//! transfer/CMS cast — BREAKAGE I-02), so adjustments operate on
//! straight encoded values until the M1 CMS lane lands; the kernels'
//! math is unchanged when it does.

use std::sync::Arc;

use image_codecs::raw::{RawSource, RawTarget};
use image_codecs::{ImageSource, JpegSource, MemoryByteSource, Orientation, PngSource};
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileSliceMut, Transfer,
};
use image_gpu::GpuContext;
use image_kernels::families::adjust::{
    AdjustBrightnessContrastParams, AdjustExposureParams, AdjustSaturationParams,
    ADJUST_BRIGHTNESS_CONTRAST, ADJUST_EXPOSURE, ADJUST_SATURATION,
};
use image_pipeline::Pipeline;
use image_psd::PsdFile;

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("decode: {0}")]
    Decode(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("pipeline: {0}")]
    Pipeline(String),
}

/// One decoded image held behind a handle on the wasm surface (pixels
/// stay engine-side between calls; `Arc` so re-adjust clones are free).
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    /// Tightly packed straight RGBA8, row-major.
    pub rgba: Arc<[u8]>,
}

/// The M4 adjustments parameter set (the panel's committed values).
/// Identity: ev 0, brightness 0, contrast 1, saturation 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdjustParams {
    pub exposure_ev: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
}

impl Default for AdjustParams {
    fn default() -> Self {
        AdjustParams {
            exposure_ev: 0.0,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
        }
    }
}

impl AdjustParams {
    pub fn is_identity(&self) -> bool {
        *self == AdjustParams::default()
    }
}

impl DecodedImage {
    /// C-6 (I-06) — cut a LEVEL-0 tile window `(x, y, w, h)` out of the
    /// decoded buffer as tightly packed RGBA8 (`w'*h'*4` bytes, row-major,
    /// where `w'`/`h'` are the window CLIPPED to the image extent). Returns
    /// `(bytes, w', h')`; an empty `Vec` with `(0, 0)` when the window lies
    /// fully outside the image. Pure windowing — no resampling kernel, no
    /// GPU dispatch (orchestration, spec §6); the honest subset of the
    /// resource provider until the Engine B tiled mip eval is wired to the
    /// wasm boundary.
    pub fn tile_window_rgba8(&self, x: u32, y: u32, w: u32, h: u32) -> (Vec<u8>, u32, u32) {
        let x0 = x.min(self.width);
        let y0 = y.min(self.height);
        let x1 = x.saturating_add(w).min(self.width);
        let y1 = y.saturating_add(h).min(self.height);
        if x1 <= x0 || y1 <= y0 {
            return (Vec::new(), 0, 0);
        }
        let tw = x1 - x0;
        let th = y1 - y0;
        let mut out = vec![0u8; (tw as usize) * (th as usize) * 4];
        let stride = self.width as usize * 4;
        for row in 0..th as usize {
            let src_off = (y0 as usize + row) * stride + x0 as usize * 4;
            let dst_off = row * tw as usize * 4;
            let len = tw as usize * 4;
            out[dst_off..dst_off + len].copy_from_slice(&self.rgba[src_off..src_off + len]);
        }
        (out, tw, th)
    }
}

/// The straight-RGBA8 format the ingest slice speaks on both ends
/// (mirrors the pipeline conformance stimulus; the M0 bridge maps it
/// verbatim into the working space).
const RGBA8: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Linear,
    space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
};

/// Decode PSD / PNG / JPEG bytes (sniffed by magic) to straight RGBA8.
/// The honest M4 subset: what the codec adapters + the PSD merged-
/// composite decode carry today — 8-bit, non-CMYK. Everything else is a
/// clean `Unsupported`, never a wrong-looking image.
pub fn decode_rgba8(bytes: &[u8]) -> Result<DecodedImage, IngestError> {
    match sniff(bytes) {
        Some(Format::Psd) => decode_psd(bytes),
        Some(Format::Png) => decode_source(PngSource::new(MemoryByteSource::new(bytes.to_vec()))),
        Some(Format::Jpeg) => decode_source(JpegSource::new(MemoryByteSource::new(bytes.to_vec()))),
        None => Err(IngestError::Unsupported(
            "unrecognized image container (PSD/PNG/JPEG in the M4 slice)".into(),
        )),
    }
}

enum Format {
    Psd,
    Png,
    Jpeg,
}

fn sniff(bytes: &[u8]) -> Option<Format> {
    if bytes.starts_with(b"8BPS") {
        Some(Format::Psd)
    } else if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some(Format::Png)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(Format::Jpeg)
    } else {
        None
    }
}

fn decode_psd(bytes: &[u8]) -> Result<DecodedImage, IngestError> {
    let file = PsdFile::parse(bytes).map_err(|e| IngestError::Decode(e.to_string()))?;
    let composite = file.composite_rgba8().map_err(|e| match e {
        image_psd::PsdError::Unsupported(s) => IngestError::Unsupported(s),
        other => IngestError::Decode(other.to_string()),
    })?;
    Ok(DecodedImage {
        width: composite.width,
        height: composite.height,
        rgba: composite.rgba.into(),
    })
}

/// Full-frame decode through an `ImageSource` adapter, widened to RGBA8.
fn decode_source<S: ImageSource>(mut source: S) -> Result<DecodedImage, IngestError> {
    let info = source
        .probe()
        .map_err(|e| IngestError::Decode(e.to_string()))?;
    if info.format.depth != SampleDepth::U8 {
        return Err(IngestError::Unsupported(format!(
            "depth {:?} (8-bit only in the M4 slice)",
            info.format.depth
        )));
    }
    // EXIF orientation (JPEG/TIFF carry it; PNG/PSD don't, so it parses to
    // None and the auto-orient is a no-op). Auto-orientation is the
    // architecturally honest job of the decode-to-RGBA bridge: it is a CPU
    // memory reshuffle (transpose/flip) inherent to *ingest*, not a GPU
    // kernel — it must run before the GPU adjustment pipeline so the
    // adjustments and the C-1 composite see upright, correctly-dimensioned
    // pixels. (Spec §10.3 EXIF read path; the value is also surfaced raw on
    // SourceInfo for callers that want to defer the rotation.)
    let orientation = info.exif_meta().orientation.unwrap_or(Orientation::TopLeft);
    let channels = info.format.channels;
    if matches!(channels, ChannelLayout::Cmyk | ChannelLayout::Cmyka) {
        return Err(IngestError::Unsupported(
            "CMYK placed images (the M2 cast/CMS lane)".into(),
        ));
    }
    let (w, h) = (info.width, info.height);
    let bpp = info.format.bytes_per_pixel();
    let mut buf = vec![0u8; w as usize * h as usize * bpp];
    let roi = Region::new(0, 0, w, h);
    let mut out = TileSliceMut {
        region: roi,
        format: info.format,
        row_stride: w as usize * bpp,
        bytes: &mut buf,
    };
    source
        .read_region(roi, 1, &mut out)
        .map_err(|e| IngestError::Decode(e.to_string()))?;

    let n = w as usize * h as usize;
    let rgba: Vec<u8> = match channels {
        ChannelLayout::Rgba => buf,
        ChannelLayout::Gray => {
            let mut v = Vec::with_capacity(n * 4);
            for &g in &buf {
                v.extend_from_slice(&[g, g, g, 255]);
            }
            v
        }
        ChannelLayout::GrayA => {
            let mut v = Vec::with_capacity(n * 4);
            for px in buf.chunks_exact(2) {
                v.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
            }
            v
        }
        ChannelLayout::Cmyk | ChannelLayout::Cmyka => unreachable!("rejected above"),
    };

    // Auto-orient on the straight-RGBA8 buffer. Identity short-circuits
    // (the common case — most images are TopLeft) so non-rotated ingest
    // pays nothing.
    let (rgba, w, h) = apply_orientation(rgba, w, h, orientation);
    Ok(DecodedImage {
        width: w,
        height: h,
        rgba: rgba.into(),
    })
}

/// Apply an EXIF [`Orientation`] to a tightly-packed straight-RGBA8
/// buffer, returning the reoriented pixels and the (possibly swapped)
/// dimensions. The eight cases are the CIPA transforms expressed as a
/// (flip-x, flip-y, transpose) composition over destination coordinates.
/// `TopLeft` returns the input untouched.
fn apply_orientation(rgba: Vec<u8>, w: u32, h: u32, o: Orientation) -> (Vec<u8>, u32, u32) {
    if o.is_identity() {
        return (rgba, w, h);
    }
    let (wi, hi) = (w as usize, h as usize);
    // For each orientation, (dst_w, dst_h) and a mapping from destination
    // (dx, dy) back to source (sx, sy). Derived from the CIPA table; the
    // four 90°/270° cases transpose (dst dims swap).
    let swaps = o.swaps_dimensions();
    let (dw, dh) = if swaps { (hi, wi) } else { (wi, hi) };
    let mut out = vec![0u8; dw * dh * 4];
    for dy in 0..dh {
        for dx in 0..dw {
            // Map destination → source per orientation.
            let (sx, sy) = match o {
                Orientation::TopLeft => (dx, dy), // unreachable (identity)
                Orientation::TopRight => (wi - 1 - dx, dy), // mirror H
                Orientation::BottomRight => (wi - 1 - dx, hi - 1 - dy), // 180°
                Orientation::BottomLeft => (dx, hi - 1 - dy), // mirror V
                Orientation::LeftTop => (dy, dx), // transpose
                Orientation::RightTop => (dy, hi - 1 - dx), // 90° CW
                Orientation::RightBottom => (wi - 1 - dy, hi - 1 - dx), // transverse
                Orientation::LeftBottom => (wi - 1 - dy, dx), // 270° CW
            };
            let s = (sy * wi + sx) * 4;
            let d = (dy * dw + dx) * 4;
            out[d..d + 4].copy_from_slice(&rgba[s..s + 4]);
        }
    }
    (out, dw as u32, dh as u32)
}

/// Run the M4 adjustments chain (exposure → brightness/contrast →
/// saturation, each stage only when non-neutral) through Engine A's
/// ASYNC sink and return straight RGBA8. Identity params short-circuit
/// to the decoded pixels — pure orchestration, no kernel is skipped
/// dishonestly (there is nothing to dispatch). GPU-only by construction:
/// no adapter ⇒ the caller never reaches here with a context.
pub async fn adjust_rgba8(
    ctx: &GpuContext,
    image: &DecodedImage,
    params: &AdjustParams,
) -> Result<Vec<u8>, IngestError> {
    if params.is_identity() {
        return Ok(image.rgba.to_vec());
    }
    let mut pipe = Pipeline::new();
    let src = RawSource::new(image.width, image.height, RGBA8, image.rgba.clone())
        .map_err(|e| IngestError::Pipeline(e.to_string()))?;
    let mut node = pipe.source(Box::new(src));
    if params.exposure_ev != 0.0 {
        node = pipe.apply(
            node,
            &ADJUST_EXPOSURE,
            Arc::<[u8]>::from(AdjustExposureParams::new(params.exposure_ev).as_bytes()),
        );
    }
    if params.brightness != 0.0 || params.contrast != 1.0 {
        node = pipe.apply(
            node,
            &ADJUST_BRIGHTNESS_CONTRAST,
            Arc::<[u8]>::from(
                AdjustBrightnessContrastParams::new(params.brightness, params.contrast).as_bytes(),
            ),
        );
    }
    if params.saturation != 1.0 {
        node = pipe.apply(
            node,
            &ADJUST_SATURATION,
            Arc::<[u8]>::from(AdjustSaturationParams::new(params.saturation).as_bytes()),
        );
    }

    let roi = Region::new(0, 0, image.width, image.height);
    let mut target = RawTarget::new();
    pipe.to_encoder_async(node, roi, ctx, &mut target, RGBA8)
        .await
        .map_err(|e| IngestError::Pipeline(e.to_string()))?;
    Ok(target.into_pixels())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×1 RGBA image: pixel (0,0) red, (1,0) green — a horizontal pair
    /// so flips/rotations are unambiguous. Each pixel encodes its source
    /// (x,y) in the R,G bytes for easy assertion.
    fn grid(w: u32, h: u32) -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                v[i] = x as u8;
                v[i + 1] = y as u8;
                v[i + 2] = 0;
                v[i + 3] = 255;
            }
        }
        v
    }

    /// Read the (encoded src-x, src-y) at destination (dx, dy) of a
    /// reoriented buffer of width `dw`.
    fn at(buf: &[u8], dw: u32, dx: u32, dy: u32) -> (u8, u8) {
        let i = ((dy * dw + dx) * 4) as usize;
        (buf[i], buf[i + 1])
    }

    #[test]
    fn orientation_identity_is_untouched() {
        let src = grid(3, 2);
        let (out, w, h) = apply_orientation(src.clone(), 3, 2, Orientation::TopLeft);
        assert_eq!((w, h), (3, 2));
        assert_eq!(out, src);
    }

    #[test]
    fn orientation_mirror_horizontal() {
        // TopRight (2): mirror across the vertical axis. dst(0,0) should
        // come from src(2,0) in a 3-wide image.
        let (out, w, h) = apply_orientation(grid(3, 2), 3, 2, Orientation::TopRight);
        assert_eq!((w, h), (3, 2));
        assert_eq!(at(&out, w, 0, 0), (2, 0));
        assert_eq!(at(&out, w, 2, 1), (0, 1));
    }

    #[test]
    fn orientation_rotate_180() {
        let (out, w, h) = apply_orientation(grid(3, 2), 3, 2, Orientation::BottomRight);
        assert_eq!((w, h), (3, 2));
        // dst(0,0) == src(2,1) (opposite corner).
        assert_eq!(at(&out, w, 0, 0), (2, 1));
    }

    #[test]
    fn orientation_rotate_90_cw_swaps_dims() {
        // RightTop (6): rotate 90° CW. A 3×2 source becomes 2×3.
        let (out, w, h) = apply_orientation(grid(3, 2), 3, 2, Orientation::RightTop);
        assert_eq!((w, h), (2, 3), "90° CW swaps to 2×3");
        // Under 90° CW, source top-left (0,0) lands at dst top-right
        // (dst_w-1, 0) = (1, 0).
        assert_eq!(at(&out, w, 1, 0), (0, 0));
        // Source bottom-left (0,1) lands at dst (0,0).
        assert_eq!(at(&out, w, 0, 0), (0, 1));
    }

    #[test]
    fn orientation_rotate_270_cw_swaps_dims() {
        // LeftBottom (8): rotate 270° CW. 3×2 → 2×3.
        let (out, w, h) = apply_orientation(grid(3, 2), 3, 2, Orientation::LeftBottom);
        assert_eq!((w, h), (2, 3));
        // 270° CW: source top-left (0,0) lands at dst bottom-left
        // (0, dst_h-1) = (0, 2).
        assert_eq!(at(&out, w, 0, 2), (0, 0));
    }

    #[test]
    fn orientation_transpose_and_transverse_swap_dims() {
        for o in [Orientation::LeftTop, Orientation::RightBottom] {
            let (out, w, h) = apply_orientation(grid(3, 2), 3, 2, o);
            assert_eq!((w, h), (2, 3), "transpose/transverse swap dims for {o:?}");
            assert_eq!(out.len(), (w * h * 4) as usize);
        }
    }
}
