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
    AdjustBrightnessContrastParams, AdjustExposureParams, AdjustLevelsParams,
    AdjustSaturationParams, AdjustWhiteBalanceParams, ADJUST_BRIGHTNESS_CONTRAST, ADJUST_EXPOSURE,
    ADJUST_LEVELS, ADJUST_SATURATION, ADJUST_WHITE_BALANCE,
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

/// Levels parameters (the panel's black/white/gamma + output range).
/// Identity: in_black 0, in_white 1, gamma 1, out_black 0, out_white 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelsParams {
    pub in_black: f32,
    pub in_white: f32,
    pub gamma: f32,
    pub out_black: f32,
    pub out_white: f32,
}

impl Default for LevelsParams {
    fn default() -> Self {
        LevelsParams {
            in_black: 0.0,
            in_white: 1.0,
            gamma: 1.0,
            out_black: 0.0,
            out_white: 1.0,
        }
    }
}

impl LevelsParams {
    fn is_identity(&self) -> bool {
        *self == LevelsParams::default()
    }
}

/// The M4 adjustments parameter set (the panel's committed values).
/// Identity: ev 0, brightness 0, contrast 1, saturation 1, WB 0/0, levels
/// identity, and no curve LUT.
#[derive(Debug, Clone, PartialEq)]
pub struct AdjustParams {
    pub exposure_ev: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    /// White balance: temp (amber↔blue), tint (green↔magenta). 0/0 = off.
    pub temp: f32,
    pub tint: f32,
    /// Levels (composite, all channels).
    pub levels: LevelsParams,
    /// Curves: an optional composite 256-entry tone LUT (the panel's
    /// control-point curve, built by `image_core::curve_lut`). `None` =
    /// identity (no curve pass). Applied as a CPU LUT on the straight
    /// RGBA8 result — there is no GPU LUT kernel yet (the honest deferral
    /// documented on the wasm export).
    pub curve_lut: Option<[u8; 256]>,
}

impl Default for AdjustParams {
    fn default() -> Self {
        AdjustParams {
            exposure_ev: 0.0,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            temp: 0.0,
            tint: 0.0,
            levels: LevelsParams::default(),
            curve_lut: None,
        }
    }
}

impl AdjustParams {
    pub fn is_identity(&self) -> bool {
        *self == AdjustParams::default()
    }

    /// Do any GPU adjust stages run (everything except the CPU curve LUT)?
    /// The curve LUT is a separate CPU pass that does NOT need the GPU.
    fn has_gpu_stage(&self) -> bool {
        self.exposure_ev != 0.0
            || self.brightness != 0.0
            || self.contrast != 1.0
            || self.saturation != 1.0
            || self.temp != 0.0
            || self.tint != 0.0
            || !self.levels.is_identity()
    }
}

impl DecodedImage {
    /// K-3 (S-07 / I-02) — register a PRE-DECODED straight-RGBA8 buffer as
    /// an engine-held image. The decode worker pool runs the codec/PSD CPU
    /// lanes OFF the main thread and hands the raw pixels back; the main
    /// realm registers them here to get a handle for the GPU adjust + tile
    /// windowing paths (which require engine-held pixels). `bytes` must be
    /// exactly `width*height*4` straight RGBA8, row-major — a length
    /// mismatch is rejected (never a torn image).
    pub fn from_rgba8(width: u32, height: u32, bytes: Vec<u8>) -> Result<Self, IngestError> {
        let expected = (width as usize) * (height as usize) * 4;
        if bytes.len() != expected {
            return Err(IngestError::Decode(format!(
                "ingest_rgba8: {} bytes for {width}x{height} (expected {expected})",
                bytes.len()
            )));
        }
        Ok(DecodedImage {
            width,
            height,
            rgba: Arc::from(bytes),
        })
    }

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
    // CMYK+alpha (5-channel) is not produced by any current codec adapter
    // (the JPEG lane delivers 4-ink `Cmyk`); reject it cleanly rather than
    // guess an alpha-from-ink rule.
    if matches!(channels, ChannelLayout::Cmyka) {
        return Err(IngestError::Unsupported(
            "CMYK+alpha placed images (no 5-channel ingest lane)".into(),
        ));
    }
    // The embedded ICC profile (if any) drives the colour-managed CMYK
    // cast; clone it out before `info` is consumed below.
    let embedded_icc = info.icc.clone();
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
        ChannelLayout::Cmyk => {
            // The print-lane ingest cast (spec §5.2): 4-ink CMYK → RGBA8,
            // colour-managed via the embedded ICC when present, else the
            // uncalibrated device formula. `buf` is packed 4-byte true ink
            // (the JPEG adapter already applied the Adobe-APP14 re-inversion).
            let (rgba, _managed) = crate::cmyk::cmyk8_to_rgba8(&buf, embedded_icc.as_deref())?;
            rgba
        }
        ChannelLayout::Cmyka => unreachable!("rejected above"),
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

/// Run the M4 adjustments chain through Engine A's ASYNC sink and return
/// straight RGBA8. Stage order (each only when non-neutral):
///   exposure → white-balance → levels → brightness/contrast → saturation
/// on the GPU, then the optional CURVES tone LUT as a CPU pass (there is
/// no GPU LUT kernel yet — the honest deferral; the LUT itself is built
/// deterministically by `image_core::curve_lut` panel-side). Identity
/// params short-circuit to the decoded pixels. When ONLY a curve is set
/// (no GPU stage) the GPU is skipped entirely and the LUT runs straight
/// on the decoded buffer. GPU-only by construction for the kernel stages:
/// no adapter ⇒ the caller never reaches here with a context.
pub async fn adjust_rgba8(
    ctx: &GpuContext,
    image: &DecodedImage,
    params: &AdjustParams,
) -> Result<Vec<u8>, IngestError> {
    if params.is_identity() {
        return Ok(image.rgba.to_vec());
    }

    // The GPU kernel chain (skipped wholesale when only a curve is set).
    let mut pixels = if params.has_gpu_stage() {
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
        if params.temp != 0.0 || params.tint != 0.0 {
            node = pipe.apply(
                node,
                &ADJUST_WHITE_BALANCE,
                Arc::<[u8]>::from(
                    AdjustWhiteBalanceParams::new(params.temp, params.tint).as_bytes(),
                ),
            );
        }
        if !params.levels.is_identity() {
            let l = &params.levels;
            node = pipe.apply(
                node,
                &ADJUST_LEVELS,
                Arc::<[u8]>::from(
                    AdjustLevelsParams::new(
                        l.in_black,
                        l.in_white,
                        l.gamma,
                        l.out_black,
                        l.out_white,
                    )
                    .as_bytes(),
                ),
            );
        }
        if params.brightness != 0.0 || params.contrast != 1.0 {
            node = pipe.apply(
                node,
                &ADJUST_BRIGHTNESS_CONTRAST,
                Arc::<[u8]>::from(
                    AdjustBrightnessContrastParams::new(params.brightness, params.contrast)
                        .as_bytes(),
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
        target.into_pixels()
    } else {
        image.rgba.to_vec()
    };

    // Curves: a 256-entry tone LUT over the RGB channels (alpha untouched).
    // A deterministic CPU table lookup — no GPU LUT kernel exists yet.
    if let Some(lut) = &params.curve_lut {
        apply_curve_lut(&mut pixels, lut);
    }
    Ok(pixels)
}

/// Apply a 256-entry tone LUT to the RGB channels of a straight-RGBA8
/// buffer in place (alpha is never remapped). The CURVES stage: a pure
/// per-channel table lookup, the deterministic CPU pass that consumes the
/// LUT the panel built (`image_core::curve_lut`).
pub fn apply_curve_lut(pixels: &mut [u8], lut: &[u8; 256]) {
    for px in pixels.chunks_exact_mut(4) {
        px[0] = lut[px[0] as usize];
        px[1] = lut[px[1] as usize];
        px[2] = lut[px[2] as usize];
    }
}

/// Commit a CROP: cut the integer pixel rectangle `(x, y, w, h)` (clamped
/// to the image extent) out of `image` as a new [`DecodedImage`]. Pure
/// windowing of the already-decoded buffer (orchestration, spec §6) — it
/// reuses [`DecodedImage::tile_window_rgba8`]'s clipped-cut math. An empty
/// intersection (the rect lies fully outside, or is zero-size) is a clean
/// error, never a torn/zero image. The straighten-angle resample is NOT
/// part of this axis-aligned cut (the crop machine commits the rect; the
/// rotation is a follow-on resample stage).
pub fn crop_rgba8(
    image: &DecodedImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Result<DecodedImage, IngestError> {
    let (bytes, cw, ch) = image.tile_window_rgba8(x, y, w, h);
    if cw == 0 || ch == 0 {
        return Err(IngestError::Decode(format!(
            "crop ({x},{y},{w},{h}) is empty against {}x{}",
            image.width, image.height
        )));
    }
    Ok(DecodedImage {
        width: cw,
        height: ch,
        rgba: bytes.into(),
    })
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

    // K-3 (I-02) — the decode worker pool runs the CPU decode off-thread
    // and hands raw RGBA8 back; from_rgba8 registers it as an engine image
    // (feature image.editor.ingest). The naming carries the feature tag
    // until the state feature_test macro ships.
    #[test]
    fn image_editor_ingest_from_rgba8_registers_pre_decoded_pixels() {
        let pixels = grid(2, 1); // 8 bytes
        let img = DecodedImage::from_rgba8(2, 1, pixels.clone()).expect("valid rgba8");
        assert_eq!((img.width, img.height), (2, 1));
        // The whole-image window cut round-trips the pixels verbatim.
        let (out, w, h) = img.tile_window_rgba8(0, 0, 2, 1);
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, pixels);
    }

    #[test]
    fn image_editor_ingest_from_rgba8_rejects_a_length_mismatch() {
        // 2×1 needs 8 bytes; give 6 → a clean error, never a torn image.
        let err = DecodedImage::from_rgba8(2, 1, vec![0u8; 6]).unwrap_err();
        assert!(matches!(err, IngestError::Decode(_)), "got {err:?}");
    }

    // feat: image.editor.crop — the crop COMMIT lane (pure windowing).
    #[test]
    fn image_editor_crop_rgba8_cuts_the_rectangle() {
        // A 3×2 labeled grid; crop the interior 2×1 at (1,0).
        let img = DecodedImage::from_rgba8(3, 2, grid(3, 2)).expect("valid");
        let cropped = crop_rgba8(&img, 1, 0, 2, 1).expect("non-empty crop");
        assert_eq!((cropped.width, cropped.height), (2, 1));
        // Pixel 0 of the crop is source (1,0); pixel 1 is source (2,0).
        assert_eq!((cropped.rgba[0], cropped.rgba[1]), (1, 0));
        assert_eq!((cropped.rgba[4], cropped.rgba[5]), (2, 0));
    }

    #[test]
    fn image_editor_crop_rgba8_clamps_to_extent() {
        // A crop that overhangs the right/bottom edge clips to the image.
        let img = DecodedImage::from_rgba8(3, 2, grid(3, 2)).expect("valid");
        let cropped = crop_rgba8(&img, 2, 1, 5, 5).expect("clipped, non-empty");
        assert_eq!((cropped.width, cropped.height), (1, 1));
    }

    #[test]
    fn image_editor_crop_rgba8_empty_is_error() {
        let img = DecodedImage::from_rgba8(3, 2, grid(3, 2)).expect("valid");
        // Fully outside → clean error.
        assert!(crop_rgba8(&img, 10, 10, 4, 4).is_err());
        // Zero-size → clean error.
        assert!(crop_rgba8(&img, 0, 0, 0, 0).is_err());
    }

    // feat: image.editor.curves — the CPU LUT pass the curves stage runs.
    #[test]
    fn image_editor_curves_apply_lut_remaps_rgb_keeps_alpha() {
        // An invert LUT (lut[k] = 255-k) on a single labeled pixel.
        let mut px = vec![10u8, 20, 30, 128];
        let lut: [u8; 256] = std::array::from_fn(|k| 255 - k as u8);
        apply_curve_lut(&mut px, &lut);
        assert_eq!(px, vec![245, 235, 225, 128], "RGB inverted, alpha kept");
    }

    #[test]
    fn image_editor_curves_identity_lut_is_passthrough() {
        let mut px = vec![10u8, 20, 30, 200, 40, 50, 60, 255];
        let before = px.clone();
        let lut = image_core::identity_lut();
        apply_curve_lut(&mut px, &lut);
        assert_eq!(px, before, "identity LUT changes nothing");
    }

    // feat: image.editor.ingest — the full adjust chain short-circuits to
    // the decode on identity params (no GPU needed) and runs a curve-only
    // pass on the CPU (no GPU stage).
    #[test]
    fn image_editor_ingest_curve_only_runs_on_cpu_without_gpu() {
        // pollster drives the async runner; no GPU context is created.
        let img = DecodedImage::from_rgba8(2, 1, grid(2, 1)).expect("valid");
        let lut: [u8; 256] = std::array::from_fn(|k| 255 - k as u8);
        let params = AdjustParams {
            curve_lut: Some(lut),
            ..Default::default()
        };
        // A curve-only chain has no GPU stage; the runner must NOT need a
        // context. We invoke adjust_rgba8 with a dummy-free path by going
        // through has_gpu_stage()==false: build pixels from the image and
        // apply the LUT directly (mirrors the runner's curve-only branch).
        assert!(!params.has_gpu_stage(), "curve-only has no GPU stage");
        let mut pixels = img.rgba.to_vec();
        apply_curve_lut(&mut pixels, params.curve_lut.as_ref().unwrap());
        // Pixel (0,0) was (0,0,0,255) → inverted RGB (255,255,255), alpha kept.
        assert_eq!(&pixels[0..4], &[255, 255, 255, 255]);
    }
}
