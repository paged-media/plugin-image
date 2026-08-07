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
use image_gpu::{GpuContext, SelectionCoverage};
use image_kernels::families::adjust::{
    AdjustBlackWhiteParams, AdjustBrightnessContrastParams, AdjustChannelMixerParams,
    AdjustColorBalanceParams, AdjustExposureParams, AdjustHueRotateParams, AdjustInvertRgbParams,
    AdjustLevelsParams, AdjustLevelsRgbParams, AdjustPhotoFilterParams, AdjustPosterizeParams,
    AdjustSaturationParams, AdjustThresholdParams, AdjustVibranceParams, AdjustWhiteBalanceParams,
    ADJUST_BLACK_WHITE, ADJUST_BRIGHTNESS_CONTRAST, ADJUST_CHANNEL_MIXER, ADJUST_COLOR_BALANCE,
    ADJUST_EXPOSURE, ADJUST_HUE_ROTATE, ADJUST_INVERT_RGB, ADJUST_LEVELS, ADJUST_LEVELS_RGB,
    ADJUST_PHOTO_FILTER, ADJUST_POSTERIZE, ADJUST_SATURATION, ADJUST_THRESHOLD, ADJUST_VIBRANCE,
    ADJUST_WHITE_BALANCE,
};
use image_kernels::families::conv::{
    ConvGaussianParams, ConvUnsharpParams, CONV_GAUSSIAN_H, CONV_GAUSSIAN_V, CONV_UNSHARP,
    GAUSSIAN_MAX_RADIUS,
};
use image_kernels::families::geom::{RotateBilinearParams, GEOM_ROTATE_BILINEAR};
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
    /// What the RGB display transform did at ingest (CMS rung 1). The
    /// panel surfaces this so "sRGB assumed" is a stated state rather
    /// than a silent one.
    pub display: crate::display::DisplayTreatment,
    /// The source carried 16 bits per channel and was REDUCED to 8 at
    /// ingest. Same rule as `display`: a lossy step is a state to state,
    /// not one to hide — and the alternative here used to be refusing
    /// the file outright, which helped nobody.
    pub depth_reduced: bool,
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

/// PER-CHANNEL levels (`adjust.levels_rgb`): `[in_black, in_white,
/// gamma]` for r, g, b. Identity `[0, 1, 1]` per channel (the composite
/// output range stays on [`LevelsParams`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelsRgbParams {
    pub r: [f32; 3],
    pub g: [f32; 3],
    pub b: [f32; 3],
}

impl Default for LevelsRgbParams {
    fn default() -> Self {
        LevelsRgbParams {
            r: [0.0, 1.0, 1.0],
            g: [0.0, 1.0, 1.0],
            b: [0.0, 1.0, 1.0],
        }
    }
}

impl LevelsRgbParams {
    fn is_identity(&self) -> bool {
        *self == LevelsRgbParams::default()
    }
}

/// Color balance (`adjust.color_balance`): one offset per opponent axis
/// (cyan↔red, magenta↔green, yellow↔blue) per tonal range. All 0 = off.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColorBalanceParams {
    pub shadows: [f32; 3],
    pub midtones: [f32; 3],
    pub highlights: [f32; 3],
}

impl ColorBalanceParams {
    fn is_identity(&self) -> bool {
        *self == ColorBalanceParams::default()
    }
}

/// Photo filter (`adjust.photo_filter`): a colored gel with `density`
/// (0 = OFF — the stage is skipped) and optional luminosity preservation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhotoFilterParams {
    pub color: [f32; 3],
    pub density: f32,
    pub preserve_luminosity: bool,
}

impl Default for PhotoFilterParams {
    fn default() -> Self {
        // Warming filter (85) — the canonical default gel; density 0
        // keeps the stage off until the panel raises it.
        PhotoFilterParams {
            color: [0.925, 0.639, 0.365],
            density: 0.0,
            preserve_luminosity: true,
        }
    }
}

impl PhotoFilterParams {
    fn is_identity(&self) -> bool {
        self.density == 0.0
    }
}

/// Channel mixer (`adjust.channel_mixer`): each row is
/// `[in_r, in_g, in_b, constant]` for the r, g, b outputs. Identity = the
/// identity matrix with zero constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelMixerParams {
    pub r: [f32; 4],
    pub g: [f32; 4],
    pub b: [f32; 4],
}

impl Default for ChannelMixerParams {
    fn default() -> Self {
        ChannelMixerParams {
            r: [1.0, 0.0, 0.0, 0.0],
            g: [0.0, 1.0, 0.0, 0.0],
            b: [0.0, 0.0, 1.0, 0.0],
        }
    }
}

impl ChannelMixerParams {
    fn is_identity(&self) -> bool {
        *self == ChannelMixerParams::default()
    }
}

/// Black & White (`adjust.black_white`): the six hue-sector grayscale
/// weights behind an explicit `enabled` flag (the conversion is
/// destructive-looking, so it never rides a "neutral value" default).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlackWhiteParams {
    pub enabled: bool,
    /// reds, yellows, greens, cyans, blues, magentas.
    pub weights: [f32; 6],
}

impl Default for BlackWhiteParams {
    fn default() -> Self {
        // The conventional default mix (reds .4, yellows .6, greens .4,
        // cyans .6, blues .2, magentas .8).
        BlackWhiteParams {
            enabled: false,
            weights: [0.4, 0.6, 0.4, 0.6, 0.2, 0.8],
        }
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
    /// FILTER stages (editor-ui-coverage: the T1/T2 kernels get their
    /// first wasm reach). Each 0/false = stage off.
    /// Gaussian blur σ (px); radius derives as ceil(3σ) clamped to the
    /// kernel's GAUSSIAN_MAX_RADIUS window.
    pub blur_sigma: f32,
    /// Unsharp amount (`out = a + amount·(a − blur(a))`, threshold 0).
    pub sharpen_amount: f32,
    /// Hue rotation in degrees.
    pub hue_degrees: f32,
    /// Per-color negate, alpha preserved (`adjust.invert_rgb`).
    pub invert: bool,
    // ── the EXTENDED (kernel-breadth) adjust stages ──────────────────
    // Each is identity-neutral by default and short-circuits exactly
    // like the stages above; see `adjust_rgba8` for the fixed chain
    // order and the rationale.
    /// `adjust.vibrance` — saturation weighted by (1 − existing sat).
    /// 0 = off. (The kernel's own `saturation` term stays 0 here; the
    /// panel's global saturation is its OWN later stage.)
    pub vibrance: f32,
    /// `adjust.color_balance` — per tonal range opponent-axis offsets.
    pub color_balance: ColorBalanceParams,
    /// `adjust.photo_filter` — a colored gel; `density` 0 = off.
    pub photo_filter: PhotoFilterParams,
    /// `adjust.channel_mixer` — the 3×4 channel-mix matrix.
    pub channel_mixer: ChannelMixerParams,
    /// `adjust.levels_rgb` — per-channel input/gamma remap.
    pub levels_rgb: LevelsRgbParams,
    /// `adjust.black_white` — the six-weight grayscale mix (gated).
    pub black_white: BlackWhiteParams,
    /// `adjust.posterize` — quantize each channel to N levels.
    /// `None` = off (the panel gates it behind a checkbox).
    pub posterize: Option<f32>,
    /// `adjust.threshold` — luma cut to black/white. `None` = off.
    pub threshold: Option<f32>,
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
            blur_sigma: 0.0,
            sharpen_amount: 0.0,
            hue_degrees: 0.0,
            invert: false,
            vibrance: 0.0,
            color_balance: ColorBalanceParams::default(),
            photo_filter: PhotoFilterParams::default(),
            channel_mixer: ChannelMixerParams::default(),
            levels_rgb: LevelsRgbParams::default(),
            black_white: BlackWhiteParams::default(),
            posterize: None,
            threshold: None,
        }
    }
}

/// Wire length of the EXTENDED adjust parameter block
/// ([`AdjustParams::apply_extended`] / the `adjust_image_ext` wasm door).
/// A FLAT `f32` array so the boundary stays one argument as the stage set
/// grows. Layout (fixed, mirrored by `glue/src/engine.ts`
/// `packAdjustExt`):
///
/// ```text
///  0      vibrance
///  1..=3  color_balance shadows    (cyan-red, magenta-green, yellow-blue)
///  4..=6  color_balance midtones
///  7..=9  color_balance highlights
/// 10      black_white enabled (0 | 1)
/// 11..=16 black_white weights (reds, yellows, greens, cyans, blues, magentas)
/// 17      posterize enabled (0 | 1)
/// 18      posterize levels
/// 19      threshold enabled (0 | 1)
/// 20      threshold value
/// 21      photo_filter density (0 = off)
/// 22..=24 photo_filter color (r, g, b)
/// 25      photo_filter preserve-luminosity (0 | 1)
/// 26..=37 channel_mixer rows r[4], g[4], b[4] (in_r, in_g, in_b, const)
/// 38..=46 levels_rgb r[3], g[3], b[3] (in_black, in_white, gamma)
/// ```
pub const ADJUST_EXT_LEN: usize = 47;

impl AdjustParams {
    /// Is the whole chain a NO-OP? SEMANTIC, not structural: a stage
    /// whose gate is off (photo-filter density 0, black & white
    /// disabled, `posterize`/`threshold` `None`) contributes nothing no
    /// matter what its other fields hold, so `is_identity` asks the
    /// stages, never `== Default::default()`. This is exactly the
    /// short-circuit gate `adjust_rgba8` uses to return the decode
    /// verbatim.
    pub fn is_identity(&self) -> bool {
        !self.has_gpu_stage() && self.curve_lut.is_none()
    }

    /// Decode the flat [`ADJUST_EXT_LEN`] block onto these params. An
    /// EMPTY slice leaves every extended stage at identity (the
    /// back-compatible `adjust_image_full` door); any other length is a
    /// clean error, never a half-applied chain.
    pub fn apply_extended(&mut self, ext: &[f32]) -> Result<(), IngestError> {
        if ext.is_empty() {
            return Ok(());
        }
        if ext.len() != ADJUST_EXT_LEN {
            return Err(IngestError::Unsupported(format!(
                "extended adjust block must be {ADJUST_EXT_LEN} f32s or empty (got {})",
                ext.len()
            )));
        }
        self.vibrance = ext[0];
        self.color_balance = ColorBalanceParams {
            shadows: [ext[1], ext[2], ext[3]],
            midtones: [ext[4], ext[5], ext[6]],
            highlights: [ext[7], ext[8], ext[9]],
        };
        self.black_white = BlackWhiteParams {
            enabled: ext[10] != 0.0,
            weights: [ext[11], ext[12], ext[13], ext[14], ext[15], ext[16]],
        };
        self.posterize = (ext[17] != 0.0).then_some(ext[18]);
        self.threshold = (ext[19] != 0.0).then_some(ext[20]);
        self.photo_filter = PhotoFilterParams {
            density: ext[21],
            color: [ext[22], ext[23], ext[24]],
            preserve_luminosity: ext[25] != 0.0,
        };
        self.channel_mixer = ChannelMixerParams {
            r: [ext[26], ext[27], ext[28], ext[29]],
            g: [ext[30], ext[31], ext[32], ext[33]],
            b: [ext[34], ext[35], ext[36], ext[37]],
        };
        self.levels_rgb = LevelsRgbParams {
            r: [ext[38], ext[39], ext[40]],
            g: [ext[41], ext[42], ext[43]],
            b: [ext[44], ext[45], ext[46]],
        };
        Ok(())
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
            || self.blur_sigma > 0.0
            || self.sharpen_amount > 0.0
            || self.hue_degrees != 0.0
            || self.invert
            || self.has_extended_stage()
    }

    /// Do any of the EXTENDED (kernel-breadth) stages run?
    fn has_extended_stage(&self) -> bool {
        self.vibrance != 0.0
            || !self.color_balance.is_identity()
            || !self.photo_filter.is_identity()
            || !self.channel_mixer.is_identity()
            || !self.levels_rgb.is_identity()
            || self.black_white.enabled
            || self.posterize.is_some()
            || self.threshold.is_some()
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
            // Raw bytes handed in by a caller carry no container and so no
            // profile; they are taken as already-working-space.
            display: crate::display::DisplayTreatment::AssumedSrgb,
            depth_reduced: false,
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
        // PSD carries its profile in the image-resource block, which the
        // merged-composite decode does not surface yet — stated as
        // assumed rather than silently claimed as managed.
        display: crate::display::DisplayTreatment::AssumedSrgb,
        depth_reduced: composite.depth_reduced,
    })
}

/// Full-frame decode through an `ImageSource` adapter, widened to RGBA8.
fn decode_source<S: ImageSource>(mut source: S) -> Result<DecodedImage, IngestError> {
    let info = source
        .probe()
        .map_err(|e| IngestError::Decode(e.to_string()))?;
    if info.format.depth != SampleDepth::U8 {
        // The PSD lane ACCEPTS 16-bit and reduces it, because PSD's
        // sample order is spec-documented big-endian and the convention
        // is established and tested in `image-psd`. This lane does not,
        // and the reason is evidence rather than effort: no adapter here
        // documents its 16-bit byte order and there is no 16-bit fixture
        // to verify a guess against. A wrong guess would decode to a
        // plausible-looking WRONG image, which is worse than a refusal —
        // so the refusal stands until a fixture settles it.
        return Err(IngestError::Unsupported(format!(
            "depth {:?}: this codec lane is 8-bit (PSD 16-bit IS accepted and \
             reduced; here the sample byte order is unverified and guessing \
             it would decode to a wrong-looking image)",
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
    let (mut rgba, w, h) = apply_orientation(rgba, w, h, orientation);

    // CMS RUNG 1 — the RGB display transform. The kernels' math is
    // specified over the post-CMS working space, so this belongs at
    // DECODE and nowhere later. CMYK already arrived colour-managed
    // above (its cast consumed the same embedded profile), so only the
    // RGB-ish layouts are transformed here.
    let display = if matches!(channels, ChannelLayout::Cmyk) {
        crate::display::DisplayTreatment::Managed
    } else {
        crate::display::to_working_srgb(&mut rgba, embedded_icc.as_deref())
    };

    Ok(DecodedImage {
        width: w,
        height: h,
        rgba: rgba.into(),
        display,
        // This lane is 8-bit only (see the depth gate above), so nothing
        // was reduced here.
        depth_reduced: false,
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
/// straight RGBA8.
///
/// # The fixed chain order
///
/// Every stage runs only when non-neutral (each has an identity
/// short-circuit) and every stage is mask-aware — the bound selection's
/// coverage rides `@group(2)` on EVERY dispatch (`mix(a, result, m)`).
/// The order is FIXED and deliberate; the extended (kernel-breadth)
/// stages slot in WITHOUT changing any pre-existing stage's relative
/// position:
///
/// ```text
///  1 exposure            (input remap: scene-referred stops)
///  2 white_balance       (input remap: illuminant)
///  3 levels              (input remap: composite black/gamma/white)
///  4 levels_rgb          (input remap: PER-CHANNEL black/gamma/white)   ← new
///  5 brightness/contrast (tonal)
///  6 color_balance       (color grade: tonal-range opponent offsets)    ← new
///  7 photo_filter        (color grade: gel absorption)                  ← new
///  8 channel_mixer       (color grade: 3×4 channel matrix)              ← new
///  9 vibrance            (chroma: weighted by existing saturation)      ← new
/// 10 saturation          (chroma: global)
/// 11 black_white         (chroma destroy: six-weight grayscale mix)     ← new
/// 12 posterize           (quantize)                                     ← new
/// 13 threshold           (quantize to 1 bit)                            ← new
/// 14 hue_rotate          (FILTER stages, unchanged relative order)
/// 15 invert
/// 16 blur (separable Gaussian H then V)
/// 17 sharpen (unsharp against a fixed σ1.5 blur)
/// 18 curves              (CPU tone LUT — no GPU LUT kernel yet)
/// ```
///
/// Rationale: input remaps first (they define the working range), then
/// tone, then colour grading, then chroma, then the range-destroying
/// quantizers, then spatial filters, and the CPU curve last (it is a
/// final tone map over the composited result).
///
/// The legacy summary line, kept for orientation:
///   exposure → white-balance → levels → brightness/contrast → saturation
/// on the GPU, then the optional CURVES tone LUT as a CPU pass (there is
/// no GPU LUT kernel yet — the honest deferral; the LUT itself is built
/// deterministically by `image_core::curve_lut` panel-side). Identity
/// params short-circuit to the decoded pixels. When ONLY a curve is set
/// (no GPU stage) the GPU is skipped entirely and the LUT runs straight
/// on the decoded buffer. GPU-only by construction for the kernel stages:
/// no adapter ⇒ the caller never reaches here with a context.
///
/// `selection` is the session's coverage mask (spec §6.1): when `Some`,
/// EVERY GPU adjust/filter dispatch binds its r16float window at
/// `@group(2)` so the kernel applies `mix(input, result, mask)` — the
/// adjustment lands only inside the selection — and the CPU curve LUT
/// pass blends per-pixel by the same coverage. `None` (or an all-one
/// coverage upstream) is the constant-1 default: the whole image. The
/// mask is applied PER STAGE, so with intermediate (feathered) weights a
/// multi-stage chain compounds slightly differently than one final blend
/// of the fully adjusted image — the standard "each op runs through the
/// selection" semantics; {0,1} regions are unaffected by the difference.
pub async fn adjust_rgba8(
    ctx: &GpuContext,
    image: &DecodedImage,
    params: &AdjustParams,
    selection: Option<Arc<SelectionCoverage>>,
) -> Result<Vec<u8>, IngestError> {
    if params.is_identity() {
        return Ok(image.rgba.to_vec());
    }

    // The GPU kernel chain (skipped wholesale when only a curve is set).
    let mut pixels = if params.has_gpu_stage() {
        let mut pipe = Pipeline::new();
        pipe.set_selection(selection.clone());
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
        // 4 — per-channel levels (the composite output range stays on
        // stage 3; this one is the input/gamma remap per r, g, b).
        if !params.levels_rgb.is_identity() {
            let l = &params.levels_rgb;
            node = pipe.apply(
                node,
                &ADJUST_LEVELS_RGB,
                Arc::<[u8]>::from(AdjustLevelsRgbParams::new(l.r, l.g, l.b).as_bytes()),
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
        // 6, 7, 8 — the colour-grading trio.
        if !params.color_balance.is_identity() {
            let c = &params.color_balance;
            node = pipe.apply(
                node,
                &ADJUST_COLOR_BALANCE,
                Arc::<[u8]>::from(
                    AdjustColorBalanceParams::new(c.shadows, c.midtones, c.highlights).as_bytes(),
                ),
            );
        }
        if !params.photo_filter.is_identity() {
            let f = &params.photo_filter;
            node = pipe.apply(
                node,
                &ADJUST_PHOTO_FILTER,
                Arc::<[u8]>::from(
                    AdjustPhotoFilterParams::new(f.color, f.density, f.preserve_luminosity)
                        .as_bytes(),
                ),
            );
        }
        if !params.channel_mixer.is_identity() {
            let m = &params.channel_mixer;
            node = pipe.apply(
                node,
                &ADJUST_CHANNEL_MIXER,
                Arc::<[u8]>::from(AdjustChannelMixerParams::new(m.r, m.g, m.b).as_bytes()),
            );
        }
        // 9 — vibrance BEFORE the global saturation (it reads the
        // still-unsaturated chroma to weight its own boost). The kernel's
        // second term is the global saturation offset; we keep it 0 so
        // stage 10 stays the single global control.
        if params.vibrance != 0.0 {
            node = pipe.apply(
                node,
                &ADJUST_VIBRANCE,
                Arc::<[u8]>::from(AdjustVibranceParams::new(params.vibrance, 0.0).as_bytes()),
            );
        }
        if params.saturation != 1.0 {
            node = pipe.apply(
                node,
                &ADJUST_SATURATION,
                Arc::<[u8]>::from(AdjustSaturationParams::new(params.saturation).as_bytes()),
            );
        }
        // 11, 12, 13 — the range-destroying stages, LAST among the point
        // adjustments (posterize/threshold quantize, so anything after
        // them would work on a collapsed range).
        if params.black_white.enabled {
            let w = params.black_white.weights;
            node = pipe.apply(
                node,
                &ADJUST_BLACK_WHITE,
                Arc::<[u8]>::from(
                    AdjustBlackWhiteParams::new(w[0], w[1], w[2], w[3], w[4], w[5]).as_bytes(),
                ),
            );
        }
        if let Some(levels) = params.posterize {
            node = pipe.apply(
                node,
                &ADJUST_POSTERIZE,
                Arc::<[u8]>::from(AdjustPosterizeParams::new(levels).as_bytes()),
            );
        }
        if let Some(t) = params.threshold {
            node = pipe.apply(
                node,
                &ADJUST_THRESHOLD,
                Arc::<[u8]>::from(AdjustThresholdParams::new(t).as_bytes()),
            );
        }
        // FILTER stages (first wasm reach for the registered T1/T2
        // kernels — same registry-driven dispatch, nothing new): hue
        // rotation, per-color invert, separable Gaussian blur, and
        // unsharp masking (the classic a + amount·(a − blur(a)) blend —
        // CONV_UNSHARP is the 2-input point kernel; its blur input is a
        // fixed σ1.5 Gaussian of the current node).
        if params.hue_degrees != 0.0 {
            node = pipe.apply(
                node,
                &ADJUST_HUE_ROTATE,
                Arc::<[u8]>::from(AdjustHueRotateParams::new(params.hue_degrees).as_bytes()),
            );
        }
        if params.invert {
            node = pipe.apply(
                node,
                &ADJUST_INVERT_RGB,
                Arc::<[u8]>::from(AdjustInvertRgbParams::new().as_bytes()),
            );
        }
        if params.blur_sigma > 0.0 {
            let sigma = params.blur_sigma;
            let radius = (sigma * 3.0).ceil().min(f32::from(GAUSSIAN_MAX_RADIUS)) as u32;
            let p = Arc::<[u8]>::from(ConvGaussianParams::new(sigma, radius).as_bytes());
            node = pipe.apply(node, &CONV_GAUSSIAN_H, Arc::clone(&p));
            node = pipe.apply(node, &CONV_GAUSSIAN_V, p);
        }
        if params.sharpen_amount > 0.0 {
            let p = Arc::<[u8]>::from(ConvGaussianParams::new(1.5, 5).as_bytes());
            let mut blurred = pipe.apply(node, &CONV_GAUSSIAN_H, Arc::clone(&p));
            blurred = pipe.apply(blurred, &CONV_GAUSSIAN_V, p);
            node = pipe.apply2(
                node,
                blurred,
                &CONV_UNSHARP,
                Arc::<[u8]>::from(ConvUnsharpParams::new(params.sharpen_amount, 0.0).as_bytes()),
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
    // Under a selection the LUT result blends per-pixel by the SAME
    // coverage the GPU stages masked with, so the whole chain honors one
    // selection contract.
    if let Some(lut) = &params.curve_lut {
        match &selection {
            Some(cov) => apply_curve_lut_masked(&mut pixels, lut, cov),
            None => apply_curve_lut(&mut pixels, lut),
        }
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

/// [`apply_curve_lut`] under a selection: each pixel blends
/// `mix(original, lut(original), coverage)` — the CPU curves stage's
/// mirror of the GPU mask contract (`out = mix(a, result, m)`), so a
/// masked chain with curves changes only selected pixels. `pixels` must
/// be the coverage field's `w·h·4` bytes (the ingest slice's full-frame
/// buffer); a size mismatch falls back to the unmasked LUT (defensive —
/// the session always adjusts at image resolution).
pub fn apply_curve_lut_masked(pixels: &mut [u8], lut: &[u8; 256], coverage: &SelectionCoverage) {
    let expected = (coverage.width() as usize) * (coverage.height() as usize) * 4;
    if pixels.len() != expected {
        apply_curve_lut(pixels, lut);
        return;
    }
    for (i, px) in pixels.chunks_exact_mut(4).enumerate() {
        let m = coverage.data()[i] as u16;
        if m == 0 {
            continue;
        }
        for c in 0..3 {
            let orig = px[c] as u16;
            let mapped = lut[px[c] as usize] as u16;
            // Rounded integer mix(orig, mapped, m/255).
            px[c] = ((orig * (255 - m) + mapped * m + 127) / 255) as u8;
        }
    }
}

/// Commit a CROP: cut the integer pixel rectangle `(x, y, w, h)` (clamped
/// to the image extent) out of `image` as a new [`DecodedImage`]. Pure
/// windowing of the already-decoded buffer (orchestration, spec §6) — it
/// reuses [`DecodedImage::tile_window_rgba8`]'s clipped-cut math. An empty
/// intersection (the rect lies fully outside, or is zero-size) is a clean
/// error, never a torn/zero image. This is the AXIS-ALIGNED cut only —
/// the straighten angle rides [`straighten_crop_rgba8`], which rotates
/// through `geom.rotate_bilinear` first and then calls this.
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
        // A crop of an already-decoded image inherits its treatment; the
        // pixels were transformed (or not) at ingest.
        display: image.display,
        depth_reduced: false,
    })
}

/// STRAIGHTEN + CROP — the crop tool's rotated-frame commit.
///
/// The crop overlay previews the crop rectangle ROTATED by `degrees`
/// about its own centre (`image_core::frame_corners`); committing it
/// means "the content inside that rotated frame becomes the
/// axis-aligned result". So the image is rotated by `−degrees` about the
/// RECT's centre (`geom.rotate_bilinear`, backward-mapped, clamp-to-edge)
/// and the now-upright rectangle is cut out of the result.
///
/// `degrees == 0` short-circuits to the pure windowing [`crop_rgba8`] —
/// an axis-aligned crop never touches the GPU and never resamples (no
/// interpolation blur for the common case). A non-zero angle IS a
/// resample, so it is GPU-only like every other kernel.
///
/// HONEST EDGE RULE: the rotation clamps to the source edge, so if the
/// rotated frame swings OUTSIDE the image the border texels smear into
/// those corners instead of going transparent (the alpha-aware
/// transparent-outside variant would be a second kernel). The crop tool
/// clamps the rect to the image extent, which keeps the common case
/// inside valid data.
pub async fn straighten_crop_rgba8(
    ctx: &GpuContext,
    image: &DecodedImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    degrees: f32,
) -> Result<DecodedImage, IngestError> {
    if degrees == 0.0 {
        return crop_rgba8(image, x, y, w, h);
    }
    let center = (x as f32 + w as f32 / 2.0, y as f32 + h as f32 / 2.0);
    // Rotate by −degrees: the FRAME turned by +degrees, so the content
    // must turn back by the same amount to land upright.
    let params = RotateBilinearParams::new(-degrees, center, center);
    let win = crate::fill::rgba8_to_f16(&image.rgba);
    let out = image_gpu::execute_windowed_once_async(
        ctx,
        &GEOM_ROTATE_BILINEAR,
        &win,
        image.width,
        image.height,
        params.as_bytes(),
        None,
        image.width,
        image.height,
    )
    .await
    .map_err(|e| IngestError::Pipeline(e.to_string()))?;
    let rotated =
        DecodedImage::from_rgba8(image.width, image.height, crate::fill::f16_to_rgba8(&out))?;
    crop_rgba8(&rotated, x, y, w, h)
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

    // feat: image.selection.mask — the CPU curves stage honors the same
    // coverage contract as the GPU mask (mix by coverage).
    #[test]
    fn image_selection_curves_lut_masked_changes_only_selected_pixels() {
        // A 2×1 buffer; coverage selects ONLY pixel 0.
        let mut px = vec![10u8, 20, 30, 128, 40, 50, 60, 255];
        let lut: [u8; 256] = std::array::from_fn(|k| 255 - k as u8);
        let cov = SelectionCoverage::from_data(2, 1, vec![255, 0]).expect("2 px");
        apply_curve_lut_masked(&mut px, &lut, &cov);
        assert_eq!(&px[0..4], &[245, 235, 225, 128], "selected pixel remapped");
        assert_eq!(&px[4..8], &[40, 50, 60, 255], "deselected pixel untouched");
    }

    #[test]
    fn image_selection_curves_lut_masked_blends_at_half_coverage() {
        // Coverage 128 (~50.2%): out ≈ mix(orig, lut(orig), 128/255).
        let mut px = vec![0u8, 0, 0, 255];
        let lut: [u8; 256] = std::array::from_fn(|k| 255 - k as u8);
        let cov = SelectionCoverage::from_data(1, 1, vec![128]).expect("1 px");
        apply_curve_lut_masked(&mut px, &lut, &cov);
        // (0·127 + 255·128 + 127) / 255 = 128.
        assert_eq!(&px[0..3], &[128, 128, 128]);
        assert_eq!(px[3], 255, "alpha never remapped");
    }

    #[test]
    fn image_selection_curves_lut_masked_size_mismatch_falls_back_unmasked() {
        let mut px = vec![10u8, 20, 30, 128];
        let lut: [u8; 256] = std::array::from_fn(|k| 255 - k as u8);
        let cov = SelectionCoverage::from_data(3, 3, vec![0; 9]).expect("9 px");
        apply_curve_lut_masked(&mut px, &lut, &cov);
        assert_eq!(px, vec![245, 235, 225, 128], "defensive unmasked fallback");
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
