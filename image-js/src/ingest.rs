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
use image_codecs::{ImageSource, JpegSource, MemoryByteSource, PngSource};
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
    Ok(DecodedImage {
        width: w,
        height: h,
        rgba: rgba.into(),
    })
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
