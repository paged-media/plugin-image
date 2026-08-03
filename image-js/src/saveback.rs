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

//! SAVE-BACK — turn the ADJUSTED full-resolution pixels into file bytes.
//!
//! Two lanes, one contract (`bytes in → bytes out`, no host I/O here —
//! the bundle hands the result to the exporter registry):
//!
//! * **PSD** ([`psd_write_adjusted`]) — the composite is re-encoded into
//!   the retained parse's merged-image section, and the LAYER structure
//!   is handled per [`PsdSaveBackShape`]. Everything the model does not
//!   touch (resources, ICC, unmodeled blocks) still rides the
//!   preservation writer verbatim.
//! * **PNG / JPEG** ([`encode_rgba8`]) — a straight re-encode through the
//!   `image-codecs` targets (CPU entropy coding, spec §1).
//!
//! # HONEST SCOPE (stated in the panel string, never silently)
//!
//! * PSD save-back is **8-bit RGB only**. 16/32-bit, Grayscale, CMYK,
//!   Lab and Indexed answer a clean `Unsupported` — the same cut the
//!   composite DECODE already declares.
//! * PSD save-back is **single-layer / flattened**. A file whose only
//!   content layer already covers the canvas gets that layer's channels
//!   replaced in place ([`PsdSaveBackShape::LayerReplaced`] — the
//!   `replace_channel_pixels` path, nothing else moves). A MULTI-layer
//!   file cannot have adjusted pixels attributed to any one layer
//!   without a layer graph (a recorded deferral), so it is flattened
//!   into a NEW single-layer PSD ([`PsdSaveBackShape::Flattened`]) —
//!   announced in the UI, never silent.
//! * Spot / extra channels beyond RGB+alpha are dropped by the flatten
//!   (the header channel count is normalized to what we actually write).
//! * The **zero-edit guarantee is untouched**: nothing here runs unless
//!   the user explicitly asks for a save-back, so a plain PSD export is
//!   still byte-identical.

use image_codecs::{ImageTarget, JpegTarget, PngTarget, TargetInfo};
use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileSliceRef, Transfer,
};
use image_psd::model::{
    BlendRanges, ChannelData, ChannelInfo, ColorMode, Compression, GlobalImageData, LayerRecord,
    PascalString, PsdFile,
};

use crate::ingest::IngestError;

/// What the PSD save-back did to the layer structure — the panel turns
/// this into the sentence the user reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsdSaveBackShape {
    /// The file's single canvas-sized content layer had its channel
    /// pixels replaced (`image_psd::edit::replace_channel_pixels`); every
    /// other record, resource and unmodeled block is untouched.
    LayerReplaced,
    /// The file had multiple layers, or no addressable canvas-sized
    /// content layer at all (a layer that does not cover the canvas,
    /// channels we cannot address, or an extra-channel header): the
    /// adjusted composite was written into a NEW single-layer PSD. Any
    /// original layer structure is GONE — the caller MUST say so.
    Flattened,
}

impl PsdSaveBackShape {
    /// The user-facing sentence (the panel/status string).
    pub fn describe(self) -> &'static str {
        match self {
            PsdSaveBackShape::LayerReplaced => {
                "the adjusted pixels were written into the file's single content layer \
                 (layer structure preserved)"
            }
            PsdSaveBackShape::Flattened => {
                "the adjusted composite was FLATTENED into a NEW single-layer PSD — \
                 any original layer structure is NOT in this file"
            }
        }
    }
}

/// The raster re-encode formats the non-PSD lane offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterFormat {
    Png,
    Jpeg,
}

impl RasterFormat {
    pub fn from_wire(s: &str) -> Option<RasterFormat> {
        Some(match s {
            "png" => RasterFormat::Png,
            "jpeg" | "jpg" => RasterFormat::Jpeg,
            _ => return None,
        })
    }

    pub fn extension(self) -> &'static str {
        match self {
            RasterFormat::Png => ".png",
            RasterFormat::Jpeg => ".jpg",
        }
    }

    pub fn mime(self) -> &'static str {
        match self {
            RasterFormat::Png => "image/png",
            RasterFormat::Jpeg => "image/jpeg",
        }
    }
}

/// The fixed v0 JPEG quality for the save-back lane. A quality slider is
/// a follow-up; 90 is the conventional "visually lossless enough for a
/// working file" point.
pub const JPEG_QUALITY_DEFAULT: u8 = 90;

/// The straight-RGBA8 format the save-back speaks (the ingest slice's
/// working shape).
const RGBA8: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::U8,
    alpha: AlphaMode::Straight,
    transfer: Transfer::Linear,
    space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
};

/// Re-encode straight RGBA8 as PNG or JPEG through the codec targets.
/// One full-frame strip (the targets accumulate and encode at `finish`).
pub fn encode_rgba8(
    rgba: &[u8],
    width: u32,
    height: u32,
    format: RasterFormat,
) -> Result<Vec<u8>, IngestError> {
    let expected = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected {
        return Err(IngestError::Decode(format!(
            "encode: {} bytes for {width}x{height} (expected {expected})",
            rgba.len()
        )));
    }
    let info = TargetInfo {
        width,
        height,
        format: RGBA8,
        icc: None,
    };
    let region = Region::new(0, 0, width, height);
    let slice = TileSliceRef {
        region,
        format: RGBA8,
        row_stride: width as usize * 4,
        bytes: rgba,
    };
    let err = |e: image_codecs::CodecError| IngestError::Decode(e.to_string());
    match format {
        RasterFormat::Png => {
            let mut t = PngTarget::new();
            t.begin(info).map_err(err)?;
            t.write_strip(region, &slice).map_err(err)?;
            t.finish().map_err(err)?;
            Ok(t.into_bytes())
        }
        RasterFormat::Jpeg => {
            let mut t = JpegTarget::new(JPEG_QUALITY_DEFAULT);
            t.begin(info).map_err(err)?;
            t.write_strip(region, &slice).map_err(err)?;
            t.finish().map_err(err)?;
            Ok(t.into_bytes())
        }
    }
}

/// Split interleaved straight RGBA8 into the planar channels the PSD
/// sections want: `[R, G, B]`, plus `A` when `with_alpha`.
fn planes_from_rgba8(rgba: &[u8], n: usize, with_alpha: bool) -> Vec<Vec<u8>> {
    let count = if with_alpha { 4 } else { 3 };
    let mut planes = vec![vec![0u8; n]; count];
    for (c, plane) in planes.iter_mut().enumerate() {
        for (i, out) in plane.iter_mut().enumerate().take(n) {
            *out = rgba[i * 4 + c];
        }
    }
    planes
}

/// Encode the MERGED composite section: one compression tag for ALL
/// channels, then — for RLE — a single row-count table covering
/// `channels · height` scanlines, then the packed rows in channel-major
/// order (`image_psd::composite` documents the read side).
fn encode_composite_rle(
    planes: &[Vec<u8>],
    file: &PsdFile,
    width: u32,
    height: u32,
) -> Result<GlobalImageData, IngestError> {
    let mut counts: Vec<u8> = Vec::new();
    let mut rows_out: Vec<u8> = Vec::new();
    for plane in planes {
        let cd = ChannelData::encode_rle(plane, file.container, height, width)
            .map_err(|e| IngestError::Decode(e.to_string()))?;
        // `encode_rle` emits [count table | packed rows] for ONE plane;
        // the composite section wants ALL count tables first, so split at
        // the table width (`rows · count_width`).
        let cw = match file.container {
            image_psd::Container::Psb => 4usize,
            _ => 2usize,
        };
        let split = (height as usize) * cw;
        if cd.bytes.len() < split {
            return Err(IngestError::Decode(
                "composite RLE encode produced a short count table".into(),
            ));
        }
        counts.extend_from_slice(&cd.bytes[..split]);
        rows_out.extend_from_slice(&cd.bytes[split..]);
    }
    counts.extend_from_slice(&rows_out);
    Ok(GlobalImageData {
        compression: Compression::Rle.code(),
        raw: counts,
    })
}

/// Is this record a section divider / folder marker rather than pixel
/// content? (`lsct` kinds — a folder open/closed record or its bounding
/// divider carries no meaningful canvas pixels.)
fn is_divider(layer: &LayerRecord) -> bool {
    layer.addl.iter().any(|a| a.lsct().is_some())
}

/// Build the ONE layer record a flattened save-back emits.
fn single_layer(
    planes: &[Vec<u8>],
    file_container: image_psd::Container,
    width: u32,
    height: u32,
    with_alpha: bool,
) -> Result<LayerRecord, IngestError> {
    // Conventional order: transparency (-1) first, then R/G/B (0/1/2).
    let mut ids: Vec<i16> = Vec::new();
    let mut order: Vec<usize> = Vec::new();
    if with_alpha {
        ids.push(-1);
        order.push(3);
    }
    for (k, id) in [0i16, 1, 2].iter().enumerate() {
        ids.push(*id);
        order.push(k);
    }
    let mut channels = Vec::with_capacity(ids.len());
    let mut channel_data = Vec::with_capacity(ids.len());
    for (id, plane_idx) in ids.iter().zip(order.iter()) {
        let cd = ChannelData::encode_rle(&planes[*plane_idx], file_container, height, width)
            .map_err(|e| IngestError::Decode(e.to_string()))?;
        channels.push(ChannelInfo {
            id: *id,
            data_len: 2 + cd.bytes.len() as u64,
        });
        channel_data.push(cd);
    }
    Ok(LayerRecord {
        top: 0,
        left: 0,
        bottom: height as i32,
        right: width as i32,
        channels,
        blend_sig: *b"8BIM",
        blend_key: *b"norm",
        opacity: 255,
        clipping: 0,
        flags: 0,
        filler: 0,
        mask: None,
        blend_ranges: BlendRanges::default(),
        name_legacy: PascalString::new("Adjusted"),
        addl: Vec::new(),
        extra_raw: None,
        channel_data,
    })
}

/// Write the ADJUSTED full-resolution `rgba` (straight RGBA8, row-major)
/// back into the retained PSD parse. Returns the shape the caller must
/// report to the user. The file is left ready for
/// `image_psd::PsdFile::write` (the preservation writer).
///
/// Rejects — cleanly, never a wrong-looking file — anything outside the
/// 8-bit RGB cut or a dimension mismatch against the parsed header.
pub fn psd_write_adjusted(
    file: &mut PsdFile,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<PsdSaveBackShape, IngestError> {
    if file.header.depth != 8 {
        return Err(IngestError::Unsupported(format!(
            "PSD save-back at depth {} (8-bit only)",
            file.header.depth
        )));
    }
    if file.header.color_mode != ColorMode::Rgb {
        return Err(IngestError::Unsupported(format!(
            "PSD save-back for color mode {:?} (RGB only; Grayscale/CMYK/Lab are the \
             M2 cast lane)",
            file.header.color_mode
        )));
    }
    if file.header.width != width || file.header.height != height {
        return Err(IngestError::Unsupported(format!(
            "PSD save-back size mismatch: the engine image is {width}x{height}, the PSD \
             is {}x{} (crop/resize the PSD lane is a follow-up)",
            file.header.width, file.header.height
        )));
    }
    let n = (width as usize) * (height as usize);
    if rgba.len() != n * 4 {
        return Err(IngestError::Decode(format!(
            "PSD save-back: {} bytes for {width}x{height} (expected {})",
            rgba.len(),
            n * 4
        )));
    }

    // Alpha travels only when the parse already declared a merged
    // transparency channel (the layer-count sign flag) — inventing one
    // would change how every reader interprets the extra plane.
    let with_alpha = file.header.channels >= 4 && file.layer_mask.transparency_in_merged;
    let planes = planes_from_rgba8(rgba, n, with_alpha);
    let plane_count = planes.len() as u16;

    // Can the single-layer in-place path run? It needs exactly ONE
    // content record, covering the canvas, whose channel ids we can all
    // address, and a header channel count that already matches what we
    // write (a spot-channel file cannot keep its composite consistent).
    let content: Vec<usize> = file
        .layer_mask
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| !is_divider(l))
        .map(|(i, _)| i)
        .collect();
    let in_place = content.len() == 1 && file.header.channels == plane_count && {
        let l = &file.layer_mask.layers[content[0]];
        l.top == 0
            && l.left == 0
            && l.right == width as i32
            && l.bottom == height as i32
            && l.channels.iter().all(|c| (-1..=2).contains(&c.id))
            && l.channels.len() == planes.len()
    };

    // The merged composite is rewritten in EVERY path — it is the file's
    // own render oracle, and a stale one would show the un-adjusted image
    // in every reader that trusts it.
    file.composite = encode_composite_rle(&planes, file, width, height)?;
    file.header.channels = plane_count;

    if in_place {
        let idx = content[0];
        let ids: Vec<i16> = file.layer_mask.layers[idx]
            .channels
            .iter()
            .map(|c| c.id)
            .collect();
        for (ci, id) in ids.iter().enumerate() {
            let plane = match id {
                0 => &planes[0],
                1 => &planes[1],
                2 => &planes[2],
                -1 => &planes[3],
                other => {
                    return Err(IngestError::Decode(format!(
                        "unexpected layer channel id {other} after the in-place gate"
                    )))
                }
            };
            image_psd::edit::replace_channel_pixels(
                file,
                idx,
                ci,
                plane,
                Compression::Rle,
                height,
                width,
            )
            .map_err(|e| IngestError::Decode(e.to_string()))?;
        }
        return Ok(PsdSaveBackShape::LayerReplaced);
    }

    // Flatten: ONE synthesized canvas-sized layer carrying the adjusted
    // pixels replaces the record list. The header/resources (ICC!) and
    // every document-level block survive; the layer TREE does not — the
    // caller announces that.
    let layer = single_layer(&planes, file.container, width, height, with_alpha)?;
    file.layer_mask.layers = vec![layer];
    file.layer_mask.transparency_in_merged = with_alpha;
    file.layer_mask.section_raw = None;
    Ok(PsdSaveBackShape::Flattened)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×1 8-bit RGB PSD (RAW composite, no layers) — the same shape
    /// the glue's `psdBytes()` fixture builds.
    fn flat_psd() -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"8BPS");
        b.extend_from_slice(&1u16.to_be_bytes()); // version
        b.extend_from_slice(&[0u8; 6]);
        b.extend_from_slice(&3u16.to_be_bytes()); // channels
        b.extend_from_slice(&1u32.to_be_bytes()); // height
        b.extend_from_slice(&2u32.to_be_bytes()); // width
        b.extend_from_slice(&8u16.to_be_bytes()); // depth
        b.extend_from_slice(&3u16.to_be_bytes()); // RGB
        b.extend_from_slice(&0u32.to_be_bytes()); // color mode data
        b.extend_from_slice(&0u32.to_be_bytes()); // resources
        b.extend_from_slice(&0u32.to_be_bytes()); // layer & mask
        b.extend_from_slice(&0u16.to_be_bytes()); // RAW
        b.extend_from_slice(&[10, 20, 30, 40, 50, 60]); // R, G, B planes
        b
    }

    // feat: image.editor.saveback — the PSD lane writes the adjusted
    // composite and (with no addressable single layer) flattens.
    #[test]
    fn image_editor_saveback_psd_round_trips_the_adjusted_pixels() {
        let mut file = PsdFile::parse(&flat_psd()).expect("parse");
        // Adjusted pixels: (1,2,3,255) and (4,5,6,255).
        let rgba = vec![1u8, 2, 3, 255, 4, 5, 6, 255];
        let shape = psd_write_adjusted(&mut file, 2, 1, &rgba).expect("save-back");
        assert_eq!(
            shape,
            PsdSaveBackShape::Flattened,
            "a layerless PSD gains the synthesized single layer"
        );
        let bytes = file.write().expect("write");
        let back = PsdFile::parse(&bytes).expect("reparse");
        let comp = back.composite_rgba8().expect("composite");
        assert_eq!(
            comp.rgba, rgba,
            "the merged composite carries the adjustment"
        );
        assert_eq!(back.layer_mask.layers.len(), 1, "single-layer result");
        assert_eq!(back.layer_mask.layers[0].name(), "Adjusted");
    }

    #[test]
    fn image_editor_saveback_psd_replaces_a_single_full_canvas_layer_in_place() {
        // Build a PSD with exactly one canvas-sized RGB layer, then adjust.
        let mut file = PsdFile::parse(&flat_psd()).expect("parse");
        let planes = planes_from_rgba8(&[9u8, 9, 9, 255, 9, 9, 9, 255], 2, false);
        let layer = single_layer(&planes, file.container, 2, 1, false).expect("layer");
        file.layer_mask.layers = vec![layer];
        file.layer_mask.section_raw = None;
        let bytes = file.write().expect("write seed");
        let mut seeded = PsdFile::parse(&bytes).expect("reparse seed");

        let rgba = vec![7u8, 8, 9, 255, 10, 11, 12, 255];
        let shape = psd_write_adjusted(&mut seeded, 2, 1, &rgba).expect("save-back");
        assert_eq!(shape, PsdSaveBackShape::LayerReplaced);
        assert_eq!(
            seeded.layer_mask.layers.len(),
            1,
            "no records added/removed"
        );
        let out = seeded.write().expect("write");
        let back = PsdFile::parse(&out).expect("reparse");
        assert_eq!(back.composite_rgba8().expect("composite").rgba, rgba);
        assert_eq!(back.layer_mask.layers[0].name(), "Adjusted", "name kept");
    }

    #[test]
    fn image_editor_saveback_psd_rejects_a_size_mismatch() {
        let mut file = PsdFile::parse(&flat_psd()).expect("parse");
        let err = psd_write_adjusted(&mut file, 4, 4, &[0u8; 64]).unwrap_err();
        assert!(matches!(err, IngestError::Unsupported(_)), "got {err:?}");
    }

    // feat: image.editor.saveback — the PNG/JPEG lane.
    #[test]
    fn image_editor_saveback_png_encodes_a_readable_png() {
        let rgba = vec![255u8, 0, 0, 255, 0, 255, 0, 255];
        let png = encode_rgba8(&rgba, 2, 1, RasterFormat::Png).expect("png");
        assert_eq!(&png[..4], &[0x89, b'P', b'N', b'G'], "PNG magic");
        // And it decodes back to the same pixels through the ingest lane.
        let back = crate::ingest::decode_rgba8(&png).expect("decode");
        assert_eq!((back.width, back.height), (2, 1));
        assert_eq!(&back.rgba[..], &rgba[..]);
    }

    #[test]
    fn image_editor_saveback_jpeg_encodes_a_readable_jpeg() {
        let rgba = vec![255u8, 0, 0, 255, 0, 255, 0, 255];
        let jpg = encode_rgba8(&rgba, 2, 1, RasterFormat::Jpeg).expect("jpeg");
        assert_eq!(&jpg[..3], &[0xFF, 0xD8, 0xFF], "JPEG SOI");
        // Lossy: assert it DECODES at the right size, not bit equality.
        let back = crate::ingest::decode_rgba8(&jpg).expect("decode");
        assert_eq!((back.width, back.height), (2, 1));
    }

    #[test]
    fn image_editor_saveback_encode_rejects_a_length_mismatch() {
        assert!(encode_rgba8(&[0u8; 6], 2, 1, RasterFormat::Png).is_err());
    }

    #[test]
    fn image_editor_saveback_raster_format_names_round_trip() {
        assert_eq!(RasterFormat::from_wire("png"), Some(RasterFormat::Png));
        assert_eq!(RasterFormat::from_wire("jpg"), Some(RasterFormat::Jpeg));
        assert_eq!(RasterFormat::from_wire("jpeg"), Some(RasterFormat::Jpeg));
        assert_eq!(RasterFormat::from_wire("webp"), None);
        assert_eq!(RasterFormat::Png.extension(), ".png");
        assert_eq!(RasterFormat::Jpeg.mime(), "image/jpeg");
    }
}
