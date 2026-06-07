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

//! Layer & mask information section. PSD stores groups as a FLAT layer
//! list with `lsct` section-divider markers; the model preserves that
//! flat list in source order (round-trip-faithful) and derives the
//! group tree as a view — write never reorders or restructures.
//!
//! Provenance: Adobe Photoshop File Format specification, "Layer and
//! Mask Information Section".

use super::addl::AdditionalLayerInfo;
use super::channel::ChannelData;
use super::resources::PascalString;
use super::Raw;

/// Channel kind ids: 0..n = composite channels, -1 = transparency
/// mask (alpha), -2 = user-supplied layer mask, -3 = real user mask
/// (when both vector and raster masks exist).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelInfo {
    pub id: i16,
    /// Byte length of this channel's image data (incl. the 2-byte
    /// compression tag) — u32 in PSD, u64 in PSB.
    pub data_len: u64,
}

/// Layer mask / adjustment-mask data. Variable-size (0 / 20 / 36+
/// bytes); parsed view + verbatim payload (the section content after
/// its 4-byte size field) so producer quirks survive round-trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerMaskData {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
    pub default_color: u8,
    pub flags: u8,
    /// The full mask-data payload as read (everything inside the size
    /// frame). Verbatim re-emit unit; also carries the real-mask /
    /// parameter variants we don't decompose in M0.
    pub raw: Vec<u8>,
}

/// Layer blending ranges — opaque-verbatim (no editing semantics in
/// scope; the payload after the 4-byte length field).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlendRanges {
    pub raw: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerRecord {
    // Bounding rectangle (top, left, bottom, right) — note the spec's
    // T/L/B/R field order, kept verbatim.
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
    pub channels: Vec<ChannelInfo>,
    /// Always `8BIM`.
    pub blend_sig: [u8; 4],
    /// Blend mode key (`norm`, `mul `, `scrn`, …) — kept as the raw
    /// fourcc; semantic mapping to compose kernels is M1.
    pub blend_key: [u8; 4],
    pub opacity: u8,
    /// 0 = base, 1 = non-base (clipping group member).
    pub clipping: u8,
    pub flags: u8,
    pub filler: u8,
    pub mask: Option<LayerMaskData>,
    pub blend_ranges: BlendRanges,
    /// Legacy Pascal name (padded to 4 inside the record); the
    /// canonical name is the `luni` block when present.
    pub name_legacy: PascalString,
    pub addl: Vec<AdditionalLayerInfo>,
    /// The record's extra-data section exactly as read (mask + blend
    /// ranges + name + addl blocks, including all padding). Verbatim
    /// re-emit unit for the zero-edit path; `None` re-encodes from the
    /// parsed fields above.
    pub extra_raw: Raw,
    /// Channel image data, parallel to `channels`, stored as on-disk
    /// compressed payloads (decode-on-demand).
    pub channel_data: Vec<ChannelData>,
}

impl LayerRecord {
    pub fn name(&self) -> String {
        self.addl
            .iter()
            .find_map(|a| a.unicode_name())
            .unwrap_or_else(|| self.name_legacy.text_lossy())
    }
}

/// Global layer mask info — opaque-verbatim (payload after its length
/// field; zero-length sections are common and meaningful).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GlobalLayerMask {
    pub raw: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LayerAndMaskInfo {
    /// Source-order flat layer list (bottom-most first, as stored).
    pub layers: Vec<LayerRecord>,
    /// The stored layer count was negative (sign flag: first alpha
    /// channel contains the transparency data for the merged result).
    pub transparency_in_merged: bool,
    pub global_mask: Option<GlobalLayerMask>,
    /// File-level additional layer info blocks after the global mask.
    pub addl_global: Vec<AdditionalLayerInfo>,
    /// The ENTIRE section payload as read (inside the section length
    /// frame). The zero-edit writer emits this verbatim — the strongest
    /// byte-identity guarantee against producer length/padding quirks;
    /// `None` re-encodes everything from the model.
    pub section_raw: Raw,
}
