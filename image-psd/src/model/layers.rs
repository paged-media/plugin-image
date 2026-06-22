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

use super::addl::{parse_addl_run, AddlPad};
use super::channel::Compression;
use crate::container::{Container, LenWidth};
use crate::reader::ByteReader;
use crate::writer::ByteWriter;
use crate::{PsdError, Result};

const BLEND_SIGNATURE: [u8; 4] = *b"8BIM";

impl LayerAndMaskInfo {
    /// Parse the length-framed layer & mask info section. The full payload
    /// is captured into `section_raw` for the zero-edit verbatim path; the
    /// typed sub-structure is parsed in parallel for the re-encode path.
    pub fn parse(r: &mut ByteReader, container: Container) -> Result<LayerAndMaskInfo> {
        let width = container.section_len_width();
        let total = r.len_field(width)? as usize;
        let section_bytes = r.take(total)?.to_vec();

        let mut sub = ByteReader::new(&section_bytes);
        let (layers, transparency_in_merged) = parse_layer_info(&mut sub, container)?;
        let global_mask = if sub.remaining() >= 4 {
            Some(GlobalLayerMask::parse(&mut sub)?)
        } else {
            None
        };
        // Document-level additional layer info blocks, padded to 4.
        let addl_global = parse_addl_run(&mut sub, container, AddlPad::Four)?;

        Ok(LayerAndMaskInfo {
            layers,
            transparency_in_merged,
            global_mask,
            addl_global,
            section_raw: Some(section_bytes),
        })
    }

    /// Emit the section: verbatim when `section_raw` is present, else
    /// re-encode the whole section length-framed.
    pub fn emit(&self, w: &mut ByteWriter, container: Container) {
        let width = container.section_len_width();
        if let Some(raw) = &self.section_raw {
            w.len_field(width, raw.len() as u64);
            w.bytes(raw);
            return;
        }
        w.framed(width, |w| {
            emit_layer_info(self, w, container);
            match &self.global_mask {
                Some(gm) => gm.emit(w),
                // The zero-length global mask is common and meaningful; emit
                // the 4-byte zero length explicitly (brief §4b).
                None => w.u32(0),
            }
            for a in &self.addl_global {
                a.emit(w, container, AddlPad::Four);
            }
        });
    }
}

/// Parse the layer info sub-block: its own length frame, the signed layer
/// count, all layer records, then all channel image data in layer/channel
/// order. Returns the records and the transparency flag.
fn parse_layer_info(r: &mut ByteReader, container: Container) -> Result<(Vec<LayerRecord>, bool)> {
    let width = container.section_len_width();
    // The layer info length may be zero (no layers) — then the section is
    // composite-only and there is nothing more here.
    if r.remaining() < width.bytes() {
        return Ok((Vec::new(), false));
    }
    let len = r.len_field(width)? as usize;
    if len == 0 {
        return Ok((Vec::new(), false));
    }
    let mut sub = r.sub(len)?;
    let raw_count = sub.i16()?;
    let transparency_in_merged = raw_count < 0;
    let count = raw_count.unsigned_abs() as usize;

    let mut layers = Vec::with_capacity(count);
    for _ in 0..count {
        layers.push(LayerRecord::parse(&mut sub, container)?);
    }
    // Channel image data follows in layer/channel order. Each channel's
    // declared `data_len` includes the 2-byte compression tag.
    for layer in &mut layers {
        for ci in 0..layer.channels.len() {
            let data_len = layer.channels[ci].data_len as usize;
            let cd = ChannelData::parse(&mut sub, data_len)?;
            layer.channel_data.push(cd);
        }
    }
    Ok((layers, transparency_in_merged))
}

/// Emit the layer info sub-block (length-framed), the signed count, all
/// records, then all channel image data.
fn emit_layer_info(info: &LayerAndMaskInfo, w: &mut ByteWriter, container: Container) {
    let width = container.section_len_width();
    // An empty layer list re-encodes as a zero-length layer info block (no
    // count word) — the form the parser treats as "no layers", so the
    // round-trip is symmetric. The count word + records form is only
    // emitted when there is at least one layer.
    if info.layers.is_empty() {
        w.len_field(width, 0);
        return;
    }
    let anchor = w.len() + width.bytes();
    w.framed(width, |w| {
        let count = info.layers.len() as i16;
        let signed = if info.transparency_in_merged {
            -count
        } else {
            count
        };
        w.i16(signed);
        for layer in &info.layers {
            layer.emit_record(w, container);
        }
        for layer in &info.layers {
            for cd in &layer.channel_data {
                cd.emit(w);
            }
        }
        // Layer info content is rounded up to even (brief §4a). Anchor on
        // the content start (just after the length field).
        w.pad_to(2, anchor);
    });
}

impl ChannelData {
    /// Read one channel's image data of exactly `data_len` bytes
    /// (compression tag + payload). The payload is kept verbatim; M0 never
    /// decodes it (preservation + streaming budget).
    fn parse(r: &mut ByteReader, data_len: usize) -> Result<ChannelData> {
        if data_len < 2 {
            return Err(PsdError::Malformed {
                section: "channel image data",
                detail: format!("channel data length {data_len} < 2"),
            });
        }
        let comp_code = r.u16()?;
        let compression = Compression::from_code(comp_code).ok_or_else(|| PsdError::Malformed {
            section: "channel image data",
            detail: format!("unknown compression {comp_code}"),
        })?;
        let bytes = r.take(data_len - 2)?.to_vec();
        Ok(ChannelData { compression, bytes })
    }

    fn emit(&self, w: &mut ByteWriter) {
        w.u16(self.compression.code());
        w.bytes(&self.bytes);
    }
}

impl GlobalLayerMask {
    fn parse(r: &mut ByteReader) -> Result<GlobalLayerMask> {
        let len = r.u32()? as usize;
        let raw = r.take(len)?.to_vec();
        Ok(GlobalLayerMask { raw })
    }

    fn emit(&self, w: &mut ByteWriter) {
        w.u32(self.raw.len() as u32);
        w.bytes(&self.raw);
    }
}

impl LayerRecord {
    fn parse(r: &mut ByteReader, container: Container) -> Result<LayerRecord> {
        let width = container.section_len_width();
        let top = r.i32()?;
        let left = r.i32()?;
        let bottom = r.i32()?;
        let right = r.i32()?;
        let channel_count = r.u16()? as usize;
        let mut channels = Vec::with_capacity(channel_count);
        for _ in 0..channel_count {
            let id = r.i16()?;
            let data_len = r.len_field(width)?;
            channels.push(ChannelInfo { id, data_len });
        }
        let blend_sig = r.fourcc()?;
        if blend_sig != BLEND_SIGNATURE {
            return Err(PsdError::Malformed {
                section: "layer record",
                detail: format!(
                    "bad blend signature {}",
                    String::from_utf8_lossy(&blend_sig)
                ),
            });
        }
        let blend_key = r.fourcc()?;
        let opacity = r.u8()?;
        let clipping = r.u8()?;
        let flags = r.u8()?;
        let filler = r.u8()?;

        // Extra data: u32 length frame holding mask + blend ranges + name +
        // addl blocks. Capture it verbatim AND parse it.
        let extra_len = r.u32()? as usize;
        let extra_bytes = r.take(extra_len)?.to_vec();
        let (mask, blend_ranges, name_legacy, addl) = parse_extra(&extra_bytes, container)?;

        Ok(LayerRecord {
            top,
            left,
            bottom,
            right,
            channels,
            blend_sig,
            blend_key,
            opacity,
            clipping,
            flags,
            filler,
            mask,
            blend_ranges,
            name_legacy,
            addl,
            extra_raw: Some(extra_bytes),
            channel_data: Vec::new(),
        })
    }

    fn emit_record(&self, w: &mut ByteWriter, container: Container) {
        let width = container.section_len_width();
        w.i32(self.top);
        w.i32(self.left);
        w.i32(self.bottom);
        w.i32(self.right);
        w.u16(self.channels.len() as u16);
        for ch in &self.channels {
            w.i16(ch.id);
            w.len_field(width, ch.data_len);
        }
        w.fourcc(self.blend_sig);
        w.fourcc(self.blend_key);
        w.u8(self.opacity);
        w.u8(self.clipping);
        w.u8(self.flags);
        w.u8(self.filler);
        // Extra data: verbatim when present, else re-encode the components.
        if let Some(extra) = &self.extra_raw {
            w.u32(extra.len() as u32);
            w.bytes(extra);
        } else {
            w.framed(LenWidth::U32, |w| emit_extra(self, w, container));
        }
    }
}

/// Parse a layer record's extra-data span: layer mask data, blending
/// ranges, the legacy Pascal name (padded to a multiple of 4 INCLUDING the
/// length byte), then additional-layer-info blocks (each padded to even).
fn parse_extra(
    bytes: &[u8],
    container: Container,
) -> Result<(
    Option<LayerMaskData>,
    BlendRanges,
    PascalString,
    Vec<AdditionalLayerInfo>,
)> {
    let mut r = ByteReader::new(bytes);
    let mask = LayerMaskData::parse(&mut r)?;
    let blend_ranges = BlendRanges::parse(&mut r)?;
    let name_legacy = read_pascal_pad4(&mut r)?;
    let addl = parse_addl_run(&mut r, container, AddlPad::Even)?;
    Ok((mask, blend_ranges, name_legacy, addl))
}

fn emit_extra(layer: &LayerRecord, w: &mut ByteWriter, container: Container) {
    match &layer.mask {
        Some(m) => m.emit(w),
        None => w.u32(0),
    }
    layer.blend_ranges.emit(w);
    write_pascal_pad4(&layer.name_legacy, w);
    for a in &layer.addl {
        a.emit(w, container, AddlPad::Even);
    }
}

impl LayerMaskData {
    /// `u32 size` (0 | 20 | 36). 0 ⇒ no mask. Otherwise the rect + default
    /// color + flags are typed; the rest of the size-framed payload (the
    /// real-mask / parameter variants) lives in `raw` verbatim.
    fn parse(r: &mut ByteReader) -> Result<Option<LayerMaskData>> {
        let size = r.u32()? as usize;
        if size == 0 {
            return Ok(None);
        }
        let payload = r.take(size)?;
        let mut pr = ByteReader::new(payload);
        let top = pr.i32()?;
        let left = pr.i32()?;
        let bottom = pr.i32()?;
        let right = pr.i32()?;
        let default_color = pr.u8()?;
        let flags = pr.u8()?;
        Ok(Some(LayerMaskData {
            top,
            left,
            bottom,
            right,
            default_color,
            flags,
            raw: payload.to_vec(),
        }))
    }

    fn emit(&self, w: &mut ByteWriter) {
        // `raw` is the full size-framed payload; re-emit it under its own
        // length so the 20/36-byte variants survive untouched.
        w.u32(self.raw.len() as u32);
        w.bytes(&self.raw);
    }
}

impl BlendRanges {
    /// `u32 length` + opaque payload.
    fn parse(r: &mut ByteReader) -> Result<BlendRanges> {
        let len = r.u32()? as usize;
        let raw = r.take(len)?.to_vec();
        Ok(BlendRanges { raw })
    }

    fn emit(&self, w: &mut ByteWriter) {
        w.u32(self.raw.len() as u32);
        w.bytes(&self.raw);
    }
}

/// Read the legacy Pascal name, padded to a multiple of 4 INCLUDING the
/// length byte (brief §5). The stored `PascalString` is trimmed (len byte +
/// content); padding is re-applied by `write_pascal_pad4`.
fn read_pascal_pad4(r: &mut ByteReader) -> Result<PascalString> {
    // The length byte + content is at most 256 bytes; capture it as a slice
    // before consuming the pad so we keep only [len][content].
    let n = r.u8()? as usize;
    let content = r.take(n)?;
    let mut raw = Vec::with_capacity(1 + n);
    raw.push(n as u8);
    raw.extend_from_slice(content);
    // Pad the (length byte + content) span to a multiple of 4.
    let field = 1 + n;
    let pad = (4 - (field % 4)) % 4;
    if pad != 0 {
        r.take(pad)?;
    }
    Ok(PascalString { raw })
}

fn write_pascal_pad4(name: &PascalString, w: &mut ByteWriter) {
    let start = w.len();
    w.bytes(&name.raw);
    w.pad_to(4, start);
}
