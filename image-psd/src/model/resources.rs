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

//! Image resources section: `8BIM` blocks keyed by u16 id. Typed
//! bodies for the ids the engine consumes (ICC 1039, resolution 1005);
//! everything else opaque-verbatim. Every block additionally keeps its
//! complete source bytes (`raw_block`, signature through padding) for
//! the lazy-verbatim guard.
//!
//! Provenance: Adobe Photoshop File Format specification, "Image
//! Resources Section" + "Image Resource IDs".

/// Resource ids the model types (everything else is `Opaque`).
pub const RES_RESOLUTION_INFO: u16 = 1005;
pub const RES_ICC_PROFILE: u16 = 1039;

/// A Pascal string as stored: length byte + bytes, NO padding (padding
/// is contextual and applied by the emitters). Kept raw because PSD
/// names are MacRoman-ish legacy bytes, not guaranteed UTF-8.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PascalString {
    pub raw: Vec<u8>,
}

impl PascalString {
    /// Build from ASCII-safe text (lossy for anything else — the
    /// canonical name lives in `luni` anyway).
    pub fn new(s: &str) -> Self {
        let bytes: Vec<u8> = s.bytes().take(255).collect();
        let mut raw = Vec::with_capacity(bytes.len() + 1);
        raw.push(bytes.len() as u8);
        raw.extend_from_slice(&bytes);
        PascalString { raw }
    }

    pub fn text_lossy(&self) -> String {
        if self.raw.is_empty() {
            return String::new();
        }
        let n = self.raw[0] as usize;
        String::from_utf8_lossy(&self.raw[1..1 + n.min(self.raw.len() - 1)]).into_owned()
    }
}

/// Fixed-point resolution record (id 1005). Stored fields are the raw
/// 32-bit fixed-point values + display units — typed but lossless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionInfo {
    /// Horizontal resolution, 16.16 fixed point, pixels per inch.
    pub h_res_fixed: u32,
    /// 1 = ppi, 2 = ppcm.
    pub h_res_unit: u16,
    /// 1 = inches, 2 = cm, 3 = points, 4 = picas, 5 = columns.
    pub width_unit: u16,
    pub v_res_fixed: u32,
    pub v_res_unit: u16,
    pub height_unit: u16,
}

impl ResolutionInfo {
    pub fn h_ppi(&self) -> f64 {
        self.h_res_fixed as f64 / 65536.0
    }

    pub fn v_ppi(&self) -> f64 {
        self.v_res_fixed as f64 / 65536.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceBody {
    IccProfile(Vec<u8>),
    ResolutionInfo(ResolutionInfo),
    Opaque,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageResourceBlock {
    pub id: u16,
    pub name: PascalString,
    /// Parsed view (`Opaque` for unmodeled ids; the bytes live in
    /// `raw_block`).
    pub body: ResourceBody,
    /// The complete block as read — signature, id, name (padded), size,
    /// data, padding. Verbatim re-emit unit (strategy 3); `None` only
    /// for constructed blocks, which re-encode canonically.
    pub raw_block: super::Raw,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImageResources {
    pub blocks: Vec<ImageResourceBlock>,
}

impl ImageResources {
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.blocks.iter().find_map(|b| match &b.body {
            ResourceBody::IccProfile(p) => Some(p.as_slice()),
            _ => None,
        })
    }

    pub fn resolution(&self) -> Option<&ResolutionInfo> {
        self.blocks.iter().find_map(|b| match &b.body {
            ResourceBody::ResolutionInfo(r) => Some(r),
            _ => None,
        })
    }
}

use crate::reader::ByteReader;
use crate::writer::ByteWriter;
use crate::{PsdError, Result};

pub const RESOURCE_SIGNATURE: [u8; 4] = *b"8BIM";

impl ResolutionInfo {
    fn parse(data: &[u8]) -> Option<ResolutionInfo> {
        // Fixed 16-byte record; anything else stays Opaque (raw_block
        // still preserves the bytes verbatim).
        if data.len() != 16 {
            return None;
        }
        let mut r = ByteReader::new(data);
        Some(ResolutionInfo {
            h_res_fixed: r.u32().ok()?,
            h_res_unit: r.u16().ok()?,
            width_unit: r.u16().ok()?,
            v_res_fixed: r.u32().ok()?,
            v_res_unit: r.u16().ok()?,
            height_unit: r.u16().ok()?,
        })
    }

    fn emit_data(&self, w: &mut ByteWriter) {
        w.u32(self.h_res_fixed);
        w.u16(self.h_res_unit);
        w.u16(self.width_unit);
        w.u32(self.v_res_fixed);
        w.u16(self.v_res_unit);
        w.u16(self.height_unit);
    }
}

impl ImageResources {
    /// Parse the length-framed resources section: `u32 total length` then
    /// `8BIM` blocks to the section end. Each block's complete source span
    /// (signature → trailing pad) is captured in `raw_block`.
    pub fn parse(r: &mut ByteReader) -> Result<ImageResources> {
        let total = r.u32()? as usize;
        let mut sub = r.sub(total)?;
        let mut blocks = Vec::new();
        while !sub.is_at_end() {
            blocks.push(ImageResourceBlock::parse(&mut sub)?);
        }
        Ok(ImageResources { blocks })
    }

    /// Emit the section length-framed; each block re-emits verbatim if it
    /// still carries `raw_block`, else re-encodes canonically.
    pub fn emit(&self, w: &mut ByteWriter) {
        w.framed(crate::container::LenWidth::U32, |w| {
            for b in &self.blocks {
                b.emit(w);
            }
        });
    }
}

impl ImageResourceBlock {
    fn parse(r: &mut ByteReader) -> Result<ImageResourceBlock> {
        // Probe the header on a clone to learn the full block length, then
        // take the whole block as one slice — that slice IS `raw_block`
        // (signature through trailing pad), so verbatim re-emit is exact.
        let mut probe = r.clone();
        let sig = probe.fourcc()?;
        if sig != RESOURCE_SIGNATURE {
            return Err(PsdError::Malformed {
                section: "image resource",
                detail: format!("bad block signature {}", String::from_utf8_lossy(&sig)),
            });
        }
        let id = probe.u16()?;
        let name_field_len = pascal_padded_even_len(&mut probe)?;
        let size = probe.u32()? as usize;
        // Data padded to even; the size field excludes the pad byte.
        let data_padded = size + (size & 1);
        let block_len = 4 + 2 + name_field_len + 4 + data_padded;
        let block = r.take(block_len)?;

        // Re-parse the captured slice into typed fields.
        let mut br = ByteReader::new(block);
        br.fourcc()?; // signature (validated above)
        br.u16()?; // id
        let name = read_pascal_field(&mut br, name_field_len)?;
        br.u32()?; // size
        let data = br.take(size)?;
        let body = match id {
            RES_ICC_PROFILE => ResourceBody::IccProfile(data.to_vec()),
            RES_RESOLUTION_INFO => match ResolutionInfo::parse(data) {
                Some(ri) => ResourceBody::ResolutionInfo(ri),
                None => ResourceBody::Opaque,
            },
            _ => ResourceBody::Opaque,
        };
        Ok(ImageResourceBlock {
            id,
            name,
            body,
            raw_block: Some(block.to_vec()),
        })
    }

    fn emit(&self, w: &mut ByteWriter) {
        if let Some(raw) = &self.raw_block {
            w.bytes(raw);
            return;
        }
        w.fourcc(RESOURCE_SIGNATURE);
        w.u16(self.id);
        // Name field padded to even (anchored at the name's first byte).
        let name_start = w.len();
        w.bytes(&self.name.raw);
        w.pad_to(2, name_start);
        // Data length-framed via a manual size write so we can re-encode
        // typed bodies; pad the data span to even, excluding the pad byte
        // from the stored size.
        match &self.body {
            ResourceBody::Opaque => {
                // Opaque blocks must carry raw_block; reaching here means a
                // constructed Opaque block, which is empty by definition.
                w.u32(0);
            }
            ResourceBody::IccProfile(p) => {
                w.u32(p.len() as u32);
                let data_start = w.len();
                w.bytes(p);
                w.pad_to(2, data_start);
            }
            ResourceBody::ResolutionInfo(ri) => {
                w.u32(16);
                let data_start = w.len();
                ri.emit_data(w);
                w.pad_to(2, data_start);
            }
        }
    }
}

/// Total byte length of a resource-name field: length byte + content,
/// padded so the field occupies an even byte count (2 minimum for the
/// empty name). Advances `r` past the field.
fn pascal_padded_even_len(r: &mut ByteReader) -> Result<usize> {
    let n = r.u8()? as usize;
    let _ = r.take(n)?;
    let content = 1 + n;
    let field = content + (content & 1);
    // Consume the pad byte so the probe stays aligned with the real read.
    if field != content {
        r.take(1)?;
    }
    Ok(field)
}

/// Read a resource-name field of known total length `field_len` (content
/// plus pad), returning the trimmed `PascalString` (length byte + content,
/// no pad — padding is contextual and re-applied by the emitter).
fn read_pascal_field(r: &mut ByteReader, field_len: usize) -> Result<PascalString> {
    let field = r.take(field_len)?;
    let n = field[0] as usize;
    Ok(PascalString {
        raw: field[..1 + n].to_vec(),
    })
}
