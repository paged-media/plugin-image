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

//! Additional layer information blocks (`8BIM`/`8B64` + fourcc key).
//! Typed in M0: `lsct` (section divider — the group structure), `luni`
//! (Unicode name), `lyid` (layer id). EVERYTHING else — `lfx2`, `SoCo`,
//! `TySh`, `SoLd`, `vmsk`, Adobe-private, undocumented — is preserved
//! opaquely (the constitutive §10.4 invariant). Each block keeps its
//! complete source bytes for verbatim re-emit; the parsed body is a
//! view.
//!
//! Provenance: Adobe Photoshop File Format specification, "Additional
//! Layer Information"; `luni` count/padding variance from black-box
//! corpus observation (synthesized fixtures pin both variants).

use super::Raw;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// 0 — any other type of layer.
    Other,
    /// 1 — open folder.
    OpenFolder,
    /// 2 — closed folder.
    ClosedFolder,
    /// 3 — bounding section divider (hidden marker closing a group).
    BoundingDivider,
}

impl SectionKind {
    pub const fn code(self) -> u32 {
        match self {
            SectionKind::Other => 0,
            SectionKind::OpenFolder => 1,
            SectionKind::ClosedFolder => 2,
            SectionKind::BoundingDivider => 3,
        }
    }

    pub fn from_code(c: u32) -> Option<Self> {
        Some(match c {
            0 => SectionKind::Other,
            1 => SectionKind::OpenFolder,
            2 => SectionKind::ClosedFolder,
            3 => SectionKind::BoundingDivider,
            _ => return None,
        })
    }
}

/// `lsct` payload: kind (+ optional blend key when length ≥ 12,
/// + optional sub-type when length ≥ 16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LsctData {
    pub kind: SectionKind,
    pub blend_key: Option<[u8; 4]>,
    pub sub_kind: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AddlBody {
    SectionDivider(LsctData),
    /// `luni` — UTF-16BE name, decoded.
    UnicodeName(String),
    /// `lyid`.
    LayerId(u32),
    /// Everything unmodeled. The bytes live in `raw_block`.
    Opaque,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdditionalLayerInfo {
    /// `8BIM` or `8B64`.
    pub sig: [u8; 4],
    pub key: [u8; 4],
    pub body: AddlBody,
    /// The complete block as read — signature through trailing padding.
    /// Verbatim re-emit unit; `None` only for constructed blocks.
    pub raw_block: Raw,
}

impl AdditionalLayerInfo {
    pub fn unicode_name(&self) -> Option<String> {
        match &self.body {
            AddlBody::UnicodeName(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn layer_id(&self) -> Option<u32> {
        match &self.body {
            AddlBody::LayerId(id) => Some(*id),
            _ => None,
        }
    }

    pub fn lsct(&self) -> Option<&LsctData> {
        match &self.body {
            AddlBody::SectionDivider(d) => Some(d),
            _ => None,
        }
    }
}

use crate::container::{Container, LenWidth};
use crate::reader::ByteReader;
use crate::writer::ByteWriter;
use crate::{PsdError, Result};

/// Padding alignment of an additional-layer-info block's *outer span*.
/// Inside a layer record's extra data the canonical pad is to EVEN; at
/// document level it is to a multiple of 4 (spec §10.4 / brief §6). The
/// parser is tolerant — it reads exactly the stored length and then aligns
/// to the context boundary, so any padding the producer chose lands inside
/// the captured `raw_block` and is re-emitted verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddlPad {
    Even,
    Four,
}

impl AddlPad {
    const fn align(self) -> usize {
        match self {
            AddlPad::Even => 2,
            AddlPad::Four => 4,
        }
    }
}

const SIG_8BIM: [u8; 4] = *b"8BIM";
const SIG_8B64: [u8; 4] = *b"8B64";

impl AdditionalLayerInfo {
    /// Parse one additional-layer-info block at `r`. The block length
    /// field is u64 only in PSB and only for the keys the container marks
    /// wide. The captured `raw_block` spans signature → trailing pad.
    pub fn parse(
        r: &mut ByteReader,
        container: Container,
        pad: AddlPad,
    ) -> Result<AdditionalLayerInfo> {
        // Probe length on a clone, then take the whole block as one slice.
        let mut probe = r.clone();
        let sig = probe.fourcc()?;
        if sig != SIG_8BIM && sig != SIG_8B64 {
            return Err(PsdError::Malformed {
                section: "additional layer info",
                detail: format!("bad block signature {}", String::from_utf8_lossy(&sig)),
            });
        }
        let key = probe.fourcc()?;
        let width = if container.addl_len_is_wide(key) {
            LenWidth::U64
        } else {
            LenWidth::U32
        };
        let data_len = probe.len_field(width)? as usize;
        let header_len = 4 + 4 + width.bytes();
        // The stored length excludes padding; pad the (data) span to the
        // context boundary, anchored at the data start.
        let pad_bytes = {
            let rem = data_len % pad.align();
            if rem == 0 {
                0
            } else {
                pad.align() - rem
            }
        };
        let block_len = header_len + data_len + pad_bytes;
        let block = r.take(block_len)?;

        // Re-parse the captured slice for the typed view.
        let mut br = ByteReader::new(block);
        br.fourcc()?; // sig
        br.fourcc()?; // key
        br.len_field(width)?; // length
        let data = br.take(data_len)?;
        let body = decode_body(key, data);
        Ok(AdditionalLayerInfo {
            sig,
            key,
            body,
            raw_block: Some(block.to_vec()),
        })
    }

    /// Emit one block: verbatim if `raw_block` is present, else re-encode
    /// canonically (length-framed, then padded to the context boundary;
    /// the stored length excludes the pad).
    pub fn emit(&self, w: &mut ByteWriter, container: Container, pad: AddlPad) {
        if let Some(raw) = &self.raw_block {
            w.bytes(raw);
            return;
        }
        let width = if container.addl_len_is_wide(self.key) {
            LenWidth::U64
        } else {
            LenWidth::U32
        };
        w.fourcc(self.sig);
        w.fourcc(self.key);
        // The stored length covers only the data; pad the data span to the
        // context boundary AFTER back-patching, anchored at the data start.
        let data_anchor = w.len() + width.bytes();
        w.framed(width, |w| encode_body(&self.body, w));
        w.pad_to(pad.align(), data_anchor);
    }
}

/// Re-encode helper used by tests that reconstruct blocks; kept separate
/// so `emit` stays a thin verbatim/anchor wrapper.
fn encode_body(body: &AddlBody, w: &mut ByteWriter) {
    match body {
        AddlBody::SectionDivider(d) => {
            w.u32(d.kind.code());
            if let Some(bk) = d.blend_key {
                w.fourcc(*b"8BIM");
                w.fourcc(bk);
                if let Some(sk) = d.sub_kind {
                    w.u32(sk);
                }
            }
        }
        AddlBody::UnicodeName(s) => {
            let units: Vec<u16> = s.encode_utf16().collect();
            w.u32(units.len() as u32);
            for u in units {
                w.u16(u);
            }
        }
        AddlBody::LayerId(id) => w.u32(*id),
        AddlBody::Opaque => {}
    }
}

fn decode_body(key: [u8; 4], data: &[u8]) -> AddlBody {
    match &key {
        b"lsct" => decode_lsct(data)
            .map(AddlBody::SectionDivider)
            .unwrap_or(AddlBody::Opaque),
        b"luni" => decode_luni(data)
            .map(AddlBody::UnicodeName)
            .unwrap_or(AddlBody::Opaque),
        b"lyid" => decode_lyid(data)
            .map(AddlBody::LayerId)
            .unwrap_or(AddlBody::Opaque),
        _ => AddlBody::Opaque,
    }
}

/// `lsct`: u32 kind; if len≥12 a `8BIM`+blend key; if len≥16 a u32 sub-kind.
fn decode_lsct(data: &[u8]) -> Option<LsctData> {
    let mut r = ByteReader::new(data);
    let kind = SectionKind::from_code(r.u32().ok()?)?;
    let blend_key = if data.len() >= 12 {
        let _sig = r.fourcc().ok()?; // `8BIM`
        Some(r.fourcc().ok()?)
    } else {
        None
    };
    let sub_kind = if data.len() >= 16 {
        Some(r.u32().ok()?)
    } else {
        None
    };
    Some(LsctData {
        kind,
        blend_key,
        sub_kind,
    })
}

/// `luni`: u32 code-unit count + UTF-16BE. The count traditionally
/// EXCLUDES a trailing null; some producers append one and include it in
/// the count. We trim a single trailing null so the decoded string is the
/// same for both variants — `raw_block` preserves the on-disk bytes, so
/// the variant survives round-trip regardless.
fn decode_luni(data: &[u8]) -> Option<String> {
    let mut r = ByteReader::new(data);
    let count = r.u32().ok()? as usize;
    let mut units = Vec::with_capacity(count);
    for _ in 0..count {
        units.push(r.u16().ok()?);
    }
    if matches!(units.last(), Some(0)) {
        units.pop();
    }
    String::from_utf16(&units).ok()
}

/// `lyid`: a single u32 layer id.
fn decode_lyid(data: &[u8]) -> Option<u32> {
    if data.len() < 4 {
        return None;
    }
    ByteReader::new(data).u32().ok()
}

/// Parse a run of additional-layer-info blocks filling the remainder of
/// `r`, each padded to `pad`. Used both inside layer-record extra data
/// (Even) and at document level (Four).
pub fn parse_addl_run(
    r: &mut ByteReader,
    container: Container,
    pad: AddlPad,
) -> Result<Vec<AdditionalLayerInfo>> {
    let mut out = Vec::new();
    // A block needs at least sig(4)+key(4)+len(4) = 12 bytes; a shorter
    // tail is residual pad the caller already accounts for via raw_block
    // at the enclosing level, so we stop cleanly.
    while r.remaining() >= 12 {
        out.push(AdditionalLayerInfo::parse(r, container, pad)?);
    }
    Ok(out)
}
