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

//! Layer-record + extra-data emission (brief §5–§9). A [`LayerSpec`]
//! holds everything one flat layer-list entry needs; [`emit_layer_record`]
//! writes the fixed scalars and the extra-data tail (mask, blend ranges,
//! Pascal name, additional-layer-info blocks), and [`emit_layer_channels`]
//! writes that layer's channel payloads in declared order.
//!
//! Group structure is the FLAT-list convention of brief §7: the folder
//! record (`lsct` kind 1|2) appears ABOVE its members and a bounding
//! divider (`lsct` kind 3, named `</Layer group>`) BELOW them. Since PSD
//! stores bottom-most first, the builder appends the divider, then the
//! members, then the folder — the manifest records nesting depth so the
//! parser can be checked against intent.

use image_psd::container::{Container, LenWidth};
use image_psd::model::Compression;

use super::channels::{encode_channel, Plane};
use super::emit::Emit;

/// The conventional bounding-divider name (brief §7).
pub const GROUP_DIVIDER_NAME: &str = "</Layer group>";

/// One channel within a layer: its PSD channel id (0/1/2 composite, -1
/// alpha, -2 user mask, -3 real mask) + its source plane + how to
/// compress it. The on-disk per-channel length frames the encoded body.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub id: i16,
    pub plane: Plane,
    pub compression: Compression,
}

/// Layer mask data (brief §5): size 0 (absent), 20, or 36.
#[derive(Debug, Clone)]
pub struct MaskSpec {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
    pub default_color: u8,
    pub flags: u8,
    /// When set, emits the size-36 form: real_flags + real_background +
    /// the real rect (brief §5).
    pub real: Option<RealMask>,
}

#[derive(Debug, Clone)]
pub struct RealMask {
    pub real_flags: u8,
    pub real_background: u8,
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
}

/// One additional-layer-info block to attach to a layer record. Typed
/// kinds the builder constructs canonically; `Opaque` carries verbatim
/// payload bytes for unknown-key fixtures.
#[derive(Debug, Clone)]
pub enum AddlSpec {
    /// `lsct` (brief §7): kind code + optional blend key + optional
    /// sub-kind. The presence of `blend_key`/`sub_kind` drives the
    /// emitted length (12 / 16).
    Lsct {
        kind: u32,
        blend_key: Option<[u8; 4]>,
        sub_kind: Option<u32>,
    },
    /// `luni` (brief §8): UTF-16BE name. `count_includes_null` pins the
    /// producer variant where the trailing NUL is counted (and emitted).
    Luni {
        name: String,
        count_includes_null: bool,
    },
    /// `lyid` (brief §9).
    Lyid(u32),
    /// Anything else: signature ('8BIM'/'8B64'), key, raw payload.
    Opaque {
        sig: [u8; 4],
        key: [u8; 4],
        payload: Vec<u8>,
    },
}

/// A flat layer-list entry (brief §5). `extra_addl` are written into the
/// layer record's extra data after the name; `channels` follow in the
/// per-layer channel image-data run.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
    /// The legacy Pascal name in the layer record (brief §5). The
    /// Unicode name, when present, is an additional `luni` block.
    pub name: String,
    pub blend_key: [u8; 4],
    pub opacity: u8,
    pub clipping: u8,
    pub flags: u8,
    pub mask: Option<MaskSpec>,
    /// Opaque blending-ranges payload (brief §5); empty = the 0-length
    /// form (`u32 0`).
    pub blend_ranges: Vec<u8>,
    pub channels: Vec<ChannelSpec>,
    pub extra_addl: Vec<AddlSpec>,
}

/// UTF-16BE bytes for a string (no BOM, big-endian — brief §8).
fn utf16be(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for u in s.encode_utf16() {
        out.extend_from_slice(&u.to_be_bytes());
    }
    out
}

/// Lower an [`AddlSpec`] to its on-disk (signature, key, payload) triple
/// (brief §6–§9). The single source of truth for the typed payload
/// shapes — both the layer-record encoder and the document-level encoder
/// in `mod.rs` consume this, so the two padding contexts share one body
/// definition.
pub fn addl_parts(spec: &AddlSpec) -> ([u8; 4], [u8; 4], Vec<u8>) {
    match spec {
        AddlSpec::Lsct {
            kind,
            blend_key,
            sub_kind,
        } => {
            let mut p = Emit::new();
            p.u32(*kind);
            if let Some(bk) = blend_key {
                p.fourcc(*b"8BIM").fourcc(*bk);
                if let Some(sk) = sub_kind {
                    p.u32(*sk);
                }
            }
            (*b"8BIM", *b"lsct", p.into_bytes())
        }
        AddlSpec::Luni {
            name,
            count_includes_null,
        } => {
            let units = name.encode_utf16().count() as u32;
            let mut p = Emit::new();
            if *count_includes_null {
                // The variant some producers write: count includes the
                // trailing NUL, and the NUL bytes are present (brief §8).
                p.u32(units + 1);
                p.raw(&utf16be(name));
                p.u16(0);
            } else {
                p.u32(units);
                p.raw(&utf16be(name));
            }
            (*b"8BIM", *b"luni", p.into_bytes())
        }
        AddlSpec::Lyid(id) => {
            let mut p = Emit::new();
            p.u32(*id);
            (*b"8BIM", *b"lyid", p.into_bytes())
        }
        AddlSpec::Opaque { sig, key, payload } => (*sig, *key, payload.clone()),
    }
}

/// Container-correct length width for an additional-layer-info `key`
/// (brief §6): PSB widens only the keys [`Container::addl_len_is_wide`]
/// enumerates.
pub fn addl_len_width(c: Container, key: [u8; 4]) -> LenWidth {
    if c.addl_len_is_wide(key) {
        LenWidth::U64
    } else {
        LenWidth::U32
    }
}

/// Emit one additional-layer-info block into `e` (brief §6). At the
/// layer-record level the data is padded to EVEN; the stored length
/// EXCLUDES that pad.
fn emit_addl(e: &mut Emit, c: Container, spec: &AddlSpec) {
    let (sig, key, payload) = addl_parts(spec);
    e.fourcc(sig).fourcc(key);
    e.len_field(addl_len_width(c, key), payload.len() as u64);
    e.raw(&payload);
    // Canonical even padding inside layer-record extra data (brief §6).
    e.pad_to(2);
}

/// Emit the layer mask data sub-block (brief §5): a u32 size of 0, 20,
/// or 36 followed by that many bytes.
fn emit_mask(e: &mut Emit, mask: &Option<MaskSpec>) {
    match mask {
        None => {
            e.u32(0);
        }
        Some(m) => {
            let mut body = Emit::new();
            body.i32(m.top).i32(m.left).i32(m.bottom).i32(m.right);
            body.u8(m.default_color).u8(m.flags);
            match &m.real {
                None => {
                    // size-20 form: 2 pad bytes complete the fixed block.
                    body.u16(0);
                }
                Some(r) => {
                    body.u8(r.real_flags).u8(r.real_background);
                    body.i32(r.top).i32(r.left).i32(r.bottom).i32(r.right);
                }
            }
            let bytes = body.into_bytes();
            e.u32(bytes.len() as u32);
            e.raw(&bytes);
        }
    }
}

/// Emit one layer record: the fixed scalars + the extra-data tail
/// (brief §5). Channel image data is emitted separately, after ALL
/// records, by [`emit_layer_channels`].
pub fn emit_layer_record(e: &mut Emit, c: Container, layer: &LayerSpec) {
    let w = c.section_len_width();

    e.i32(layer.top)
        .i32(layer.left)
        .i32(layer.bottom)
        .i32(layer.right);

    e.u16(layer.channels.len() as u16);
    for ch in &layer.channels {
        e.i16(ch.id);
        // The per-channel length INCLUDES the 2-byte compression tag,
        // which encode_channel already prepends (brief §5).
        let body = encode_channel(&ch.plane, ch.compression, w);
        e.len_field(w, body.len() as u64);
    }

    e.fourcc(*b"8BIM").fourcc(layer.blend_key);
    e.u8(layer.opacity).u8(layer.clipping).u8(layer.flags).u8(0);

    // Extra data: build into a sub-emitter so its u32 length is exact.
    let mut extra = Emit::new();
    emit_mask(&mut extra, &layer.mask);
    // Blending ranges: u32 length + opaque bytes (brief §5).
    extra.u32(layer.blend_ranges.len() as u32);
    extra.raw(&layer.blend_ranges);
    // Layer name: Pascal string padded to a multiple of 4 INCLUDING its
    // length byte (brief §5).
    extra.pascal_string(layer.name.as_bytes(), 4);
    for a in &layer.extra_addl {
        emit_addl(&mut extra, c, a);
    }
    let extra_bytes = extra.into_bytes();
    e.u32(extra_bytes.len() as u32);
    e.raw(&extra_bytes);
}

/// Emit one layer's channel image data (brief §5): each channel's body
/// (compression tag + payload) in the same order the record declared.
pub fn emit_layer_channels(e: &mut Emit, c: Container, layer: &LayerSpec) {
    let w = c.section_len_width();
    for ch in &layer.channels {
        let body = encode_channel(&ch.plane, ch.compression, w);
        e.raw(&body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(w: u32, h: u32) -> Vec<ChannelSpec> {
        [(0i16, 200u8), (1, 100), (2, 50)]
            .iter()
            .map(|&(id, v)| ChannelSpec {
                id,
                plane: Plane::solid(w, h, v),
                compression: Compression::Raw,
            })
            .collect()
    }

    fn base_layer() -> LayerSpec {
        LayerSpec {
            top: 0,
            left: 0,
            bottom: 2,
            right: 2,
            name: "L".into(),
            blend_key: *b"norm",
            opacity: 255,
            clipping: 0,
            flags: 0,
            mask: None,
            blend_ranges: Vec::new(),
            channels: solid_rgb(2, 2),
            extra_addl: Vec::new(),
        }
    }

    #[test]
    fn psd_builder_layer_record_scalar_prefix() {
        let mut e = Emit::new();
        emit_layer_record(&mut e, Container::Psd, &base_layer());
        // rect: 0,0,2,2 then channel count 3.
        assert_eq!(
            &e.bytes[0..16],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2]
        );
        assert_eq!(&e.bytes[16..18], &[0, 3]);
    }

    #[test]
    fn psd_builder_layer_mask_size_20() {
        let mut e = Emit::new();
        emit_mask(
            &mut e,
            &Some(MaskSpec {
                top: 1,
                left: 2,
                bottom: 3,
                right: 4,
                default_color: 255,
                flags: 0,
                real: None,
            }),
        );
        // u32 size = 20, then rect(16) + default(1) + flags(1) + pad(2).
        assert_eq!(&e.bytes[0..4], &[0, 0, 0, 20]);
        assert_eq!(e.bytes.len(), 24);
    }

    #[test]
    fn psd_builder_layer_mask_size_36() {
        let mut e = Emit::new();
        emit_mask(
            &mut e,
            &Some(MaskSpec {
                top: 0,
                left: 0,
                bottom: 1,
                right: 1,
                default_color: 0,
                flags: 0,
                real: Some(RealMask {
                    real_flags: 1,
                    real_background: 255,
                    top: 0,
                    left: 0,
                    bottom: 1,
                    right: 1,
                }),
            }),
        );
        assert_eq!(&e.bytes[0..4], &[0, 0, 0, 36]);
        assert_eq!(e.bytes.len(), 40);
    }

    #[test]
    fn psd_builder_luni_count_excludes_null_by_default() {
        let mut e = Emit::new();
        emit_addl(
            &mut e,
            Container::Psd,
            &AddlSpec::Luni {
                name: "Hi".into(),
                count_includes_null: false,
            },
        );
        // 8BIM luni, u32 payload-len = 4 (count u32) + 4 (2 UTF-16 units)
        // = 8; payload count = 2.
        assert_eq!(&e.bytes[0..4], b"8BIM");
        assert_eq!(&e.bytes[4..8], b"luni");
        assert_eq!(&e.bytes[8..12], &[0, 0, 0, 8]); // payload len
        assert_eq!(&e.bytes[12..16], &[0, 0, 0, 2]); // unit count
        assert_eq!(&e.bytes[16..20], &[0, b'H', 0, b'i']);
    }

    #[test]
    fn psd_builder_luni_count_includes_null_variant() {
        let mut e = Emit::new();
        emit_addl(
            &mut e,
            Container::Psd,
            &AddlSpec::Luni {
                name: "Hi".into(),
                count_includes_null: true,
            },
        );
        // payload = count(4) + units(4) + NUL(2) = 10; count = 3.
        assert_eq!(&e.bytes[8..12], &[0, 0, 0, 10]);
        assert_eq!(&e.bytes[12..16], &[0, 0, 0, 3]);
        assert_eq!(&e.bytes[20..22], &[0, 0]); // the counted NUL
    }

    #[test]
    fn psd_builder_lsct_length_grows_with_blend_and_subkind() {
        // kind only ⇒ 4-byte payload.
        let mut e = Emit::new();
        emit_addl(
            &mut e,
            Container::Psd,
            &AddlSpec::Lsct {
                kind: 1,
                blend_key: None,
                sub_kind: None,
            },
        );
        assert_eq!(&e.bytes[8..12], &[0, 0, 0, 4]);

        // kind + blend key ⇒ 12-byte payload.
        let mut e = Emit::new();
        emit_addl(
            &mut e,
            Container::Psd,
            &AddlSpec::Lsct {
                kind: 1,
                blend_key: Some(*b"pass"),
                sub_kind: None,
            },
        );
        assert_eq!(&e.bytes[8..12], &[0, 0, 0, 12]);

        // kind + blend + sub ⇒ 16-byte payload.
        let mut e = Emit::new();
        emit_addl(
            &mut e,
            Container::Psd,
            &AddlSpec::Lsct {
                kind: 1,
                blend_key: Some(*b"pass"),
                sub_kind: Some(0),
            },
        );
        assert_eq!(&e.bytes[8..12], &[0, 0, 0, 16]);
    }

    #[test]
    fn psd_builder_addl_pads_payload_to_even() {
        // An opaque block with an odd payload gets a single pad byte; the
        // STORED length still excludes it (brief §6).
        let mut e = Emit::new();
        emit_addl(
            &mut e,
            Container::Psd,
            &AddlSpec::Opaque {
                sig: *b"8BIM",
                key: *b"xpgi",
                payload: vec![0xAB, 0xCD, 0xEF],
            },
        );
        assert_eq!(&e.bytes[8..12], &[0, 0, 0, 3]); // stored len = 3
        assert_eq!(e.bytes.len(), 8 + 4 + 4); // sig+key+len+3+1 pad = 16
    }

    #[test]
    fn psd_builder_addl_psb_wide_key() {
        // PSB widens the length for the keys addl_len_is_wide lists
        // (here 'Layr'); the opaque payload then frames with a u64.
        let mut e = Emit::new();
        emit_addl(
            &mut e,
            Container::Psb,
            &AddlSpec::Opaque {
                sig: *b"8BIM",
                key: *b"Layr",
                payload: vec![1, 2],
            },
        );
        assert_eq!(&e.bytes[0..4], b"8BIM");
        assert_eq!(&e.bytes[4..8], b"Layr");
        assert_eq!(&e.bytes[8..16], &[0, 0, 0, 0, 0, 0, 0, 2]); // u64 len
    }
}
