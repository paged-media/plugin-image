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

//! The INDEPENDENT byte emitter (spec §10.4 oracle 1). A deliberately
//! separate code path from image-psd's production writer — its own
//! big-endian primitives and its own PackBits encoder — so the PSD
//! round-trip suite is never self-referential: if both sides shared a
//! bug it would hide, so they share nothing but the on-disk format.
//!
//! All integers are big-endian (the PSD/PSB byte order). PSB widens
//! exactly the fields the brief calls out; that width is carried as a
//! [`LenWidth`] argument, never inferred.

use image_psd::container::LenWidth;

/// A growable big-endian byte sink. Every PSD scalar has a method; there
/// is no host-endianness assumption anywhere — `to_be_bytes` is the only
/// integer→bytes path.
#[derive(Debug, Default, Clone)]
pub struct Emit {
    pub bytes: Vec<u8>,
}

impl Emit {
    pub fn new() -> Self {
        Emit { bytes: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.bytes.push(v);
        self
    }

    pub fn i8(&mut self, v: i8) -> &mut Self {
        self.bytes.push(v as u8);
        self
    }

    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_be_bytes());
        self
    }

    pub fn i16(&mut self, v: i16) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_be_bytes());
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_be_bytes());
        self
    }

    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_be_bytes());
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_be_bytes());
        self
    }

    /// A container-parameterized length field: 4 bytes (PSD) or 8 bytes
    /// (PSB). The single place PSB widening is realized.
    pub fn len_field(&mut self, w: LenWidth, v: u64) -> &mut Self {
        match w {
            LenWidth::U32 => self.u32(v as u32),
            LenWidth::U64 => self.u64(v),
        }
    }

    /// An RLE per-row byte count: u16 in PSD, u32 in PSB (brief §10/§11).
    pub fn row_count(&mut self, w: LenWidth, v: u32) -> &mut Self {
        match w {
            LenWidth::U32 => self.u16(v as u16),
            LenWidth::U64 => self.u32(v),
        }
    }

    pub fn raw(&mut self, b: &[u8]) -> &mut Self {
        self.bytes.extend_from_slice(b);
        self
    }

    /// A 4-byte ASCII tag ('8BPS', '8BIM', a blend key, an addl key).
    pub fn fourcc(&mut self, tag: [u8; 4]) -> &mut Self {
        self.bytes.extend_from_slice(&tag);
        self
    }

    /// Pad with zero bytes until the total length is a multiple of
    /// `align` (1 = no-op). Returns the count of pad bytes written.
    pub fn pad_to(&mut self, align: usize) -> usize {
        let pad = (align - (self.bytes.len() % align)) % align;
        self.bytes.resize(self.bytes.len() + pad, 0);
        pad
    }

    /// A Pascal string whose *whole field* (length byte + content +
    /// padding) is rounded up to a multiple of `align`. The empty name
    /// at align=2 occupies 2 bytes (the brief's minimum); a layer name
    /// at align=4 follows brief §5.
    pub fn pascal_string(&mut self, s: &[u8], align: usize) -> &mut Self {
        let start = self.bytes.len();
        let n = s.len().min(255);
        self.bytes.push(n as u8);
        self.bytes.extend_from_slice(&s[..n]);
        let field = self.bytes.len() - start;
        let pad = (align - (field % align)) % align;
        self.bytes.resize(self.bytes.len() + pad, 0);
        self
    }
}

/// PackBits-encode one independent row (brief §12). Control byte `n` as
/// i8: `0..=127` ⇒ copy the next `n+1` literals; `-127..=-1` ⇒ repeat
/// the next byte `1-n` times; `-128` is reserved (never emitted).
///
/// This is the encoder's own greedy strategy — runs of ≥ 3 equal bytes
/// become repeats, everything else accumulates into literal spans
/// (capped at 128). It is NOT image-psd's encoder; the round-trip suite
/// decodes these bytes with the production decoder, which is the point.
pub fn pack_bits_row(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let n = src.len();
    while i < n {
        // Length of the run of identical bytes starting at i.
        let mut run = 1usize;
        while i + run < n && src[i + run] == src[i] && run < 128 {
            run += 1;
        }
        if run >= 3 {
            // Worth a repeat packet: control = 1 - run (i.e. -(run-1)).
            out.push((1i32 - run as i32) as i8 as u8);
            out.push(src[i]);
            i += run;
        } else {
            // Accumulate a literal span up to 128 bytes, stopping early
            // when a ≥3 run begins (so the run is encoded as a repeat).
            let lit_start = i;
            let mut lit_len = 0usize;
            while i < n && lit_len < 128 {
                let mut look = 1usize;
                while i + look < n && src[i + look] == src[i] && look < 3 {
                    look += 1;
                }
                if look >= 3 {
                    break;
                }
                i += 1;
                lit_len += 1;
            }
            out.push((lit_len as i32 - 1) as i8 as u8);
            out.extend_from_slice(&src[lit_start..lit_start + lit_len]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psd_builder_emit_be_scalars() {
        let mut e = Emit::new();
        e.u16(0x1234).u32(0x89AB_CDEF).i32(-1);
        assert_eq!(
            e.bytes,
            vec![0x12, 0x34, 0x89, 0xAB, 0xCD, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF]
        );
    }

    #[test]
    fn psd_builder_emit_len_field_widths() {
        let mut psd = Emit::new();
        psd.len_field(LenWidth::U32, 0x0102_0304);
        assert_eq!(psd.bytes, vec![0x01, 0x02, 0x03, 0x04]);

        let mut psb = Emit::new();
        psb.len_field(LenWidth::U64, 0x0102_0304);
        assert_eq!(psb.bytes, vec![0, 0, 0, 0, 0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn psd_builder_emit_row_count_widths() {
        let mut psd = Emit::new();
        psd.row_count(LenWidth::U32, 0x0102);
        assert_eq!(psd.bytes, vec![0x01, 0x02]);

        let mut psb = Emit::new();
        psb.row_count(LenWidth::U64, 0x0102);
        assert_eq!(psb.bytes, vec![0, 0, 0x01, 0x02]);
    }

    #[test]
    fn psd_builder_emit_pascal_even_padding() {
        // Empty name at align 2 ⇒ length byte + one pad = 2 bytes.
        let mut e = Emit::new();
        e.pascal_string(b"", 2);
        assert_eq!(e.bytes, vec![0x00, 0x00]);

        // "ab": length(1) + 2 content = 3 bytes, padded to 4.
        let mut e = Emit::new();
        e.pascal_string(b"ab", 2);
        assert_eq!(e.bytes, vec![0x02, b'a', b'b', 0x00]);

        // "abc": length(1) + 3 = 4, already even ⇒ no pad.
        let mut e = Emit::new();
        e.pascal_string(b"abc", 2);
        assert_eq!(e.bytes, vec![0x03, b'a', b'b', b'c']);
    }

    #[test]
    fn psd_builder_emit_pascal_align4_layer_name() {
        // Layer name padded to a multiple of 4 INCLUDING the length byte.
        // "x": length(1)+1 = 2 → pad to 4.
        let mut e = Emit::new();
        e.pascal_string(b"x", 4);
        assert_eq!(e.bytes, vec![0x01, b'x', 0x00, 0x00]);

        // "abc": length(1)+3 = 4 → no pad.
        let mut e = Emit::new();
        e.pascal_string(b"abc", 4);
        assert_eq!(e.bytes, vec![0x03, b'a', b'b', b'c']);

        // "abcd": field = length byte(1) + 4 content = 5 → pad to 8. The
        // length byte stores the CONTENT count (4), not the field width.
        let mut e = Emit::new();
        e.pascal_string(b"abcd", 4);
        assert_eq!(e.bytes.len(), 8);
        assert_eq!(&e.bytes[..5], &[0x04, b'a', b'b', b'c', b'd']);
    }

    #[test]
    fn psd_builder_packbits_repeat_run() {
        // 5 identical bytes ⇒ one repeat packet: control = 1-5 = -4.
        let row = vec![0xAA; 5];
        let packed = pack_bits_row(&row);
        assert_eq!(packed, vec![(-4i8) as u8, 0xAA]);
    }

    #[test]
    fn psd_builder_packbits_literal_run() {
        // 4 distinct bytes ⇒ one literal packet: control = 4-1 = 3.
        let row = vec![1, 2, 3, 4];
        let packed = pack_bits_row(&row);
        assert_eq!(packed, vec![3, 1, 2, 3, 4]);
    }

    #[test]
    fn psd_builder_packbits_mixed() {
        // [1,2, AA,AA,AA, 9]: literal(1,2) then repeat(AA×3) then
        // literal(9). control bytes: 1 (copy 2), -2 (repeat 3), 0 (copy 1).
        let row = vec![1, 2, 0xAA, 0xAA, 0xAA, 9];
        let packed = pack_bits_row(&row);
        assert_eq!(packed, vec![1, 1, 2, (-2i8) as u8, 0xAA, 0, 9]);
    }

    #[test]
    fn psd_builder_packbits_long_literal_caps_at_128() {
        // 200 strictly increasing-then-wrapping bytes never form a ≥3
        // run, so they split into a 128-literal packet + a 72-literal
        // packet.
        let row: Vec<u8> = (0..200u32).map(|i| (i % 251) as u8).collect();
        let packed = pack_bits_row(&row);
        assert_eq!(packed[0], 127); // 128 literals
        assert_eq!(&packed[1..129], &row[..128]);
        assert_eq!(packed[129], 71); // 72 literals
    }
}
