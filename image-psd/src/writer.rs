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

//! Big-endian emitter with length back-patching. The writer never
//! trusts a stored length: `framed()` writes a placeholder, emits the
//! children, then back-patches — so section lengths are correct by
//! construction (spec §10.4 "the writer maintains section lengths").
//!
//! Padding rules are explicit, anchored helpers (`pad_to`) because PSD
//! mixes 2-byte and 4-byte alignment per structure — never inline
//! ad-hoc `+1 & !1` math (the documented risk surface).

use crate::container::LenWidth;

#[derive(Debug, Default)]
pub struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    pub fn new() -> Self {
        ByteWriter { buf: Vec::new() }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn i16(&mut self, v: i16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn fourcc(&mut self, v: [u8; 4]) {
        self.buf.extend_from_slice(&v);
    }

    pub fn len_field(&mut self, width: LenWidth, v: u64) {
        match width {
            LenWidth::U32 => self.u32(v as u32),
            LenWidth::U64 => self.u64(v),
        }
    }

    /// Length-framed section: writes a `width` placeholder, runs `f`,
    /// back-patches the byte count of everything `f` emitted.
    pub fn framed(&mut self, width: LenWidth, f: impl FnOnce(&mut Self)) {
        let len_pos = self.buf.len();
        self.len_field(width, 0);
        let start = self.buf.len();
        f(self);
        let len = (self.buf.len() - start) as u64;
        match width {
            LenWidth::U32 => {
                self.buf[len_pos..len_pos + 4].copy_from_slice(&(len as u32).to_be_bytes());
            }
            LenWidth::U64 => {
                self.buf[len_pos..len_pos + 8].copy_from_slice(&len.to_be_bytes());
            }
        }
    }

    /// Zero-pad so that (len - from) is a multiple of `align`. `from`
    /// is the anchor the alignment is measured against (e.g. the start
    /// of a resource block, the start of an addl-info payload).
    pub fn pad_to(&mut self, align: usize, from: usize) {
        debug_assert!(self.buf.len() >= from);
        let rem = (self.buf.len() - from) % align;
        if rem != 0 {
            self.buf.resize(self.buf.len() + (align - rem), 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framed_backpatches() {
        let mut w = ByteWriter::new();
        w.framed(LenWidth::U32, |w| {
            w.u16(0xBEEF);
            w.u8(0x01);
        });
        assert_eq!(w.into_bytes(), vec![0, 0, 0, 3, 0xBE, 0xEF, 0x01]);
    }

    #[test]
    fn framed_u64_nested() {
        let mut w = ByteWriter::new();
        w.framed(LenWidth::U64, |w| {
            w.framed(LenWidth::U32, |w| w.u8(7));
        });
        // outer len = 4 (inner len field) + 1 (payload) = 5
        assert_eq!(w.into_bytes(), vec![0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 7]);
    }

    #[test]
    fn pad_anchored() {
        let mut w = ByteWriter::new();
        w.u8(0xAA); // 1 byte before the anchored structure
        let anchor = w.len();
        w.bytes(&[1, 2, 3]);
        w.pad_to(4, anchor);
        assert_eq!(w.len() - anchor, 4);
        w.pad_to(4, anchor); // aligned: no-op
        assert_eq!(w.len() - anchor, 4);
    }
}
