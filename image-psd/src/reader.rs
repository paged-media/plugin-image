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

//! Big-endian cursor with explicit padding helpers. All multi-byte PSD
//! fields are big-endian (Adobe spec, "File Header Section").

use crate::container::LenWidth;
use crate::{PsdError, Result};

#[derive(Debug, Clone)]
pub struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        ByteReader { buf, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn is_at_end(&self) -> bool {
        self.pos == self.buf.len()
    }

    fn need(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            Err(PsdError::Truncated {
                offset: self.pos,
                needed: n,
                available: self.remaining(),
            })
        } else {
            Ok(())
        }
    }

    /// Borrow `n` bytes and advance.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// A sub-reader over the next `n` bytes (length-framed section);
    /// the parent cursor advances past them.
    pub fn sub(&mut self, n: usize) -> Result<ByteReader<'a>> {
        Ok(ByteReader::new(self.take(n)?))
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn fourcc(&mut self) -> Result<[u8; 4]> {
        Ok(self.take(4)?.try_into().unwrap())
    }

    /// Container-parameterized length field (the PSB-widened ones).
    pub fn len_field(&mut self, width: LenWidth) -> Result<u64> {
        match width {
            LenWidth::U32 => Ok(self.u32()? as u64),
            LenWidth::U64 => self.u64(),
        }
    }

    /// Skip forward so that (pos - from) is a multiple of `align`.
    /// PSD's padding rules are positional: the padded extent starts at
    /// a known anchor (`from`), not at the buffer origin.
    pub fn align(&mut self, align: usize, from: usize) -> Result<()> {
        debug_assert!(self.pos >= from);
        let rem = (self.pos - from) % align;
        if rem != 0 {
            self.take(align - rem)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn be_reads() {
        let mut r = ByteReader::new(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        assert_eq!(r.u16().unwrap(), 0x0102);
        assert_eq!(r.u32().unwrap(), 0x0304_0506);
        assert_eq!(r.remaining(), 2);
        assert!(r.u32().is_err()); // truncation reported, cursor safe
    }

    #[test]
    fn align_from_anchor() {
        let data = [0u8; 8];
        let mut r = ByteReader::new(&data);
        let anchor = r.pos();
        r.take(3).unwrap();
        r.align(4, anchor).unwrap();
        assert_eq!(r.pos(), 4);
        r.align(4, anchor).unwrap(); // already aligned: no-op
        assert_eq!(r.pos(), 4);
    }

    #[test]
    fn sub_reader_isolates() {
        let mut r = ByteReader::new(&[1, 2, 3, 4, 5]);
        let mut s = r.sub(3).unwrap();
        assert_eq!(s.take(3).unwrap(), &[1, 2, 3]);
        assert!(s.u8().is_err());
        assert_eq!(r.take(2).unwrap(), &[4, 5]);
    }
}
