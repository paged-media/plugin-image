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

//! The sans-IO byte seam (spec §10.2). M0 ships the memory backing;
//! native file and OPFS/ReadableStream backings are M1 (the OPFS one
//! additionally gated on BREAKAGE I-03).

use crate::{CodecError, Result};

pub trait ByteSource {
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `buf` from `offset`. Short reads are errors — callers size
    /// `buf` from `len()`/container metadata.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct MemoryByteSource {
    bytes: std::sync::Arc<[u8]>,
}

impl MemoryByteSource {
    pub fn new(bytes: impl Into<std::sync::Arc<[u8]>>) -> Self {
        MemoryByteSource {
            bytes: bytes.into(),
        }
    }
}

impl ByteSource for MemoryByteSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .filter(|&e| e <= self.bytes.len() as u64)
            .ok_or(CodecError::OutOfBounds {
                offset,
                len: buf.len(),
                source_len: self.bytes.len() as u64,
            })?;
        buf.copy_from_slice(&self.bytes[offset as usize..end as usize]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_read_at() {
        let mut s = MemoryByteSource::new(vec![1u8, 2, 3, 4].into_boxed_slice());
        let mut buf = [0u8; 2];
        s.read_at(1, &mut buf).unwrap();
        assert_eq!(buf, [2, 3]);
        assert!(s.read_at(3, &mut buf).is_err());
    }
}
