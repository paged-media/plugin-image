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

//! CPU-side pixel slice views — the codec bridge surface (spec §10.2):
//! `ImageSource::read_region` fills a `TileSliceMut`,
//! `ImageTarget::write_strip` reads a `TileSliceRef`. Interleaved
//! layout, explicit row stride (in BYTES), format always carried.

use crate::format::PixelFormat;
use crate::region::Region;

#[derive(Debug)]
pub struct TileSliceMut<'a> {
    pub region: Region,
    pub format: PixelFormat,
    /// Interleaved pixel bytes; rows separated by `row_stride` bytes.
    pub bytes: &'a mut [u8],
    pub row_stride: usize,
}

#[derive(Debug)]
pub struct TileSliceRef<'a> {
    pub region: Region,
    pub format: PixelFormat,
    pub bytes: &'a [u8],
    pub row_stride: usize,
}

impl TileSliceMut<'_> {
    /// Minimal coherence check: the buffer must hold `h` rows of at
    /// least `w * bpp` bytes at the declared stride.
    pub fn validate(&self) -> bool {
        let bpp = self.format.bytes_per_pixel();
        let row_bytes = self.region.w as usize * bpp;
        self.row_stride >= row_bytes
            && self.bytes.len()
                >= self.row_stride * self.region.h.saturating_sub(1) as usize + row_bytes
    }
}

impl<'a> TileSliceRef<'a> {
    pub fn row(&self, y: u32) -> &'a [u8] {
        let bpp = self.format.bytes_per_pixel();
        let start = y as usize * self.row_stride;
        &self.bytes[start..start + self.region.w as usize * bpp]
    }
}
