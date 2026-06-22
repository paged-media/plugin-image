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

//! `ImageTarget` — streaming encode on CPU workers (spec §10.2,
//! frozen). Fed strip-by-strip by the `to_encoder` sink (M1), the only
//! structured GPU-readback path in the system (§7.3).

use image_core::{PixelFormat, Region, TileSliceRef};

use crate::Result;

#[derive(Debug, Clone)]
pub struct TargetInfo {
    pub width: u32,
    pub height: u32,
    /// Format of the strips that will be written.
    pub format: PixelFormat,
    /// ICC profile to embed, if any.
    pub icc: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodedStats {
    pub bytes_written: u64,
}

pub trait ImageTarget {
    fn begin(&mut self, info: TargetInfo) -> Result<()>;

    /// Strips arrive top-to-bottom, non-overlapping, jointly covering
    /// the full target extent.
    fn write_strip(&mut self, region: Region, data: &TileSliceRef<'_>) -> Result<()>;

    fn finish(&mut self) -> Result<EncodedStats>;
}
