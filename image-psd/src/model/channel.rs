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

//! Per-channel image data. Stored as the on-disk compressed payload —
//! decode-on-demand (preservation + the 500 MB-PSB streaming budget).
//! RAW (0) and RLE (1) decode↔encode in M0; ZIP (2/3) is parse-only
//! opaque until M1 (`rendered` needs it; `preserved` doesn't).
//!
//! Provenance: Adobe Photoshop File Format specification, "Channel
//! Image Data".

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    Raw,
    Rle,
    Zip,
    ZipPrediction,
}

impl Compression {
    pub const fn code(self) -> u16 {
        match self {
            Compression::Raw => 0,
            Compression::Rle => 1,
            Compression::Zip => 2,
            Compression::ZipPrediction => 3,
        }
    }

    pub fn from_code(c: u16) -> Option<Self> {
        Some(match c {
            0 => Compression::Raw,
            1 => Compression::Rle,
            2 => Compression::Zip,
            3 => Compression::ZipPrediction,
            _ => return None,
        })
    }
}

/// One channel's image data: the compression tag + the payload exactly
/// as stored (for RLE this INCLUDES the per-row byte-count table that
/// precedes the packed rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelData {
    pub compression: Compression,
    pub bytes: Vec<u8>,
}
