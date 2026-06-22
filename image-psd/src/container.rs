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

//! PSD vs PSB: one model, two containers (spec §10.4). The container
//! is set once from the header version (1 = PSD, 2 = PSB) and threaded
//! as a plain `Copy` parameter through every length/offset read and
//! write — never inferred ad hoc.
//!
//! Provenance: Adobe Photoshop File Format specification, "PSB" notes —
//! PSB widens specific length fields to 8 bytes and raises the canvas
//! limit from 30,000 to 300,000 px.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Psd,
    Psb,
}

impl Container {
    pub const fn version(self) -> u16 {
        match self {
            Container::Psd => 1,
            Container::Psb => 2,
        }
    }

    pub fn from_version(v: u16) -> Option<Self> {
        match v {
            1 => Some(Container::Psd),
            2 => Some(Container::Psb),
            _ => None,
        }
    }

    /// Max canvas edge (px) — used for validation only.
    pub const fn max_dimension(self) -> u32 {
        match self {
            Container::Psd => 30_000,
            Container::Psb => 300_000,
        }
    }

    /// Width in bytes of the *widened* length fields (layer & mask info
    /// section length, layer info length, per-channel data length).
    pub const fn section_len_width(self) -> LenWidth {
        match self {
            Container::Psd => LenWidth::U32,
            Container::Psb => LenWidth::U64,
        }
    }

    /// Whether an additional-layer-info block with `key` carries an
    /// 8-byte length in THIS container. In PSD all keys use 4 bytes;
    /// in PSB the spec enumerates the keys whose data can exceed 4 GiB.
    /// Provenance: Adobe spec, "Layer records / Additional Layer
    /// Information" PSB note.
    pub fn addl_len_is_wide(self, key: [u8; 4]) -> bool {
        match self {
            Container::Psd => false,
            Container::Psb => matches!(
                &key,
                b"LMsk"
                    | b"Lr16"
                    | b"Lr32"
                    | b"Layr"
                    | b"Mt16"
                    | b"Mt32"
                    | b"Mtrn"
                    | b"Alph"
                    | b"FMsk"
                    | b"lnk2"
                    | b"FEid"
                    | b"FXid"
                    | b"PxSD"
            ),
        }
    }
}

/// Length-field width for the container-parameterized framing helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LenWidth {
    U32,
    U64,
}

impl LenWidth {
    pub const fn bytes(self) -> usize {
        match self {
            LenWidth::U32 => 4,
            LenWidth::U64 => 8,
        }
    }
}
