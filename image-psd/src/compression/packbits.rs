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

//! PackBits RLE — decode-tolerant, encode-canonical. We decode any
//! valid PackBits stream (including `-128` no-op bytes, which are
//! skipped); our encoder emits exactly one canonical form. Unedited
//! channels round-trip byte-identically anyway via the verbatim
//! payload (`ChannelData.bytes`), which is why canonical-only encoding
//! is safe (crate-level strategy 3).
//!
//! Provenance: Apple PackBits as referenced by the Adobe Photoshop
//! File Format specification ("RLE compressed ... with the byte counts
//! ... rows are padded to an even size"); TIFF 6.0 spec, PackBits
//! appendix.
//!
//! IMPLEMENTATION LANDS IN M0 FAN-OUT (unit U1) with the full edge-case
//! matrix: empty input, single byte, max literal run (128), max
//! replicate run (128), `-128` no-op skip, run-boundary row ends,
//! encode(decode(x)) canonical idempotence.

use crate::{PsdError, Result};

/// Decode one PackBits stream into `out`. `out.len()` must be the
/// exact expected unpacked size (PSD rows are independent streams).
pub fn decode(src: &[u8], out: &mut [u8]) -> Result<()> {
    let (_, _) = (src, out);
    Err(PsdError::Unsupported(
        "packbits decode lands with M0 fan-out unit U1".into(),
    ))
}

/// Encode `src` into the canonical PackBits form.
pub fn encode(src: &[u8]) -> Vec<u8> {
    let _ = src;
    unimplemented!("packbits encode lands with M0 fan-out unit U1")
}
