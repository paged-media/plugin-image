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

//! The `samp` section — sampled tips.
//!
//! **This is the payload a brush engine actually paints with**; the rest
//! of the file is metadata about how to stamp it. Behaviour spec §2.

use crate::reader::ByteReader;
use crate::vm_array::{read_vm_array_list, VmRect, VmSamples};
use crate::Result;

use super::AbrWarning;

/// One sampled tip: an id, a rectangle, and a single-channel coverage
/// mask.
#[derive(Debug, Clone, PartialEq)]
pub struct AbrSample {
    /// The record's Pascal-string id — a 36-character UUID in every
    /// observed record, so the field occupies exactly 37 bytes.
    ///
    /// The public GIMP-lineage loader hardcodes a 37-byte skip and never
    /// reads the value; that is a coincidence of UUID length, not a
    /// format guarantee — and the id is load-bearing, because it is the
    /// join key a brush's `sampledData` resolves against (spec §2.2
    /// NOTE, §3.3).
    pub id: String,
    /// The plane's own rectangle, stored **y-first**. `top`/`left` are
    /// frequently non-zero and are the bounding box of the sampled
    /// region in the document the brush was defined from — provenance,
    /// NOT a canvas offset to honour when stamping (spec §2.2 trap,
    /// settled by the published-PNG oracle: each PNG is exactly `w`×`h`
    /// with no margin while the origins range over thousands of pixels).
    pub bounds: VmRect,
    pub width: u32,
    pub height: u32,
    /// 8 or 16. Every corpus tip is 8 (3,202/3,202).
    pub depth: u16,
    /// **Coverage.** 255 = fully painted, 0 = no paint. NOT inverted.
    ///
    /// Settled by a published-artwork oracle: 238 tips decoded this way
    /// are byte-identical to 238 independently exported transparent PNGs
    /// of the same artwork (mean absolute difference 0.000; the inverted
    /// reading differs by ~225). GIMP-lineage readers invert because
    /// GIMP's brush-mask convention is the opposite of coverage — that
    /// inversion belongs to their output stage and porting it inward
    /// paints the negative of the artwork (spec §2.5).
    pub coverage: VmSamples,
}

impl AbrSample {
    /// The origin the tip was sampled from. Preserve it for round-trip;
    /// ignore it when rendering.
    pub fn origin(&self) -> (i32, i32) {
        (self.bounds.left, self.bounds.top)
    }

    /// Coverage as 8-bit, ready for a brush engine. 16-bit tips are
    /// reduced by taking the high byte — explicitly lossy, and done here
    /// at the engine boundary rather than on the read path.
    pub fn coverage8(&self) -> Vec<u8> {
        self.coverage.to_eight_lossy()
    }
}

/// Parse a whole `samp` section body.
///
/// Records are a packed run read to the section end:
///
/// ```text
/// u32 brush_length
/// -- round brush_length UP to the next multiple of 4 --
/// record_end = (offset AFTER the length field) + rounded
/// ... body ...
/// seek to record_end
/// ```
///
/// Both details of that arithmetic are easy to get backwards and both
/// are load-bearing (spec §2.1 `[OBS]`, 3,202 records: declared lengths
/// are frequently 1–3 short of a multiple of 4 and the next record
/// starts on the rounded boundary). Seeking to `record_end` rather than
/// trusting the body handler is what makes the section robust against
/// unmodelled trailing fields.
pub(crate) fn parse_samp_section(
    r: &mut ByteReader,
    warnings: &mut Vec<AbrWarning>,
) -> Result<Vec<AbrSample>> {
    let mut out = Vec::new();
    // A record needs at least its own 4-byte length field.
    while r.remaining() >= 4 {
        let declared = r.u32()? as usize;
        let rounded = declared.div_ceil(4) * 4;
        let take = rounded.min(r.remaining());
        if rounded > r.remaining() {
            warnings.push(AbrWarning::SampleRecordOvershoot {
                declared,
                rounded,
                available: r.remaining(),
            });
        }
        // The declared length is the body WITHOUT the pad; the record
        // occupies the rounded length. Anything past `declared` inside
        // the record is that pad and is not "unaccounted for".
        let pad = take.saturating_sub(declared);
        let mut body = r.sub(take)?;
        if let Some(sample) = parse_record(&mut body, pad, warnings)? {
            out.push(sample);
        }
    }
    Ok(out)
}

fn parse_record(
    body: &mut ByteReader,
    pad: usize,
    warnings: &mut Vec<AbrWarning>,
) -> Result<Option<AbrSample>> {
    // Pascal string, pad-to-1 (no padding) — the simple case.
    let n = body.u8()? as usize;
    let id = String::from_utf8_lossy(body.take(n)?).into_owned();

    let list = read_vm_array_list(body)?;

    // THE STRUCTURAL SELF-CHECK (spec §2.2): a correct parse of the
    // array-list ends the record with ZERO bytes unaccounted for. The
    // eight bytes every public description calls "unexplained trailing
    // bytes" are the container's last two empty slots, written after the
    // pixel data. A leftover here means the layout was mis-parsed, and
    // that is exactly what we want to hear about.
    if body.remaining() != pad {
        warnings.push(AbrWarning::SampleRecordTrailingBytes {
            id: id.clone(),
            remaining: body.remaining(),
            expected_pad: pad,
        });
    }

    if list.arrays.len() > 1 {
        // A brush tip is a single-channel mask; more than one written
        // plane is not something any observed record does.
        warnings.push(AbrWarning::MultiplePlanes {
            id: id.clone(),
            planes: list.arrays.len(),
        });
    }
    let Some(plane) = list.arrays.first() else {
        warnings.push(AbrWarning::SampleHasNoPlane { id });
        return Ok(None);
    };
    if plane.depth_disagrees() {
        warnings.push(AbrWarning::PlaneDepthDisagrees {
            id: id.clone(),
            pixel_depth: plane.pixel_depth,
            depth: plane.depth,
        });
    }
    if plane.depth == 16 {
        warnings.push(AbrWarning::SixteenBitPlaneUnverified { id: id.clone() });
    }
    let (Some(width), Some(height)) = (plane.width(), plane.height()) else {
        warnings.push(AbrWarning::SampleDecodeFailed {
            id,
            detail: format!("non-positive plane bounds {:?}", plane.bounds),
        });
        return Ok(None);
    };
    // A decode failure degrades to "this tip is unavailable" plus a
    // named diagnostic — it does not fail the file, because one bad
    // record must not cost a 2,000-tip library.
    let coverage = match plane.decode() {
        Ok(c) => c,
        Err(e) => {
            warnings.push(AbrWarning::SampleDecodeFailed {
                id,
                detail: e.to_string(),
            });
            return Ok(None);
        }
    };
    Ok(Some(AbrSample {
        id,
        bounds: plane.bounds,
        width,
        height,
        depth: plane.depth,
        coverage,
    }))
}
