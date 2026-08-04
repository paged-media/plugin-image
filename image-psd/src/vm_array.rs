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

//! The PSD **virtual memory array list** — the self-describing container
//! Photoshop uses to carry a set of channel planes with their own
//! bounds, depth and per-plane compression.
//!
//! It is the body of an `.abr` sampled-tip record (behaviour spec §2.2)
//! and of a PSD/ABR pattern record (§8.1), so it is built here once,
//! next to the descriptor tree, rather than in either consumer.
//!
//! # Provenance
//!
//! `thoughts/docs/paged/plugin-image/abr-brush-format-spec.md` §2.2 —
//! `[OBS]`, decoded byte-exactly from 3,202 sampled-tip records across
//! 7 files; §2.3/§2.4 for the raw and PackBits-with-row-table plane
//! payloads. The Adobe Photoshop File Formats Specification documents
//! the pattern record that embeds this container `[PUB]` but not the
//! container's own field layout. `references/` is never read here.
//!
//! # Why the structure and not the constant
//!
//! Every public description of the ABR sampled-tip record — and revision
//! 1 of the behaviour spec — calls the region between the record id and
//! the bounds an opaque **264-byte skip**. It is not a skip. 264 is what
//! this container's header measures when `array_count == 56` and the one
//! written plane sits in slot 55; the arithmetic is
//! `4 + 4 + 4 + 16 + 4` header bytes, `55 × 4` empty slot words, and
//! `4 + 4 + 4` of the written slot's own prefix. A file that writes a
//! different `array_count`, a different slot, or more than one plane
//! breaks the constant *silently* and reads its bounds out of the middle
//! of the slot table.
//!
//! Parsing the structure also explains the eight bytes every record was
//! said to carry "unexplained" at the end: the slot loop runs
//! `array_count + 2` times, and the last two empty slots are written
//! AFTER the pixel data. A correct parse therefore ends a record with
//! **zero** bytes unaccounted for, which is the cheap, strong self-check
//! this module exposes as [`VmArrayList::consumed`].

use crate::compression::packbits;
use crate::reader::ByteReader;
use crate::{PsdError, Result};

/// Absolute ceiling on the pixel count of one plane. A brush tip is a
/// few thousand pixels on a side; this is a hostile-input guard so a
/// crafted bounds rectangle cannot request a huge allocation. `w` and
/// `h` are differences of two attacker-controlled `i32`s (spec §2.2
/// trap 21), so the product is computed widened and bounded before any
/// allocation happens.
pub const MAX_PLANE_PIXELS: u64 = 64 * 1024 * 1024;

/// Maximum expansion factor of a PackBits stream: the densest possible
/// packet is a two-byte replicate run producing 128 output bytes. Used
/// to bound a declared plane size against its actual compressed payload,
/// so an RLE plane claiming absurd dimensions is rejected before it is
/// allocated rather than after it fails to decode.
pub const MAX_PACKBITS_EXPANSION: u64 = 128;

/// The `vm_version` word every observed container carries (`[OBS]`,
/// 3,202/3,202).
pub const VM_VERSION: u32 = 3;

/// A bounding rectangle as stored: **y first** (spec §2.2 trap 26 —
/// reading these as `(x, y, w, h)` in source order transposes the plane
/// and looks plausible on a round tip).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VmRect {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
}

impl VmRect {
    /// `right - left`, or `None` when the rectangle is degenerate or
    /// inverted. Never returns 0.
    pub fn width(&self) -> Option<u32> {
        let w = (self.right as i64) - (self.left as i64);
        (w > 0 && w <= u32::MAX as i64).then_some(w as u32)
    }

    /// `bottom - top`, or `None` when degenerate or inverted.
    pub fn height(&self) -> Option<u32> {
        let h = (self.bottom as i64) - (self.top as i64);
        (h > 0 && h <= u32::MAX as i64).then_some(h as u32)
    }

    fn read(r: &mut ByteReader) -> Result<VmRect> {
        Ok(VmRect {
            top: r.i32()?,
            left: r.i32()?,
            bottom: r.i32()?,
            right: r.i32()?,
        })
    }
}

/// Compression tag of one plane's payload.
pub const COMPRESSION_RAW: u8 = 0;
pub const COMPRESSION_RLE: u8 = 1;

/// One written plane (one array slot).
#[derive(Debug, Clone, PartialEq)]
pub struct VmArray {
    /// The slot index this plane occupied. Retained because the plane's
    /// position in the slot table is the only thing that distinguishes
    /// planes in a multi-plane container, and because it is what makes
    /// the "264" arithmetic reproducible.
    pub slot: u32,
    /// A 32-bit copy of the bit depth. Agreed with `depth` in every
    /// observed record (`[OBS]` 3,202/3,202); disagreement is reported
    /// by [`VmArray::depth_disagrees`] rather than silently resolved.
    pub pixel_depth: u32,
    /// The plane's OWN rectangle — the one that dimensions its pixel
    /// data. The container repeats an outer rectangle before the slot
    /// table; if the two ever disagree, this is the authoritative one
    /// (spec §2.2 trap).
    pub bounds: VmRect,
    /// 8 or 16.
    pub depth: u16,
    /// [`COMPRESSION_RAW`] or [`COMPRESSION_RLE`].
    pub compression: u8,
    /// The payload exactly as stored — compressed for RLE. Kept
    /// undecoded so a plane is only expanded when someone asks.
    pub data: Vec<u8>,
}

/// Decoded plane samples. 16-bit planes are NOT truncated to 8 bits:
/// the format stores 16, `image-core` has 16-bit-capable buffers, and
/// quantising on read would be a lossy decision dressed as a format
/// fact (spec §2.3 trap).
#[derive(Debug, Clone, PartialEq)]
pub enum VmSamples {
    Eight(Vec<u8>),
    Sixteen(Vec<u16>),
}

impl VmSamples {
    pub fn len(&self) -> usize {
        match self {
            VmSamples::Eight(v) => v.len(),
            VmSamples::Sixteen(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The samples as 8-bit. 16-bit input is reduced by taking the high
    /// byte. LOSSY and deliberately explicit — call it at a boundary
    /// that genuinely wants 8 bits (the brush-engine bridge), never on
    /// the read path.
    pub fn to_eight_lossy(&self) -> Vec<u8> {
        match self {
            VmSamples::Eight(v) => v.clone(),
            VmSamples::Sixteen(v) => v.iter().map(|s| (s >> 8) as u8).collect(),
        }
    }
}

impl VmArray {
    pub fn width(&self) -> Option<u32> {
        self.bounds.width()
    }

    pub fn height(&self) -> Option<u32> {
        self.bounds.height()
    }

    /// `true` when the redundant 32-bit depth copy disagrees with the
    /// 16-bit one. Never observed; worth a diagnostic if it happens.
    pub fn depth_disagrees(&self) -> bool {
        self.pixel_depth != self.depth as u32
    }

    /// Expand the plane.
    ///
    /// * `compression == 0` — `w × h` samples, row-major, no row padding.
    ///   Rare but real (spec §2.3 `[OBS]`, 19 records): not a branch to
    ///   defer.
    /// * `compression == 1` — PackBits with an up-front table of `h`
    ///   `u16` compressed row lengths, ALL rows first, then the row
    ///   payloads back to back (spec §2.4). NOT a 2-byte prefix per row;
    ///   interleaving decodes row 0 correctly and garbles the rest.
    ///
    /// 16-bit RLE is REFUSED with a specific diagnostic: no fixture in
    /// the corpus is 16-bit at all, and whether the row table counts
    /// compressed bytes of 16-bit samples or Photoshop splits high and
    /// low byte planes as PSD does is genuinely unknown (spec §2.4 GAP /
    /// §14.2 item 1). Guessing here would be inventing a format.
    pub fn decode(&self) -> Result<VmSamples> {
        let (w, h) = self.dimensions()?;
        let pixels = (w as u64) * (h as u64);
        if pixels > MAX_PLANE_PIXELS {
            return Err(malformed(format!(
                "plane of {w}×{h} = {pixels} pixels exceeds the {MAX_PLANE_PIXELS}-pixel guard"
            )));
        }
        match (self.compression, self.depth) {
            (COMPRESSION_RAW, 8) => {
                let need = pixels as usize;
                if self.data.len() < need {
                    return Err(malformed(format!(
                        "raw 8-bit plane needs {need} byte(s), payload has {}",
                        self.data.len()
                    )));
                }
                Ok(VmSamples::Eight(self.data[..need].to_vec()))
            }
            (COMPRESSION_RAW, 16) => {
                // Layout stated by the behaviour spec (§2.3) and by the
                // PSD 16-bit channel convention, but NEVER exercised by
                // any fixture — no 16-bit tip exists in the corpus. The
                // caller is told via `AbrWarning::SixteenBitPlaneUnverified`.
                let need = pixels as usize * 2;
                if self.data.len() < need {
                    return Err(malformed(format!(
                        "raw 16-bit plane needs {need} byte(s), payload has {}",
                        self.data.len()
                    )));
                }
                let mut out = Vec::with_capacity(pixels as usize);
                for chunk in self.data[..need].chunks_exact(2) {
                    out.push(u16::from_be_bytes([chunk[0], chunk[1]]));
                }
                Ok(VmSamples::Sixteen(out))
            }
            (COMPRESSION_RLE, 8) => Ok(VmSamples::Eight(self.decode_rle8(w, h)?)),
            (COMPRESSION_RLE, 16) => Err(PsdError::Unsupported(
                "16-bit RLE plane: the row-length table's meaning for 16-bit samples is not \
                 established (no 16-bit tip exists in the fixture corpus), and guessing between \
                 'compressed bytes of 16-bit samples' and PSD's split high/low byte planes would \
                 silently produce wrong pixels"
                    .into(),
            )),
            (c, d) => Err(PsdError::Unsupported(format!(
                "plane compression {c} at bit depth {d}"
            ))),
        }
    }

    fn dimensions(&self) -> Result<(u32, u32)> {
        match (self.width(), self.height()) {
            (Some(w), Some(h)) => Ok((w, h)),
            _ => Err(malformed(format!(
                "non-positive plane bounds {:?}",
                self.bounds
            ))),
        }
    }

    fn decode_rle8(&self, w: u32, h: u32) -> Result<Vec<u8>> {
        let pixels = (w as u64) * (h as u64);
        // A PackBits stream expands by at most 128×, so a payload of N
        // bytes cannot legitimately produce more than 128·N samples.
        // Reject before allocating rather than after decoding fails.
        if pixels > MAX_PACKBITS_EXPANSION * self.data.len() as u64 {
            return Err(malformed(format!(
                "RLE plane claims {w}×{h} = {pixels} pixels, which a {}-byte payload cannot \
                 produce (PackBits expands by at most {MAX_PACKBITS_EXPANSION}×)",
                self.data.len()
            )));
        }
        let mut r = ByteReader::new(&self.data);
        let mut row_lengths = Vec::with_capacity(h as usize);
        for _ in 0..h {
            row_lengths.push(r.u16()? as usize);
        }
        let mut out = vec![0u8; pixels as usize];
        for (y, len) in row_lengths.iter().enumerate() {
            let row = r.take(*len)?;
            let start = y * w as usize;
            // Each row decompresses to exactly `w` bytes (spec §2.4
            // `[OBS]`, 3,183 records / ~1.1M rows, zero short rows and
            // zero overruns). `image-psd`'s existing PackBits decoder
            // errors on either, and the corpus says it would have
            // accepted every record — so it is reused UNCHANGED. An
            // overrun is a warning-worthy anomaly, not an expected
            // condition, and surfacing it as an error is the honest
            // behaviour for a hostile-input reader.
            packbits::decode(row, &mut out[start..start + w as usize])?;
        }
        Ok(out)
    }
}

/// A parsed virtual memory array list.
#[derive(Debug, Clone, PartialEq)]
pub struct VmArrayList {
    /// Always `0x0001_0000` in every observed record. Retained, not
    /// interpreted.
    pub unknown: u32,
    /// Always 3 (`[OBS]`).
    pub version: u32,
    /// The container's declared byte length, counted from the field
    /// after it to the end of the last array.
    pub declared_length: u32,
    /// The OUTER rectangle, which duplicates the per-array one. Equal in
    /// 3,202 of 3,202 observed records; kept so the duplication is
    /// visible rather than assumed away.
    pub bounds: VmRect,
    /// Always 56 (`[OBS]`) — a serializer's channel ceiling, not a
    /// meaningful count. The slot loop runs `array_count + 2` times.
    pub array_count: u32,
    /// The written planes, in slot order. Empty slots are not
    /// represented; each plane records its own [`VmArray::slot`].
    pub arrays: Vec<VmArray>,
    /// Bytes consumed by this container, measured from the `unknown`
    /// word. A correct structural parse of a real sampled-tip record
    /// accounts for **every** byte of the record body after the id —
    /// see the module docs.
    pub consumed: usize,
}

impl VmArrayList {
    /// The single written plane, when there is exactly one.
    ///
    /// A brush tip is a single-channel mask and every observed record
    /// writes exactly one plane (spec §2.2 `[OBS]`), but the container
    /// is general and the count carries no meaning, so this is a
    /// convenience — not an assumption baked into the parse.
    pub fn single_plane(&self) -> Option<&VmArray> {
        match self.arrays.as_slice() {
            [one] => Some(one),
            _ => None,
        }
    }
}

fn malformed(detail: String) -> PsdError {
    PsdError::Malformed {
        section: "vm array list",
        detail,
    }
}

/// Read a virtual memory array list from the cursor's current position.
pub fn read_vm_array_list(r: &mut ByteReader) -> Result<VmArrayList> {
    let start = r.pos();
    let unknown = r.u32()?;
    let version = r.u32()?;
    let declared_length = r.u32()?;
    let bounds = VmRect::read(r)?;
    let array_count = r.u32()?;

    // The loop runs `array_count + 2` times and each slot costs at least
    // its 4-byte `is_written` word, so a slot count the input cannot
    // hold is malformed.
    let slots = (array_count as u64).checked_add(2).ok_or_else(|| {
        malformed(format!(
            "array_count {array_count} overflows the slot count"
        ))
    })?;
    if slots > (r.remaining() / 4) as u64 {
        return Err(malformed(format!(
            "array_count {array_count} implies {slots} slots, more than the {} byte(s) available",
            r.remaining()
        )));
    }

    let mut arrays = Vec::new();
    for slot in 0..slots as u32 {
        let is_written = r.u32()?;
        if is_written == 0 {
            continue;
        }
        let array_length = r.u32()? as usize;
        // `array_length` counts from the field after it (pixel_depth)
        // through the end of the pixel data: 4 + 16 + 2 + 1 = 23 bytes
        // of header, then the payload.
        const ARRAY_HEADER_AFTER_LENGTH: usize = 4 + 16 + 2 + 1;
        let payload_len = array_length
            .checked_sub(ARRAY_HEADER_AFTER_LENGTH)
            .ok_or_else(|| {
                malformed(format!(
                    "array_length {array_length} is shorter than the \
                     {ARRAY_HEADER_AFTER_LENGTH}-byte array header"
                ))
            })?;
        let pixel_depth = r.u32()?;
        let abounds = VmRect::read(r)?;
        let depth = r.u16()?;
        let compression = r.u8()?;
        let data = r.take(payload_len)?.to_vec();
        arrays.push(VmArray {
            slot,
            pixel_depth,
            bounds: abounds,
            depth,
            compression,
            data,
        });
    }

    Ok(VmArrayList {
        unknown,
        version,
        declared_length,
        bounds,
        array_count,
        arrays,
        consumed: r.pos() - start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a minimal one-plane container by hand. This is the
    /// module's OWN unit-level scaffold; the real independent emitter
    /// (used by the round-trip suite) lives in `image-conformance`.
    fn container(array_count: u32, slot: u32, w: i32, h: i32, payload: &[u8], comp: u8) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        b.extend_from_slice(&VM_VERSION.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes()); // declared_length (unchecked)
        for v in [0i32, 0, h, w] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b.extend_from_slice(&array_count.to_be_bytes());
        for i in 0..array_count + 2 {
            if i != slot {
                b.extend_from_slice(&0u32.to_be_bytes());
                continue;
            }
            b.extend_from_slice(&1u32.to_be_bytes());
            b.extend_from_slice(&((23 + payload.len()) as u32).to_be_bytes());
            b.extend_from_slice(&8u32.to_be_bytes());
            for v in [0i32, 0, h, w] {
                b.extend_from_slice(&v.to_be_bytes());
            }
            b.extend_from_slice(&8u16.to_be_bytes());
            b.push(comp);
            b.extend_from_slice(payload);
        }
        b
    }

    #[test]
    fn image_psd_vm_array_list_the_264_constant_is_arithmetic_not_a_format_fact() {
        // The public "264-byte skip" is exactly the header of a container
        // with array_count 56 whose written slot is 55. Reproduce it.
        let px = vec![7u8; 4];
        let bytes = container(56, 55, 2, 2, &px, COMPRESSION_RAW);
        let mut r = ByteReader::new(&bytes);
        let list = read_vm_array_list(&mut r).unwrap();
        // 4 + 4 + 4 + 16 + 4 header, 55 empty slots × 4, then the written
        // slot's is_written + array_length + pixel_depth = 12 ⇒ 264 bytes
        // land on the `top` field of the plane's own rectangle.
        let prefix = 4 + 4 + 4 + 16 + 4 + 55 * 4 + 12;
        assert_eq!(prefix, 264, "the constant everybody hardcodes");
        assert_eq!(list.arrays.len(), 1);
        assert_eq!(list.arrays[0].slot, 55);
        // …and the two trailing empty slots after the pixel data are the
        // "8 unexplained trailing bytes" of every public description.
        assert_eq!(list.consumed, bytes.len());
        assert_eq!(bytes.len(), prefix + 16 + 2 + 1 + 4 + 8);
    }

    #[test]
    fn image_psd_vm_array_list_a_different_slot_would_break_the_constant() {
        // Same array_count, plane written in slot 3 instead of 55: a
        // reader that skipped 264 bytes reads garbage; the structural
        // parse is unaffected.
        let px = vec![9u8; 6];
        let bytes = container(56, 3, 3, 2, &px, COMPRESSION_RAW);
        let mut r = ByteReader::new(&bytes);
        let list = read_vm_array_list(&mut r).unwrap();
        assert_eq!(list.arrays[0].slot, 3);
        assert_eq!(list.arrays[0].bounds.width(), Some(3));
        assert_eq!(list.arrays[0].bounds.height(), Some(2));
        assert_eq!(list.consumed, bytes.len());
        assert_eq!(list.arrays[0].decode().unwrap(), VmSamples::Eight(px));
    }

    #[test]
    fn image_psd_vm_array_list_a_different_array_count_would_break_the_constant() {
        let px = vec![1u8; 1];
        let bytes = container(4, 2, 1, 1, &px, COMPRESSION_RAW);
        let mut r = ByteReader::new(&bytes);
        let list = read_vm_array_list(&mut r).unwrap();
        assert_eq!(list.array_count, 4);
        assert_eq!(list.arrays.len(), 1);
        assert_eq!(list.consumed, bytes.len());
    }

    #[test]
    fn image_psd_vm_array_list_bounds_are_y_first() {
        // top=0 left=0 bottom=5 right=2 ⇒ a 2-wide, 5-tall plane. Read
        // x-first it would be 5×2 and every non-square tip transposes.
        let px = vec![0u8; 10];
        let bytes = container(2, 0, 2, 5, &px, COMPRESSION_RAW);
        let mut r = ByteReader::new(&bytes);
        let list = read_vm_array_list(&mut r).unwrap();
        let a = &list.arrays[0];
        assert_eq!((a.bounds.top, a.bounds.left), (0, 0));
        assert_eq!((a.bounds.bottom, a.bounds.right), (5, 2));
        assert_eq!((a.width(), a.height()), (Some(2), Some(5)));
    }

    #[test]
    fn image_psd_vm_array_list_non_positive_bounds_are_refused() {
        let a = VmArray {
            slot: 0,
            pixel_depth: 8,
            bounds: VmRect {
                top: 10,
                left: 10,
                bottom: 5,
                right: 5,
            },
            depth: 8,
            compression: COMPRESSION_RAW,
            data: vec![0; 16],
        };
        assert!(a.decode().is_err());
    }

    #[test]
    fn image_psd_vm_array_list_rle_row_table_is_up_front_not_per_row() {
        // Two rows of 4 px. The table carries BOTH lengths first.
        let row0 = packbits::encode(&[1, 1, 1, 1]);
        let row1 = packbits::encode(&[2, 3, 4, 5]);
        let mut payload = Vec::new();
        payload.extend_from_slice(&(row0.len() as u16).to_be_bytes());
        payload.extend_from_slice(&(row1.len() as u16).to_be_bytes());
        payload.extend_from_slice(&row0);
        payload.extend_from_slice(&row1);
        let bytes = container(56, 55, 4, 2, &payload, COMPRESSION_RLE);
        let mut r = ByteReader::new(&bytes);
        let list = read_vm_array_list(&mut r).unwrap();
        assert_eq!(
            list.arrays[0].decode().unwrap(),
            VmSamples::Eight(vec![1, 1, 1, 1, 2, 3, 4, 5])
        );
        assert_eq!(list.consumed, bytes.len());
    }

    #[test]
    fn image_psd_vm_array_list_rle_expansion_bound_rejects_absurd_dimensions() {
        let a = VmArray {
            slot: 0,
            pixel_depth: 8,
            bounds: VmRect {
                top: 0,
                left: 0,
                bottom: 4096,
                right: 4096,
            },
            depth: 8,
            compression: COMPRESSION_RLE,
            data: vec![0; 8], // could expand to 1 KiB at most, not 16 MiB
        };
        let err = a.decode().unwrap_err();
        assert!(format!("{err}").contains("PackBits expands"), "{err}");
    }

    #[test]
    fn image_psd_vm_array_list_sixteen_bit_rle_is_refused_by_name() {
        let a = VmArray {
            slot: 0,
            pixel_depth: 16,
            bounds: VmRect {
                top: 0,
                left: 0,
                bottom: 2,
                right: 2,
            },
            depth: 16,
            compression: COMPRESSION_RLE,
            data: vec![0; 32],
        };
        match a.decode().unwrap_err() {
            PsdError::Unsupported(m) => assert!(m.contains("16-bit RLE"), "{m}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn image_psd_vm_array_list_sixteen_bit_raw_keeps_full_precision() {
        let mut payload = Vec::new();
        for v in [0u16, 0x1234, 0xFFFF, 0x00FF] {
            payload.extend_from_slice(&v.to_be_bytes());
        }
        let mut b = Vec::new();
        b.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        b.extend_from_slice(&VM_VERSION.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        for v in [0i32, 0, 2, 2] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b.extend_from_slice(&1u32.to_be_bytes()); // array_count 1 ⇒ 3 slots
        b.extend_from_slice(&1u32.to_be_bytes()); // slot 0 written
        b.extend_from_slice(&((23 + payload.len()) as u32).to_be_bytes());
        b.extend_from_slice(&16u32.to_be_bytes());
        for v in [0i32, 0, 2, 2] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b.extend_from_slice(&16u16.to_be_bytes());
        b.push(COMPRESSION_RAW);
        b.extend_from_slice(&payload);
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());

        let mut r = ByteReader::new(&b);
        let list = read_vm_array_list(&mut r).unwrap();
        let s = list.arrays[0].decode().unwrap();
        assert_eq!(s, VmSamples::Sixteen(vec![0, 0x1234, 0xFFFF, 0x00FF]));
        // The 8-bit reduction is available but is a caller's decision.
        assert_eq!(s.to_eight_lossy(), vec![0x00, 0x12, 0xFF, 0x00]);
        assert_eq!(list.consumed, b.len());
    }
}
