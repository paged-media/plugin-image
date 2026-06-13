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

//! A tiny, dependency-free EXIF/TIFF reader for the three facts the
//! ingest lane needs: **orientation**, **DPI**, and the **color-space
//! tag**. It parses the raw EXIF payload codec adapters already capture
//! (`SourceInfo.exif`) — the byte stream that *starts at the TIFF header*
//! (`II*\0` little-endian or `MM\0*` big-endian), which is exactly what
//! zune-jpeg hands back from APP1 (it strips the `Exif\0\0` marker
//! prefix) and what TIFF files lead with.
//!
//! ## Why hand-rolled rather than a crate
//!
//! A full EXIF library (kamadak-exif et al.) parses hundreds of tags and
//! drags a dependency + a second TIFF reader into a workspace that
//! already pins its codec deps tightly (spec §10.3) and watches an 8 MiB
//! wasm budget (BREAKAGE I-07). We need exactly three tags off IFD0 (+
//! one off the Exif sub-IFD). A ~150-line bounded IFD walk over a byte
//! slice is the architecturally honest cost for that, and it keeps the
//! `#![forbid(unsafe_code)]` codec crate dependency-flat. It is
//! deliberately *lenient*: any malformed / truncated / unknown structure
//! yields `None` for that field, never an error — EXIF is advisory
//! metadata, and a broken APP1 must not fail an otherwise-valid decode.

/// EXIF orientation (TIFF tag 0x0112). The eight CIPA values; `TopLeft`
/// (1) is the no-op identity. Names describe where row-0/col-0 of the
/// *stored* pixels belongs in the *displayed* image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// 1 — no transform (row 0 = visual top, col 0 = visual left).
    TopLeft,
    /// 2 — mirror horizontal.
    TopRight,
    /// 3 — rotate 180°.
    BottomRight,
    /// 4 — mirror vertical.
    BottomLeft,
    /// 5 — mirror horizontal then rotate 270° CW (transpose).
    LeftTop,
    /// 6 — rotate 90° CW.
    RightTop,
    /// 7 — mirror horizontal then rotate 90° CW (transverse).
    RightBottom,
    /// 8 — rotate 270° CW (== 90° CCW).
    LeftBottom,
}

impl Orientation {
    fn from_u16(v: u16) -> Option<Orientation> {
        Some(match v {
            1 => Orientation::TopLeft,
            2 => Orientation::TopRight,
            3 => Orientation::BottomRight,
            4 => Orientation::BottomLeft,
            5 => Orientation::LeftTop,
            6 => Orientation::RightTop,
            7 => Orientation::RightBottom,
            8 => Orientation::LeftBottom,
            _ => return None,
        })
    }

    /// The identity orientation — nothing to apply.
    pub fn is_identity(self) -> bool {
        self == Orientation::TopLeft
    }

    /// Whether applying this orientation swaps width and height (the four
    /// 90°/270° cases). The ingest auto-orient uses this to report the
    /// post-orientation dimensions.
    pub fn swaps_dimensions(self) -> bool {
        matches!(
            self,
            Orientation::LeftTop
                | Orientation::RightTop
                | Orientation::RightBottom
                | Orientation::LeftBottom
        )
    }
}

/// The EXIF `ColorSpace` tag (0xA001, in the Exif sub-IFD). Only the two
/// defined values plus the de-facto "uncalibrated / Adobe RGB" sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpaceTag {
    /// 1 — sRGB.
    Srgb,
    /// 0xFFFF — uncalibrated (commonly Adobe RGB when a real ICC profile
    /// is also embedded; the ICC, when present, is authoritative).
    Uncalibrated,
    /// Any other value the spec doesn't define — surfaced raw for
    /// provenance.
    Other(u16),
}

/// The advisory facts a single EXIF/TIFF block yields. Every field is
/// independently optional — a present-but-unreadable block still returns
/// `Exif::default()`-shaped `None`s, never an error.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Exif {
    pub orientation: Option<Orientation>,
    /// Pixels-per-inch, X and Y, derived from XResolution/YResolution +
    /// ResolutionUnit (cm is converted to inch; ResolutionUnit==1 "no
    /// unit" yields `None` — the value is a meaningless aspect ratio).
    pub dpi_x: Option<f32>,
    pub dpi_y: Option<f32>,
    pub color_space: Option<ColorSpaceTag>,
}

impl Exif {
    /// Parse a raw EXIF payload (starting at the TIFF header). Always
    /// returns an `Exif`; unreadable fields are `None`.
    pub fn parse(payload: &[u8]) -> Exif {
        parse_tiff(payload).unwrap_or_default()
    }
}

// ---- TIFF/IFD walk (bounded, lenient) ------------------------------------

/// Byte order of the TIFF stream.
#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

struct Reader<'a> {
    /// The whole TIFF block; all offsets are from its start.
    buf: &'a [u8],
    endian: Endian,
}

impl<'a> Reader<'a> {
    fn u16(&self, off: usize) -> Option<u16> {
        let b = self.buf.get(off..off + 2)?;
        Some(match self.endian {
            Endian::Little => u16::from_le_bytes([b[0], b[1]]),
            Endian::Big => u16::from_be_bytes([b[0], b[1]]),
        })
    }

    fn u32(&self, off: usize) -> Option<u32> {
        let b = self.buf.get(off..off + 4)?;
        Some(match self.endian {
            Endian::Little => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Endian::Big => u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
        })
    }
}

/// A decoded IFD entry's salient fields (we only ever read SHORT and
/// RATIONAL values, plus sub-IFD pointers stored as LONG).
struct Entry {
    tag: u16,
    field_type: u16,
    count: u32,
    /// Byte offset of the value: either inline (the 4-byte value field
    /// itself) or, when the value doesn't fit in 4 bytes, the offset it
    /// points at. Resolved here so callers read uniformly.
    value_off: usize,
}

const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_RATIONAL: u16 = 5;

fn type_size(field_type: u16) -> usize {
    match field_type {
        1 | 2 | 6 | 7 => 1, // BYTE/ASCII/SBYTE/UNDEFINED
        TYPE_SHORT | 8 => 2, // SHORT/SSHORT
        TYPE_LONG | 9 | 11 => 4, // LONG/SLONG/FLOAT
        TYPE_RATIONAL | 10 | 12 => 8, // RATIONAL/SRATIONAL/DOUBLE
        _ => 0,
    }
}

fn parse_tiff(buf: &[u8]) -> Option<Exif> {
    if buf.len() < 8 {
        return None;
    }
    let endian = match &buf[0..2] {
        b"II" => Endian::Little,
        b"MM" => Endian::Big,
        _ => return None,
    };
    let r = Reader { buf, endian };
    // Magic 42 confirms byte order was read correctly.
    if r.u16(2)? != 42 {
        return None;
    }
    let ifd0 = r.u32(4)? as usize;

    let mut exif = Exif::default();
    let mut exif_ifd_ptr: Option<usize> = None;

    // --- IFD0: orientation, resolution, the Exif sub-IFD pointer. ---
    let (mut xres, mut yres, mut res_unit): (Option<f64>, Option<f64>, Option<u16>) =
        (None, None, None);
    for entry in iter_ifd(&r, ifd0) {
        match entry.tag {
            0x0112 => {
                // Orientation (SHORT).
                if entry.field_type == TYPE_SHORT {
                    exif.orientation = r.u16(entry.value_off).and_then(Orientation::from_u16);
                }
            }
            0x011A => xres = read_rational(&r, &entry), // XResolution
            0x011B => yres = read_rational(&r, &entry), // YResolution
            0x0128 => {
                // ResolutionUnit (SHORT): 1 none, 2 inch, 3 cm.
                if entry.field_type == TYPE_SHORT {
                    res_unit = r.u16(entry.value_off);
                }
            }
            0x8769 => {
                // Exif IFD pointer (LONG).
                if entry.field_type == TYPE_LONG {
                    exif_ifd_ptr = r.u32(entry.value_off).map(|v| v as usize);
                }
            }
            _ => {}
        }
    }

    // Resolution → DPI. Unit 2 = inch (verbatim); 3 = cm (×2.54);
    // 1/none/absent → a bare aspect ratio, not a real density → None.
    let to_inch = match res_unit {
        Some(3) => Some(2.54_f64),
        Some(2) | None => Some(1.0_f64), // default unit is inch per TIFF
        _ => None,
    };
    if let Some(factor) = to_inch {
        exif.dpi_x = xres.map(|v| (v * factor) as f32);
        exif.dpi_y = yres.map(|v| (v * factor) as f32);
    }

    // --- Exif sub-IFD: ColorSpace (0xA001). ---
    if let Some(ptr) = exif_ifd_ptr {
        for entry in iter_ifd(&r, ptr) {
            if entry.tag == 0xA001 && entry.field_type == TYPE_SHORT {
                exif.color_space = r.u16(entry.value_off).map(|v| match v {
                    1 => ColorSpaceTag::Srgb,
                    0xFFFF => ColorSpaceTag::Uncalibrated,
                    other => ColorSpaceTag::Other(other),
                });
            }
        }
    }

    Some(exif)
}

/// Iterate the entries of the IFD at `ifd_off`. Bounded: an out-of-range
/// offset or count yields an empty iterator (collected eagerly into a
/// `Vec` so the borrow is simple and the walk is obviously terminating).
fn iter_ifd<'a>(r: &Reader<'a>, ifd_off: usize) -> Vec<Entry> {
    let mut out = Vec::new();
    let Some(count) = r.u16(ifd_off) else {
        return out;
    };
    // A sane cap: a real IFD has a few dozen entries. Guard against a
    // corrupt count steering a huge loop.
    let count = count.min(512) as usize;
    for i in 0..count {
        let e = ifd_off + 2 + i * 12;
        let (Some(tag), Some(field_type), Some(cnt)) =
            (r.u16(e), r.u16(e + 2), r.u32(e + 4))
        else {
            break;
        };
        let value_field = e + 8;
        let bytes = type_size(field_type).saturating_mul(cnt as usize);
        let value_off = if bytes <= 4 {
            value_field
        } else {
            // Value doesn't fit inline: the 4-byte field is an offset.
            match r.u32(value_field) {
                Some(o) => o as usize,
                None => break,
            }
        };
        out.push(Entry {
            tag,
            field_type,
            count: cnt,
            value_off,
        });
    }
    out
}

/// Read a single RATIONAL (two u32: numerator/denominator) as f64.
/// Returns `None` on a zero denominator or short buffer.
fn read_rational(r: &Reader<'_>, entry: &Entry) -> Option<f64> {
    if entry.field_type != TYPE_RATIONAL || entry.count == 0 {
        return None;
    }
    let num = r.u32(entry.value_off)? as f64;
    let den = r.u32(entry.value_off + 4)? as f64;
    if den == 0.0 {
        None
    } else {
        Some(num / den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal little-endian TIFF/EXIF block: IFD0 with the given
    /// (tag, type, value) SHORT entries, plus optional resolution and a
    /// ColorSpace tag in an Exif sub-IFD. Returns bytes that start at the
    /// TIFF header (exactly what `Exif::parse` expects).
    fn build_le(
        orientation: Option<u16>,
        xres: Option<(u32, u32)>,
        res_unit: Option<u16>,
        color_space: Option<u16>,
    ) -> Vec<u8> {
        // Layout: [0..8] header. IFD0 at 8. Each entry 12 bytes. We place
        // any RATIONAL value blocks and the sub-IFD after IFD0.
        let mut entries: Vec<(u16, u16, u32, [u8; 4])> = Vec::new();
        if let Some(o) = orientation {
            entries.push((0x0112, TYPE_SHORT, 1, pack_short(o)));
        }
        // XResolution / ResolutionUnit need an external RATIONAL block;
        // compute its offset after the whole IFD0 + next-ifd pointer.
        let n_ifd0 = entries.len()
            + xres.is_some() as usize
            + res_unit.is_some() as usize
            + color_space.is_some() as usize; // Exif-IFD pointer entry
        // IFD0 spans: 2 (count) + 12*n + 4 (next-IFD offset).
        let ifd0_end = 8 + 2 + 12 * n_ifd0 + 4;
        let mut tail = Vec::new();
        let mut next = ifd0_end;

        let rational_off = if let Some((num, den)) = xres {
            let off = next;
            tail.extend_from_slice(&num.to_le_bytes());
            tail.extend_from_slice(&den.to_le_bytes());
            next += 8;
            Some(off)
        } else {
            None
        };
        if let Some(off) = rational_off {
            entries.push((0x011A, TYPE_RATIONAL, 1, (off as u32).to_le_bytes()));
        }
        if let Some(u) = res_unit {
            entries.push((0x0128, TYPE_SHORT, 1, pack_short(u)));
        }

        // Exif sub-IFD (ColorSpace) placed after the rational block.
        let exif_ifd_off = if color_space.is_some() {
            Some(next)
        } else {
            None
        };
        if let Some(off) = exif_ifd_off {
            entries.push((0x8769, TYPE_LONG, 1, (off as u32).to_le_bytes()));
        }

        // Now we know n_ifd0 == entries.len(); assert layout assumption.
        assert_eq!(entries.len(), n_ifd0, "entry count drifted");

        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // IFD0 at offset 8
        // IFD0.
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        // sort entries by tag (TIFF requires ascending tag order; our
        // reader doesn't depend on it, but keep it realistic).
        entries.sort_by_key(|e| e.0);
        for (tag, ty, cnt, val) in &entries {
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&ty.to_le_bytes());
            buf.extend_from_slice(&cnt.to_le_bytes());
            buf.extend_from_slice(val);
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        assert_eq!(buf.len(), ifd0_end, "IFD0 size mismatch");
        buf.extend_from_slice(&tail);

        // Append the Exif sub-IFD if requested.
        if let Some(cs) = color_space {
            assert_eq!(buf.len(), exif_ifd_off.unwrap(), "exif-ifd offset drift");
            buf.extend_from_slice(&1u16.to_le_bytes()); // one entry
            buf.extend_from_slice(&0xA001u16.to_le_bytes());
            buf.extend_from_slice(&TYPE_SHORT.to_le_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&pack_short(cs));
            buf.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        }
        buf
    }

    /// A SHORT inline value: the 2-byte value left-justified in the
    /// 4-byte little-endian value field.
    fn pack_short(v: u16) -> [u8; 4] {
        let b = v.to_le_bytes();
        [b[0], b[1], 0, 0]
    }

    #[test]
    fn exif_reads_orientation_rotate_90() {
        let bytes = build_le(Some(6), None, None, None);
        let e = Exif::parse(&bytes);
        assert_eq!(e.orientation, Some(Orientation::RightTop));
        assert!(e.orientation.unwrap().swaps_dimensions());
    }

    #[test]
    fn exif_reads_all_eight_orientations() {
        let expect = [
            (1, Orientation::TopLeft),
            (2, Orientation::TopRight),
            (3, Orientation::BottomRight),
            (4, Orientation::BottomLeft),
            (5, Orientation::LeftTop),
            (6, Orientation::RightTop),
            (7, Orientation::RightBottom),
            (8, Orientation::LeftBottom),
        ];
        for (v, o) in expect {
            let bytes = build_le(Some(v), None, None, None);
            assert_eq!(Exif::parse(&bytes).orientation, Some(o), "value {v}");
        }
        assert!(Orientation::TopLeft.is_identity());
    }

    #[test]
    fn exif_reads_dpi_inch() {
        // 300 dpi (XResolution 300/1), unit = inch (2).
        let bytes = build_le(None, Some((300, 1)), Some(2), None);
        let e = Exif::parse(&bytes);
        assert_eq!(e.dpi_x, Some(300.0));
    }

    #[test]
    fn exif_reads_dpi_cm_converts() {
        // 118.11 px/cm ≈ 300 dpi; unit = cm (3) → ×2.54.
        let bytes = build_le(None, Some((11811, 100)), Some(3), None);
        let e = Exif::parse(&bytes);
        let dpi = e.dpi_x.unwrap();
        assert!((dpi - 300.0).abs() < 0.5, "got {dpi}");
    }

    #[test]
    fn exif_resolution_unit_none_is_no_dpi() {
        // Unit 1 (no unit) → the value is a bare aspect ratio, not DPI.
        let bytes = build_le(None, Some((4, 3)), Some(1), None);
        assert_eq!(Exif::parse(&bytes).dpi_x, None);
    }

    #[test]
    fn exif_reads_color_space_srgb() {
        let bytes = build_le(None, None, None, Some(1));
        assert_eq!(Exif::parse(&bytes).color_space, Some(ColorSpaceTag::Srgb));
    }

    #[test]
    fn exif_reads_color_space_uncalibrated() {
        let bytes = build_le(None, None, None, Some(0xFFFF));
        assert_eq!(
            Exif::parse(&bytes).color_space,
            Some(ColorSpaceTag::Uncalibrated)
        );
    }

    #[test]
    fn exif_big_endian_orientation() {
        // Hand-build a minimal big-endian (MM) block: orientation = 8.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"MM");
        buf.extend_from_slice(&42u16.to_be_bytes());
        buf.extend_from_slice(&8u32.to_be_bytes());
        buf.extend_from_slice(&1u16.to_be_bytes()); // 1 entry
        buf.extend_from_slice(&0x0112u16.to_be_bytes());
        buf.extend_from_slice(&TYPE_SHORT.to_be_bytes());
        buf.extend_from_slice(&1u32.to_be_bytes());
        // SHORT value 8, big-endian left-justified in the 4-byte field.
        buf.extend_from_slice(&[0x00, 0x08, 0x00, 0x00]);
        buf.extend_from_slice(&0u32.to_be_bytes());
        assert_eq!(
            Exif::parse(&buf).orientation,
            Some(Orientation::LeftBottom)
        );
    }

    #[test]
    fn exif_garbage_is_none_not_panic() {
        assert_eq!(Exif::parse(&[]), Exif::default());
        assert_eq!(Exif::parse(b"not a tiff header at all"), Exif::default());
        // Right magic, then truncated garbage offsets.
        assert_eq!(
            Exif::parse(&[b'I', b'I', 42, 0, 0xFF, 0xFF, 0xFF, 0xFF]),
            Exif::default()
        );
    }

    #[test]
    fn exif_unknown_orientation_value_is_none() {
        let bytes = build_le(Some(99), None, None, None);
        assert_eq!(Exif::parse(&bytes).orientation, None);
    }
}
