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

//! VECTOR MASKS — the `vmsk` / `vsms` additional-layer-info block as
//! paths in canvas pixels, plus everything Photoshop reads from it when
//! it draws the mask. Plain data: rasterizing is the consumer's job
//! (`image-js`), where the coverage type lives.
//!
//! # The layout ([PUB] Adobe Photoshop File Formats Specification)
//!
//! "Vector mask setting": `u32` version (3), `u32` flags (bit 0 invert,
//! bit 1 not linked, bit 2 disabled), then path records in the "Path
//! resource format": 26 bytes each, a `u16` selector first. Selectors
//! 0/3 are closed/open subpath LENGTH records, 1/2 closed knots (linked,
//! unlinked), 4/5 open knots, 6 the path fill-rule record, 7 the
//! clipboard record, 8 the initial fill rule. A knot is three points —
//! the control point preceding the anchor, the anchor, the control point
//! leaving it — each `(vertical, horizontal)` as signed fixed-point 8.24
//! fractions of the document height and width.
//!
//! # What Photoshop does with it ([OBS], Photoshop 27.10)
//!
//! The specification documents the length record's knot count only.
//! Photoshop 27 writes, and reads, three more fields there — observed by
//! having it build masks with each choice and diffing the bytes, then by
//! patching single fields of its own files and having it re-open them:
//!
//! * **bytes 4–5: the path operation** Photoshop's "path operations"
//!   menu offers per path COMPONENT — 0 exclude overlapping shapes,
//!   1 combine, 2 subtract front shape, 3 intersect. `0xFFFF` marks a
//!   subpath that belongs to the component started by the previous
//!   length record (what "Merge Shape Components" produces).
//! * **bytes 6–7: the component's fill rule** — 2 non-zero winding,
//!   1 even-odd (a five-point star self-intersecting in one subpath has
//!   a hole under 1 and none under 2; an inner rectangle in the same
//!   component is a hole under 1 whatever its direction). Continuation
//!   subpaths carry 0 and take their component's rule; a head carrying
//!   0, which Photoshop never writes, renders even-odd.
//! * **bytes 14–15: the component index** (0, 1, … shared by a
//!   component's subpaths). Informational; not needed to draw.
//!
//! Components fold in file order. The first one's operation decides the
//! starting state: SUBTRACT starts from a FULL mask (a lone subtract
//! component is a hole in an otherwise opaque mask), anything else from
//! an empty one and adds the component. The rest apply their operation
//! to the running result. The fill-rule record (selector 6) carries no
//! data, and the initial-fill record (selector 8) has no visible effect
//! on read: patched to 1, the mask draws exactly as with 0. OPEN
//! subpaths fill as if closed by a STRAIGHT line, whatever their end
//! handles say. Flag bit 0 inverts, bit 2 disables (the mask is kept,
//! not applied), bit 1 (unlinked) does not change the pixels.

use crate::reader::ByteReader;
use crate::{PsdError, Result};

/// A point in canvas pixels, `x` right, `y` down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathPoint {
    pub x: f64,
    pub y: f64,
}

/// One Bezier knot: the anchor and its two control points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Knot {
    /// Selector 1/4 (linked) rather than 2/5 (unlinked). Editing only.
    pub linked: bool,
    /// The control point of the segment ARRIVING at the anchor.
    pub control_in: PathPoint,
    pub anchor: PathPoint,
    /// The control point of the segment LEAVING the anchor.
    pub control_out: PathPoint,
}

/// One subpath: knots in order. A closed subpath's last segment runs
/// back to the first knot as a Bezier; an open one is filled as if a
/// straight line closed it.
#[derive(Debug, Clone, PartialEq)]
pub struct SubPath {
    pub closed: bool,
    pub knots: Vec<Knot>,
}

/// How a component folds into the components before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathOp {
    /// 0: exclude overlapping shapes (XOR).
    Exclude,
    /// 1: combine shapes (union).
    Combine,
    /// 2: subtract front shape.
    Subtract,
    /// 3: intersect shape areas.
    Intersect,
}

/// The fill rule inside one component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRule {
    EvenOdd,
    NonZero,
}

/// One path component: an operation, a fill rule, and its subpaths.
#[derive(Debug, Clone, PartialEq)]
pub struct PathComponent {
    pub op: PathOp,
    pub fill_rule: FillRule,
    pub subpaths: Vec<SubPath>,
}

/// A parsed vector mask.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorMask {
    /// Always 3 in every file observed; anything else is refused.
    pub version: u32,
    /// Flags bit 0.
    pub invert: bool,
    /// Flags bit 1: the mask does not move with the layer. No pixel
    /// effect.
    pub not_linked: bool,
    /// Flags bit 2: kept, not applied.
    pub disabled: bool,
    /// The initial-fill record's value (selector 8). Recorded for
    /// fidelity; Photoshop draws the same mask whatever it says.
    pub initial_fill: bool,
    pub components: Vec<PathComponent>,
}

/// Bytes per path record.
pub const PATH_RECORD_LEN: usize = 26;

/// Signed fixed-point 8.24 → `f64`.
fn fixed_8_24(v: i32) -> f64 {
    f64::from(v) / f64::from(1u32 << 24)
}

fn read_point(r: &mut ByteReader, width: u32, height: u32) -> Result<PathPoint> {
    let y = fixed_8_24(r.i32()?) * f64::from(height);
    let x = fixed_8_24(r.i32()?) * f64::from(width);
    Ok(PathPoint { x, y })
}

fn malformed(detail: String) -> PsdError {
    PsdError::Malformed {
        section: "vector mask",
        detail,
    }
}

impl VectorMask {
    /// Parse a `vmsk`/`vsms` payload (after the block's signature, key
    /// and length) for a `width`×`height` document. Path coordinates
    /// come back in canvas pixels.
    ///
    /// Refuses (rather than guesses) a version other than 3, an
    /// operation or fill rule outside the observed values, and records
    /// that do not frame (knots without a length record, a count that
    /// runs past the block, an unknown selector).
    pub fn parse(payload: &[u8], width: u32, height: u32) -> Result<VectorMask> {
        let mut r = ByteReader::new(payload);
        let version = r.u32()?;
        if version != 3 {
            return Err(PsdError::Unsupported(format!(
                "vector mask version {version} (only 3 is known)"
            )));
        }
        let flags = r.u32()?;
        let mut mask = VectorMask {
            version,
            invert: flags & 0x01 != 0,
            not_linked: flags & 0x02 != 0,
            disabled: flags & 0x04 != 0,
            initial_fill: false,
            components: Vec::new(),
        };
        // Knots still owed to the subpath whose length record came last.
        let mut owed = 0usize;
        let mut open = false;
        while r.remaining() >= PATH_RECORD_LEN {
            let mut rec = r.sub(PATH_RECORD_LEN)?;
            let selector = rec.u16()?;
            match selector {
                0 | 3 => {
                    if owed != 0 {
                        return Err(malformed(format!(
                            "a subpath length record arrived {owed} knot(s) early"
                        )));
                    }
                    let count = usize::from(rec.u16()?);
                    let op = rec.u16()?;
                    let rule = rec.u16()?;
                    let sub = SubPath {
                        closed: selector == 0,
                        knots: Vec::with_capacity(count),
                    };
                    if op == 0xFFFF {
                        let Some(c) = mask.components.last_mut() else {
                            return Err(PsdError::Unsupported(
                                "a vector mask whose first subpath continues a component \
                                 (operation 0xFFFF) — there is no component before it"
                                    .into(),
                            ));
                        };
                        c.subpaths.push(sub);
                    } else {
                        let op = match op {
                            0 => PathOp::Exclude,
                            1 => PathOp::Combine,
                            2 => PathOp::Subtract,
                            3 => PathOp::Intersect,
                            other => {
                                return Err(PsdError::Unsupported(format!(
                                    "vector mask path operation {other} (0–3 are known)"
                                )))
                            }
                        };
                        let fill_rule = match rule {
                            0 | 1 => FillRule::EvenOdd,
                            2 => FillRule::NonZero,
                            other => {
                                return Err(PsdError::Unsupported(format!(
                                    "vector mask fill rule {other} (1 even-odd and 2 \
                                     non-zero are known)"
                                )))
                            }
                        };
                        mask.components.push(PathComponent {
                            op,
                            fill_rule,
                            subpaths: vec![sub],
                        });
                    }
                    owed = count;
                    open = selector == 3;
                }
                1 | 2 | 4 | 5 => {
                    let open_knot = selector >= 4;
                    if owed == 0 || open_knot != open {
                        return Err(malformed(format!(
                            "knot record (selector {selector}) outside a matching subpath"
                        )));
                    }
                    let control_in = read_point(&mut rec, width, height)?;
                    let anchor = read_point(&mut rec, width, height)?;
                    let control_out = read_point(&mut rec, width, height)?;
                    let sub = mask
                        .components
                        .last_mut()
                        .and_then(|c| c.subpaths.last_mut())
                        .ok_or_else(|| malformed("knot before any subpath".into()))?;
                    sub.knots.push(Knot {
                        linked: selector == 1 || selector == 4,
                        control_in,
                        anchor,
                        control_out,
                    });
                    owed -= 1;
                }
                6 | 7 => {}
                8 => mask.initial_fill = rec.u16()? != 0,
                other => {
                    return Err(malformed(format!("unknown path record selector {other}")));
                }
            }
        }
        if owed != 0 {
            return Err(malformed(format!(
                "the last subpath is missing {owed} knot(s)"
            )));
        }
        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 26-byte record: a selector and up to 24 payload bytes.
    fn rec(selector: u16, body: &[u8]) -> Vec<u8> {
        let mut v = selector.to_be_bytes().to_vec();
        v.extend_from_slice(body);
        v.resize(PATH_RECORD_LEN, 0);
        v
    }

    fn length(selector: u16, count: u16, op: u16, rule: u16) -> Vec<u8> {
        let mut b = count.to_be_bytes().to_vec();
        b.extend_from_slice(&op.to_be_bytes());
        b.extend_from_slice(&rule.to_be_bytes());
        rec(selector, &b)
    }

    /// A corner knot at canvas `(x, y)` of a `w`×`h` document.
    fn knot(selector: u16, x: f64, y: f64, w: f64, h: f64) -> Vec<u8> {
        let fy = ((y / h) * f64::from(1u32 << 24)).round() as i32;
        let fx = ((x / w) * f64::from(1u32 << 24)).round() as i32;
        let mut b = Vec::new();
        for _ in 0..3 {
            b.extend_from_slice(&fy.to_be_bytes());
            b.extend_from_slice(&fx.to_be_bytes());
        }
        rec(selector, &b)
    }

    fn block(flags: u32, records: &[Vec<u8>]) -> Vec<u8> {
        let mut v = 3u32.to_be_bytes().to_vec();
        v.extend_from_slice(&flags.to_be_bytes());
        for r in records {
            v.extend_from_slice(r);
        }
        v
    }

    fn rect(op: u16, rule: u16, x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Vec<u8>> {
        vec![
            length(0, 4, op, rule),
            knot(2, x0, y0, 64.0, 32.0),
            knot(2, x1, y0, 64.0, 32.0),
            knot(2, x1, y1, 64.0, 32.0),
            knot(2, x0, y1, 64.0, 32.0),
        ]
    }

    #[test]
    #[allow(non_snake_case)]
    fn vector_mask_records_parse_to_canvas_pixels__feat__image_psd_layer_import() {
        let mut records = vec![rec(6, &[]), rec(8, &[0, 1])];
        records.extend(rect(1, 1, 10.25, 4.5, 40.0, 20.0));
        let mut bytes = block(0, &records);
        bytes.extend_from_slice(&[0, 0]); // the block's padding
        let m = VectorMask::parse(&bytes, 64, 32).expect("parses");
        assert!(m.initial_fill);
        assert!(!m.invert && !m.disabled && !m.not_linked);
        assert_eq!(m.components.len(), 1);
        let c = &m.components[0];
        assert_eq!((c.op, c.fill_rule), (PathOp::Combine, FillRule::EvenOdd));
        let k = &c.subpaths[0].knots;
        assert_eq!(k.len(), 4);
        // 8.24 fractions of WIDTH for x and HEIGHT for y, stored y first.
        assert!((k[0].anchor.x - 10.25).abs() < 1e-5, "{:?}", k[0].anchor);
        assert!((k[0].anchor.y - 4.5).abs() < 1e-5, "{:?}", k[0].anchor);
        assert!((k[2].anchor.x - 40.0).abs() < 1e-5 && (k[2].anchor.y - 20.0).abs() < 1e-5);
        assert!(!k[0].linked);
    }

    #[test]
    #[allow(non_snake_case)]
    fn vector_mask_flags_and_operations_decode__feat__image_psd_layer_import() {
        let mut records = rect(2, 2, 0.0, 0.0, 8.0, 8.0);
        records.extend(rect(3, 1, 4.0, 4.0, 12.0, 12.0));
        records.extend(rect(0, 1, 2.0, 2.0, 6.0, 6.0));
        let m = VectorMask::parse(&block(0b101, &records), 64, 32).expect("parses");
        assert!(m.invert && m.disabled && !m.not_linked);
        let ops: Vec<_> = m.components.iter().map(|c| (c.op, c.fill_rule)).collect();
        assert_eq!(
            ops,
            [
                (PathOp::Subtract, FillRule::NonZero),
                (PathOp::Intersect, FillRule::EvenOdd),
                (PathOp::Exclude, FillRule::EvenOdd),
            ]
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn vector_mask_continuation_subpaths_join_their_component__feat__image_psd_layer_import() {
        let mut records = rect(1, 2, 0.0, 0.0, 30.0, 30.0);
        let mut inner = rect(0xFFFF, 0, 10.0, 10.0, 20.0, 20.0);
        records.append(&mut inner);
        // An open subpath (selector 3, knots 4/5) as its own component.
        records.push(length(3, 2, 1, 1));
        records.push(knot(5, 40.0, 2.0, 64.0, 32.0));
        records.push(knot(4, 50.0, 9.0, 64.0, 32.0));
        let m = VectorMask::parse(&block(0, &records), 64, 32).expect("parses");
        assert_eq!(m.components.len(), 2);
        assert_eq!(
            m.components[0].subpaths.len(),
            2,
            "the 0xFFFF subpath joins"
        );
        assert_eq!(m.components[0].fill_rule, FillRule::NonZero);
        let open = &m.components[1].subpaths[0];
        assert!(!open.closed);
        assert!(open.knots[1].linked && !open.knots[0].linked);
    }

    #[test]
    #[allow(non_snake_case)]
    fn vector_mask_refuses_what_it_cannot_frame__feat__image_psd_layer_import() {
        // Version.
        let mut bad = block(0, &rect(1, 1, 0.0, 0.0, 1.0, 1.0));
        bad[3] = 2;
        assert!(matches!(
            VectorMask::parse(&bad, 64, 32),
            Err(PsdError::Unsupported(_))
        ));
        // Unknown operation, unknown fill rule.
        for (op, rule) in [(4u16, 1u16), (1, 3)] {
            let b = block(0, &rect(op, rule, 0.0, 0.0, 1.0, 1.0));
            assert!(matches!(
                VectorMask::parse(&b, 64, 32),
                Err(PsdError::Unsupported(_))
            ));
        }
        // A continuation with nothing to continue.
        let b = block(0, &rect(0xFFFF, 0, 0.0, 0.0, 1.0, 1.0));
        assert!(VectorMask::parse(&b, 64, 32).is_err());
        // A count that runs past the block.
        let mut short = rect(1, 1, 0.0, 0.0, 1.0, 1.0);
        short.pop();
        assert!(matches!(
            VectorMask::parse(&block(0, &short), 64, 32),
            Err(PsdError::Malformed { .. })
        ));
        // A knot with no subpath, an open knot in a closed subpath, an
        // unknown selector.
        let stray = vec![knot(2, 0.0, 0.0, 64.0, 32.0)];
        assert!(VectorMask::parse(&block(0, &stray), 64, 32).is_err());
        let mut mixed = rect(1, 1, 0.0, 0.0, 1.0, 1.0);
        mixed[1] = knot(5, 0.0, 0.0, 64.0, 32.0);
        assert!(VectorMask::parse(&block(0, &mixed), 64, 32).is_err());
        assert!(VectorMask::parse(&block(0, &[rec(9, &[])]), 64, 32).is_err());
    }
}
