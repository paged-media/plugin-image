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

//! ADJUSTMENT LAYERS — the additional-layer-info blocks that make a layer
//! an adjustment, read into plain values the layer stack can map onto
//! its own adjust chain.
//!
//! Read: Curves (`curv`), Levels (`levl`), Exposure (`expA`), Invert
//! (`nvrt`) and Hue/Saturation (`hue2`). Every other adjustment key is
//! left to the import's refusal list. A block that parses but carries
//! something the chain cannot reproduce (Levels records past the RGB
//! channels, custom Hue/Saturation range bounds) is reported by [`Adjustment::unmodelled`] so the import can
//! decline it by name rather than approximate it.
//!
//! Provenance: Adobe Photoshop File Formats specification — Adjustment
//! Layers (Levels, Curves, Exposure, Invert, Hue/Saturation 2) [PUB];
//! the layouts checked byte for byte against adjustment layers Photoshop
//! 27.10 wrote itself (`image-conformance/fixtures/photoshop/
//! adjustment-layers/`) [OBS].

use crate::reader::ByteReader;
use crate::{PsdError, Result};

/// One Levels record: input floor/ceiling, output floor/ceiling (0–255)
/// and gamma (×100, so 100 = 1.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelsRecord {
    pub in_black: u16,
    pub in_white: u16,
    pub out_black: u16,
    pub out_white: u16,
    pub gamma_x100: u16,
}

impl LevelsRecord {
    pub const IDENTITY: LevelsRecord = LevelsRecord {
        in_black: 0,
        in_white: 255,
        out_black: 0,
        out_white: 255,
        gamma_x100: 100,
    };

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }
}

/// Photoshop's default Hue/Saturation range bounds — begin ramp, begin
/// sustain, end sustain, end ramp, in degrees — for reds, yellows,
/// greens, cyans, blues and magentas.
pub const HUE_RANGE_DEFAULTS: [[i16; 4]; 6] = [
    [315, 345, 15, 45],
    [15, 45, 75, 105],
    [75, 105, 135, 165],
    [135, 165, 195, 225],
    [195, 225, 255, 285],
    [255, 285, 315, 345],
];

/// A parsed adjustment.
#[derive(Debug, Clone, PartialEq)]
pub enum Adjustment {
    /// Curves: `curves[0]` is the composite (RGB) curve, `[1..=3]` red,
    /// green and blue. A curve is its control points as (input, output),
    /// 0–255, in input order.
    Curves { curves: [Option<Vec<(u8, u8)>>; 4] },
    /// Levels: the composite record, then red, green and blue.
    /// `extra_records` is true when records past those four (the legacy
    /// 29 plus the `Lvls` extension) are not the identity.
    Levels {
        records: [LevelsRecord; 4],
        extra_records: bool,
    },
    /// Exposure: stops, offset, gamma.
    Exposure {
        exposure: f32,
        offset: f32,
        gamma: f32,
    },
    /// Invert: no parameters.
    Invert,
    /// Hue/Saturation: `master` and each range as (hue −180..180,
    /// saturation −100..100, lightness −100..100); `colorize` as (hue
    /// 0..360, saturation 0..100, lightness −100..100), applied instead of
    /// the master when `colorized`. `bounds` are the ranges' extents.
    HueSaturation {
        colorized: bool,
        colorize: [i16; 3],
        master: [i16; 3],
        ranges: [[i16; 3]; 6],
        bounds: [[i16; 4]; 6],
    },
}

impl Adjustment {
    /// The additional-layer-info keys this module reads.
    pub const KEYS: [&'static [u8; 4]; 5] = [b"curv", b"levl", b"expA", b"nvrt", b"hue2"];

    /// Parse the payload of block `key` (the bytes after its length).
    /// `None` for a key this module does not read.
    pub fn parse(key: &[u8; 4], payload: &[u8]) -> Result<Option<Adjustment>> {
        let mut r = ByteReader::new(payload);
        Ok(Some(match key {
            b"curv" => parse_curves(&mut r)?,
            b"levl" => parse_levels(&mut r)?,
            b"expA" => {
                let _version = r.u16()?;
                Adjustment::Exposure {
                    exposure: f32::from_bits(r.u32()?),
                    offset: f32::from_bits(r.u32()?),
                    gamma: f32::from_bits(r.u32()?),
                }
            }
            b"nvrt" => Adjustment::Invert,
            b"hue2" => parse_hue_saturation(&mut r)?,
            _ => return Ok(None),
        }))
    }

    /// What this adjustment carries that the layer stack's adjust chain
    /// does not reproduce, as a phrase for the refusal; `None` when it is
    /// fully modelled.
    pub fn unmodelled(&self) -> Option<&'static str> {
        match self {
            Adjustment::Levels { extra_records, .. } => {
                extra_records.then_some("Levels on channels other than red, green and blue")
            }
            Adjustment::HueSaturation { bounds, .. } => (*bounds != HUE_RANGE_DEFAULTS)
                .then_some("Hue/Saturation colour ranges moved off their defaults"),
            Adjustment::Curves { .. } | Adjustment::Exposure { .. } | Adjustment::Invert => None,
        }
    }
}

fn malformed(detail: String) -> PsdError {
    PsdError::Malformed {
        section: "adjustment layer",
        detail,
    }
}

/// `curv`: a pad byte, a version word, a 32-bit map of the curves
/// present (bit 0 the composite, then the channels in order), then per
/// curve its point count and points as (OUTPUT, INPUT) pairs [PUB, OBS:
/// output first]. An optional `Crv ` extension repeats the curves keyed
/// by channel index; the legacy section is complete for RGB, so it is
/// what is read.
fn parse_curves(r: &mut ByteReader) -> Result<Adjustment> {
    let _pad = r.u8()?;
    let _version = r.u16()?;
    let map = r.u32()?;
    let mut curves: [Option<Vec<(u8, u8)>>; 4] = Default::default();
    for bit in 0..32u32 {
        if map & (1 << bit) == 0 {
            continue;
        }
        let n = r.u16()? as usize;
        if n.saturating_mul(4) > r.remaining() {
            return Err(malformed(format!(
                "curve claims {n} points, more than the {} byte(s) left",
                r.remaining()
            )));
        }
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            let out = r.u16()?;
            let inp = r.u16()?;
            pts.push((inp.min(255) as u8, out.min(255) as u8));
        }
        if bit >= 4 {
            return Err(PsdError::Unsupported(format!(
                "a Curves adjustment on channel {bit} (composite and RGB only)"
            )));
        }
        curves[bit as usize] = Some(pts);
    }
    Ok(Adjustment::Curves { curves })
}

/// `levl`: a version word, then 29 records of five words (the composite,
/// then the channels); an optional `Lvls` extension (version 3, a total
/// record count) carries the records past 29.
fn parse_levels(r: &mut ByteReader) -> Result<Adjustment> {
    let _version = r.u16()?;
    let read = |r: &mut ByteReader| -> Result<LevelsRecord> {
        Ok(LevelsRecord {
            in_black: r.u16()?,
            in_white: r.u16()?,
            out_black: r.u16()?,
            out_white: r.u16()?,
            gamma_x100: r.u16()?,
        })
    };
    let mut records = [LevelsRecord::IDENTITY; 4];
    let mut extra_records = false;
    for slot in &mut records {
        *slot = read(r)?;
    }
    for _ in 4..29 {
        if !read(r)?.is_identity() {
            extra_records = true;
        }
    }
    if r.remaining() >= 8 && r.fourcc()? == *b"Lvls" {
        let _version = r.u16()?;
        let total = r.u16()? as usize;
        for _ in 0..total.saturating_sub(29) {
            if r.remaining() < 10 {
                break;
            }
            if !read(r)?.is_identity() {
                extra_records = true;
            }
        }
    }
    Ok(Adjustment::Levels {
        records,
        extra_records,
    })
}

/// `hue2`: version, colorize flag, pad, the colorize triple, the master
/// triple, then six ranges of four bounds and a triple.
fn parse_hue_saturation(r: &mut ByteReader) -> Result<Adjustment> {
    let _version = r.u16()?;
    let colorized = r.u8()? != 0;
    let _pad = r.u8()?;
    let triple = |r: &mut ByteReader| -> Result<[i16; 3]> { Ok([r.i16()?, r.i16()?, r.i16()?]) };
    let colorize = triple(r)?;
    let master = triple(r)?;
    let mut ranges = [[0i16; 3]; 6];
    let mut bounds = [[0i16; 4]; 6];
    for k in 0..6 {
        bounds[k] = [r.i16()?, r.i16()?, r.i16()?, r.i16()?];
        ranges[k] = triple(r)?;
    }
    Ok(Adjustment::HueSaturation {
        colorized,
        colorize,
        master,
        ranges,
        bounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn be(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_be_bytes()).collect()
    }

    #[test]
    fn image_psd_adjustment_curves_points_are_output_then_input() {
        // pad, version 1, map = composite + blue, then two curves.
        let mut b = vec![0u8];
        b.extend(be(&[1]));
        b.extend(0b1001u32.to_be_bytes());
        b.extend(be(&[2, 0, 0, 255, 255]));
        b.extend(be(&[2, 30, 0, 220, 255]));
        let Some(Adjustment::Curves { curves }) = Adjustment::parse(b"curv", &b).unwrap() else {
            panic!()
        };
        assert_eq!(curves[0].as_deref(), Some(&[(0, 0), (255, 255)][..]));
        assert_eq!(curves[1], None);
        assert_eq!(curves[3].as_deref(), Some(&[(0, 30), (255, 220)][..]));
    }

    #[test]
    fn image_psd_adjustment_levels_reads_composite_then_channels() {
        let mut words = vec![2];
        words.extend([10, 240, 0, 255, 120]);
        words.extend([0, 200, 0, 255, 80]);
        for _ in 2..29 {
            words.extend([0, 255, 0, 255, 100]);
        }
        let Some(adj) = Adjustment::parse(b"levl", &be(&words)).unwrap() else {
            panic!()
        };
        let Adjustment::Levels {
            records,
            extra_records,
        } = &adj
        else {
            panic!()
        };
        assert_eq!(records[0].gamma_x100, 120);
        assert_eq!(records[1].in_white, 200);
        assert!(records[2].is_identity());
        assert!(!extra_records);
        assert_eq!(adj.unmodelled(), None);
    }

    #[test]
    fn image_psd_adjustment_unmodelled_parts_are_named() {
        let exposure = Adjustment::Exposure {
            exposure: 0.4,
            offset: -0.03,
            gamma: 1.2,
        };
        assert_eq!(
            exposure.unmodelled(),
            None,
            "an offset and gamma are a table"
        );
        let mut bounds = HUE_RANGE_DEFAULTS;
        bounds[0][0] = 300;
        let moved = Adjustment::HueSaturation {
            colorized: false,
            colorize: [0, 25, 0],
            master: [0; 3],
            ranges: [[0; 3]; 6],
            bounds,
        };
        assert!(moved.unmodelled().unwrap().contains("ranges"));
    }
}
