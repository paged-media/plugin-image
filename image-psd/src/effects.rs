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

//! LAYER EFFECTS — the `lfx2` descriptor read into what the layer import
//! needs to know: is any effect actually drawn, which ones, and the
//! parameters of the one it models (Color Overlay).
//!
//! A layer can carry an effects block whose effects are all switched off
//! (or whose master switch is off); Photoshop draws nothing for it, and
//! neither does the import. The corpus has 183 such layers — every one of
//! them used to refuse its whole file.
//!
//! Provenance: Adobe Photoshop File Formats specification — Additional
//! Layer Information, Object-based effects layer info (`lfx2`: a version
//! word, then a versioned descriptor) and the descriptor value structure
//! [PUB]; the effect keys (`SoFi` Color Overlay, `FrFX` Stroke, `GrFl`
//! Gradient Overlay, `DrSh` Drop Shadow, `IrSh` Inner Shadow, `OrGl` /
//! `IrGl` glows, `ebbl` Bevel & Emboss, `ChFX` Satin, `patternFill`), each
//! effect's `enab` switch, the `*Multi` list form and `masterFXSwitch`
//! checked against effects Photoshop 27.10 wrote itself
//! (`image-conformance/fixtures/photoshop/layer-effects/`) [OBS].

use crate::descriptor::{read_versioned_descriptor, Descriptor, DescriptorValue};
use crate::reader::ByteReader;
use crate::Result;

/// A Color Overlay as the stack draws it: a solid colour blended onto
/// the layer's own content, inside its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorOverlay {
    /// The colour in RGB. For a colour given in CMYK this is filled in by
    /// the importer through the document's ink transform (`cmyk`).
    pub rgb: [u8; 3],
    /// The colour as INK (0–255 = 0–100 %), when the descriptor gives it
    /// in CMYK (`CMYC`), as in a CMYK document.
    pub cmyk: Option<[u8; 4]>,
    /// The layer-record blend key (`norm`, `mul `, …) of the overlay's
    /// mode.
    pub blend_key: [u8; 4],
    /// 0–255.
    pub opacity: u8,
}

/// What a layer's `lfx2` block draws.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Effects {
    /// The enabled effects' keys, in descriptor order, without the
    /// `Multi` suffix (`SoFi`, `FrFX`, …). Empty when nothing is drawn.
    pub enabled: Vec<String>,
    /// The Color Overlay, when one is enabled and representable; `Err`
    /// names why an enabled one is not.
    pub color_overlay: Option<std::result::Result<ColorOverlay, String>>,
}

/// The descriptor enum value of a blend mode (`BlnM`) as the layer
/// record's four-character key. Both dialects occur: the corpus writes
/// the four-character ids (`Nrml`, `Mltp`), Photoshop 27.10 writes the
/// string ids (`normal`, `multiply`) for the same modes [OBS].
pub fn blend_key_of(mode: &str) -> Option<[u8; 4]> {
    Some(match mode {
        "Nrml" | "normal" => *b"norm",
        "Drkn" | "darken" => *b"dark",
        "Mltp" | "multiply" => *b"mul ",
        "CBrn" | "colorBurn" => *b"idiv",
        "linearBurn" => *b"lbrn",
        "darkerColor" => *b"dkCl",
        "Lghn" | "lighten" => *b"lite",
        "Scrn" | "screen" => *b"scrn",
        "CDdg" | "colorDodge" => *b"div ",
        "linearDodge" => *b"lddg",
        "lighterColor" => *b"lgCl",
        "Ovrl" | "overlay" => *b"over",
        "SftL" | "softLight" => *b"sLit",
        "HrdL" | "hardLight" => *b"hLit",
        "vividLight" => *b"vLit",
        "linearLight" => *b"lLit",
        "pinLight" => *b"pLit",
        "hardMix" => *b"hMix",
        "Dfrn" | "difference" => *b"diff",
        "Xclu" | "exclusion" => *b"smud",
        "blendSubtraction" => *b"fsub",
        "blendDivide" => *b"fdiv",
        "H   " | "hue" => *b"hue ",
        "Strt" | "saturation" => *b"sat ",
        "Clr " | "color" => *b"colr",
        "Lmns" | "luminosity" => *b"lum ",
        _ => return None,
    })
}

impl Effects {
    /// Read an `lfx2` payload (the bytes after the block length): a
    /// version word, then a versioned descriptor.
    pub fn parse_lfx2(payload: &[u8]) -> Result<Effects> {
        let mut r = ByteReader::new(payload);
        let _version = r.u32()?;
        let (_, d) = read_versioned_descriptor(&mut r)?;
        Ok(Self::from_descriptor(&d))
    }

    fn from_descriptor(d: &Descriptor) -> Effects {
        let mut out = Effects::default();
        if d.bool(b"masterFXSwitch") == Some(false) {
            return out;
        }
        let on =
            |e: &Descriptor| e.bool(b"enab").unwrap_or(true) && e.bool(b"present").unwrap_or(true);
        for (key, value) in &d.items {
            let instances: Vec<&Descriptor> = match value {
                DescriptorValue::Descriptor(e) => vec![e],
                DescriptorValue::List(l) => l
                    .iter()
                    .filter_map(DescriptorValue::as_descriptor)
                    .collect(),
                _ => continue,
            };
            let name = key.text_lossy();
            let kind = name.strip_suffix("Multi").unwrap_or(&name).to_string();
            let kind = match kind.as_str() {
                "solidFill" => "SoFi".to_string(),
                "frameFX" => "FrFX".to_string(),
                "gradientFill" => "GrFl".to_string(),
                "dropShadow" => "DrSh".to_string(),
                "innerShadow" => "IrSh".to_string(),
                _ => kind,
            };
            let enabled: Vec<&Descriptor> = instances.into_iter().filter(|e| on(e)).collect();
            if enabled.is_empty() {
                continue;
            }
            if kind == "SoFi" {
                out.color_overlay = Some(if enabled.len() > 1 {
                    Err("more than one Color Overlay".into())
                } else {
                    color_overlay(enabled[0])
                });
            }
            if !out.enabled.contains(&kind) {
                out.enabled.push(kind);
            }
        }
        out
    }
}

fn color_overlay(e: &Descriptor) -> std::result::Result<ColorOverlay, String> {
    let mode = e
        .enum_value(b"Md  ")
        .map(|(_, v)| v.text_lossy())
        .unwrap_or_else(|| "Nrml".into());
    let blend_key =
        blend_key_of(&mode).ok_or_else(|| format!("a Color Overlay in mode {mode:?}"))?;
    let opacity = e
        .unit_float(b"Opct")
        .map_or(100.0, |(_, v)| v)
        .clamp(0.0, 100.0);
    let clr = e
        .descriptor(b"Clr ")
        .ok_or("a Color Overlay without a colour")?;
    let (rgb, cmyk) = if clr.class_id.matches(b"RGBC") {
        let ch = |k: &[u8]| clr.number(k).unwrap_or(0.0).round().clamp(0.0, 255.0) as u8;
        ([ch(b"Rd  "), ch(b"Grn "), ch(b"Bl  ")], None)
    } else if clr.class_id.matches(b"CMYC") {
        // [PUB] a CMYK colour: Cyn / Mgnt / Ylw / Blck in percent.
        let ink = |k: &[u8]| {
            (clr.number(k).unwrap_or(0.0) * 255.0 / 100.0)
                .round()
                .clamp(0.0, 255.0) as u8
        };
        (
            [0, 0, 0],
            Some([ink(b"Cyn "), ink(b"Mgnt"), ink(b"Ylw "), ink(b"Blck")]),
        )
    } else {
        return Err(format!(
            "a Color Overlay whose colour is given as {:?}",
            clr.class_id.text_lossy()
        ));
    };
    Ok(ColorOverlay {
        rgb,
        cmyk,
        blend_key,
        opacity: (opacity * 255.0 / 100.0).round() as u8,
    })
}
