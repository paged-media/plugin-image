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

//! Blend-mode identifiers, in **both dialects**.
//!
//! Modern Photoshop writes enum identifiers as long-form names
//! (`normal`) where it once wrote four-character codes (`Nrml`) — and
//! writes both **in the same key, in the same file** (behaviour spec §9
//! `[OBS]`: `textureBlendMode` carried long-form `linearHeight` and
//! 4-byte `Sbtr` across one corpus). A reader must accept both, for
//! every enumerated value, indefinitely.
//!
//! The resolution order is the spec's: legacy 4-character table first
//! (exact and unambiguous), long-form table second, then a documented
//! default with the unrecognised value REPORTED — never a silent
//! fallback, because new long-form ids will keep appearing.
//!
//! # Provenance
//!
//! `[PUB]` for the classic four-character vocabulary (Adobe Photoshop
//! File Formats Specification documents the blend-mode keys) and for
//! the long-form direction of travel (Adobe UXP/batchPlay documents the
//! preference for long-form string identifiers with an escape for the
//! legacy codes). `[REF]` for the exact spelling of the later long-form
//! ids. `[OBS]` for `linearHeight` and `Sbtr`; `Hght` is `[REF]` and has
//! never been seen (spec §9, §14.2 item 7).

/// A blend mode as it appears in `.abr` (`textureBlendMode`, the dual
/// brush's `BlnM`, and `toolOptions`' `Md  ` — one vocabulary, three
/// sites).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    Normal,
    Dissolve,
    Darken,
    Multiply,
    ColorBurn,
    LinearBurn,
    DarkerColor,
    Lighten,
    Screen,
    ColorDodge,
    LinearDodge,
    LighterColor,
    Overlay,
    SoftLight,
    HardLight,
    VividLight,
    LinearLight,
    PinLight,
    HardMix,
    Difference,
    Exclusion,
    Subtract,
    Divide,
    Hue,
    Saturation,
    Color,
    Luminosity,
    /// ABR-only: texture blending against a height field. `[OBS]`
    LinearHeight,
    /// ABR-only. `[REF]` — never observed.
    Height,
    /// ABR-only. `[OBS]`
    SubtractionTexture,
}

/// The legacy four-character vocabulary. `[PUB]` for the layer modes,
/// `[OBS]`/`[REF]` for the three ABR-only ones (see the module docs).
const LEGACY: &[(&[u8; 4], BlendMode)] = &[
    (b"Nrml", BlendMode::Normal),
    (b"Dslv", BlendMode::Dissolve),
    (b"Drkn", BlendMode::Darken),
    (b"Mltp", BlendMode::Multiply),
    (b"CBrn", BlendMode::ColorBurn),
    (b"Lghn", BlendMode::Lighten),
    (b"Scrn", BlendMode::Screen),
    (b"CDdg", BlendMode::ColorDodge),
    (b"Ovrl", BlendMode::Overlay),
    (b"SftL", BlendMode::SoftLight),
    (b"HrdL", BlendMode::HardLight),
    (b"Dfrn", BlendMode::Difference),
    (b"Xclu", BlendMode::Exclusion),
    (b"H   ", BlendMode::Hue),
    (b"Strt", BlendMode::Saturation),
    (b"Clr ", BlendMode::Color),
    (b"Lmns", BlendMode::Luminosity),
    (b"Hght", BlendMode::Height),
    (b"Sbtr", BlendMode::SubtractionTexture),
];

/// The long-form vocabulary. The modes added after the 4-character era
/// were *born* long-form and have no legacy code at all.
const LONG_FORM: &[(&str, BlendMode)] = &[
    ("normal", BlendMode::Normal),
    ("dissolve", BlendMode::Dissolve),
    ("darken", BlendMode::Darken),
    ("multiply", BlendMode::Multiply),
    ("colorBurn", BlendMode::ColorBurn),
    ("linearBurn", BlendMode::LinearBurn),
    ("darkerColor", BlendMode::DarkerColor),
    ("lighten", BlendMode::Lighten),
    ("screen", BlendMode::Screen),
    ("colorDodge", BlendMode::ColorDodge),
    ("linearDodge", BlendMode::LinearDodge),
    ("lighterColor", BlendMode::LighterColor),
    ("overlay", BlendMode::Overlay),
    ("softLight", BlendMode::SoftLight),
    ("hardLight", BlendMode::HardLight),
    ("vividLight", BlendMode::VividLight),
    ("linearLight", BlendMode::LinearLight),
    ("pinLight", BlendMode::PinLight),
    ("hardMix", BlendMode::HardMix),
    ("difference", BlendMode::Difference),
    ("exclusion", BlendMode::Exclusion),
    ("blendSubtraction", BlendMode::Subtract),
    ("blendDivide", BlendMode::Divide),
    ("hue", BlendMode::Hue),
    ("saturation", BlendMode::Saturation),
    ("color", BlendMode::Color),
    ("luminosity", BlendMode::Luminosity),
    ("linearHeight", BlendMode::LinearHeight),
    ("height", BlendMode::Height),
];

impl BlendMode {
    /// Resolve an enum VALUE key (the second of the two `key-or-4cc`
    /// fields — there is no `.`-joined form in the bytes, spec §9
    /// CORRECTION).
    ///
    /// Returns `None` for an unrecognised identifier so the caller can
    /// report it and substitute [`BlendMode::Normal`]. Callers must not
    /// fail the file on a miss.
    pub fn from_key(key: &[u8]) -> Option<BlendMode> {
        if key.len() == 4 {
            if let Some((_, m)) = LEGACY.iter().find(|(c, _)| c.as_slice() == key) {
                return Some(*m);
            }
        }
        let s = std::str::from_utf8(key).ok()?;
        LONG_FORM.iter().find(|(n, _)| *n == s).map(|(_, m)| *m)
    }

    /// A stable identifier for logs and for the wasm/TS surface.
    pub fn name(self) -> &'static str {
        LONG_FORM
            .iter()
            .find(|(_, m)| *m == self)
            .map(|(n, _)| *n)
            .unwrap_or(match self {
                BlendMode::Subtract => "blendSubtraction",
                BlendMode::Divide => "blendDivide",
                BlendMode::SubtractionTexture => "subtractionTexture",
                _ => "normal",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_abr_brush_model_blend_accepts_both_dialects_for_the_same_mode() {
        assert_eq!(BlendMode::from_key(b"Nrml"), Some(BlendMode::Normal));
        assert_eq!(BlendMode::from_key(b"normal"), Some(BlendMode::Normal));
        assert_eq!(BlendMode::from_key(b"CBrn"), Some(BlendMode::ColorBurn));
        assert_eq!(
            BlendMode::from_key(b"colorBurn"),
            Some(BlendMode::ColorBurn)
        );
    }

    #[test]
    fn image_abr_brush_model_blend_legacy_wins_on_a_four_byte_key() {
        // `H   ` is the legacy code for Hue. It is also a brush-descriptor
        // key meaning hue jitter (spec §6.6 trap) — which is why the
        // table is scoped to blend-mode SITES and never shared by 4-char
        // code across contexts.
        assert_eq!(BlendMode::from_key(b"H   "), Some(BlendMode::Hue));
        assert_eq!(BlendMode::from_key(b"Strt"), Some(BlendMode::Saturation));
    }

    #[test]
    fn image_abr_brush_model_blend_abr_only_modes_are_present() {
        assert_eq!(
            BlendMode::from_key(b"linearHeight"),
            Some(BlendMode::LinearHeight)
        );
        assert_eq!(
            BlendMode::from_key(b"Sbtr"),
            Some(BlendMode::SubtractionTexture)
        );
        assert_eq!(BlendMode::from_key(b"Hght"), Some(BlendMode::Height));
    }

    #[test]
    fn image_abr_brush_model_blend_modes_born_long_form_have_no_legacy_code() {
        for born_long in ["linearBurn", "vividLight", "hardMix", "blendDivide"] {
            assert!(BlendMode::from_key(born_long.as_bytes()).is_some());
        }
        // …and nothing in the legacy table maps to them.
        for (_, m) in LEGACY {
            assert!(!matches!(
                m,
                BlendMode::LinearBurn
                    | BlendMode::VividLight
                    | BlendMode::HardMix
                    | BlendMode::Divide
            ));
        }
    }

    #[test]
    fn image_abr_brush_model_blend_unknown_identifier_is_reported_not_guessed() {
        assert_eq!(BlendMode::from_key(b"noSuchModeYet"), None);
        assert_eq!(BlendMode::from_key(b"ZZZZ"), None);
    }

    #[test]
    fn image_abr_brush_model_blend_there_is_no_dotted_form() {
        // Revision 1 of the behaviour spec described enum values as
        // `BlnM.Nrml`; the bytes contain zero `.` characters. A reader
        // that stripped up to the first dot would do nothing on
        // well-formed input and mangle a future dotted key.
        assert_eq!(BlendMode::from_key(b"BlnM.Nrml"), None);
    }
}
