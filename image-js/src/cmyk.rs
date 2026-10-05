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

//! The CMYK ingest cast (spec §5.2, the M2 print lane reached from the
//! M4 ingest slice). The JPEG codec already delivers true 4-ink CMYK
//! (`ChannelLayout::Cmyk`, post the Adobe-APP14 re-inversion); this
//! module is the missing CMS step that turns that ink into the straight
//! RGBA8 the ingest lane speaks — so a CMYK placed image decodes instead
//! of being rejected `Unsupported`.
//!
//! Two cases, both honest:
//!
//!  - **Embedded ICC present** — the colour-managed path. The embedded
//!    CMYK *device* profile is the moxcms transform source; the working
//!    RGB destination is the canonical sRGB profile moxcms synthesises
//!    (`ColorProfile::new_srgb`). This is the same
//!    [`image_cms::CmsEngine::compile_cmyk_to_rgba8`] print lane the
//!    conformance suite (`image-cms/tests/cmyk_lane.rs`) gates against
//!    lcms2. Intent is Perceptual (the ingest default; the panel does not
//!    yet surface an intent control — a follow-up, not a wrong result).
//!
//!  - **No embedded ICC** — the uncalibrated fallback. Many CMYK JPEGs
//!    ship with no profile, and there is no free, redistributable
//!    reference CMYK profile to assume on their behalf (US Web Coated
//!    SWOP and FOGRA profiles are not freely licensed). Rather than
//!    reject the image (the old behaviour) we apply the naive device
//!    CMYK→RGB formula `c = (1 - ink/255)`, `R = 255·(1-C')·(1-K')` …,
//!    i.e. the multiplicative ink model. This is NOT colour-managed and
//!    is deliberately flagged as such (the caller can surface a
//!    "no profile — uncalibrated" note); it is the standard device
//!    interpretation and never produces a torn or inverted image.
//!
//! Both paths run on the CPU: CMS transform compilation/application and
//! the codec decode are inherently-CPU work (spec §6), not GPU kernels.
//!
//! Callers: the CMYK JPEG decode, and the CMYK PSD — its merged composite
//! and, for a layered open, every layer plate ([`InkTransform`]).

use image_cms::moxcms_engine::MoxcmsEngine;
use image_cms::{working_srgb_profile, CmsEngine, Intent, Profile};
use image_core::{ContentHash, IccHash};

use crate::ingest::IngestError;

/// Build an [`image_cms::Profile`] from raw ICC bytes (the interner's
/// content-hash identity, inlined — the ingest path holds one transient
/// profile, not a document-wide interner).
fn profile_from_bytes(bytes: Vec<u8>) -> Profile {
    let bytes: std::sync::Arc<[u8]> = bytes.into();
    Profile {
        hash: IccHash(ContentHash::of(&bytes).0),
        bytes,
    }
}

/// One CMYK→RGBA8 conversion, compiled ONCE per source and applied to
/// every buffer of it — a layered CMYK PSD converts its merged composite
/// and every layer plate, and they must all go through the SAME
/// transform (or the layered open and the flattened open would disagree
/// by construction).
///
/// # The intent, and what Photoshop does
///
/// Photoshop's RGB view of a CMYK document — Edit > Convert to Profile,
/// with the Color Settings' defaults — is Adobe ACE, RELATIVE
/// COLORIMETRIC, with BLACK-POINT COMPENSATION (read from the app's
/// Color Settings by `scripts/photoshop/probes/cmyk-stacks.jsx`,
/// "Europe General Purpose 3"). moxcms 0.8.1 has no BPC, and relative
/// colorimetric WITHOUT it lands 28 levels from Photoshop on the
/// recorded ink patches (the profile's black is lifted: L* ≈ 9 printed
/// black has nowhere to go in sRGB but up). PERCEPTUAL — the profile's
/// A2B0, whose black is already mapped to the PCS black — lands within
/// 4 levels (mean 0.44) of Photoshop's relative colorimetric + BPC on
/// the same patches, the same as lcms2's own relcol+BPC does. So the
/// lane uses Perceptual, and the convention is stated: "Photoshop's
/// default view, approximated by the profile's perceptual table" — the
/// replay (`image-conformance/tests/psd_cmyk_photoshop.rs`) holds it to
/// those numbers. Saturated greens outside sRGB are where ACE and the
/// ICC-standard CMMs clip differently (up to ~23 levels on one channel,
/// identical for moxcms and lcms2).
pub struct InkTransform {
    compiled: Option<image_cms::CompiledCmykTransform>,
}

impl InkTransform {
    /// Compile from the source's embedded ICC profile. `None`, or a
    /// profile that does not compile as a CMYK source, yields the
    /// uncalibrated device formula — and [`Self::is_managed`] says so.
    pub fn for_profile(icc: Option<&[u8]>) -> InkTransform {
        let compiled = icc.and_then(|bytes| {
            let src = profile_from_bytes(bytes.to_vec());
            let dst = working_srgb_profile().ok()?;
            MoxcmsEngine
                .compile_cmyk_to_rgba8(&src, &dst, Intent::Perceptual, false)
                .ok()
        });
        InkTransform { compiled }
    }

    /// `true` when the embedded profile drove the conversion; `false` for
    /// the uncalibrated device formula.
    pub fn is_managed(&self) -> bool {
        self.compiled.is_some()
    }

    /// The treatment to report for pixels this converted.
    pub fn treatment(&self) -> crate::display::DisplayTreatment {
        if self.is_managed() {
            crate::display::DisplayTreatment::CmykConverted
        } else {
            crate::display::DisplayTreatment::CmykUncalibrated
        }
    }

    /// Packed 4-ink CMYK8 (`4·n` bytes, 0 = no ink) → straight RGBA8
    /// (`4·n` bytes, A = 255).
    pub fn to_rgba8(&self, cmyk: &[u8]) -> Vec<u8> {
        debug_assert_eq!(cmyk.len() % 4, 0, "CMYK input must be 4 bytes per pixel");
        match &self.compiled {
            Some(t) => t.cmyk_to_rgba8_vec(cmyk),
            None => cmyk_device_to_rgba8(cmyk),
        }
    }
}

/// Convert a packed 4-ink CMYK8 buffer (`4·n` bytes, true ink amounts) to
/// straight RGBA8 (`4·n` bytes, A = 255) using the embedded ICC profile
/// when present, else the uncalibrated device-CMYK fallback. Returns the
/// RGBA8 bytes and whether the conversion was colour-managed (`true`) or
/// the uncalibrated fallback (`false`) — the caller may surface that.
pub fn cmyk8_to_rgba8(cmyk: &[u8], icc: Option<&[u8]>) -> Result<(Vec<u8>, bool), IngestError> {
    // A bad/non-CMYK embedded profile falls back to the device formula
    // rather than failing the decode (an image with a broken profile is
    // still a valid image).
    let t = InkTransform::for_profile(icc);
    Ok((t.to_rgba8(cmyk), t.is_managed()))
}

/// The naive, uncalibrated device CMYK→RGBA8 conversion: the standard
/// multiplicative ink model `R = 255·(1-C')·(1-K')` (and M/Y likewise),
/// where `C' = C/255`. Alpha is synthesised to 255 (CMYK ink carries no
/// transparency). Used only when no embedded ICC profile is available;
/// it is colour-INcorrect by definition but never produces a torn or
/// inverted image.
pub fn cmyk_device_to_rgba8(cmyk: &[u8]) -> Vec<u8> {
    let n = cmyk.len() / 4;
    let mut rgba = vec![0u8; n * 4];
    for (px, out) in cmyk.chunks_exact(4).zip(rgba.chunks_exact_mut(4)) {
        let c = px[0] as f32 / 255.0;
        let m = px[1] as f32 / 255.0;
        let y = px[2] as f32 / 255.0;
        let k = px[3] as f32 / 255.0;
        let one_k = 1.0 - k;
        out[0] = ((1.0 - c) * one_k * 255.0 + 0.5) as u8;
        out[1] = ((1.0 - m) * one_k * 255.0 + 0.5) as u8;
        out[2] = ((1.0 - y) * one_k * 255.0 + 0.5) as u8;
        out[3] = 255;
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    // feat: image.editor.ingest (the CMYK ingest cast) — naming carries
    // the feature tag until the state feature_test macro ships.

    /// The device fallback maps the ink corners the way the multiplicative
    /// model demands, and always synthesises opaque alpha.
    #[test]
    fn image_editor_ingest_cmyk_device_fallback_maps_ink_corners() {
        // paper white (no ink), solid C, solid M, solid Y, solid K.
        let cmyk = vec![
            0, 0, 0, 0, // white  -> 255,255,255
            255, 0, 0, 0, // cyan   -> 0,255,255
            0, 255, 0, 0, // magenta-> 255,0,255
            0, 0, 255, 0, // yellow -> 255,255,0
            0, 0, 0, 255, // black  -> 0,0,0
        ];
        let rgba = cmyk_device_to_rgba8(&cmyk);
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255], "paper white");
        assert_eq!(&rgba[4..8], &[0, 255, 255, 255], "solid cyan");
        assert_eq!(&rgba[8..12], &[255, 0, 255, 255], "solid magenta");
        assert_eq!(&rgba[12..16], &[255, 255, 0, 255], "solid yellow");
        assert_eq!(&rgba[16..20], &[0, 0, 0, 255], "solid K");
        for px in rgba.chunks_exact(4) {
            assert_eq!(px[3], 255, "alpha synthesised to opaque");
        }
    }

    /// With no ICC, `cmyk8_to_rgba8` takes the uncalibrated path and
    /// reports `colour_managed == false` (the caller can surface it).
    #[test]
    fn image_editor_ingest_cmyk_no_icc_is_uncalibrated_fallback() {
        let cmyk = vec![0u8, 0, 0, 0, 255, 0, 0, 0];
        let (rgba, managed) = cmyk8_to_rgba8(&cmyk, None).expect("device fallback never fails");
        assert!(!managed, "no ICC must take the uncalibrated path");
        assert_eq!(rgba.len(), cmyk.len(), "pixel-for-pixel");
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255], "paper white");
    }

    /// With a real sRGB-as-source ICC (the wrong colour class) the cast
    /// rejects internally and falls back to the device formula — a broken
    /// profile must NEVER fail the decode. (A genuine CMYK device profile
    /// taking the managed path is gated by `image-cms/tests/cmyk_lane.rs`,
    /// which has the lcms2 oracle to author one.)
    #[test]
    fn image_editor_ingest_cmyk_bad_profile_falls_back_not_errors() {
        // sRGB bytes are a valid ICC but an RGB source — the CMYK lane
        // refuses it, so we must fall back rather than error.
        let srgb = working_srgb_profile().expect("synthesise sRGB");
        let cmyk = vec![0u8, 0, 0, 0];
        let (rgba, managed) = cmyk8_to_rgba8(&cmyk, Some(&srgb.bytes)).expect("must fall back");
        assert!(!managed, "an RGB source profile cannot drive the CMYK lane");
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255]);
    }
}
