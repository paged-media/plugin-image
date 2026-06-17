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

/// Convert a packed 4-ink CMYK8 buffer (`4·n` bytes, true ink amounts) to
/// straight RGBA8 (`4·n` bytes, A = 255) using the embedded ICC profile
/// when present, else the uncalibrated device-CMYK fallback. Returns the
/// RGBA8 bytes and whether the conversion was colour-managed (`true`) or
/// the uncalibrated fallback (`false`) — the caller may surface that.
pub fn cmyk8_to_rgba8(cmyk: &[u8], icc: Option<&[u8]>) -> Result<(Vec<u8>, bool), IngestError> {
    debug_assert_eq!(cmyk.len() % 4, 0, "CMYK input must be 4 bytes per pixel");
    if let Some(icc_bytes) = icc {
        // The colour-managed lane. A bad/non-CMYK embedded profile falls
        // back to the device formula rather than failing the decode (an
        // image with a broken profile is still a valid image).
        let src = profile_from_bytes(icc_bytes.to_vec());
        let dst = working_srgb_profile()
            .map_err(|e| IngestError::Unsupported(format!("CMYK ingest destination: {e}")))?;
        match MoxcmsEngine.compile_cmyk_to_rgba8(&src, &dst, Intent::Perceptual, false) {
            Ok(t) => return Ok((t.cmyk_to_rgba8_vec(cmyk), true)),
            Err(_) => { /* fall through to the uncalibrated device formula */ }
        }
    }
    Ok((cmyk_device_to_rgba8(cmyk), false))
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
        let (rgba, managed) =
            cmyk8_to_rgba8(&cmyk, Some(&srgb.bytes)).expect("must fall back");
        assert!(!managed, "an RGB source profile cannot drive the CMYK lane");
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255]);
    }
}
