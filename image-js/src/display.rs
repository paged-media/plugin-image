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

//! The RGB DISPLAY transform at ingest — CMS rung 1.
//!
//! Until now `decode_rgba8` handed RGB pixels through verbatim. The CMS
//! lanes were built and conformance-gated, but nothing called them on the
//! RGB path, so an image tagged AdobeRGB or ProPhoto decoded to its raw
//! encoded numbers and every kernel downstream — whose math is specified
//! over the post-CMS working space — operated on values that were not in
//! that space. The result looked plausible and was wrong, which is the
//! worst kind of wrong: nothing reports it.
//!
//! This module is the missing step, and it is deliberately the SAME shape
//! as [`crate::cmyk`]: transform at decode, keep every kernel untouched.
//!
//! Two honest states, and the caller surfaces which one happened:
//!
//! * **Managed** — the source carried an embedded ICC profile, and it
//!   compiled. Pixels are transformed into the working sRGB space.
//! * **sRGB assumed** — no embedded profile (the overwhelmingly common
//!   case for web PNG/JPEG), or a profile that failed to compile. The
//!   pixels pass through unchanged, which is the correct treatment for
//!   untagged sRGB and the only defensible guess otherwise.
//!
//! "sRGB assumed" is a state to REPORT, not to hide. A broken profile
//! falls back rather than failing the decode, for the same reason the
//! CMYK lane does: an image with a bad profile is still a valid image.
//!
//! Runs on the CPU, like the rest of ingest — CMS compilation and codec
//! decode are inherently-CPU work (spec §6), not GPU kernels.

use image_cms::qcms_engine::QcmsEngine;
use image_cms::{working_srgb_profile, CmsEngine, Intent, Profile};
use image_core::{ContentHash, IccHash};

/// What the display transform did, so the panel can say it rather than
/// leave the user guessing which numbers they are looking at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayTreatment {
    /// An embedded profile compiled and the pixels were transformed.
    Managed,
    /// No embedded profile — the bytes are taken to be sRGB already.
    AssumedSrgb,
    /// A profile was present but did not compile; treated as sRGB.
    ProfileRejected,
}

impl DisplayTreatment {
    /// A short phrase for the panel's colour row.
    pub fn label(self) -> &'static str {
        match self {
            Self::Managed => "ICC managed",
            Self::AssumedSrgb => "sRGB assumed",
            Self::ProfileRejected => "sRGB assumed (embedded profile rejected)",
        }
    }

    /// Whether real colour management happened. `false` is not an error —
    /// it is the honest majority case.
    pub fn is_managed(self) -> bool {
        matches!(self, Self::Managed)
    }
}

/// Build an [`image_cms::Profile`] from raw ICC bytes. Mirrors
/// `cmyk::profile_from_bytes` — the ingest path holds one transient
/// profile, not a document-wide interner.
fn profile_from_bytes(bytes: Vec<u8>) -> Profile {
    let bytes: std::sync::Arc<[u8]> = bytes.into();
    Profile {
        hash: IccHash(ContentHash::of(&bytes).0),
        bytes,
    }
}

/// Transform a straight RGBA8 buffer from its embedded profile into the
/// working sRGB space, in place. Returns which treatment was applied.
///
/// `icc` is the profile the codec read (EXIF/ICC parsing already happens
/// at probe time); `None` means the file carried none.
pub fn to_working_srgb(rgba: &mut [u8], icc: Option<&[u8]>) -> DisplayTreatment {
    debug_assert_eq!(rgba.len() % 4, 0, "RGBA input must be 4 bytes per pixel");
    let Some(icc_bytes) = icc else {
        return DisplayTreatment::AssumedSrgb;
    };
    let src = profile_from_bytes(icc_bytes.to_vec());
    let Ok(dst) = working_srgb_profile() else {
        // The destination profile is ours and should always build; if it
        // does not, transforming would be worse than not transforming.
        return DisplayTreatment::ProfileRejected;
    };
    // Perceptual, no black-point compensation — the display lane's
    // documented default (BPC is recorded but inert in this backend, see
    // the `image.cms.display` registry note).
    match QcmsEngine.compile(&src, &dst, Intent::Perceptual, false) {
        Ok(t) => {
            t.apply_rgba8(rgba);
            DisplayTreatment::Managed
        }
        Err(_) => DisplayTreatment::ProfileRejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No profile is the common case and must be a pass-through, not a
    /// guess dressed up as management.
    #[test]
    fn untagged_pixels_pass_through_as_assumed_srgb() {
        let mut rgba = vec![10, 20, 30, 255, 200, 100, 50, 255];
        let before = rgba.clone();
        let treatment = to_working_srgb(&mut rgba, None);
        assert_eq!(treatment, DisplayTreatment::AssumedSrgb);
        assert!(!treatment.is_managed());
        assert_eq!(rgba, before, "untagged pixels must not be altered");
    }

    /// A malformed profile must not fail the decode — an image with a
    /// broken profile is still a valid image. Same rule as the CMYK lane.
    #[test]
    fn a_broken_profile_falls_back_rather_than_failing() {
        let mut rgba = vec![10, 20, 30, 255];
        let before = rgba.clone();
        let treatment = to_working_srgb(&mut rgba, Some(b"not an icc profile at all"));
        assert_eq!(treatment, DisplayTreatment::ProfileRejected);
        assert_eq!(rgba, before, "a rejected profile leaves pixels untouched");
        // And it says so — the label must not read like a success.
        assert!(treatment.label().contains("rejected"));
    }

    /// The labels are user-facing; pin that the unmanaged ones say so.
    #[test]
    fn labels_report_the_unmanaged_states_honestly() {
        assert_eq!(DisplayTreatment::AssumedSrgb.label(), "sRGB assumed");
        assert!(DisplayTreatment::Managed.is_managed());
        assert!(!DisplayTreatment::ProfileRejected.is_managed());
    }
}
