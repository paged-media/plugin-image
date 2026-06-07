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

//! The moxcms backend — the print lane (D-11 ruling §F.2 in the A-0
//! audit). Pure-Rust CMS (no C FFI), chosen for the capabilities core's
//! qcms build demonstrably lacks (audit §A): genuine handling of all
//! four rendering intents (selecting the per-intent A2B/B2A LUTs that
//! qcms collapses to Perceptual/RelativeColorimetric), CMYK ingest, and
//! ICC v4 pipelines — the print-grade ingest / soft-proof / export lane
//! (spec §10.1 constraint 2).
//!
//! ## Intents
//!
//! Honored for real. moxcms keys LUT selection off
//! [`moxcms::TransformOptions::rendering_intent`]: a LUT-bearing profile
//! routes Perceptual → A2B0, Relative/Absolute colorimetric → A2B1
//! (the colorimetric table), and Saturation → A2B2. All four
//! [`Intent`] variants map 1:1 to [`moxcms::RenderingIntent`]; none is
//! silently degraded the way qcms degrades Saturation/Absolute (audit
//! §A). Matrix-shaper profiles (e.g. plain sRGB) carry no per-intent
//! tables, so for those the result is intent-independent by
//! construction — that is correct CMS behaviour, not a degradation:
//! there is nothing intent-specific to select. `tests/print_lane.rs`
//! proves the difference on a LUT profile that does carry distinct
//! A2B0/A2B1 tables.
//!
//! ## Black-point compensation
//!
//! moxcms 0.8.1 does **not** expose BPC as a runtime option: the
//! `black_point_compensation` field of `TransformOptions` is commented
//! out upstream and the BPC conversion module is dead code in this
//! release. A `bpc: true` request is therefore accepted and recorded on
//! the handle for provenance, but the produced transform is identical
//! to `bpc: false`. The BPC-fidelity conformance row stays pinned to a
//! future moxcms release (or the lcms2-shaped oracle) that exposes the
//! knob; `tests/print_lane.rs` documents the degradation and the
//! registry note rather than asserting an effect that does not exist
//! yet.
//!
//! ## Bit depth
//!
//! This backend implements the shared [`ExactTransform`] contract,
//! which is 8-bit RGBA in M1 (the qcms reality the trait was frozen
//! around). moxcms additionally offers float transforms
//! (`create_transform_f32`) — that is the **M2 high-bit-depth lane**:
//! once the trait grows a float endpoint (a versioned amendment, not a
//! drive-by edit), the print lane can keep the full precision moxcms
//! computes internally instead of quantising to 8-bit codes here.

use moxcms::{ColorProfile, Layout, RenderingIntent, Transform8BitExecutor, TransformOptions};
use std::sync::Arc;

use crate::{CmsEngine, CmsError, CompiledTransform, ExactTransform, Intent, Profile};

#[derive(Debug, Default)]
pub struct MoxcmsEngine;

/// Map our [`Intent`] onto moxcms's. All four are real; moxcms uses this
/// to pick the per-intent LUT (see the module docs).
fn map_intent(i: Intent) -> RenderingIntent {
    match i {
        Intent::Perceptual => RenderingIntent::Perceptual,
        Intent::RelativeColorimetric => RenderingIntent::RelativeColorimetric,
        Intent::Saturation => RenderingIntent::Saturation,
        Intent::AbsoluteColorimetric => RenderingIntent::AbsoluteColorimetric,
    }
}

/// The compiled moxcms RGBA8 transform. moxcms transforms are
/// out-of-place (distinct `src`/`dst` slices), so the in-place
/// [`ExactTransform`] contract copies the input once — same shape as the
/// qcms backend.
struct MoxcmsRgba8 {
    transform: Arc<Transform8BitExecutor>,
}

impl ExactTransform for MoxcmsRgba8 {
    fn apply_rgba8(&self, pixels: &mut [u8]) {
        let src = pixels.to_vec();
        // The executor maps RGBA→RGBA with matching sample counts; alpha
        // is carried through by moxcms for alpha-bearing layouts.
        // `transform` only errs on a length mismatch, which cannot happen
        // here (src and dst are the same buffer length).
        self.transform
            .transform(&src, pixels)
            .expect("moxcms RGBA8 transform: equal-length src/dst cannot fail");
    }
}

impl CmsEngine for MoxcmsEngine {
    fn compile(
        &self,
        src: &Profile,
        dst: &Profile,
        intent: Intent,
        bpc: bool,
    ) -> Result<CompiledTransform, CmsError> {
        let src_profile = ColorProfile::new_from_slice(&src.bytes)
            .map_err(|e| CmsError::BadProfile(format!("moxcms rejected source profile: {e:?}")))?;
        let dst_profile = ColorProfile::new_from_slice(&dst.bytes).map_err(|e| {
            CmsError::BadProfile(format!("moxcms rejected destination profile: {e:?}"))
        })?;

        // Intent is honored via TransformOptions (the only knob moxcms
        // 0.8.1 reads for LUT selection). BPC is NOT exposed by this
        // release (see module docs) — `bpc` rides on the handle for
        // provenance but does not change the transform.
        let options = TransformOptions {
            rendering_intent: map_intent(intent),
            ..Default::default()
        };

        let transform = src_profile
            .create_transform_8bit(Layout::Rgba, &dst_profile, Layout::Rgba, options)
            .map_err(|e| {
                CmsError::Unsupported(format!("moxcms could not build the RGBA8 transform: {e:?}"))
            })?;

        Ok(CompiledTransform {
            src: src.hash,
            dst: dst.hash,
            intent,
            bpc,
            backend: Box::new(MoxcmsRgba8 { transform }),
        })
    }
}
