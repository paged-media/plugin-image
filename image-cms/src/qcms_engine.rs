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

//! The qcms backend — the display-consistency engine (D-11 ruling §F.1
//! in the A-0 audit): the exact build core uses (qcms 0.3, `cmyk` +
//! `iccv4-enabled`), so "the page" and "the image editor" agree by
//! construction. Known limits inherited deliberately (audit §A): no
//! BPC, 8-bit endpoints, Perceptual/RelCol-centric intents — the print
//! lane (moxcms, M1) covers what this backend can't.
//!
//! `bpc: true` requests are accepted and recorded on the handle but
//! not honored by qcms — the conformance row for BPC fidelity is
//! pinned to the moxcms lane; display-path consistency tests compare
//! against core's identical non-BPC behavior.

use crate::{CmsEngine, CmsError, CompiledTransform, ExactTransform, Intent, Profile};

#[derive(Debug, Default)]
pub struct QcmsEngine;

fn map_intent(i: Intent) -> qcms::Intent {
    match i {
        Intent::Perceptual => qcms::Intent::Perceptual,
        Intent::RelativeColorimetric => qcms::Intent::RelativeColorimetric,
        Intent::Saturation => qcms::Intent::Saturation,
        Intent::AbsoluteColorimetric => qcms::Intent::AbsoluteColorimetric,
    }
}

struct QcmsRgba8 {
    transform: qcms::Transform,
}

impl ExactTransform for QcmsRgba8 {
    fn apply_rgba8(&self, pixels: &mut [u8]) {
        // qcms converts src → dst; the in-place contract copies once.
        let src = pixels.to_vec();
        self.transform.convert(&src, pixels);
    }
}

impl CmsEngine for QcmsEngine {
    fn compile(
        &self,
        src: &Profile,
        dst: &Profile,
        intent: Intent,
        bpc: bool,
    ) -> Result<CompiledTransform, CmsError> {
        let src_profile = qcms::Profile::new_from_slice(&src.bytes, false)
            .ok_or_else(|| CmsError::BadProfile("source profile rejected by qcms".into()))?;
        let mut dst_profile = qcms::Profile::new_from_slice(&dst.bytes, false)
            .ok_or_else(|| CmsError::BadProfile("destination profile rejected by qcms".into()))?;
        dst_profile.precache_output_transform();

        let transform = qcms::Transform::new(
            &src_profile,
            &dst_profile,
            qcms::DataType::RGBA8,
            map_intent(intent),
        )
        .ok_or_else(|| CmsError::Unsupported("qcms could not build the RGBA8 transform".into()))?;

        Ok(CompiledTransform {
            src: src.hash,
            dst: dst.hash,
            intent,
            bpc,
            backend: Box::new(QcmsRgba8 { transform }),
        })
    }
}
