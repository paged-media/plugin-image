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

//! Color management (spec §10.1) behind the swappable [`CmsEngine`]
//! trait — the D-11 seam (A-0 audit ruling: HYBRID). The qcms backend
//! serves the display-consistency path with core's exact build; the
//! moxcms print lane (CMYK ingest / soft-proof with intent+BPC /
//! export) is an additive M1 backend behind the same trait.
//!
//! **Compile on CPU, apply on GPU** stands whatever the backend: the
//! engine builds the transform; the GPU path bakes it into a 3D LUT
//! (+ 1D shapers) sampled by the `cms.apply` kernel (T1/M1). Exact
//! CPU transforms only where byte-exactness is contractual (export
//! encode, conformance goldens).

#![forbid(unsafe_code)]

mod interner;
mod lut;
pub mod qcms_engine;

pub use interner::ProfileInterner;
pub use lut::GpuLut;

use image_core::IccHash;

#[derive(Debug, thiserror::Error)]
pub enum CmsError {
    #[error("profile rejected: {0}")]
    BadProfile(String),
    #[error("unsupported transform: {0}")]
    Unsupported(String),
}

/// Rendering intent (ICC). The qcms backend accepts all four tags but
/// is Perceptual/RelativeColorimetric-centric (A-0 audit §A) — the
/// print lane's intent fidelity is the moxcms backend's job (M1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    Perceptual,
    RelativeColorimetric,
    Saturation,
    AbsoluteColorimetric,
}

/// An interned ICC profile: the bytes + the identity image-core knows
/// it by (`ColorSpaceRef::Icc`).
#[derive(Debug, Clone)]
pub struct Profile {
    pub hash: IccHash,
    pub bytes: std::sync::Arc<[u8]>,
}

/// A compiled transform handle. Engine-opaque; consumers either bake
/// it to a [`GpuLut`] (the production path) or apply it exactly on CPU
/// (export/goldens; backend-provided).
pub struct CompiledTransform {
    pub src: IccHash,
    pub dst: IccHash,
    pub intent: Intent,
    pub bpc: bool,
    pub(crate) backend: Box<dyn ExactTransform>,
}

/// Backend-internal exact application (8-bit endpoints in M0 — the
/// qcms reality; the float path arrives with moxcms). Deliberately NOT
/// `Send + Sync` in M0: the engine is single-threaded until the
/// worker-capability RFC lands (BREAKAGE I-02), and qcms's transform
/// handle shouldn't be declared shareable on our say-so.
pub(crate) trait ExactTransform {
    /// Transform interleaved RGBA8 in place (alpha passed through).
    fn apply_rgba8(&self, pixels: &mut [u8]);
}

impl CompiledTransform {
    /// Exact CPU application — export encode / conformance goldens.
    pub fn apply_rgba8(&self, pixels: &mut [u8]) {
        self.backend.apply_rgba8(pixels);
    }

    /// Bake to the GPU-sampleable LUT (production apply path). M0
    /// bakes by sampling the exact transform over the lattice.
    pub fn bake_lut(&self, dim: u32) -> GpuLut {
        lut::bake_from_exact(self.backend.as_ref(), dim)
    }
}

/// The swappable engine seam (D-11). Mirrors core's narrow `Cmm`
/// trait: build once per (src, dst, intent, bpc), apply many.
pub trait CmsEngine {
    fn compile(
        &self,
        src: &Profile,
        dst: &Profile,
        intent: Intent,
        bpc: bool,
    ) -> Result<CompiledTransform, CmsError>;
}
