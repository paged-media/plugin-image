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
pub mod moxcms_engine;
pub mod qcms_engine;

pub use interner::ProfileInterner;
pub use lut::GpuLut;

use image_core::IccHash;

/// The canonical sRGB profile as a ready-to-use [`Profile`] — the working
/// RGB destination for the ingest casts (CMYK→RGBA8) when no document
/// working-space profile is supplied. Synthesised by the moxcms backend
/// (`ColorProfile::new_srgb`), so no external ICC asset is bundled. The
/// returned profile's `hash` is the content hash of the synthesised bytes
/// (stable across calls within a build). Errors only if moxcms cannot
/// encode its own canonical sRGB profile (a should-never-happen).
pub fn working_srgb_profile() -> Result<Profile, CmsError> {
    let bytes = moxcms::ColorProfile::new_srgb()
        .encode()
        .map_err(|e| CmsError::BadProfile(format!("could not synthesise sRGB profile: {e:?}")))?;
    let bytes: std::sync::Arc<[u8]> = bytes.into();
    let hash = IccHash(image_core::ContentHash::of(&bytes).0);
    Ok(Profile { hash, bytes })
}

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

/// A compiled CMYK→RGBA8 transform handle — the print-lane ingest path
/// (spec §5.2: "CMYK sources transform to working space at ingest").
/// Distinct from [`CompiledTransform`] because the input is 4-channel ink
/// (one byte per C, M, Y, K), not RGBA: the [`ExactTransform`] in-place
/// contract cannot express a channel-count change. Engine-opaque; the
/// backend converts a packed CMYK8 buffer to a packed straight-RGBA8
/// buffer (alpha synthesised as 255 — CMYK ink carries no transparency).
pub struct CompiledCmykTransform {
    pub src: IccHash,
    pub dst: IccHash,
    pub intent: Intent,
    pub bpc: bool,
    pub(crate) backend: Box<dyn CmykTransform>,
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

/// Backend-internal exact CMYK→RGBA8 application: reads `cmyk` as packed
/// 4-byte ink (C, M, Y, K) and writes `rgba` as packed straight RGBA8
/// (A = 255). The two slices have independent channel counts, so this is
/// a separate out-of-place contract rather than an in-place one. 8-bit
/// endpoints in M1 (the float path arrives with the M2 high-bit lane).
pub(crate) trait CmykTransform {
    /// `cmyk.len()` must be `4 * n` and `rgba.len()` must be `4 * n` for
    /// the same `n`; the caller guarantees this (see
    /// [`CompiledCmykTransform::apply_cmyk_to_rgba8`]).
    fn apply(&self, cmyk: &[u8], rgba: &mut [u8]);
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

impl CompiledCmykTransform {
    /// Exact CPU application of the print-lane ingest transform: packed
    /// CMYK8 (`cmyk`, `4 * n` bytes) → packed straight RGBA8 (`rgba`,
    /// `4 * n` bytes, A = 255). Panics on a length mismatch — callers hold
    /// the contract; conversions are compiled paths, never ad-hoc per-op
    /// code (spec §5.1).
    pub fn apply_cmyk_to_rgba8(&self, cmyk: &[u8], rgba: &mut [u8]) {
        assert_eq!(cmyk.len() % 4, 0, "CMYK input must be 4 bytes per pixel");
        assert_eq!(rgba.len() % 4, 0, "RGBA output must be 4 bytes per pixel");
        assert_eq!(
            cmyk.len(),
            rgba.len(),
            "CMYK→RGBA is pixel-for-pixel (both 4 channels)"
        );
        self.backend.apply(cmyk, rgba);
    }

    /// Convenience: allocate the RGBA8 output for a CMYK8 input and apply.
    pub fn cmyk_to_rgba8_vec(&self, cmyk: &[u8]) -> Vec<u8> {
        let mut rgba = vec![0u8; cmyk.len()];
        self.apply_cmyk_to_rgba8(cmyk, &mut rgba);
        rgba
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

    /// Compile a CMYK→RGBA8 transform: a CMYK *device* source profile
    /// (`src`) to an RGB destination (`dst`) — the print-lane ingest path
    /// (spec §5.2). Additive over [`Self::compile`]: backends that cannot
    /// honour CMYK ingest (the qcms display lane, audit §A) keep the
    /// frozen surface by inheriting the default, which returns
    /// [`CmsError::Unsupported`]. The moxcms print lane overrides it.
    fn compile_cmyk_to_rgba8(
        &self,
        _src: &Profile,
        _dst: &Profile,
        _intent: Intent,
        _bpc: bool,
    ) -> Result<CompiledCmykTransform, CmsError> {
        Err(CmsError::Unsupported(
            "CMYK ingest is the moxcms print lane; this backend is RGB-only".into(),
        ))
    }
}
