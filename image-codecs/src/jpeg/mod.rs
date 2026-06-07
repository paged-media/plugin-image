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

//! JPEG codec adapter (spec §10.3) — decode via **zune-jpeg** (baseline
//! + progressive, pure-Rust), encode via **jpeg-encoder**.
//!
//! Scope, M0:
//!  - **Depth: 8-bit only.** JPEG is an 8-bit-sample format here; there
//!    is no 12-bit lane.
//!  - **Channel mapping is spec-verbatim (§5.1).** zune reports the
//!    *input* colorspace (after honouring the Adobe APP14 transform
//!    marker, §10.3): `Luma → Gray`; `YCbCr`/`RGB → Rgba` (the 3-channel
//!    layouts are not in the `ChannelLayout` set, so we widen to RGBA
//!    with alpha synthesised at 255 at the slice boundary and record the
//!    native truth in `native_format`); `CMYK`/`YCCK → Cmyk`.
//!  - **CMYK / the Adobe inversion rule.** Adobe-authored CMYK and YCCK
//!    JPEGs (APP14 marker present, transform 0 = CMYK, 2 = YCCK) store
//!    *inverted* ink samples: the stored byte is `255 - ink` (so a
//!    full-ink channel is stored as 0). This is the de-facto JFIF/Adobe
//!    convention — Photoshop writes APP14 and inverts; the JPEG bitstream
//!    itself has no colour-model field. We therefore re-invert
//!    (`ink = 255 - stored`) on decode **whenever the APP14 marker is
//!    present**, delivering true CMYK ink amounts. A 4-component JPEG
//!    *without* APP14 is taken as already-non-inverted CMYK (no APP14 →
//!    no inversion). YCCK is first taken to inverted-CMY via the inverse
//!    JFIF YCbCr→RGB matrix (CMY' = RGB), then the same APP14 inversion
//!    applies. Documented in `decode.rs::cmyk_from_*`.
//!  - **ICC** is read from the APP2 `ICC_PROFILE` markers — zune-jpeg's
//!    `icc_profile()` concatenates the sequence-numbered chunks for us
//!    (no M1.5 follow-up needed for the read path).
//!  - **native_shrink == [1].** zune-jpeg exposes no public DCT-scaled
//!    (1/2, 1/4, 1/8) decode entry point, so the decoder advertises no
//!    native downscale; the shrink-on-load planner (§7.2) must resample
//!    post-decode for JPEG (the consequence: a "shrink to 1/8" request
//!    still pays a full-resolution IDCT here — the DCT-scaled fast path
//!    is a follow-up gated on a zune-jpeg API, not available at 0.5.15).
//!  - Decode is **whole-image-then-window** (like PNG): the first
//!    `read_region` decodes the full frame, caches it in the spec
//!    layout, and serves windows by copy.
//!
//! Encode (`JpegTarget`, jpeg-encoder):
//!  - **RGB + Gray only** (`Rgba` strips have their alpha stripped —
//!    JPEG has no alpha; `Gray` encodes as Luma). CMYK *encode* is out
//!    of M0 scope (jpeg-encoder supports it, but we have no CMYK strip
//!    producer to exercise it; left as a follow-up).
//!  - **Quality** is a `JpegTarget::new(quality)` constructor parameter —
//!    the frozen `TargetInfo` carries no codec options, so per-codec knobs
//!    live on the adapter constructor.
//!  - **Chroma subsampling defaults to 4:2:0** (`R_4_2_0`).

mod decode;
mod encode;

pub use decode::JpegSource;
pub use encode::JpegTarget;

use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, SampleDepth, Transfer,
    TransferCurve,
};

pub(crate) const JPEG: &str = "jpeg";

/// The fixed transfer/space/alpha JPEG decodes into at M0: JPEG carries
/// no colour-management contract in the baseline bitstream, so — like the
/// PNG-without-gamma ruling (§10.3) — we map to sRGB-encoded, straight
/// alpha. Only `channels` and (for CMYK) the meaning of the samples vary
/// by the file's colour transform; depth is always U8.
///
/// For `Cmyk`, `AlphaMode::Straight` with opaque-everywhere semantics is
/// a placeholder: CMYK has no alpha channel, the colorant amounts are the
/// payload. The CMS lane (image-cms, M1) is what turns these device-CMYK
/// inks into the working space; M0 only carries them faithfully.
pub(crate) const fn jpeg_format(channels: ChannelLayout) -> PixelFormat {
    PixelFormat {
        channels,
        depth: SampleDepth::U8,
        alpha: AlphaMode::Straight,
        transfer: Transfer::Gamma(TransferCurve::Srgb),
        space: ColorSpaceRef::Named(NamedSpace::Srgb),
    }
}
