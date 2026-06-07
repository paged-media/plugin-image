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

//! PNG codec adapter (spec §10.3) — the D-4 winner is **zune-png**
//! (decode + encode, pure-Rust, no_std-capable; fastest decode and
//! competitive encode in the synthetic sweep, registry/codecs.yaml).
//!
//! Scope, M0:
//!  - 8-bit only. PNG carries no embedded depth conversion contract we
//!    want to fake here; 16-bit input errors `Unsupported` (the M1
//!    `native_format "…16"` lane is where U16 lands).
//!  - Channel mapping is spec-verbatim (§5.1): PNG Gray→`Gray`,
//!    GrayA→`GrayA`, RGB→`Rgba` (alpha synthesised at 255 — RGB is not
//!    in the `ChannelLayout` set, so we widen at the slice boundary and
//!    record the native truth in `SourceInfo.native_format`), RGBA→`Rgba`
//!    straight. Palette images decode to RGB/RGBA upstream in zune.
//!  - Color: PNG without a cICP/iCCP/sRGB chunk is treated as sRGB —
//!    `Transfer::Gamma(Srgb)`, `space Srgb`, `AlphaMode::Straight`. A
//!    gAMA/cHRM-only file is still mapped to sRGB at M0 (the CMS lane
//!    that honours arbitrary gamma is image-cms, M1).
//!  - M0 decode is **full-image-then-window**: `read_region` decodes the
//!    whole frame once, caches it, and serves windows by copy. Real
//!    streaming (zune has no row-incremental public API; the M1 plan is
//!    a chunked/interlaced reader) is BREAKAGE-tracked. PNG advertises
//!    `native_shrink == [1]` — no decoder-side downscale.

mod decode;
mod encode;

pub use decode::PngSource;
pub use encode::PngTarget;

use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, SampleDepth, Transfer,
    TransferCurve,
};

pub(crate) const PNG: &str = "png";

/// The fixed transfer/space/alpha PNG decodes into at M0: sRGB-encoded,
/// straight alpha, per the §10.3 ruling for PNG-without-gamma-info. Only
/// `channels` varies by the file's color type; depth is always U8 here.
pub(crate) const fn png_format(channels: ChannelLayout) -> PixelFormat {
    PixelFormat {
        channels,
        depth: SampleDepth::U8,
        alpha: AlphaMode::Straight,
        transfer: Transfer::Gamma(TransferCurve::Srgb),
        space: ColorSpaceRef::Named(NamedSpace::Srgb),
    }
}
