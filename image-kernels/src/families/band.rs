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

//! band family (T0, spec §11) — channel plumbing: identity passthrough,
//! single-channel extraction, alpha override, alpha broadcast. Every op
//! is a pure rearrangement of the input texel's f16 lanes (plus the one
//! exact-in-f16 literal `1.0`), so all four are bit-exact gpu↔ref.
//!
//! Provenance: elementary pointwise algebra / no reference reading.

use crate::{KernelClass, KernelDef, Tolerance};

kernel_family! {
    /// out = a (the cache-identity op — materialize a node's pixels).
    static BAND_COPY, params BandCopyParams, ref band_copy {
        id: "band.copy",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| a,
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = one channel splayed across rgb, opaque alpha (band → gray).
    static BAND_EXTRACT, params BandExtractParams, ref band_extract {
        id: "band.extract",
        class: KernelClass::Point,
        inputs: 1,
        params: { channel: u32 },
        eval: |a, b, p| pack4(a[p.channel], a[p.channel], a[p.channel], 1.0),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = rgb of a with alpha replaced by the param.
    static BAND_SET_ALPHA, params BandSetAlphaParams, ref band_set_alpha {
        id: "band.set_alpha",
        class: KernelClass::Point,
        inputs: 1,
        params: { alpha: f32 },
        eval: |a, b, p| pack4(a[0], a[1], a[2], p.alpha),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = alpha broadcast to all four channels (alpha → coverage map).
    static BAND_BROADCAST_ALPHA, params BandBroadcastAlphaParams, ref band_broadcast_alpha {
        id: "band.broadcast_alpha",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| splat4(a[3]),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

pub static FAMILY: &[&KernelDef] = &[
    &BAND_COPY,
    &BAND_EXTRACT,
    &BAND_SET_ALPHA,
    &BAND_BROADCAST_ALPHA,
];
