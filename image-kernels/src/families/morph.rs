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

//! Morphology + rank filters (T3, spec §11) — handwritten WGSL modules
//! under the ABI v1.1 contract (`abi::assemble` docs). All three are
//! 3×3-neighbourhood windowed kernels (`radius (1, 1)`), following the
//! `conv.box` windowed convention: output texel `(x, y)` maps to window
//! center `(x + 1, y + 1)` in `in0` coords, and the module applies the
//! ABI selection mask itself as `mix(center, result, m)`.
//!
//! - `morph.dilate`: per-channel MAX over the 3×3 window.
//! - `morph.erode`:  per-channel MIN over the 3×3 window.
//! - `rank.median3`: per-channel MEDIAN of the 9 window samples, via a
//!   fixed comparator network (see [`MEDIAN3_WGSL`] docs).
//!
//! All three are EXACT (`Tolerance::Exact`): max/min select an existing
//! f16 window value, and the median network is built purely from
//! componentwise `min`/`max` of existing samples — so every output texel
//! is one of the input f16 values reproduced bit-for-bit. None are
//! `mip_exact`: neighbourhood max/min/median do not commute with mip
//! downsampling (a max over a box of averages ≠ the average of a box of
//! maxes), so the engine recomputes per level rather than scaling params.
//!
//! 3×3 ONLY (M0): larger radii are a params-driven follow-up — a
//! `radius` uniform plus a dynamic window loop (dilate/erode generalize
//! trivially; a larger rank filter needs a histogram/selection method,
//! not a fixed network). The `KernelClass::Windowed` radius here is the
//! fixed `(1, 1)` ROI bound.
//!
//! Provenance: mathematical morphology (dilation/erosion as per-channel
//! sup/inf over a flat structuring element) and rank filtering (the
//! median as the order-statistic at rank 5 of 9) are standard textbook
//! material; the median selection network is a standard 19-comparator
//! median-of-9 network. No reference reading.

use crate::{KernelClass, KernelDef, ParamsLayout, Tolerance};

/// Bare ABI-pad params — morphology/median3 take no parameters in the
/// 3×3 form. (A `radius` field arrives with the larger-window follow-up.)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct MorphParams {
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl MorphParams {
    pub fn new() -> Self {
        Self { _abi_pad: 0 }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

// ───────────────────────────── dilate ──────────────────────────────

/// out = per-channel MAX of the 3×3 window (radius (1, 1)); mask-mixed
/// against the window center per the windowed convention. Exact: `max`
/// selects an existing f16 sample.
pub static MORPH_DILATE: KernelDef = KernelDef {
    id: "morph.dilate",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<MorphParams>(),
        fields: &[],
    },
    wgsl: MORPH_DILATE_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::Exact,
};

// Reduction order: dy outer ascending, dx inner ascending (the
// `conv.box` convention); the scalar reference folds the 9 samples with
// `max` in that exact order. `max` is associative/commutative over real
// values, but the order is fixed anyway to keep the two lanes
// byte-identical by construction (§6.3).
const MORPH_DILATE_WGSL: &str = "\
// paged.image kernel `morph.dilate` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // Window center: output (x, y) maps to in0 (x + 1, y + 1).
    let c = xy + vec2<i32>(1, 1);
    var acc = textureLoad(in0, c + vec2<i32>(-1, -1), 0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            acc = max(acc, textureLoad(in0, c + vec2<i32>(dx, dy), 0));
        }
    }
    let center = textureLoad(in0, c, 0);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(center, acc, vec4<f32>(m)));
}
";

// ────────────────────────────── erode ──────────────────────────────

/// out = per-channel MIN of the 3×3 window (radius (1, 1)); mask-mixed
/// against the window center. Exact: `min` selects an existing f16
/// sample.
pub static MORPH_ERODE: KernelDef = KernelDef {
    id: "morph.erode",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<MorphParams>(),
        fields: &[],
    },
    wgsl: MORPH_ERODE_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::Exact,
};

const MORPH_ERODE_WGSL: &str = "\
// paged.image kernel `morph.erode` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // Window center: output (x, y) maps to in0 (x + 1, y + 1).
    let c = xy + vec2<i32>(1, 1);
    var acc = textureLoad(in0, c + vec2<i32>(-1, -1), 0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            acc = min(acc, textureLoad(in0, c + vec2<i32>(dx, dy), 0));
        }
    }
    let center = textureLoad(in0, c, 0);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(center, acc, vec4<f32>(m)));
}
";

// ───────────────────────────── median3 ─────────────────────────────
//
// Per-channel median of the 9 samples of the 3×3 window. The 9 window
// taps are loaded into s0..s8 in raster order (dy outer ascending, dx
// inner ascending), i.e.:
//
//     s0 s1 s2        (dy = -1: dx = -1, 0, +1)
//     s3 s4 s5        (dy =  0)
//     s6 s7 s8        (dy = +1)
//
// The median (the rank-5 order statistic of 9) is selected by a FIXED
// 19-comparator selection network — the standard median-of-9 network
// that produces only the median (not a full sort). Each comparator is a
// componentwise compare-exchange `op2(a, b)` that replaces (a, b) with
// (min(a, b), max(a, b)); applied per channel via the vec4 builtins.
// Because the network is built purely from `min`/`max` of existing
// samples, the result is bit-for-bit one of the input f16 values — hence
// `Tolerance::Exact`. The Rust reference mirrors this network step for
// step, in the same order, so both lanes select the same sample (§6.3).
//
// Network (s4 holds the median after the final step):
//   op2(s1,s2) op2(s4,s5) op2(s7,s8)
//   op2(s0,s1) op2(s3,s4) op2(s6,s7)
//   op2(s1,s2) op2(s4,s5) op2(s7,s8)
//   op2(s0,s3) op2(s5,s8) op2(s4,s7)
//   op2(s3,s6) op2(s1,s4) op2(s2,s5)
//   op2(s4,s7) op2(s4,s2) op2(s6,s4)
//   op2(s4,s2)
//   median = s4

/// out = per-channel median of the 9 samples of the 3×3 window (radius
/// (1, 1)); mask-mixed against the window center. Exact: the selection
/// network is pure `min`/`max` of existing samples, so the median is one
/// of the input f16 values reproduced bit-for-bit.
pub static RANK_MEDIAN3: KernelDef = KernelDef {
    id: "rank.median3",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<MorphParams>(),
        fields: &[],
    },
    wgsl: MEDIAN3_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::Exact,
};

const MEDIAN3_WGSL: &str = "\
// paged.image kernel `rank.median3` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outp);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    // Window center: output (x, y) maps to in0 (x + 1, y + 1).
    let c = xy + vec2<i32>(1, 1);
    // 9 taps in raster order (dy outer asc, dx inner asc).
    var s0 = textureLoad(in0, c + vec2<i32>(-1, -1), 0);
    var s1 = textureLoad(in0, c + vec2<i32>( 0, -1), 0);
    var s2 = textureLoad(in0, c + vec2<i32>( 1, -1), 0);
    var s3 = textureLoad(in0, c + vec2<i32>(-1,  0), 0);
    var s4 = textureLoad(in0, c + vec2<i32>( 0,  0), 0);
    var s5 = textureLoad(in0, c + vec2<i32>( 1,  0), 0);
    var s6 = textureLoad(in0, c + vec2<i32>(-1,  1), 0);
    var s7 = textureLoad(in0, c + vec2<i32>( 0,  1), 0);
    var s8 = textureLoad(in0, c + vec2<i32>( 1,  1), 0);
    // 19-comparator median-of-9 selection network. Each line is a
    // componentwise compare-exchange (lo := min, hi := max).
    var lo : vec4<f32>;
    var hi : vec4<f32>;
    lo = min(s1, s2); hi = max(s1, s2); s1 = lo; s2 = hi;
    lo = min(s4, s5); hi = max(s4, s5); s4 = lo; s5 = hi;
    lo = min(s7, s8); hi = max(s7, s8); s7 = lo; s8 = hi;
    lo = min(s0, s1); hi = max(s0, s1); s0 = lo; s1 = hi;
    lo = min(s3, s4); hi = max(s3, s4); s3 = lo; s4 = hi;
    lo = min(s6, s7); hi = max(s6, s7); s6 = lo; s7 = hi;
    lo = min(s1, s2); hi = max(s1, s2); s1 = lo; s2 = hi;
    lo = min(s4, s5); hi = max(s4, s5); s4 = lo; s5 = hi;
    lo = min(s7, s8); hi = max(s7, s8); s7 = lo; s8 = hi;
    lo = min(s0, s3); hi = max(s0, s3); s0 = lo; s3 = hi;
    lo = min(s5, s8); hi = max(s5, s8); s5 = lo; s8 = hi;
    lo = min(s4, s7); hi = max(s4, s7); s4 = lo; s7 = hi;
    lo = min(s3, s6); hi = max(s3, s6); s3 = lo; s6 = hi;
    lo = min(s1, s4); hi = max(s1, s4); s1 = lo; s4 = hi;
    lo = min(s2, s5); hi = max(s2, s5); s2 = lo; s5 = hi;
    lo = min(s4, s7); hi = max(s4, s7); s4 = lo; s7 = hi;
    lo = min(s4, s2); hi = max(s4, s2); s4 = lo; s2 = hi;
    lo = min(s6, s4); hi = max(s6, s4); s6 = lo; s4 = hi;
    lo = min(s4, s2); hi = max(s4, s2); s4 = lo; s2 = hi;
    let result = s4;
    let center = textureLoad(in0, c, 0);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(center, result, vec4<f32>(m)));
}
";

pub static FAMILY: &[&KernelDef] = &[&MORPH_DILATE, &MORPH_ERODE, &RANK_MEDIAN3];
