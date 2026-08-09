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

//! The Noise-family rank kernels (`rank.despeckle`, `rank.dust_scratches`)
//! — registration, ABI conformance, and the ALGORITHMS behind their
//! identity claims.
//!
//! Two kinds of test live here, and the difference matters:
//!
//! 1. STRUCTURAL — the kernel is in `morph::FAMILY`, its params block is
//!    the size the layout declares, its assembled WGSL validates under
//!    naga, and it declares exactly the ABI v1.1 binding interface plus
//!    the mandatory mask epilogue. These test the shipped artifact.
//! 2. MODEL — a scalar re-statement of the shader's selection logic,
//!    used to prove the claims the doc comments make (count-below really
//!    selects the median; the despeckle gate really keeps an edge and
//!    really kills an impulse; the threshold-1 case really is identity).
//!    These test the ALGORITHM, not the WGSL: the model is written here
//!    by hand and cannot catch a divergence in the shader text. The real
//!    GPU↔reference parity lane is `image-conformance` (which owns the
//!    scalar twins) — until a `family_rank_noise.rs` lands there, these
//!    model tests are the argument that the design is correct, not
//!    evidence that the GPU agrees with it.

use image_kernels::families::morph::{
    RankDespeckleParams, RankDustScratchesParams, DUST_SCRATCHES_MAX_RADIUS, FAMILY,
    RANK_DESPECKLE, RANK_DUST_SCRATCHES,
};
use image_kernels::{abi, KernelClass, KernelDef, Tolerance};

// ───────────────────────────── structural ───────────────────────────

#[test]
fn rank_noise_kernels_are_registered_in_family() {
    let ids: Vec<&str> = FAMILY.iter().map(|d| d.id).collect();
    assert!(
        ids.contains(&"rank.despeckle"),
        "rank.despeckle must be in morph::FAMILY (families/mod.rs feeds \
         all_defined() from the family lists): {ids:?}"
    );
    assert!(
        ids.contains(&"rank.dust_scratches"),
        "rank.dust_scratches must be in morph::FAMILY: {ids:?}"
    );

    // Despeckle is the fixed 3×3 form; dust & scratches carries the
    // runtime radius and so declares the MAX as its ROI bound.
    assert_eq!(RANK_DESPECKLE.inputs, 1);
    assert_eq!(RANK_DUST_SCRATCHES.inputs, 1);
    assert!(RANK_DESPECKLE.module, "handwritten ABI v1.1 module");
    assert!(RANK_DUST_SCRATCHES.module, "handwritten ABI v1.1 module");
    assert_eq!(
        RANK_DESPECKLE.class,
        KernelClass::Windowed { radius: (1, 1) }
    );
    assert_eq!(
        RANK_DUST_SCRATCHES.class,
        KernelClass::Windowed {
            radius: (DUST_SCRATCHES_MAX_RADIUS, DUST_SCRATCHES_MAX_RADIUS)
        }
    );
    // A rank filter never commutes with mip downsampling (§8.3).
    assert!(!RANK_DESPECKLE.mip_exact);
    assert!(!RANK_DUST_SCRATCHES.mip_exact);
    // Dust & scratches only ever writes back an existing sample; the
    // despeckle `amount` lerp is the family's one inexact step.
    assert_eq!(RANK_DUST_SCRATCHES.gpu_tolerance, Tolerance::Exact);
    assert_eq!(
        RANK_DESPECKLE.gpu_tolerance,
        Tolerance::ChannelEpsF16(2),
        "the lerp is the only inexact step — if this widens, say why"
    );
}

#[test]
fn rank_noise_params_layouts_match_the_rust_blocks() {
    // `min_binding_size` on the uniform binding is built from
    // `params.size`, so a drift here is a runtime validation error, not
    // a compile error.
    assert_eq!(
        RANK_DESPECKLE.params.size,
        std::mem::size_of::<RankDespeckleParams>()
    );
    assert_eq!(
        RANK_DESPECKLE.params.size, 12,
        "edge_threshold + amount + _abi_pad"
    );
    assert_eq!(
        RANK_DUST_SCRATCHES.params.size,
        std::mem::size_of::<RankDustScratchesParams>()
    );
    assert_eq!(
        RANK_DUST_SCRATCHES.params.size, 12,
        "radius + threshold + _abi_pad"
    );

    // The handwritten WGSL `struct Params` must name the same fields in
    // the same order as the repr(C) block (ABI v1.1 module contract);
    // `assemble` does not generate it for module kernels, so nothing
    // else checks this.
    for def in [&RANK_DESPECKLE, &RANK_DUST_SCRATCHES] {
        let mut cursor = def
            .wgsl
            .find("struct Params {")
            .unwrap_or_else(|| panic!("{}: no `struct Params`", def.id));
        for f in def.params.fields {
            let decl = format!("{}: {}", f.name, f.wgsl_ty);
            let at = def.wgsl[cursor..]
                .find(&decl)
                .unwrap_or_else(|| panic!("{}: `{decl}` missing or out of order", def.id));
            cursor += at + decl.len();
        }
        assert!(
            def.wgsl[cursor..].starts_with(",\n    _abi_pad: u32,\n}"),
            "{}: the params block must end with the trailing ABI pad",
            def.id
        );
    }
}

#[test]
fn rank_noise_modules_naga_validate() {
    for def in [&RANK_DESPECKLE, &RANK_DUST_SCRATCHES] {
        let src = abi::assemble(def);
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{}: WGSL parse failed: {e}\n{src}", def.id));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}: WGSL validation failed: {e:?}\n{src}", def.id));
    }
}

#[test]
fn rank_noise_modules_declare_the_abi_binding_interface() {
    for def in [&RANK_DESPECKLE, &RANK_DUST_SCRATCHES] {
        let src = abi::assemble(def);
        for needle in [
            "@group(0) @binding(0) var in0 : texture_2d<f32>;",
            "@group(1) @binding(0) var<uniform> params : Params;",
            "@group(2) @binding(0) var mask : texture_2d<f32>;",
            "@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;",
            "@compute @workgroup_size(16, 16, 1)",
            "if (gid.x >= dims.x || gid.y >= dims.y) { return; }",
            // The MANDATORY mask epilogue — a windowed module applies
            // the selection mask itself, against the window CENTER.
            "let m = textureLoad(mask, xy, 0).r;",
            "textureStore(outp, xy, mix(center, result, vec4<f32>(m)));",
        ] {
            assert!(
                src.contains(needle),
                "{}: ABI v1.1 requires `{needle}`\n{src}",
                def.id
            );
        }
    }
}

#[test]
fn rank_dust_scratches_identity_is_structural_not_incidental() {
    // The identity at threshold 1.0 rests on TWO textual facts, either
    // of which a well-meaning edit could quietly break:
    //   * the comparison is STRICT (`>`), so |c − m| == 1 does NOT fire;
    //   * the radius is CLAMPED to the ROI bound, so an over-large
    //     radius degrades instead of reading outside its window.
    let src = RANK_DUST_SCRATCHES.wgsl;
    assert!(
        src.contains("let outlier = abs(center - med) > vec4<f32>(t);"),
        "the substitution gate must stay a STRICT `>` — `>=` would make \
         threshold 1.0 replace every centre that is exactly 1 away"
    );
    assert!(
        src.contains("let r = i32(min(params.radius, u32(R_MAX)));"),
        "radius must be clamped to the ROI bound, never trusted"
    );
    assert!(
        src.contains(&format!("const R_MAX : i32 = {DUST_SCRATCHES_MAX_RADIUS};")),
        "the shader bound and DUST_SCRATCHES_MAX_RADIUS must agree — the \
         engine plans the ROI from the KernelDef, the shader indexes with \
         its own constant"
    );
    // The window center offset IS the class radius (windowed convention).
    assert!(src.contains("let c = xy + vec2<i32>(R_MAX, R_MAX);"));

    // And the params a caller must send for that identity.
    let p = RankDustScratchesParams::new(1, 1.0);
    assert_eq!(p.as_bytes().len(), RANK_DUST_SCRATCHES.params.size);
    assert_eq!(
        p.as_bytes(),
        &[1, 0, 0, 0, 0x00, 0x00, 0x80, 0x3f, 0, 0, 0, 0][..],
        "radius=1u, threshold=1.0f32 (0x3f800000 LE), pad=0"
    );
    // radius = 0 is the second identity: a one-tap median is the centre.
    assert_eq!(
        RankDustScratchesParams::new(0, 0.0).as_bytes(),
        &[0u8; 12][..]
    );
}

#[test]
fn rank_despeckle_identity_is_the_zero_amount_lerp() {
    let src = RANK_DESPECKLE.wgsl;
    assert!(
        src.contains("let smoothed = mix(center, med, clamp(params.amount, 0.0, 1.0));"),
        "amount must reach the output ONLY through a lerp from the \
         centre — that is what makes amount = 0 an exact identity"
    );
    // amount = 0 ⇒ mix(center, med, 0) == center exactly, so the ABI
    // epilogue mixes center with center and the texel is untouched for
    // ANY mask value. Encode that identity call site.
    let p = RankDespeckleParams::new(0.25, 0.0);
    assert_eq!(p.as_bytes().len(), RANK_DESPECKLE.params.size);
    assert_eq!(
        p.as_bytes(),
        &[0x00, 0x00, 0x80, 0x3e, 0, 0, 0, 0, 0, 0, 0, 0][..],
        "edge_threshold=0.25f32 (0x3e800000 LE), amount=0.0, pad=0"
    );
    // The lerp identity in arithmetic, not in prose: mix(a, b, 0) == a
    // bit-for-bit for every finite a, b.
    for (a, b) in [(0.0f32, 1.0f32), (0.375, 0.125), (1.0, 0.0), (0.7, 0.7)] {
        assert_eq!(a + 0.0 * (b - a), a);
    }
}

// ─────────────────────────────── model ──────────────────────────────

/// The shader's count-below selection, restated scalar-wise: for each
/// candidate, count taps below and at-or-below it; the median is the
/// candidate whose rank interval brackets ⌊n/2⌋.
fn count_below_median(win: &[f32]) -> f32 {
    let n = win.len();
    assert!(n % 2 == 1, "odd window only");
    let k = (n / 2) as f32;
    // Seeded with the centre, exactly like the shader — the defined
    // fallback when no candidate qualifies (a NaN window).
    let mut med = win[n / 2];
    for &cand in win {
        let mut lt = 0.0f32;
        let mut le = 0.0f32;
        for &s in win {
            if s < cand {
                lt += 1.0;
            }
            if s <= cand {
                le += 1.0;
            }
        }
        if lt <= k && le > k {
            med = cand;
        }
    }
    med
}

/// A tiny deterministic PRNG — no `rand` dependency, and the same
/// stream every run so a failure is reproducible.
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

#[test]
fn rank_dust_scratches_count_below_selects_the_true_median() {
    let mut st = 0x5eed_1234u32;
    for r in 0..=(DUST_SCRATCHES_MAX_RADIUS as usize) {
        let n = (2 * r + 1) * (2 * r + 1);
        for trial in 0..64 {
            // Quantize hard on some trials so DUPLICATES (the case the
            // rank bracket exists to handle) are common, not rare.
            let levels = if trial % 2 == 0 { 3u32 } else { 256u32 };
            let win: Vec<f32> = (0..n)
                .map(|_| (lcg(&mut st) % levels) as f32 / (levels - 1) as f32)
                .collect();
            let mut sorted = win.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(
                count_below_median(&win),
                sorted[n / 2],
                "r={r} trial={trial} window={win:?}"
            );
        }
    }
}

/// The shader's substitution rule.
fn dust_scratches_model(win: &[f32], threshold: f32) -> f32 {
    let center = win[win.len() / 2];
    let med = count_below_median(win);
    let t = threshold.max(0.0);
    if (center - med).abs() > t {
        med
    } else {
        center
    }
}

#[test]
fn rank_dust_scratches_threshold_one_is_identity_and_zero_is_a_median() {
    let mut st = 0xd057_c001u32;
    for r in 1..=(DUST_SCRATCHES_MAX_RADIUS as usize) {
        let n = (2 * r + 1) * (2 * r + 1);
        for _ in 0..64 {
            let win: Vec<f32> = (0..n)
                .map(|_| (lcg(&mut st) % 256) as f32 / 255.0)
                .collect();
            let center = win[n / 2];
            let mut sorted = win.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // IDENTITY: |c − m| ≤ 1 for [0, 1] data, and the gate is
            // strict, so nothing is ever substituted.
            assert_eq!(
                dust_scratches_model(&win, 1.0),
                center,
                "threshold 1.0 must be the identity for [0, 1] data"
            );
            // The extreme case of that bound: a full-range difference.
            let mut extreme = vec![1.0f32; n];
            extreme[n / 2] = 0.0;
            assert_eq!(dust_scratches_model(&extreme, 1.0), 0.0);
            // ...which threshold 0 does substitute.
            assert_eq!(dust_scratches_model(&extreme, 0.0), 1.0);

            // threshold 0 ⇒ plain median.
            assert_eq!(dust_scratches_model(&win, 0.0), sorted[n / 2]);
        }
    }
    // radius 0: the one-tap window's median is the centre, so every
    // threshold is an identity.
    assert_eq!(dust_scratches_model(&[0.42], 0.0), 0.42);
}

/// The shader's despeckle rule: sort the 9 taps, gate on the INNER
/// range (rank 8 − rank 2), substitute the median through `amount`.
fn despeckle_model(win: &[f32; 9], edge_threshold: f32, amount: f32) -> f32 {
    let mut s = *win;
    // The same data-oblivious bubble schedule the shader runs.
    for i in 0..8 {
        for j in 0..(8 - i) {
            let (lo, hi) = (s[j].min(s[j + 1]), s[j].max(s[j + 1]));
            s[j] = lo;
            s[j + 1] = hi;
        }
    }
    let center = win[4];
    let spread = s[7] - s[1];
    if spread <= edge_threshold.max(0.0) {
        // WGSL defines `mix(a, b, t)` as `a*(1 - t) + b*t`, which is
        // EXACT at both ends — `a` at t = 0, `b` at t = 1. Written the
        // other common way (`a + t*(b - a)`) the t = 1 end is off by an
        // ULP, which is precisely the rounding `ChannelEpsF16(2)` is
        // declared for; the t = 0 IDENTITY is exact under either form,
        // so it does not depend on which one a driver picks.
        let t = amount.clamp(0.0, 1.0);
        center * (1.0 - t) + s[4] * t
    } else {
        center
    }
}

#[test]
fn rank_despeckle_kills_an_impulse_but_keeps_an_edge() {
    // An isolated speckle on a flat field: it is an EXTREME of its
    // window, so the inner range stays 0 and the gate opens even at
    // edge_threshold 0.
    let speckle = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(despeckle_model(&speckle, 0.0, 1.0), 0.0, "speckle removed");

    // The corner of a bright square — the classic case a plain median
    // ROUNDS OFF. Six samples on one side, three on the other, so the
    // inner range is the full step and the gate stays shut.
    let corner = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let mut sorted = corner;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted[4], 0.0, "a plain median would erase this corner");
    assert_eq!(
        despeckle_model(&corner, 0.5, 1.0),
        1.0,
        "despeckle must NOT soften an edge/corner"
    );

    // A step edge straight through the window: same verdict.
    let edge = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    assert_eq!(despeckle_model(&edge, 0.5, 1.0), edge[4]);

    // Noise-scale variation on a flat field: the gate opens once the
    // threshold covers the wobble, and the median smooths it.
    let noisy = [0.50, 0.52, 0.49, 0.51, 0.60, 0.48, 0.52, 0.50, 0.51];
    assert_ne!(despeckle_model(&noisy, 0.05, 1.0), noisy[4]);

    // IDENTITY at amount = 0, in every one of those regimes.
    for w in [speckle, corner, edge, noisy] {
        for t in [0.0, 0.05, 0.5, 2.0] {
            assert_eq!(despeckle_model(&w, t, 0.0), w[4]);
        }
    }
}

#[test]
fn rank_despeckle_large_threshold_degrades_to_a_plain_median() {
    // The documented far end of the parameter range: a threshold at or
    // above the data range opens the gate everywhere.
    let mut st = 0xbeef_0042u32;
    for _ in 0..128 {
        let mut w = [0.0f32; 9];
        for v in w.iter_mut() {
            *v = (lcg(&mut st) % 256) as f32 / 255.0;
        }
        let mut sorted = w;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Approximate ON PURPOSE: at amount = 1 the output still goes
        // through the lerp, so "is the median" is a tolerance claim
        // (ChannelEpsF16(2)), not a bit-exact one — unlike
        // `rank.median3`, which is `Tolerance::Exact` because nothing
        // arithmetic touches the selected sample. f16 has ~2^-11
        // relative precision, so 1e-6 is far inside the shipped bound.
        assert!(
            (despeckle_model(&w, 1.0, 1.0) - sorted[4]).abs() < 1e-6,
            "amount 1 + threshold above the data range should be a median"
        );
    }
}

/// Sanity: the two new kernels are the only rank/morph entries the
/// family gained, and every one of them is a windowed module (so the
/// engine inflates an ROI for it).
#[test]
fn rank_noise_family_stays_windowed() {
    let defs: Vec<&KernelDef> = FAMILY.to_vec();
    assert_eq!(defs.len(), 5, "dilate, erode, median3, despeckle, dust");
    for d in defs {
        assert!(
            matches!(d.class, KernelClass::Windowed { .. }),
            "{}: the morph/rank family is windowed by definition",
            d.id
        );
        assert!(d.module, "{}: handwritten ABI v1.1 module", d.id);
    }
}
