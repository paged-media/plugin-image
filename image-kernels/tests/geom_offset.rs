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

//! `geom.offset` — the Move tool's kernel (and Photoshop's Offset
//! filter at `edge == Wrap`). These are the DEFINITION-level gates that
//! do not need a GPU: the ABI contract, the param encoding, the family
//! registration, the edge-policy encoding, and a scalar model of the
//! shader that pins the behavioural claims made in its doc-comment.
//!
//! feat: geom.offset (registry/kernels.yaml — row PENDING, orchestrator
//! wiring; until it lands `image_kernels::lookup("geom.offset")` is
//! `None` and the crate's `registry_and_definitions_agree` unit test
//! reports it as a code-defined kernel with no row).
//!
//! WHY A SCALAR MODEL LIVES IN THIS FILE. The gpu↔ref parity for this
//! family runs in `image-conformance/tests/family_geom.rs`, which this
//! agent does not own. The model below mirrors the WGSL term-for-term —
//! same backward map, same clamp/wrap/zero taps, same fixed
//! `mix(mix(p00,p10,fx), mix(p01,p11,fx), fy)` blend order — so the
//! behavioural assertions are meaningful now and the same reference
//! transplants into the conformance harness unchanged.

use image_kernels::families::geom::{EdgePolicy, OffsetParams, GEOM_OFFSET};
use image_kernels::families::{geom, ALL_FAMILIES};
use image_kernels::{KernelClass, Tolerance};

// ─────────────────── scalar model of the shader ───────────────────

type Px = [f32; 4];

/// `mix(a, b, t)` = `a*(1-t) + b*t`, WGSL's definition. At `t == 0.0`
/// this is bit-exactly `a` — the fact the identity condition rests on.
fn mix(a: Px, b: Px, t: f32) -> Px {
    std::array::from_fn(|i| a[i] * (1.0 - t) + b[i] * t)
}

/// One tap under the edge policy — mirrors the WGSL `tap()`, INCLUDING
/// its fallback: anything that is not `Wrap` or `Transparent` clamps.
fn tap(img: &[Px], w: i32, h: i32, p: (i32, i32), edge: u32) -> Px {
    match edge {
        2 => {
            let (dw, dh) = (w.max(1), h.max(1));
            let x = ((p.0 % dw) + dw) % dw;
            let y = ((p.1 % dh) + dh) % dh;
            img[(y * w + x) as usize]
        }
        0 => {
            if p.0 < 0 || p.1 < 0 || p.0 >= w || p.1 >= h {
                [0.0; 4]
            } else {
                img[(p.1 * w + p.0) as usize]
            }
        }
        _ => {
            let x = p.0.clamp(0, w - 1);
            let y = p.1.clamp(0, h - 1);
            img[(y * w + x) as usize]
        }
    }
}

/// The whole kernel, mask epilogue included. `mask` is the constant
/// selection coverage (the ABI's group-2 texture, `.r`).
fn offset_model(img: &[Px], w: i32, h: i32, p: &OffsetParams, mask: f32) -> Vec<Px> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let sx = x as f32 - p.dx;
            let sy = y as f32 - p.dy;
            let x0 = sx.floor();
            let y0 = sy.floor();
            let fx = sx - x0;
            let fy = sy - y0;
            let i0 = (x0 as i32, y0 as i32);

            let p00 = tap(img, w, h, i0, p.edge);
            let p10 = tap(img, w, h, (i0.0 + 1, i0.1), p.edge);
            let p01 = tap(img, w, h, (i0.0, i0.1 + 1), p.edge);
            let p11 = tap(img, w, h, (i0.0 + 1, i0.1 + 1), p.edge);

            let top = mix(p00, p10, fx);
            let bot = mix(p01, p11, fx);
            let result = mix(top, bot, fy);

            let a = img[(y * w + x) as usize];
            out.push(mix(a, result, mask));
        }
    }
    out
}

/// A labeled source: each texel encodes its own coordinate, so a
/// translation is legible per channel. Values are f16-exact at these
/// dims, as the rest of the family's fixtures are.
fn labeled(w: i32, h: i32) -> Vec<Px> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            [
                x as f32 / 256.0,
                y as f32 / 256.0,
                (x + y) as f32 / 512.0,
                1.0,
            ]
        })
        .collect()
}

const W: i32 = 9;
const H: i32 = 7;

// ───────────────────────── IDENTITY ────────────────────────────────

/// THE identity condition from the doc-comment: `dx == 0, dy == 0`
/// returns the input UNCHANGED — for every edge policy, and for every
/// mask value, bit-exactly (not within a tolerance).
#[test]
fn image_editor_geom_offset_zero_is_the_identity() {
    let img = labeled(W, H);
    for edge in [EdgePolicy::Transparent, EdgePolicy::Clamp, EdgePolicy::Wrap] {
        for mask in [0.0f32, 0.25, 0.5, 1.0] {
            let p = OffsetParams::identity(edge);
            assert!(p.is_identity());
            let out = offset_model(&img, W, H, &p, mask);
            assert_eq!(
                out, img,
                "dx=0,dy=0 must return the input unchanged (edge={edge:?}, mask={mask})"
            );
        }
    }
}

/// The identity survives the edge policies' out-of-range fallback too:
/// a zero offset with a policy value no shader branch recognises is
/// still the input.
#[test]
fn image_editor_geom_offset_zero_is_the_identity_under_a_bogus_policy() {
    let img = labeled(W, H);
    let p = OffsetParams {
        dx: 0.0,
        dy: 0.0,
        edge: 99,
        _abi_pad: 0,
    };
    assert_eq!(offset_model(&img, W, H, &p, 1.0), img);
}

// ──────────────────── translation behaviour ────────────────────────

/// A whole-pixel move is LOSSLESS: `fx == fy == 0.0` exactly, so the
/// blend collapses to a single tap and the moved texels are the source
/// texels bit-for-bit. This is the claim that lets a Move drag apply
/// repeatedly without softening the layer.
#[test]
fn image_editor_geom_offset_integer_move_is_bit_exact() {
    let img = labeled(W, H);
    let p = OffsetParams::new(3.0, -2.0, EdgePolicy::Clamp);
    let out = offset_model(&img, W, H, &p, 1.0);
    for y in 0..H {
        for x in 0..W {
            let sx = (x - 3).clamp(0, W - 1);
            let sy = (y + 2).clamp(0, H - 1);
            assert_eq!(
                out[(y * W + x) as usize],
                img[(sy * W + sx) as usize],
                "({x},{y}) must be the source texel verbatim"
            );
        }
    }
}

/// Round-tripping a whole-pixel move is the identity under `Wrap` —
/// nothing has left the buffer, so nothing may have been resampled.
#[test]
fn image_editor_geom_offset_integer_move_round_trips_under_wrap() {
    let img = labeled(W, H);
    let fwd = offset_model(
        &img,
        W,
        H,
        &OffsetParams::new(4.0, 3.0, EdgePolicy::Wrap),
        1.0,
    );
    let back = offset_model(
        &fwd,
        W,
        H,
        &OffsetParams::new(-4.0, -3.0, EdgePolicy::Wrap),
        1.0,
    );
    assert_eq!(back, img);
}

/// SUB-PIXEL IS BILINEAR, NOT ROUNDED. A half-pixel move must land
/// strictly BETWEEN the two neighbouring source texels — if it were
/// rounded it would equal one of them exactly, and the layer would
/// stutter under the pointer instead of tracking it.
#[test]
fn image_editor_geom_offset_subpixel_is_bilinear_not_rounded() {
    let img = labeled(W, H);
    let out = offset_model(
        &img,
        W,
        H,
        &OffsetParams::new(0.5, 0.0, EdgePolicy::Clamp),
        1.0,
    );
    for y in 0..H {
        for x in 2..W {
            let here = out[(y * W + x) as usize][0];
            let left = img[(y * W + x - 1) as usize][0];
            let right = img[(y * W + x) as usize][0];
            let expected = 0.5 * left + 0.5 * right;
            assert!(
                (here - expected).abs() < 1e-6,
                "({x},{y}) expected the 50/50 blend {expected}, got {here}"
            );
            assert!(
                here > left && here < right,
                "({x},{y}) a rounded move would equal a neighbour; \
                 got {here} against {left}/{right}"
            );
        }
    }
}

/// A quarter-pixel move weights the two taps 3:1 — the fraction is used,
/// not just its presence.
#[test]
fn image_editor_geom_offset_subpixel_weights_follow_the_fraction() {
    let img = labeled(W, H);
    let out = offset_model(
        &img,
        W,
        H,
        &OffsetParams::new(0.25, 0.0, EdgePolicy::Clamp),
        1.0,
    );
    let (x, y) = (5, 3);
    let expected = 0.25 * img[(y * W + x - 1) as usize][0] + 0.75 * img[(y * W + x) as usize][0];
    assert!((out[(y * W + x) as usize][0] - expected).abs() < 1e-6);
}

// ─────────────────────── edge policies ─────────────────────────────

/// `Transparent` vacates the uncovered band — a moved layer must expose
/// emptiness, and in the PREMULTIPLIED working space that is `vec4(0)`.
#[test]
fn image_editor_geom_offset_transparent_vacates_the_uncovered_band() {
    let img = labeled(W, H);
    let out = offset_model(
        &img,
        W,
        H,
        &OffsetParams::new(3.0, 0.0, EdgePolicy::Transparent),
        1.0,
    );
    for y in 0..H {
        for x in 0..3 {
            assert_eq!(
                out[(y * W + x) as usize],
                [0.0; 4],
                "({x},{y}) is outside the moved content and must be transparent"
            );
        }
        // …and the covered part is still the source, verbatim.
        for x in 3..W {
            assert_eq!(out[(y * W + x) as usize], img[(y * W + x - 3) as usize]);
        }
    }
}

/// `Clamp` extends the edge texel instead of vacating.
#[test]
fn image_editor_geom_offset_clamp_extends_the_edge() {
    let img = labeled(W, H);
    let out = offset_model(
        &img,
        W,
        H,
        &OffsetParams::new(3.0, 0.0, EdgePolicy::Clamp),
        1.0,
    );
    for y in 0..H {
        for x in 0..3 {
            assert_eq!(out[(y * W + x) as usize], img[(y * W) as usize]);
        }
    }
}

/// `Wrap` is the Offset FILTER: content leaving one side re-enters the
/// other, and a shift by exactly the source width is the identity.
#[test]
fn image_editor_geom_offset_wrap_is_periodic() {
    let img = labeled(W, H);
    let out = offset_model(
        &img,
        W,
        H,
        &OffsetParams::new(3.0, 0.0, EdgePolicy::Wrap),
        1.0,
    );
    for y in 0..H {
        for x in 0..W {
            let sx = ((x - 3) % W + W) % W;
            assert_eq!(out[(y * W + x) as usize], img[(y * W + sx) as usize]);
        }
    }

    let full = offset_model(
        &img,
        W,
        H,
        &OffsetParams::new(W as f32, H as f32, EdgePolicy::Wrap),
        1.0,
    );
    assert_eq!(full, img, "a shift by the full period is the identity");
}

/// The three policies must actually DIFFER where the content is not
/// covered — otherwise the param is decorative.
#[test]
fn image_editor_geom_offset_policies_are_distinguishable() {
    let img = labeled(W, H);
    let of = |e| offset_model(&img, W, H, &OffsetParams::new(3.0, 0.0, e), 1.0);
    let (t, c, w) = (
        of(EdgePolicy::Transparent),
        of(EdgePolicy::Clamp),
        of(EdgePolicy::Wrap),
    );
    assert_ne!(t, c);
    assert_ne!(c, w);
    assert_ne!(t, w);
}

// ─────────────── edge-policy encoding: reject / clamp ──────────────

/// The frozen wire encoding. These numbers cross the uniform block and
/// the wasm boundary; changing one silently re-maps every stored param
/// block.
#[test]
fn image_editor_geom_offset_edge_policy_encoding_is_frozen() {
    assert_eq!(EdgePolicy::Transparent.as_u32(), 0);
    assert_eq!(EdgePolicy::Clamp.as_u32(), 1);
    assert_eq!(EdgePolicy::Wrap.as_u32(), 2);
    for e in [EdgePolicy::Transparent, EdgePolicy::Clamp, EdgePolicy::Wrap] {
        assert_eq!(EdgePolicy::from_u32(e.as_u32()), Some(e));
    }
}

/// REJECTION: an out-of-range policy arriving from JS/wasm is refused,
/// not silently reinterpreted.
#[test]
fn image_editor_geom_offset_edge_policy_rejects_out_of_range() {
    for v in [3u32, 4, 7, 99, u32::MAX] {
        assert_eq!(
            EdgePolicy::from_u32(v),
            None,
            "{v} is not a policy and must not decode"
        );
    }
}

/// CLAMPING: the degrade-rather-than-fail decoder lands on `Clamp`, and
/// that choice matches the SHADER — an unrecognised value falls through
/// both `if`s into the clamp tap. Verified against the model, not just
/// asserted about the enum.
#[test]
fn image_editor_geom_offset_edge_policy_clamps_out_of_range() {
    for v in [3u32, 7, 99, u32::MAX] {
        assert_eq!(EdgePolicy::from_u32_or_clamp(v), EdgePolicy::Clamp);
    }
    for v in [0u32, 1, 2] {
        assert_eq!(
            EdgePolicy::from_u32_or_clamp(v),
            EdgePolicy::from_u32(v).unwrap()
        );
    }

    let img = labeled(W, H);
    let bogus = OffsetParams {
        dx: 3.0,
        dy: -2.0,
        edge: 99,
        _abi_pad: 0,
    };
    let clamped = OffsetParams::new(3.0, -2.0, EdgePolicy::Clamp);
    assert_eq!(
        offset_model(&img, W, H, &bogus, 1.0),
        offset_model(&img, W, H, &clamped, 1.0),
        "the shader's fallback branch is CLAMP; the Rust decoder must agree"
    );
}

// ───────────────────── params encoding / layout ────────────────────

/// The uniform block: 16 bytes, `dx | dy | edge | _abi_pad`, and the
/// `ParamsLayout` field list matches the struct field-for-field IN
/// ORDER (the WGSL `struct Params` is generated from nothing — it is
/// handwritten — so this is the only thing holding the two together
/// besides the module-text check below).
#[test]
fn image_editor_geom_offset_params_encoding() {
    assert_eq!(std::mem::size_of::<OffsetParams>(), 16);
    assert_eq!(GEOM_OFFSET.params.size, std::mem::size_of::<OffsetParams>());
    assert_eq!(GEOM_OFFSET.params.size % 16, 0, "16-aligned uniform block");

    let names: Vec<&str> = GEOM_OFFSET.params.fields.iter().map(|f| f.name).collect();
    let types: Vec<&str> = GEOM_OFFSET
        .params
        .fields
        .iter()
        .map(|f| f.wgsl_ty)
        .collect();
    assert_eq!(names, ["dx", "dy", "edge"]);
    assert_eq!(types, ["f32", "f32", "u32"]);
    // fields + the trailing `_abi_pad`, all 4-byte scalars.
    assert_eq!(GEOM_OFFSET.params.size, 4 * (names.len() + 1));

    let p = OffsetParams::new(1.5, -2.25, EdgePolicy::Wrap);
    let b = p.as_bytes();
    assert_eq!(b.len(), 16);
    assert_eq!(&b[0..4], &1.5f32.to_le_bytes());
    assert_eq!(&b[4..8], &(-2.25f32).to_le_bytes());
    assert_eq!(&b[8..12], &2u32.to_le_bytes());
    assert_eq!(&b[12..16], &0u32.to_le_bytes(), "_abi_pad is always 0");

    // Byte identity IS param identity (the op-cache key, spec §6.2).
    assert_ne!(
        OffsetParams::new(1.0, 0.0, EdgePolicy::Clamp).as_bytes(),
        OffsetParams::new(1.0, 0.0, EdgePolicy::Wrap).as_bytes(),
        "the edge policy must participate in the cache key"
    );
}

// ──────────────────── registration + ABI contract ──────────────────

/// Registered in `geom::FAMILY` — and therefore in `ALL_FAMILIES`,
/// which is what `all_defined()` and the registry-drift gate read.
#[test]
fn image_editor_geom_offset_is_registered_in_the_family() {
    assert!(
        geom::FAMILY.iter().any(|d| d.id == "geom.offset"),
        "geom.offset must be in geom::FAMILY"
    );
    assert!(
        ALL_FAMILIES
            .iter()
            .flat_map(|f| f.iter())
            .any(|d| d.id == "geom.offset"),
        "geom.offset must be reachable from ALL_FAMILIES"
    );
    // Exactly once — a duplicated entry would double every count the
    // registry quotes.
    assert_eq!(
        ALL_FAMILIES
            .iter()
            .flat_map(|f| f.iter())
            .filter(|d| d.id == "geom.offset")
            .count(),
        1
    );
}

/// The metadata the engines dispatch on.
#[test]
fn image_editor_geom_offset_kernel_metadata() {
    assert_eq!(GEOM_OFFSET.id, "geom.offset");
    assert_eq!(GEOM_OFFSET.inputs, 1);
    assert!(GEOM_OFFSET.module, "handwritten ABI v1.1 module");
    assert!(!GEOM_OFFSET.mip_exact, "a coordinate remap runs at level 0");
    assert_eq!(
        GEOM_OFFSET.class,
        KernelClass::Resample { support: 1.0 },
        "a 2x2 bilinear footprint, like rotate_bilinear/warp_backward"
    );
    assert_eq!(GEOM_OFFSET.gpu_tolerance, Tolerance::ChannelEpsF16(4));
}

/// The ABI v1.1 binding contract, checked against the module TEXT: the
/// four groups, the workgroup size, the bounds prologue, and the
/// mandatory mask epilogue. `abi::assemble` returns a `module: true`
/// kernel's WGSL verbatim, so this is the shipped source.
#[test]
fn image_editor_geom_offset_wgsl_honours_the_abi() {
    let src = image_kernels::abi::assemble(&GEOM_OFFSET);
    for needle in [
        "@group(0) @binding(0) var in0 : texture_2d<f32>;",
        "@group(1) @binding(0) var<uniform> params : Params;",
        "@group(2) @binding(0) var mask : texture_2d<f32>;",
        "@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;",
        "@compute @workgroup_size(16, 16, 1)",
        "if (gid.x >= dims.x || gid.y >= dims.y) { return; }",
        "let m = textureLoad(mask, xy, 0).r;",
        "textureStore(outp, xy, mix(a, result, vec4<f32>(m)));",
    ] {
        assert!(src.contains(needle), "missing from the module: {needle}");
    }
    assert!(
        !src.contains("textureSample"),
        "inputs are textureLoad-only under the ABI (no samplers)"
    );
}

/// The handwritten WGSL `struct Params` matches `ParamsLayout` field for
/// field in order, plus the trailing `_abi_pad`. A drift here is a
/// silently misdecoded uniform block, which is why it gets its own gate.
#[test]
fn image_editor_geom_offset_wgsl_struct_matches_the_layout() {
    let src = image_kernels::abi::assemble(&GEOM_OFFSET);
    let body = src
        .split_once("struct Params {")
        .expect("module declares struct Params")
        .1
        .split_once('}')
        .expect("struct Params closes")
        .0;
    let decls: Vec<String> = body
        .lines()
        .map(|l| l.trim().trim_end_matches(',').to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let mut expected: Vec<String> = GEOM_OFFSET
        .params
        .fields
        .iter()
        .map(|f| format!("{}: {}", f.name, f.wgsl_ty))
        .collect();
    expected.push("_abi_pad: u32".to_string());
    assert_eq!(decls, expected);
}

/// Naga parses and validates the module — the real gate for handwritten
/// WGSL (the crate-wide `wgsl_validate` suite covers this too; keeping a
/// local copy means a broken module fails in the file that owns it).
#[test]
fn image_editor_geom_offset_wgsl_validates() {
    let src = image_kernels::abi::assemble(&GEOM_OFFSET);
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("geom.offset: WGSL parse failed: {e}\n{src}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("geom.offset: WGSL validation failed: {e:?}\n{src}"));
}
