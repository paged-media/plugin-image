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

//! Golden-expansion guard: snapshots BOTH emissions of one
//! `kernel_family!` row so any macro change that diverges the WGSL
//! from the scalar reference is caught immediately (the highest-value
//! phase-0 test per the plan — WGSL ≡ Rust is the whole codegen
//! thesis).

use image_conformance::Px;
use image_kernels::families::linear::{math_linear, MathLinearParams, MATH_LINEAR};
use image_kernels::{abi, ParamField};

#[test]
fn wgsl_body_is_the_stringified_dsl() {
    assert_eq!(MATH_LINEAR.wgsl, "a * splat4(p.gain) + splat4(p.bias)");
}

#[test]
fn params_layout_matches_struct() {
    assert_eq!(MATH_LINEAR.params.size, 12); // gain + bias + _abi_pad
    assert_eq!(
        MATH_LINEAR.params.fields,
        &[
            ParamField {
                name: "gain",
                wgsl_ty: "f32"
            },
            ParamField {
                name: "bias",
                wgsl_ty: "f32"
            },
        ][..]
    );
}

#[test]
fn assembled_module_snapshot() {
    let wgsl = abi::assemble(&MATH_LINEAR);
    // Structural anchors of ABI v1 — a frozen contract, not styling.
    for anchor in [
        "struct Params {",
        "    gain: f32,",
        "    bias: f32,",
        "    _abi_pad: u32,",
        "@group(0) @binding(0) var in0 : texture_2d<f32>;",
        "@group(1) @binding(0) var<uniform> params : Params;",
        "@group(2) @binding(0) var mask : texture_2d<f32>;",
        "@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;",
        "@compute @workgroup_size(16, 16, 1)",
        "let result : vec4<f32> = a * splat4(p.gain) + splat4(p.bias);",
        "textureStore(outp, xy, mix(a, result, splat4(m)));",
    ] {
        assert!(
            wgsl.contains(anchor),
            "missing ABI anchor {anchor:?} in:\n{wgsl}"
        );
    }
    // Unary kernel: no second input binding.
    assert!(!wgsl.contains("in1"), "unary kernel must not bind in1");
}

#[test]
fn reference_twin_matches_hand_math() {
    let p = MathLinearParams::new(2.0, 0.5);
    let out = math_linear(Px([0.25, 0.5, 1.0, 1.0]), Px([0.0; 4]), &p);
    assert_eq!(out, Px([1.0, 1.5, 2.5, 2.5]));
}
