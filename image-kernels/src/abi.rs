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

//! The frozen WGSL kernel ABI (spec §9.2, ABI v1 — M0 phase 0):
//!
//! ```text
//! @group(0): input tile textures (arity per KernelDef.inputs)
//! @group(1): params uniform block  (repr(C) layout shared with Rust)
//! @group(2): selection mask texture (constant-1 default)
//! @group(3): output storage texture (rgba16float, write)
//! workgroup: 16×16 over the tile grid (256² tile ⇒ 16×16 workgroups)
//! ```
//!
//! Inputs are sampled with `textureLoad` ONLY (no samplers — avoids
//! filterability traps; resampling kernels do their own footprint
//! math). The selection mask is applied at the ABI level for pointwise
//! kernels: `out = mix(a, result, mask)` — selection-ready from day one
//! (spec §6.1); Engine A binds the constant-1 mask.

use crate::KernelDef;

/// Workgroup edge; a 256² tile dispatches 16×16 workgroups.
pub const WORKGROUP_SIZE: u32 = 16;

/// Workgroups per tile edge.
pub const GROUPS_PER_TILE: u32 = image_core::TILE / WORKGROUP_SIZE;

/// Helper functions available to every kernel body — the WGSL half of
/// the restricted DSL. The Rust half lives in `reference_prelude`; the
/// two are kept in lock-step (the golden-expansion conformance test
/// guards drift). WGSL builtins (`clamp`, `mix`, `min`, `max`, `abs`,
/// `floor`, `select`) are part of the DSL whitelist and need no
/// preamble entry.
pub const WGSL_PRELUDE: &str = "\
fn splat4(x: f32) -> vec4<f32> { return vec4<f32>(x); }
";

/// Assemble the complete WGSL compute module for a kernel: ABI
/// preamble + the 1:1 `Params` struct from `ParamsLayout` + the body
/// expression spliced into the fixed entry point.
pub fn assemble(def: &KernelDef) -> String {
    assert!(
        def.inputs >= 1 && def.inputs <= 2,
        "ABI v1 assembles unary/binary kernels (generators land with T2)"
    );

    let mut params_struct = String::from("struct Params {\n");
    for f in def.params.fields {
        params_struct.push_str(&format!("    {}: {},\n", f.name, f.wgsl_ty));
    }
    // Every param block carries the trailing ABI pad (see family.rs) so
    // empty param lists still form a valid uniform struct.
    // NOTE: modern WGSL struct declarations take no trailing semicolon.
    params_struct.push_str("    _abi_pad: u32,\n}\n");

    let mut s = String::new();
    s.push_str(&format!(
        "// paged.image kernel `{}` — GENERATED under ABI v{}.\n",
        def.id,
        crate::ABI_VERSION
    ));
    s.push_str("// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.\n\n");
    s.push_str(&params_struct);
    s.push('\n');
    s.push_str("@group(0) @binding(0) var in0 : texture_2d<f32>;\n");
    if def.inputs == 2 {
        s.push_str("@group(0) @binding(1) var in1 : texture_2d<f32>;\n");
    }
    s.push_str("@group(1) @binding(0) var<uniform> params : Params;\n");
    s.push_str("@group(2) @binding(0) var mask : texture_2d<f32>;\n");
    s.push_str("@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;\n\n");
    s.push_str(WGSL_PRELUDE);
    s.push('\n');
    s.push_str(&format!(
        "@compute @workgroup_size({wg}, {wg}, 1)\n",
        wg = WORKGROUP_SIZE
    ));
    s.push_str("fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n");
    s.push_str("    let dims = textureDimensions(outp);\n");
    s.push_str("    if (gid.x >= dims.x || gid.y >= dims.y) { return; }\n");
    s.push_str("    let xy = vec2<i32>(i32(gid.x), i32(gid.y));\n");
    s.push_str("    let a = textureLoad(in0, xy, 0);\n");
    if def.inputs == 2 {
        s.push_str("    let b = textureLoad(in1, xy, 0);\n");
    } else {
        s.push_str("    let b = vec4<f32>(0.0);\n");
    }
    s.push_str("    let m = textureLoad(mask, xy, 0).r;\n");
    s.push_str("    let p = params;\n");
    s.push_str(&format!("    let result : vec4<f32> = {};\n", def.wgsl));
    s.push_str("    textureStore(outp, xy, mix(a, result, splat4(m)));\n");
    s.push_str("}\n");
    s
}
