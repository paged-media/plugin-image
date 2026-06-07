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

//! Convolution family (T1, spec §11) — handwritten WGSL modules under
//! the ABI v1.1 contract (`abi::assemble` docs). `conv.box` is the
//! amendment proof: the first windowed kernel through the module lane
//! and the windowed parity harness. Gaussian (separable two-pass) and
//! unsharp land with the M1 fan-out.
//!
//! Provenance: separable convolution and box filtering are standard
//! literature; no reference reading.

use crate::{KernelClass, KernelDef, ParamsLayout, Tolerance};

/// 3×3 box mean — params are the bare ABI pad.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct ConvBoxParams {
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl ConvBoxParams {
    pub fn new() -> Self {
        Self { _abi_pad: 0 }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// out = mean of the 3×3 window (radius 1,1); mask-mixed against the
/// window center per the windowed convention.
pub static CONV_BOX: KernelDef = KernelDef {
    id: "conv.box",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<ConvBoxParams>(),
        fields: &[],
    },
    wgsl: CONV_BOX_WGSL,
    module: true,
    mip_exact: true,
    gpu_tolerance: Tolerance::ChannelEpsF16(2),
};

// Summation order (dy outer ascending, dx inner ascending) is part of
// the kernel's determinism contract — the scalar reference mirrors it
// exactly (§6.3 fixed reduction order).
const CONV_BOX_WGSL: &str = "\
// paged.image kernel `conv.box` — handwritten under ABI v1.1.
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
    // Window center: output (x, y) maps to in0 (x + rx, y + ry).
    let c = xy + vec2<i32>(1, 1);
    var sum = vec4<f32>(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            sum = sum + textureLoad(in0, c + vec2<i32>(dx, dy), 0);
        }
    }
    let result = sum / 9.0;
    let center = textureLoad(in0, c, 0);
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(center, result, vec4<f32>(m)));
}
";

pub static FAMILY: &[&KernelDef] = &[&CONV_BOX];
