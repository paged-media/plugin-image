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

//! The reusable gpu↔ref parity harness (spec §12.4): upload → dispatch
//! → readback → run the scalar reference → quantize reference to f16
//! (final step, §5.2) → per-channel f16-ULP diff. A per-kernel test is
//! ~5 lines: build inputs, call [`parity`], [`assert_within`].

use half::f16;
use image_gpu::{execute_tile_once, TileInput};
use image_kernels::reference_prelude::Px;
use image_kernels::{KernelDef, Tolerance};

use crate::device::test_device;
use crate::quantize::{f16_ulp_distance, f32_to_f16_bits};

/// A test tile in reference precision (f32 rgba), row-major.
///
/// STIMULUS RULE (M0): values must be FINITE. The ABI applies
/// `mix(in, result, mask)` on the GPU; with NaN/Inf inputs that flow
/// is fast-math/driver-dependent (Metal may fold `x*0` ≠ IEEE), so
/// NaN/Inf probes are excluded from parity stimulus until the D-10
/// hardware-divergence policy takes them up. Kernel-body NaN semantics
/// (relational/boolean helpers) remain deterministic by construction.
#[derive(Debug, Clone)]
pub struct RefTile {
    pub w: u32,
    pub h: u32,
    pub px: Vec<Px>,
}

impl RefTile {
    pub fn from_fn(w: u32, h: u32, f: impl Fn(u32, u32) -> Px) -> RefTile {
        let mut px = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                px.push(f(x, y));
            }
        }
        RefTile { w, h, px }
    }

    /// rgba16float texel bytes for the GPU upload — inputs are
    /// quantized ONCE here, so GPU and reference consume identical
    /// stimulus (the f16 value), per §6.3.
    pub fn f16_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.px.len() * 8);
        for p in &self.px {
            for c in p.0 {
                out.extend_from_slice(&f16::from_f32(c).to_bits().to_le_bytes());
            }
        }
        out
    }

    /// The tile as the reference consumes it: the SAME f16-quantized
    /// values the GPU received, widened back to f32.
    pub fn quantized_px(&self) -> Vec<Px> {
        self.px
            .iter()
            .map(|p| Px(p.0.map(|c| f16::from_f32(c).to_f32())))
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParityResult {
    /// Worst per-channel distance in f16 ULPs.
    pub max_ulp: u32,
    /// Texel index + channel of the worst divergence (diagnostics).
    pub worst_at: (usize, usize),
}

/// Run `def` on the test GPU and the scalar reference over the same
/// stimulus; `None` when the environment has no GPU adapter (callers
/// skip — the merge-gate GPU lane runs where one is guaranteed).
pub fn parity<P: bytemuck::Pod>(
    def: &'static KernelDef,
    reference: impl Fn(Px, Px, &P) -> Px,
    inputs: &[&RefTile],
    params: &P,
) -> Option<ParityResult> {
    let ctx = test_device()?;
    assert_eq!(inputs.len(), def.inputs as usize, "input arity");
    let (w, h) = (inputs[0].w, inputs[0].h);

    // GPU lane.
    let in_bytes: Vec<Vec<u8>> = inputs.iter().map(|t| t.f16_bytes()).collect();
    let tile_inputs: Vec<TileInput<'_>> = in_bytes
        .iter()
        .map(|b| TileInput { f16_bytes: b })
        .collect();
    let gpu_out = execute_tile_once(
        ctx,
        def,
        &tile_inputs,
        bytemuck::bytes_of(params),
        None, // constant-1 mask: the Engine A binding (§6.1)
        w,
        h,
    )
    .expect("kernel execution");

    // Reference lane over the SAME (f16-quantized) stimulus.
    let quant: Vec<Vec<Px>> = inputs.iter().map(|t| t.quantized_px()).collect();
    let zero = Px([0.0; 4]);
    let mut max_ulp = 0u32;
    let mut worst_at = (0usize, 0usize);
    for i in 0..(w * h) as usize {
        let a = quant[0][i];
        let b = if def.inputs == 2 { quant[1][i] } else { zero };
        let want = reference(a, b, params);
        for c in 0..4 {
            let want_bits = f32_to_f16_bits(want.0[c]);
            let got_bits = u16::from_le_bytes([gpu_out[i * 8 + c * 2], gpu_out[i * 8 + c * 2 + 1]]);
            let d = f16_ulp_distance(want_bits, got_bits);
            if d > max_ulp {
                max_ulp = d;
                worst_at = (i, c);
            }
        }
    }
    Some(ParityResult { max_ulp, worst_at })
}

/// Windowed/resample parity (ABI v1.1 module kernels): the GPU lane
/// runs `execute_windowed_once` over the f16-quantized `window`; the
/// reference computes each output texel from the SAME quantized window
/// (`reference(window_px, win_w, win_h, out_x, out_y, params)` — and
/// must model the kernel's own mask handling: windowed kernels mix
/// against the center sample with mask 1, i.e. plain `result`).
/// `None` when the environment has no GPU adapter.
#[allow(clippy::too_many_arguments)]
pub fn parity_windowed<P: bytemuck::Pod>(
    def: &'static KernelDef,
    reference: impl Fn(&[Px], u32, u32, u32, u32, &P) -> Px,
    window: &RefTile,
    out_w: u32,
    out_h: u32,
    params: &P,
) -> Option<ParityResult> {
    let ctx = test_device()?;
    let gpu_out = image_gpu::execute_windowed_once(
        ctx,
        def,
        &window.f16_bytes(),
        window.w,
        window.h,
        bytemuck::bytes_of(params),
        None,
        out_w,
        out_h,
    )
    .expect("windowed kernel execution");

    let quant = window.quantized_px();
    let mut max_ulp = 0u32;
    let mut worst_at = (0usize, 0usize);
    for oy in 0..out_h {
        for ox in 0..out_w {
            let i = (oy * out_w + ox) as usize;
            let want = reference(&quant, window.w, window.h, ox, oy, params);
            for c in 0..4 {
                let want_bits = f32_to_f16_bits(want.0[c]);
                let got_bits =
                    u16::from_le_bytes([gpu_out[i * 8 + c * 2], gpu_out[i * 8 + c * 2 + 1]]);
                let d = f16_ulp_distance(want_bits, got_bits);
                if d > max_ulp {
                    max_ulp = d;
                    worst_at = (i, c);
                }
            }
        }
    }
    Some(ParityResult { max_ulp, worst_at })
}

/// Assert a parity result satisfies the kernel's declared tolerance.
pub fn assert_within(result: ParityResult, def: &KernelDef) {
    let limit = match def.gpu_tolerance {
        Tolerance::Exact => 0,
        Tolerance::ChannelEpsF16(n) => n,
        Tolerance::PerceptualDeltaE(_) => {
            unimplemented!("ΔE tolerances arrive with the T1 color kernels")
        }
    };
    assert!(
        result.max_ulp <= limit,
        "{}: max f16 ULP distance {} exceeds declared tolerance {} (worst at texel {}, channel {})",
        def.id,
        result.max_ulp,
        limit,
        result.worst_at.0,
        result.worst_at.1,
    );
}
