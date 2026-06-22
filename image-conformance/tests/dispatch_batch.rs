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

//! Batched-dispatch equivalence (spec §9.2): N tiles coalesced into ONE
//! command encoder / ONE compute pass / one submit must produce exactly
//! the bytes the single-tile `execute_tile_once` path produces. The
//! coalescing is a scheduling change, not a math change — so the gate is
//! byte-identity, not tolerance. feat: image.gpu.dispatch.batch.
//!
//! SKIPs (does not fail) without a GPU adapter — the merge-gate GPU lane
//! runs where one is guaranteed (spec §6.3 device-skip discipline).

use image_conformance::device::test_device;
use image_conformance::harness::RefTile;
use image_conformance::Px;
use image_gpu::{execute_tile_once, BatchTile, DispatchBatch, KernelPipeline, TileInput};
use image_kernels::families::linear::{math_linear, MathLinearParams, MATH_LINEAR};

/// Four DISTINCT 256² tiles — different gradients so a stale/duplicated
/// bind group in the batch would diverge from the single-tile path.
fn distinct_tiles() -> Vec<RefTile> {
    let t = image_core::TILE;
    (0..4u32)
        .map(|k| {
            let phase = k as f32 * 0.17;
            RefTile::from_fn(t, t, move |x, y| {
                Px([
                    (x as f32 / t as f32 + phase).fract(),
                    (y as f32 / t as f32 + phase * 2.0).fract(),
                    ((x + y) as f32 / (2 * t) as f32 + phase).fract(),
                    1.0,
                ])
            })
        })
        .collect()
}

#[test]
fn image_gpu_dispatch_batch_matches_single_tile() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let t = image_core::TILE;
    let tiles = distinct_tiles();
    let p = MathLinearParams::new(1.5, -0.125);
    let params = bytemuck::bytes_of(&p);

    // Single-tile lane: the trusted reference path, one tile per call.
    let in_bytes: Vec<Vec<u8>> = tiles.iter().map(|t| t.f16_bytes()).collect();
    let single: Vec<Vec<u8>> = in_bytes
        .iter()
        .map(|b| {
            execute_tile_once(
                ctx,
                &MATH_LINEAR,
                &[TileInput { f16_bytes: b }],
                params,
                None,
                t,
                t,
            )
            .expect("single-tile execution")
        })
        .collect();

    // Batched lane: all four tiles in ONE submit.
    let pipeline = KernelPipeline::build(ctx, &MATH_LINEAR);
    let batch = DispatchBatch::new(&pipeline, params).expect("batch params accepted");
    let tile_inputs: Vec<[TileInput<'_>; 1]> = in_bytes
        .iter()
        .map(|b| [TileInput { f16_bytes: b }])
        .collect();
    let batch_tiles: Vec<BatchTile<'_>> = tile_inputs
        .iter()
        .map(|inp| BatchTile {
            inputs: inp,
            mask: None,
            w: t,
            h: t,
        })
        .collect();
    let batched = batch
        .submit_and_read(ctx, &batch_tiles)
        .expect("batched execution");

    assert_eq!(batched.len(), single.len(), "tile count preserved");
    for (i, (b, s)) in batched.iter().zip(&single).enumerate() {
        assert_eq!(
            b, s,
            "tile {i}: batched output must be byte-identical to the single-tile path"
        );
    }

    // Sanity: the distinct tiles really do differ (a degenerate all-equal
    // stimulus would make the identity check vacuous).
    assert_ne!(single[0], single[1], "stimulus tiles must be distinct");

    // And the math is the one we think it is — spot-check against the
    // scalar reference at one texel (gain·x + bias, alpha included).
    let q = tiles[0].quantized_px();
    let want = math_linear(q[0], Px([0.0; 4]), &p);
    let got_r = half::f16::from_le_bytes([batched[0][0], batched[0][1]]).to_f32();
    let tol = 1e-2_f32; // f16 storage round-trip slack
    assert!(
        (got_r - want.0[0]).abs() < tol,
        "texel 0 red: got {got_r}, want {} (ref math.linear)",
        want.0[0]
    );
}
