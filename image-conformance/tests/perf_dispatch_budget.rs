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

//! WORK BUDGETS for a single kernel dispatch, in COUNTS
//! (`image_gpu::counters`), never wall-clock.
//!
//! A budget equals the measured value. It is lowered in the commit that
//! earns it and never raised; raising one means the change made the
//! engine do more work, which is the regression this file exists to
//! catch. The numbers below are 2026-10-04's: every dispatch compiles
//! its own shader module and pipeline, so running the same kernel twice
//! costs two compiles. The pipeline cache lowers `REPEAT_PIPELINES`.

#![allow(non_snake_case)]

use image_conformance::device::test_device;
use image_conformance::harness::RefTile;
use image_conformance::Px;
use image_gpu::{counters, execute_tile_once, GpuCounters, TileInput};
use image_kernels::families::arithmetic::{MathAddConstParams, MATH_ADD_CONST};

const W: u32 = 64;
const H: u32 = 64;

fn tile() -> Vec<u8> {
    RefTile::from_fn(W, H, |x, y| {
        Px([x as f32 / W as f32, y as f32 / H as f32, 0.5, 1.0])
    })
    .f16_bytes()
}

fn dispatch(ctx: &image_gpu::GpuContext, bytes: &[u8]) -> GpuCounters {
    let p = MathAddConstParams::new(0.25);
    let (out, cost) = counters::measure(|| {
        execute_tile_once(
            ctx,
            &MATH_ADD_CONST,
            &[TileInput { f16_bytes: bytes }],
            bytemuck::bytes_of(&p),
            None,
            W,
            H,
        )
    });
    out.expect("dispatch");
    cost
}

/// One unary point kernel over one 64×64 tile, constant mask.
#[test]
fn one_dispatch_costs_exactly_this__feat__image_conformance_harness() {
    let Some(ctx) = test_device() else { return };
    let bytes = tile();
    let c = dispatch(ctx, &bytes);
    let texels = (W * H) as u64;
    assert_eq!(
        c,
        GpuCounters {
            shader_modules: 1,
            pipelines_built: 1,
            dispatches: 1,
            dispatched_texels: texels,
            submits: 1,
            // input + mask + output
            textures_created: 3,
            // rgba16f input + r16f constant mask + the param block
            bytes_uploaded: texels * 8 + texels * 2 + MATH_ADD_CONST.params.size as u64,
            readbacks: 1,
            // 64 texels × 8 bytes = 512 per row, already 256-aligned
            bytes_read_back: texels * 8,
        },
        "one dispatch: {c:?}"
    );
}

/// BUDGET: the same kernel twice. Measured 2026-10-04: two shader
/// modules and two pipelines — nothing is reused.
const REPEAT_PIPELINES: u64 = 2;

#[test]
fn the_same_kernel_twice__feat__image_conformance_harness() {
    let Some(ctx) = test_device() else { return };
    let bytes = tile();
    let a = dispatch(ctx, &bytes);
    let b = dispatch(ctx, &bytes);
    let pipelines = a.pipelines_built + b.pipelines_built;
    assert!(
        pipelines <= REPEAT_PIPELINES,
        "pipelines built for two identical dispatches: {pipelines} > budget {REPEAT_PIPELINES}"
    );
    // Equal to the budget, not just under it: a budget that is looser
    // than the measurement cannot catch the next regression.
    assert_eq!(
        pipelines, REPEAT_PIPELINES,
        "the work went down — lower REPEAT_PIPELINES to {pipelines} in this commit"
    );
}
