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
//! catch. Measurements are of the STEADY STATE (after one warm-up run)
//! so they do not depend on which test touched the shared device first.

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

/// One unary point kernel over one 64×64 tile, constant mask, warm.
/// Until 2026-10-04 every dispatch also compiled a shader module and a
/// pipeline (1 + 1 here); the per-device pipeline cache made them 0.
/// It also created its input, mask and output textures (3) and uploaded
/// a constant-1 mask: now they come from the scratch pool and the shared
/// per-size constant, so a warm dispatch creates none and uploads only
/// its input and params.
#[test]
fn one_dispatch_costs_exactly_this__feat__image_conformance_harness() {
    let Some(ctx) = test_device() else { return };
    let bytes = tile();
    dispatch(ctx, &bytes); // warm-up
    let c = dispatch(ctx, &bytes);
    let texels = (W * H) as u64;
    assert_eq!(
        c,
        GpuCounters {
            shader_modules: 0,
            pipelines_built: 0,
            dispatches: 1,
            dispatched_texels: texels,
            submits: 1,
            // was 3 (input + mask + output): scratch pool + shared mask
            textures_created: 0,
            // rgba16f input + the param block (was + an r16f constant mask)
            bytes_uploaded: texels * 8 + MATH_ADD_CONST.params.size as u64,
            readbacks: 1,
            // 64 texels × 8 bytes = 512 per row, already 256-aligned
            bytes_read_back: texels * 8,
        },
        "one warm dispatch: {c:?}"
    );
}

/// The pipeline is built once per device: a hundred dispatches of the
/// same kernel build at most one (none if another test built it first).
#[test]
fn a_kernel_compiles_once_per_device__feat__image_conformance_harness() {
    let Some(ctx) = test_device() else { return };
    let bytes = tile();
    let (_, c) = counters::measure(|| {
        for _ in 0..100 {
            dispatch(ctx, &bytes);
        }
    });
    assert!(
        c.pipelines_built <= 1,
        "100 dispatches built {} pipelines",
        c.pipelines_built
    );
    assert_eq!(c.shader_modules, c.pipelines_built);
    assert_eq!(c.dispatches, 100);
}
