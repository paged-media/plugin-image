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

//! Engine B incremental-correctness gate (spec §12.4) — the
//! non-negotiable invariant: `request` (cache-using) MUST equal
//! `evaluate_from_scratch` after any damage sequence. Engine B's failure
//! mode is stale tiles, not wrong math, so this is the proof that
//! matters. The M2 fan-out broadens this to random graphs × random
//! damage and the mip-equivalence checks; this is the seed proof.
//! feat: image.graph.engine-b.

use std::sync::Arc;

use image_conformance::device::test_device;
use image_core::{Region, TileCoord, TILE};
use image_graph::{BufferGraph, SourceData};
use image_kernels::families::linear::{MathLinearParams, MATH_LINEAR};

fn ramp_tile(seed: u8) -> Arc<[u8]> {
    // One rgba16float tile; values vary by seed so re-pointing a param
    // actually changes the output.
    let mut v = vec![0u8; (TILE * TILE * 8) as usize];
    for (i, px) in v.chunks_exact_mut(8).enumerate() {
        let f = half::f16::from_f32(((i as u32 % 97) as f32 / 97.0 + seed as f32 * 0.001).fract());
        for c in 0..4 {
            px[c * 2..c * 2 + 2].copy_from_slice(&f.to_bits().to_le_bytes());
        }
    }
    Arc::from(v.into_boxed_slice())
}

fn linear_params(gain: f32, bias: f32) -> Vec<u8> {
    MathLinearParams::new(gain, bias).as_bytes().to_vec()
}

#[test]
fn engine_b_incremental_equals_from_scratch_after_param_edits() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    // Graph: source → linear(a) → linear(b)  (a two-op chain).
    let mut g = BufferGraph::new();
    let mut src = SourceData::new();
    for x in 0..2i32 {
        for y in 0..2i32 {
            src.set_tile(
                TileCoord { level: 0, x, y },
                ramp_tile((x * 2 + y) as u8 + 1),
                1,
            );
        }
    }
    let s = g.add_source(src);
    let n1 = g.add_op(&MATH_LINEAR, linear_params(1.5, -0.1), vec![s]);
    let n2 = g.add_op(&MATH_LINEAR, linear_params(0.8, 0.05), vec![n1]);

    let region = Region::new(0, 0, 2 * TILE, 2 * TILE);

    // Prime the caches.
    let _ = g.request(n2, region, 0, ctx).unwrap();
    assert!(g.cache_len(n2) > 0, "request should populate the cache");

    // A sequence of param edits on both nodes (gestures then a commit).
    let edits: &[(usize, (f32, f32))] = &[
        (1, (2.0, 0.0)), // gesture on n1
        (2, (1.0, 0.2)), // gesture on n2
        (1, (2.0, 0.0)), // re-set identical bytes on n1 (no-op invalidation)
        (1, (0.5, 0.3)), // commit a different value on n1
    ];
    let nodes = [s, n1, n2];
    for &(which, (gain, bias)) in edits {
        g.set_params(nodes[which], linear_params(gain, bias));

        // After EACH edit: the incremental result must equal a full
        // from-scratch evaluation, byte-for-byte AND generation-for-
        // generation.
        let incremental = g.request(n2, region, 0, ctx).unwrap();
        let scratch = g.evaluate_from_scratch(n2, region, 0, ctx).unwrap();
        assert_eq!(incremental.len(), scratch.len());
        for (a, b) in incremental.iter().zip(&scratch) {
            assert_eq!(a.coord, b.coord);
            assert_eq!(
                a.generation, b.generation,
                "generation mismatch at {:?}",
                a.coord
            );
            assert_eq!(
                &a.bytes[..],
                &b.bytes[..],
                "tile bytes mismatch at {:?}",
                a.coord
            );
        }
    }
}

#[test]
fn engine_b_write_source_tile_invalidates_downstream() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let mut g = BufferGraph::new();
    let mut src = SourceData::new();
    src.set_tile(
        TileCoord {
            level: 0,
            x: 0,
            y: 0,
        },
        ramp_tile(1),
        1,
    );
    let s = g.add_source(src);
    let n1 = g.add_op(&MATH_LINEAR, linear_params(1.0, 0.0), vec![s]);
    let region = Region::new(0, 0, TILE, TILE);

    let before = g.request(n1, region, 0, ctx).unwrap()[0].bytes.clone();
    // Overwrite the source tile (a WriteBuffer Operation) with new bytes
    // + a bumped generation.
    g.write_source_tile(
        s,
        TileCoord {
            level: 0,
            x: 0,
            y: 0,
        },
        ramp_tile(42),
        2,
    );
    let after_inc = g.request(n1, region, 0, ctx).unwrap();
    let after_scratch = g.evaluate_from_scratch(n1, region, 0, ctx).unwrap();
    assert_eq!(&after_inc[0].bytes[..], &after_scratch[0].bytes[..]);
    assert_ne!(
        &before[..],
        &after_inc[0].bytes[..],
        "downstream must reflect the source write"
    );
}
