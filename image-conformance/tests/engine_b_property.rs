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

//! Engine B broad correctness — the property fan-out the seed proof
//! (`engine_b_incremental.rs`) promised (spec §12.4). Three properties,
//! all driven by a fixed-seed LCG (no wall-clock; a failing seed reruns
//! deterministically):
//!
//! - PROPERTY 1 (`engine_b_incremental_equals_from_scratch`): a random
//!   DAG over {math.linear, math.invert, math.add, math.mul,
//!   compose.normal, conv.box} (mixed unary/binary/windowed) × a random
//!   damage sequence (set_params / gesture / write_source_tile) →
//!   `request` MUST equal `evaluate_from_scratch` byte-AND-generation-
//!   identical after EVERY step. The non-negotiable invariant: Engine B's
//!   failure mode is stale tiles, not wrong math, so a divergence here is
//!   a provenance/invalidation bug.
//! - PROPERTY 2 (`engine_b_cache_noop_stability`): priming then re-setting
//!   IDENTICAL param bytes is a true no-op — `cache_len` unchanged and the
//!   output bytes stable (the §8.2 provenance equality check, not a dirty
//!   flag).
//! - PROPERTY 3 (`engine_b_mip_exact_point_commutes`): for a point kernel
//!   (math.linear, `mip_exact: true`), evaluating the op at level 1 over a
//!   source pyramid equals applying the kernel directly to the level-1
//!   source — point kernels commute with the mip switch (§8.3). Windowed
//!   σ-halving is the deeper M3 check (documented below, not asserted
//!   here — it needs the engine to rescale windowed params per level).
//!
//! feat: image.graph.engine-b.

use std::sync::Arc;

use image_conformance::device::test_device;
use image_core::{Region, TileCoord, TILE};
use image_graph::{BufferGraph, EvaluatedTile, NodeId, SourceData};
use image_kernels::families::arithmetic::{MathAddParams, MathMulParams, MATH_ADD, MATH_MUL};
use image_kernels::families::compose::{ComposeParams, COMPOSE_NORMAL};
use image_kernels::families::conv::{ConvBoxParams, CONV_BOX};
use image_kernels::families::linear::{
    MathInvertParams, MathLinearParams, MATH_INVERT, MATH_LINEAR,
};
use image_kernels::KernelDef;

const TILE_BYTES: usize = (TILE * TILE * 8) as usize; // rgba16float

// ───────────────────────────── PRNG ─────────────────────────────
//
// A 64-bit LCG (Numerical Recipes constants). Fixed seed per case ⇒
// fully deterministic and resume-safe; no wall-clock, no thread-rng.

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero fixpoint; mix the seed in.
        Lcg(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Return the high bits (the low bits of an LCG are weak).
        self.0
    }

    /// Uniform in `[0, n)` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() >> 33) as usize % n
    }

    /// A float in roughly `[-1.0, 1.0]` — bounded stimulus, finite on
    /// both lanes (the harness rule; no NaN/Inf into the kernels).
    fn unit_signed(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        2.0 * u - 1.0
    }

    /// A float in `[0.5, 1.5]` — gains/opacities that stay sane.
    fn moderate(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        0.5 + u
    }
}

// ─────────────────────────── tile makers ──────────────────────────

/// One rgba16float tile whose values vary by `seed` (so re-pointing a
/// source or param actually changes the output, not just the cache key).
fn rand_tile(rng: &mut Lcg, seed: u64) -> Arc<[u8]> {
    let mut v = vec![0u8; TILE_BYTES];
    let base = (seed % 251) as f32 * 0.003;
    for (i, px) in v.chunks_exact_mut(8).enumerate() {
        // Per-texel value in [0,1); RGB vary, alpha kept at 1.0 so the
        // premultiplied compose path is well-conditioned.
        let f = half::f16::from_f32(((i as u64 % 89) as f32 / 89.0 + base + rng_bit(rng)).fract());
        for c in 0..3 {
            px[c * 2..c * 2 + 2].copy_from_slice(&f.to_bits().to_le_bytes());
        }
        let a = half::f16::from_f32(1.0);
        px[6..8].copy_from_slice(&a.to_bits().to_le_bytes());
    }
    Arc::from(v.into_boxed_slice())
}

fn rng_bit(rng: &mut Lcg) -> f32 {
    (rng.next_u64() >> 52) as f32 / (1u64 << 12) as f32 * 0.01
}

// ──────────────────────── param construction ──────────────────────

/// The op kinds the random DAG draws from. Each carries its arity so the
/// builder wires the right number of inputs.
#[derive(Clone, Copy)]
enum OpKind {
    Linear,  // unary point
    Invert,  // unary point
    Add,     // binary point
    Mul,     // binary point
    Compose, // binary point (premultiplied source-over)
    Box,     // unary windowed (radius 1,1)
}

impl OpKind {
    fn kernel(self) -> &'static KernelDef {
        match self {
            OpKind::Linear => &MATH_LINEAR,
            OpKind::Invert => &MATH_INVERT,
            OpKind::Add => &MATH_ADD,
            OpKind::Mul => &MATH_MUL,
            OpKind::Compose => &COMPOSE_NORMAL,
            OpKind::Box => &CONV_BOX,
        }
    }

    fn arity(self) -> usize {
        self.kernel().inputs as usize
    }

    /// Fresh random param bytes for this op kind.
    fn rand_params(self, rng: &mut Lcg) -> Vec<u8> {
        match self {
            OpKind::Linear => MathLinearParams::new(rng.moderate(), rng.unit_signed() * 0.2)
                .as_bytes()
                .to_vec(),
            OpKind::Invert => MathInvertParams::new().as_bytes().to_vec(),
            OpKind::Add => MathAddParams::new().as_bytes().to_vec(),
            OpKind::Mul => MathMulParams::new().as_bytes().to_vec(),
            OpKind::Compose => ComposeParams::new(rng.moderate().min(1.0))
                .as_bytes()
                .to_vec(),
            OpKind::Box => ConvBoxParams::new().as_bytes().to_vec(),
        }
    }

    /// Whether this op's params have any random freedom — only these are
    /// worth issuing a `set_params`/`gesture` edit against.
    fn has_params(self) -> bool {
        matches!(self, OpKind::Linear | OpKind::Compose)
    }
}

const OP_KINDS: &[OpKind] = &[
    OpKind::Linear,
    OpKind::Invert,
    OpKind::Add,
    OpKind::Mul,
    OpKind::Compose,
    OpKind::Box,
];

// ───────────────────────── graph builder ──────────────────────────

struct BuiltGraph {
    g: BufferGraph,
    /// Op nodes only (sources excluded) — the set damage edits target.
    ops: Vec<(NodeId, OpKind)>,
    /// Source nodes — the set `write_source_tile` targets.
    sources: Vec<NodeId>,
    /// The terminal node a request reads from (last node added).
    sink: NodeId,
    /// How many 256² source tiles per axis the sources carry (2 ⇒ 2×2).
    tiles_per_axis: i32,
}

/// Build a random DAG: depth 2–5, ≤ 8 nodes, mixing unary/binary/windowed
/// ops. Binary nodes pick TWO already-existing nodes as inputs (a DAG,
/// never a cycle — inputs are always earlier indices).
fn build_graph(rng: &mut Lcg) -> BuiltGraph {
    let tiles_per_axis = 2i32;
    let mut g = BufferGraph::new();

    // 1–2 source nodes, each a 2×2 grid of random level-0 tiles.
    let n_sources = 1 + rng.below(2);
    let mut sources = Vec::new();
    let mut all_nodes: Vec<NodeId> = Vec::new();
    for s in 0..n_sources {
        let mut src = SourceData::new();
        for x in 0..tiles_per_axis {
            for y in 0..tiles_per_axis {
                let seed = (s as u64) << 32 | ((x * tiles_per_axis + y) as u64 + 1);
                src.set_tile(TileCoord { level: 0, x, y }, rand_tile(rng, seed), 1);
            }
        }
        let id = g.add_source(src);
        sources.push(id);
        all_nodes.push(id);
    }

    // Op nodes until we hit a random target count (depth 2–5 worth of
    // layering), capped at 8 nodes total.
    let target_total = (n_sources + 2 + rng.below(4)).min(8);
    let mut ops = Vec::new();
    while all_nodes.len() < target_total {
        let kind = OP_KINDS[rng.below(OP_KINDS.len())];
        let arity = kind.arity();
        // Pick `arity` inputs from existing nodes (with replacement for
        // binary — a node may feed both inputs of an add/mul/compose).
        let mut inputs = Vec::with_capacity(arity);
        for _ in 0..arity {
            inputs.push(all_nodes[rng.below(all_nodes.len())]);
        }
        let id = g.add_op(kind.kernel(), kind.rand_params(rng), inputs);
        ops.push((id, kind));
        all_nodes.push(id);
    }

    // Guarantee at least one op exists (target_total ≥ n_sources + 2).
    let sink = *all_nodes.last().unwrap();
    BuiltGraph {
        g,
        ops,
        sources,
        sink,
        tiles_per_axis,
    }
}

// ──────────────────────── comparison helper ───────────────────────

/// `request` MUST equal `evaluate_from_scratch` — bytes AND generations,
/// tile-for-tile. Panics with a located message on divergence.
fn assert_incremental_equals_scratch(
    g: &mut BufferGraph,
    sink: NodeId,
    region: Region,
    level: u8,
    ctx: &image_gpu::GpuContext,
    step: &str,
) {
    let incremental = g.request(sink, region, level, ctx).unwrap();
    let scratch = g.evaluate_from_scratch(sink, region, level, ctx).unwrap();
    assert_eq!(
        incremental.len(),
        scratch.len(),
        "[{step}] tile count differs"
    );
    for (a, b) in incremental.iter().zip(&scratch) {
        assert_eq!(a.coord, b.coord, "[{step}] coord order differs");
        assert_eq!(
            a.generation, b.generation,
            "[{step}] generation mismatch at {:?}",
            a.coord
        );
        assert_eq!(
            &a.bytes[..],
            &b.bytes[..],
            "[{step}] tile bytes mismatch at {:?}",
            a.coord
        );
    }
}

/// A random sub-region inside the source extent, snapped to whole tiles
/// so it always covers at least one tile.
fn random_subregion(rng: &mut Lcg, tiles_per_axis: i32) -> Region {
    let t = TILE as i32;
    let tx = rng.below(tiles_per_axis as usize) as i32;
    let ty = rng.below(tiles_per_axis as usize) as i32;
    // 1 .. remaining tiles, in each axis.
    let tw = 1 + rng.below((tiles_per_axis - tx) as usize);
    let th = 1 + rng.below((tiles_per_axis - ty) as usize);
    Region::new(tx * t, ty * t, (tw as u32) * TILE, (th as u32) * TILE)
}

// ───────────────────────── PROPERTY 1 ─────────────────────────────

#[test]
fn engine_b_incremental_equals_from_scratch_random_dag_and_damage() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    const CASES: u64 = 32;
    for case in 0..CASES {
        let mut rng = Lcg::new(0x00EB_0000 ^ case);
        let mut built = build_graph(&mut rng);

        // Prime: an initial full-extent request populates every cache.
        let full = Region::new(
            0,
            0,
            (built.tiles_per_axis as u32) * TILE,
            (built.tiles_per_axis as u32) * TILE,
        );
        let _ = built.g.request(built.sink, full, 0, ctx).unwrap();
        assert_incremental_equals_scratch(
            &mut built.g,
            built.sink,
            full,
            0,
            ctx,
            &format!("case {case} prime"),
        );

        // 5–10 random damage steps; after EACH, a random sub-region must
        // be incremental==scratch.
        let steps = 5 + rng.below(6);
        for step in 0..steps {
            apply_random_damage(&mut rng, &mut built);
            let region = random_subregion(&mut rng, built.tiles_per_axis);
            assert_incremental_equals_scratch(
                &mut built.g,
                built.sink,
                region,
                0,
                ctx,
                &format!("case {case} step {step}"),
            );
        }
    }
}

/// One random mutation: a param edit (set_params or gesture) on a random
/// param-bearing op, OR a source-tile overwrite with a bumped generation.
/// Falls back to a source write when no op carries params.
fn apply_random_damage(rng: &mut Lcg, built: &mut BuiltGraph) {
    let param_ops: Vec<(NodeId, OpKind)> = built
        .ops
        .iter()
        .copied()
        .filter(|(_, k)| k.has_params())
        .collect();

    let do_source_write = param_ops.is_empty() || rng.below(3) == 0;
    if do_source_write {
        let src = built.sources[rng.below(built.sources.len())];
        let x = rng.below(built.tiles_per_axis as usize) as i32;
        let y = rng.below(built.tiles_per_axis as usize) as i32;
        // A fresh generation strictly above any previously used (the
        // monotone-generation contract, §5.3). Step count is small; a
        // large constant base keeps it monotone across steps.
        let gen = 1000 + rng.next_u64() % 1_000_000;
        let tile_seed = rng.next_u64();
        let bytes = rand_tile(rng, tile_seed);
        built
            .g
            .write_source_tile(src, TileCoord { level: 0, x, y }, bytes, gen);
    } else {
        let (node, kind) = param_ops[rng.below(param_ops.len())];
        let params = kind.rand_params(rng);
        // Half the edits go through `gesture` (the ephemeral path), half
        // through `set_params` (the committed path) — both must keep the
        // invariant.
        if rng.below(2) == 0 {
            built.g.gesture(node, params);
        } else {
            built.g.set_params(node, params);
        }
    }
}

// ───────────────────────── PROPERTY 2 ─────────────────────────────

#[test]
fn engine_b_cache_noop_stability_on_identical_param_reset() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    // A small fixed chain: source → linear(a) → linear(b). Both linear
    // nodes carry params, so the re-set no-op is observable at each.
    for case in 0..8u64 {
        let mut rng = Lcg::new(0xCAC4E ^ case);
        let mut g = BufferGraph::new();
        let mut src = SourceData::new();
        for x in 0..2i32 {
            for y in 0..2i32 {
                src.set_tile(
                    TileCoord { level: 0, x, y },
                    rand_tile(&mut rng, (x * 2 + y) as u64 + 1),
                    1,
                );
            }
        }
        let s = g.add_source(src);
        let p1 = MathLinearParams::new(rng.moderate(), rng.unit_signed() * 0.2);
        let p2 = MathLinearParams::new(rng.moderate(), rng.unit_signed() * 0.2);
        let n1 = g.add_op(&MATH_LINEAR, p1.as_bytes().to_vec(), vec![s]);
        let n2 = g.add_op(&MATH_LINEAR, p2.as_bytes().to_vec(), vec![n1]);

        let region = Region::new(0, 0, 2 * TILE, 2 * TILE);

        // Prime.
        let before: Vec<EvaluatedTile> = g.request(n2, region, 0, ctx).unwrap();
        let len1_before = g.cache_len(n1);
        let len2_before = g.cache_len(n2);
        assert!(len2_before > 0, "[case {case}] prime should populate cache");

        // Re-set IDENTICAL bytes on both nodes — a pure no-op by the
        // provenance equality (§8.2): no cache drop, no recompute.
        g.set_params(n1, p1.as_bytes().to_vec());
        g.gesture(n2, p2.as_bytes().to_vec());

        assert_eq!(
            g.cache_len(n1),
            len1_before,
            "[case {case}] identical re-set must not change n1 cache size"
        );
        assert_eq!(
            g.cache_len(n2),
            len2_before,
            "[case {case}] identical re-set must not change n2 cache size"
        );

        // The bytes (and generations) must be stable across the no-op.
        let after = g.request(n2, region, 0, ctx).unwrap();
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.coord, b.coord);
            assert_eq!(
                a.generation, b.generation,
                "[case {case}] no-op changed generation at {:?}",
                a.coord
            );
            assert_eq!(
                &a.bytes[..],
                &b.bytes[..],
                "[case {case}] no-op changed bytes at {:?}",
                a.coord
            );
        }
    }
}

// ───────────────────────── PROPERTY 3 ─────────────────────────────
//
// Mip-equivalence for a POINT kernel (§8.3 `mip_exact`). A point kernel
// is per-texel: it commutes with the level switch. Concretely, with a
// source whose level-1 tile is the CPU 2× box-downsample of its four
// level-0 tiles:
//
//     request(linear, level: 1)  ==  linear applied to source.tile(level 1)
//
// because evaluating the op AT level 1 pulls the level-1 source tile and
// runs the same per-texel kernel. (The DEEPER M3 check is windowed
// σ-halving: a Gaussian at level L must equal the level-0 Gaussian with
// σ/2^L over the downsampled pyramid — that needs the engine to rescale
// windowed params per level, which M2 does not yet do; documented, not
// asserted here.)

#[test]
fn engine_b_mip_exact_point_commutes_with_level_switch() {
    let Some(ctx) = test_device() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };

    for case in 0..8u64 {
        let mut rng = Lcg::new(0x319E_3AC7 ^ case);

        // Build the level-0 quad and its CPU 2× box-downsampled level-1
        // tile (a single 256² tile covering the 512² level-0 extent).
        let mut src = SourceData::new();
        let mut l0: Vec<Vec<Arc<[u8]>>> = Vec::with_capacity(2);
        for x in 0..2usize {
            let mut col = Vec::with_capacity(2);
            for y in 0..2usize {
                let t = rand_tile(&mut rng, (x * 2 + y) as u64 + 7);
                col.push(Arc::clone(&t));
                src.set_tile(
                    TileCoord {
                        level: 0,
                        x: x as i32,
                        y: y as i32,
                    },
                    t,
                    1,
                );
            }
            l0.push(col);
        }
        let l1 = box_downsample_quad(&l0);
        src.set_tile(
            TileCoord {
                level: 1,
                x: 0,
                y: 0,
            },
            Arc::clone(&l1),
            1,
        );

        let mut g = BufferGraph::new();
        let s = g.add_source(src);
        let gain = rng.moderate();
        let bias = rng.unit_signed() * 0.2;
        let p = MathLinearParams::new(gain, bias);
        let op = g.add_op(&MATH_LINEAR, p.as_bytes().to_vec(), vec![s]);

        // request(op, level: 1) over the single level-1 tile.
        let region_l1 = Region::new(0, 0, TILE, TILE);
        let got = g.request(op, region_l1, 1, ctx).unwrap();
        assert_eq!(got.len(), 1, "[case {case}] level-1 extent is one tile");

        // Oracle: run the SAME kernel directly on the level-1 source tile
        // by building a degenerate graph whose only source IS the level-1
        // bytes at level 0, then evaluating at level 0. This isolates the
        // kernel from the level machinery — the level-switch must not
        // change the per-texel math.
        let mut g2 = BufferGraph::new();
        let mut src2 = SourceData::new();
        src2.set_tile(
            TileCoord {
                level: 0,
                x: 0,
                y: 0,
            },
            Arc::clone(&l1),
            1,
        );
        let s2 = g2.add_source(src2);
        let op2 = g2.add_op(&MATH_LINEAR, p.as_bytes().to_vec(), vec![s2]);
        let expected = g2
            .request(op2, Region::new(0, 0, TILE, TILE), 0, ctx)
            .unwrap();

        assert_eq!(
            &got[0].bytes[..],
            &expected[0].bytes[..],
            "[case {case}] point kernel must commute with the mip level switch"
        );
    }
}

/// CPU 2× box-downsample of a 2×2 quad of level-0 tiles into one level-1
/// tile: each level-1 texel is the mean of the 2×2 level-0 texel block it
/// covers. rgba16float in, rgba16float out. The averaging is done in f32
/// then quantized once to f16 (the level-1 tile is the EXACT input the
/// engine's level-1 request reads — so the equivalence is about the level
/// switch, not the downsample math).
fn box_downsample_quad(l0: &[Vec<Arc<[u8]>>]) -> Arc<[u8]> {
    let t = TILE as usize;
    // Assemble the 512×512 level-0 plane in f32 first.
    let big = 2 * t;
    let mut plane = vec![0f32; big * big * 4];
    for (tx, col) in l0.iter().enumerate() {
        for (ty, tile) in col.iter().enumerate() {
            for py in 0..t {
                for px in 0..t {
                    let s = (py * t + px) * 8;
                    let gx = tx * t + px;
                    let gy = ty * t + py;
                    let d = (gy * big + gx) * 4;
                    for c in 0..4 {
                        let bits = u16::from_le_bytes([tile[s + c * 2], tile[s + c * 2 + 1]]);
                        plane[d + c] = half::f16::from_bits(bits).to_f32();
                    }
                }
            }
        }
    }
    // Downsample: each output texel = mean of its 2×2 source block.
    let mut out = vec![0u8; TILE_BYTES];
    for oy in 0..t {
        for ox in 0..t {
            let d = (oy * t + ox) * 8;
            for c in 0..4 {
                let mut acc = 0f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = ox * 2 + dx;
                        let sy = oy * 2 + dy;
                        acc += plane[(sy * big + sx) * 4 + c];
                    }
                }
                let v = half::f16::from_f32(acc / 4.0);
                out[d + c * 2..d + c * 2 + 2].copy_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }
    Arc::from(out.into_boxed_slice())
}
