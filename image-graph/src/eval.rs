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

//! Demand-driven, generation-keyed evaluation (§8.2). `request` walks
//! upstream computing exactly the requested tiles, serving from the
//! per-node cache when an entry's recorded input generations still
//! match. `evaluate_from_scratch` is the same walk with the cache
//! bypassed — the oracle for the incremental-correctness gate (§12.4).

use std::sync::Arc;

use image_core::{ParamsHash, Region, TileCoord, TILE};
use image_gpu::{execute_tile_once, execute_windowed_once, GpuContext, TileInput};
use image_kernels::KernelClass;

use crate::cache::CachedTile;
use crate::graph::{BufferGraph, GraphError, Node, NodeId};

const TILE_BYTES: usize = (TILE * TILE * 8) as usize; // rgba16float

/// One evaluated tile handed back to the caller.
#[derive(Debug, Clone)]
pub struct EvaluatedTile {
    pub coord: TileCoord,
    pub bytes: Arc<[u8]>,
    pub generation: u64,
}

impl BufferGraph {
    /// Evaluate the tiles covering `region` at mip `level` for `node`,
    /// using (and populating) the per-node caches. The interactive path.
    pub fn request(
        &mut self,
        node: NodeId,
        region: Region,
        level: u8,
        ctx: &GpuContext,
    ) -> Result<Vec<EvaluatedTile>, GraphError> {
        let coords: Vec<TileCoord> = region.tiles_at(level).collect();
        let mut out = Vec::with_capacity(coords.len());
        for c in coords {
            out.push(self.eval_tile(node, c, ctx, true)?);
        }
        Ok(out)
    }

    /// The same evaluation with every cache bypassed — the from-scratch
    /// oracle. `request` MUST equal this after any damage sequence
    /// (the §12.4 incremental-correctness invariant).
    pub fn evaluate_from_scratch(
        &mut self,
        node: NodeId,
        region: Region,
        level: u8,
        ctx: &GpuContext,
    ) -> Result<Vec<EvaluatedTile>, GraphError> {
        let coords: Vec<TileCoord> = region.tiles_at(level).collect();
        let mut out = Vec::with_capacity(coords.len());
        for c in coords {
            out.push(self.eval_tile(node, c, ctx, false)?);
        }
        Ok(out)
    }

    fn eval_tile(
        &mut self,
        node: NodeId,
        coord: TileCoord,
        ctx: &GpuContext,
        use_cache: bool,
    ) -> Result<EvaluatedTile, GraphError> {
        match &self.nodes[node] {
            Node::Source { data } => {
                let (bytes, generation) = match data.tile(coord) {
                    Some((b, g)) => (Arc::clone(b), g),
                    // Unallocated source tiles read as transparent black
                    // (the sparse-canvas rule, §5.3); generation 0.
                    None => (Arc::from(vec![0u8; TILE_BYTES].into_boxed_slice()), 0),
                };
                Ok(EvaluatedTile {
                    coord,
                    bytes,
                    generation,
                })
            }
            Node::Op { kernel, inputs, .. } => {
                let kernel = *kernel;
                let inputs = inputs.clone();

                // Gather inputs first (records their generations).
                let input_tiles = self.gather_inputs(&inputs, kernel, coord, ctx, use_cache)?;
                let input_gens: Vec<u64> = input_tiles.iter().map(|t| t.generation).collect();
                let params_hash = self.op_params_hash(node);

                if use_cache {
                    if let Node::Op { cache, .. } = &self.nodes[node] {
                        if let Some(hit) = cache.get(coord, params_hash, &input_gens) {
                            return Ok(EvaluatedTile {
                                coord,
                                bytes: Arc::clone(&hit.bytes),
                                generation: hit.generation,
                            });
                        }
                    }
                }

                let bytes = self.dispatch(node, kernel, coord, &input_tiles, ctx)?;
                // Bump generation on every recompute so downstream caches
                // observe the change. Derive deterministically from the
                // provenance so from-scratch and incremental agree on the
                // generation too (not just the bytes).
                let generation = derive_generation(params_hash, &input_gens);
                let bytes: Arc<[u8]> = Arc::from(bytes.into_boxed_slice());

                if let Node::Op { cache, .. } = &mut self.nodes[node] {
                    cache.put(
                        coord,
                        CachedTile {
                            bytes: Arc::clone(&bytes),
                            generation,
                        },
                        params_hash,
                        input_gens,
                    );
                }
                Ok(EvaluatedTile {
                    coord,
                    bytes,
                    generation,
                })
            }
        }
    }

    /// Pull the input tile(s) a kernel needs to produce `coord`. Point
    /// kernels need the same coord; windowed kernels need the
    /// neighbourhood gathered into one expanded window buffer (the
    /// engine concern, §6.1).
    fn gather_inputs(
        &mut self,
        inputs: &[NodeId],
        kernel: &'static image_kernels::KernelDef,
        coord: TileCoord,
        ctx: &GpuContext,
        use_cache: bool,
    ) -> Result<Vec<GatheredInput>, GraphError> {
        let mut out = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            match kernel.class {
                KernelClass::Windowed { radius } => {
                    out.push(self.gather_window(inp, coord, radius, ctx, use_cache)?);
                }
                _ => {
                    let t = self.eval_tile(inp, coord, ctx, use_cache)?;
                    out.push(GatheredInput {
                        bytes: t.bytes,
                        generation: t.generation,
                        w: TILE,
                        h: TILE,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Gather a `radius`-expanded window around `coord` from input
    /// `inp`, evaluating each contributing tile (so the window's
    /// generation folds every source tile it touches — the §8.2
    /// windowed-damage-inflation made concrete on the pull side).
    fn gather_window(
        &mut self,
        inp: NodeId,
        coord: TileCoord,
        radius: (u16, u16),
        ctx: &GpuContext,
        use_cache: bool,
    ) -> Result<GatheredInput, GraphError> {
        let (rx, ry) = (radius.0 as i64, radius.1 as i64);
        let win_w = TILE as i64 + 2 * rx;
        let win_h = TILE as i64 + 2 * ry;
        // Window origin in pixel space: tile origin minus the radius.
        let ox = coord.x as i64 * TILE as i64 - rx;
        let oy = coord.y as i64 * TILE as i64 - ry;

        let mut win = vec![0u8; (win_w * win_h * 8) as usize];
        let mut gen_acc: u64 = 0;
        // Which source tiles cover [ox, ox+win_w) × [oy, oy+win_h)?
        let t = TILE as i64;
        let tx0 = ox.div_euclid(t);
        let ty0 = oy.div_euclid(t);
        let tx1 = (ox + win_w - 1).div_euclid(t);
        let ty1 = (oy + win_h - 1).div_euclid(t);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let src = self.eval_tile(
                    inp,
                    TileCoord {
                        level: coord.level,
                        x: tx as i32,
                        y: ty as i32,
                    },
                    ctx,
                    use_cache,
                )?;
                gen_acc = gen_acc
                    .wrapping_mul(1099511628211)
                    .wrapping_add(src.generation);
                blit_into_window(&src.bytes, tx * t, ty * t, &mut win, win_w, win_h, ox, oy);
            }
        }
        Ok(GatheredInput {
            bytes: Arc::from(win.into_boxed_slice()),
            generation: gen_acc,
            w: win_w as u32,
            h: win_h as u32,
        })
    }

    fn dispatch(
        &self,
        node: NodeId,
        kernel: &'static image_kernels::KernelDef,
        _coord: TileCoord,
        inputs: &[GatheredInput],
        ctx: &GpuContext,
    ) -> Result<Vec<u8>, GraphError> {
        let params = self.op_params_bytes(node);
        match kernel.class {
            KernelClass::Windowed { .. } => {
                let win = &inputs[0];
                Ok(execute_windowed_once(
                    ctx, kernel, &win.bytes, win.w, win.h, &params, None, TILE, TILE,
                )?)
            }
            _ => {
                let tile_inputs: Vec<TileInput<'_>> = inputs
                    .iter()
                    .map(|i| TileInput {
                        f16_bytes: &i.bytes,
                    })
                    .collect();
                Ok(execute_tile_once(
                    ctx,
                    kernel,
                    &tile_inputs,
                    &params,
                    None,
                    TILE,
                    TILE,
                )?)
            }
        }
    }

    fn op_params_hash(&self, node: NodeId) -> ParamsHash {
        match &self.nodes[node] {
            Node::Op { params_hash, .. } => *params_hash,
            _ => ParamsHash(0),
        }
    }

    fn op_params_bytes(&self, node: NodeId) -> Vec<u8> {
        match &self.nodes[node] {
            Node::Op { params, .. } => params.to_vec(),
            _ => Vec::new(),
        }
    }
}

struct GatheredInput {
    bytes: Arc<[u8]>,
    generation: u64,
    w: u32,
    h: u32,
}

/// Deterministic output generation from the provenance: equal provenance
/// ⇒ equal generation, so incremental and from-scratch agree on
/// generations (not just bytes) — the property the gate compares.
fn derive_generation(params_hash: ParamsHash, input_gens: &[u64]) -> u64 {
    let mut h = params_hash.0 ^ 0x9e37_79b9_7f4a_7c15;
    for &g in input_gens {
        h = h.wrapping_mul(1099511628211).wrapping_add(g);
    }
    h
}

/// Copy a source tile's rows into the expanded window buffer, clipping
/// to the window bounds. Out-of-window source pixels are skipped;
/// uncovered window pixels stay zero (transparent black edge rule).
#[allow(clippy::too_many_arguments)]
fn blit_into_window(
    src: &[u8],
    src_ox: i64,
    src_oy: i64,
    win: &mut [u8],
    win_w: i64,
    win_h: i64,
    win_ox: i64,
    win_oy: i64,
) {
    let t = TILE as i64;
    for sy in 0..t {
        let wy = src_oy + sy - win_oy;
        if wy < 0 || wy >= win_h {
            continue;
        }
        for sx in 0..t {
            let wx = src_ox + sx - win_ox;
            if wx < 0 || wx >= win_w {
                continue;
            }
            let s = ((sy * t + sx) * 8) as usize;
            let d = ((wy * win_w + wx) * 8) as usize;
            win[d..d + 8].copy_from_slice(&src[s..s + 8]);
        }
    }
}
