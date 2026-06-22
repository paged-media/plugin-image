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

//! The single-threaded decode→upload→dispatch→readback bridge (spec
//! §7.1). Production overlaps codec decode (CPU workers) with kernel
//! execution (GPU queue) as a producer/consumer bridge; M0 is honest
//! and runs the stages in sequence on the calling thread — the
//! wasm-bindgen-rayon worker pool is BREAKAGE I-02, and batched dispatch
//! is the GPU agent's lane (this calls `execute_tile_once`, the simplest
//! correct realization of the same ABI).
//!
//! Region recursion: each node materializes the requested ROI by tiling
//! it (`Region::tiles_at`), propagating the required input ROI upstream
//! per `KernelClass` (`region_prop`), then producing one
//! `TileData::Heap` tile per covered coord (generation 0 — M0 has no
//! mutation, so nothing bumps generations yet). The per-node result is
//! memoized in the `OperationCache` keyed on
//! `(op id, ParamsHash, input ContentHash)`; the input hash is the
//! content hash of the upstream node's materialized result, so a re-pull
//! of an unchanged subtree is a cache hit.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use half::f16;
use image_codecs::ImageSource;
use image_core::{
    ChannelLayout, ContentHash, PixelFormat, Region, SampleDepth, Tile, TileCoord, TileData,
    TileMap, TileSliceMut, TILE,
};
use image_gpu::{execute_tile_once, execute_tile_once_async, GpuContext, TileInput};

use crate::cache::{OpKey, OperationCache};
use crate::node::{ApplyNode, OpNode};
use crate::region_prop::required_input_roi;
use crate::{NodeId, PipelineError};

/// Bytes per pixel of the rgba16float working format the GPU ABI
/// consumes (4 channels × 2 bytes).
const WORKING_BPP: usize = 8;

/// The boxed recursion shape of [`materialize_node_async`] (an async fn
/// cannot recurse unboxed; deliberately non-`Send` — see its docs).
type MaterializeFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(TileMap, ContentHash), PipelineError>> + 'a>>;

/// Materialize `node`'s output over `roi`, consulting/filling the cache.
/// Returns the node's `TileMap` plus its content hash (the cache-key
/// input component for any downstream node).
pub(crate) fn materialize_node(
    nodes: &[OpNode],
    cache: &mut OperationCache,
    node_id: NodeId,
    roi: Region,
    ctx: &GpuContext,
) -> Result<(TileMap, ContentHash), PipelineError> {
    let node = nodes
        .get(node_id.0)
        .ok_or_else(|| PipelineError::Graph(format!("dangling node {node_id:?}")))?;

    // The input-hash component of the cache key. A leaf source binds its
    // own decoded identity (op id ⊕ requested ROI); an apply node binds
    // its upstream content hash(es) — BOTH inputs for a binary kernel, so
    // a re-pull is a hit only when neither subtree changed. All computed
    // BEFORE the heavy work so a hit short-circuits it.
    let (input_hash, materialized_input) = match node {
        OpNode::Source(_) => (source_identity_hash(node, roi), None),
        OpNode::Apply(apply) => {
            let in_roi = required_input_roi(apply.def.class, roi);
            let (a_id, b_id) = apply.inputs.as_pair();
            let (a_map, a_hash) = materialize_node(nodes, cache, a_id, in_roi, ctx)?;
            match b_id {
                None => (a_hash, Some((a_map, None))),
                Some(b) => {
                    let (b_map, b_hash) = materialize_node(nodes, cache, b, in_roi, ctx)?;
                    (fold_hashes(a_hash, b_hash), Some((a_map, Some(b_map))))
                }
            }
        }
    };

    let key = OpKey {
        op_id: node.op_key(),
        params: params_hash(node),
        input: input_hash,
    };

    if let Some(hit) = cache.get(key) {
        return Ok((hit.clone(), content_hash_of(hit)));
    }

    let map = match node {
        OpNode::Source(_) => materialize_source(node, roi)?,
        OpNode::Apply(apply) => {
            let (a_map, b_map) = materialized_input.unwrap();
            materialize_apply(apply, a_map, b_map, roi, ctx)?
        }
    };
    let out_hash = content_hash_of(&map);
    cache.insert(key, map.clone());
    Ok((map, out_hash))
}

/// [`materialize_node`]'s ASYNC twin (the wasm32/WebGPU lane, where a
/// blocking readback poll cannot pump the map callback) — keep the two
/// bodies in LOCKSTEP. Recursion is boxed (an async fn cannot recurse
/// unboxed); the future is deliberately non-`Send` (it holds `&mut
/// OperationCache` across awaits and runs on one thread in both realms —
/// pollster natively, the browser microtask queue on wasm). Output is
/// byte-for-byte the sync lane's (the conformance async-parity test).
pub(crate) fn materialize_node_async<'a>(
    nodes: &'a [OpNode],
    cache: &'a mut OperationCache,
    node_id: NodeId,
    roi: Region,
    ctx: &'a GpuContext,
) -> MaterializeFuture<'a> {
    Box::pin(async move {
        let node = nodes
            .get(node_id.0)
            .ok_or_else(|| PipelineError::Graph(format!("dangling node {node_id:?}")))?;

        let (input_hash, materialized_input) = match node {
            OpNode::Source(_) => (source_identity_hash(node, roi), None),
            OpNode::Apply(apply) => {
                let in_roi = required_input_roi(apply.def.class, roi);
                let (a_id, b_id) = apply.inputs.as_pair();
                let (a_map, a_hash) =
                    materialize_node_async(nodes, cache, a_id, in_roi, ctx).await?;
                match b_id {
                    None => (a_hash, Some((a_map, None))),
                    Some(b) => {
                        let (b_map, b_hash) =
                            materialize_node_async(nodes, cache, b, in_roi, ctx).await?;
                        (fold_hashes(a_hash, b_hash), Some((a_map, Some(b_map))))
                    }
                }
            }
        };

        let key = OpKey {
            op_id: node.op_key(),
            params: params_hash(node),
            input: input_hash,
        };

        if let Some(hit) = cache.get(key) {
            return Ok((hit.clone(), content_hash_of(hit)));
        }

        let map = match node {
            OpNode::Source(_) => materialize_source(node, roi)?,
            OpNode::Apply(apply) => {
                let (a_map, b_map) = materialized_input.unwrap();
                materialize_apply_async(apply, a_map, b_map, roi, ctx).await?
            }
        };
        let out_hash = content_hash_of(&map);
        cache.insert(key, map.clone());
        Ok((map, out_hash))
    })
}

/// Decode a source leaf over `roi` and bridge its straight pixels to
/// rgba16float working bytes — one heap tile per covered coord, each
/// sized to the tile's WORK extent (`tile ∩ roi`) and zero-padded where
/// the image bounds fall short. Sizing every tile to `tile ∩ roi` is
/// the M0 invariant that lets a pointwise apply read its input tile and
/// dispatch over the SAME extent (sub-tile ROIs are the common case
/// until strip planning lands in M1).
fn materialize_source(node: &OpNode, roi: Region) -> Result<TileMap, PipelineError> {
    let OpNode::Source(src) = node else {
        unreachable!("materialize_source on non-source node")
    };
    let shrink = src.decode_shrink.max(1);
    let mut guard = src
        .source
        .lock()
        .map_err(|_| PipelineError::Graph("source mutex poisoned".into()))?;
    let info = guard.probe()?;
    // Post-shrink bounds: the decode coordinate space the pipeline tiles
    // over (shrink == 1 for PNG/raw today, so this is the native extent).
    let bounds = Region::new(
        0,
        0,
        info.width.div_ceil(shrink),
        info.height.div_ceil(shrink),
    );

    let mut map = TileMap::new(PixelFormat::GPU_WORKING);
    for coord in roi.tiles_at(0) {
        let Some(work) = tile_work_region(coord, roi) else {
            continue;
        };
        // Decode only the in-bounds sub-rect; the remainder of `work`
        // stays transparent-black background (the §5.3 sparse default).
        let mut f16_bytes = vec![0u8; (work.w as usize * work.h as usize) * WORKING_BPP];
        if let Some(decode_roi) = work.intersect(bounds) {
            decode_tile_to_working(
                &mut **guard,
                &info.format,
                shrink,
                work,
                decode_roi,
                &mut f16_bytes,
            )?;
        }
        map.insert(coord, heap_tile(f16_bytes));
    }
    Ok(map)
}

/// Run a unary OR binary kernel over `roi` against the already-
/// materialized input map(s): dispatch per covered tile at the tile's
/// work extent. `b` is `Some` exactly for binary kernels (`apply2`, the
/// compose lane); its tiles are passed as the second `TileInput`.
fn materialize_apply(
    apply: &ApplyNode,
    a: TileMap,
    b: Option<TileMap>,
    roi: Region,
    ctx: &GpuContext,
) -> Result<TileMap, PipelineError> {
    let arity = apply.def.inputs;
    let provided = if b.is_some() { 2 } else { 1 };
    if arity as usize != provided {
        // Generator (0) is T2; a kernel/wiring arity mismatch is a graph
        // construction error (use `apply` for unary, `apply2` for binary).
        return Err(PipelineError::Graph(format!(
            "kernel {} arity {} wired with {provided} input(s)",
            apply.def.id, arity
        )));
    }

    let mut map = TileMap::new(PixelFormat::GPU_WORKING);
    for coord in roi.tiles_at(0) {
        let Some(work) = tile_work_region(coord, roi) else {
            continue;
        };
        // Pointwise (Point class): the input footprint per tile is the
        // tile itself. An absent upstream tile reads as transparent-black
        // background, so the kernel runs over a zeroed input of the same
        // extent.
        let zero_len = (work.w as usize * work.h as usize) * WORKING_BPP;
        let zeros_a;
        let a_bytes: &[u8] = match a.get(coord) {
            Some(tile) => heap_bytes(tile)?,
            None => {
                zeros_a = vec![0u8; zero_len];
                &zeros_a
            }
        };

        // Build the input list: one for unary, two for binary. The second
        // input's absent tiles also read as zeroed background.
        let zeros_b;
        let inputs: Vec<TileInput> = match &b {
            None => vec![TileInput { f16_bytes: a_bytes }],
            Some(b_map) => {
                let b_bytes: &[u8] = match b_map.get(coord) {
                    Some(tile) => heap_bytes(tile)?,
                    None => {
                        zeros_b = vec![0u8; zero_len];
                        &zeros_b
                    }
                };
                vec![
                    TileInput { f16_bytes: a_bytes },
                    TileInput { f16_bytes: b_bytes },
                ]
            }
        };

        let out_bytes = execute_tile_once(
            ctx,
            apply.def,
            &inputs,
            &apply.params,
            None, // constant-1 mask: the Engine A binding (§6.1)
            work.w,
            work.h,
        )?;
        map.insert(coord, heap_tile(out_bytes));
    }
    Ok(map)
}

/// [`materialize_apply`]'s ASYNC twin — keep in LOCKSTEP (the only
/// divergence is the awaited dispatch).
async fn materialize_apply_async(
    apply: &ApplyNode,
    a: TileMap,
    b: Option<TileMap>,
    roi: Region,
    ctx: &GpuContext,
) -> Result<TileMap, PipelineError> {
    let arity = apply.def.inputs;
    let provided = if b.is_some() { 2 } else { 1 };
    if arity as usize != provided {
        return Err(PipelineError::Graph(format!(
            "kernel {} arity {} wired with {provided} input(s)",
            apply.def.id, arity
        )));
    }

    let mut map = TileMap::new(PixelFormat::GPU_WORKING);
    for coord in roi.tiles_at(0) {
        let Some(work) = tile_work_region(coord, roi) else {
            continue;
        };
        let zero_len = (work.w as usize * work.h as usize) * WORKING_BPP;
        let zeros_a;
        let a_bytes: &[u8] = match a.get(coord) {
            Some(tile) => heap_bytes(tile)?,
            None => {
                zeros_a = vec![0u8; zero_len];
                &zeros_a
            }
        };

        let zeros_b;
        let inputs: Vec<TileInput> = match &b {
            None => vec![TileInput { f16_bytes: a_bytes }],
            Some(b_map) => {
                let b_bytes: &[u8] = match b_map.get(coord) {
                    Some(tile) => heap_bytes(tile)?,
                    None => {
                        zeros_b = vec![0u8; zero_len];
                        &zeros_b
                    }
                };
                vec![
                    TileInput { f16_bytes: a_bytes },
                    TileInput { f16_bytes: b_bytes },
                ]
            }
        };

        let out_bytes = execute_tile_once_async(
            ctx,
            apply.def,
            &inputs,
            &apply.params,
            None, // constant-1 mask: the Engine A binding (§6.1)
            work.w,
            work.h,
        )
        .await?;
        map.insert(coord, heap_tile(out_bytes));
    }
    Ok(map)
}

/// Fold two input content hashes into one cache-key component for a
/// binary apply, order-sensitively (`a + b` and `b + a` are different
/// graphs and must key differently). FNV-1a over the two `u64`s.
fn fold_hashes(a: ContentHash, b: ContentHash) -> ContentHash {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&a.0.to_le_bytes());
    bytes[8..].copy_from_slice(&b.0.to_le_bytes());
    ContentHash::of(&bytes)
}

/// `ParamsHash` for a node — sources carry no params (zero), apply nodes
/// the precomputed hash of their param block.
fn params_hash(node: &OpNode) -> image_core::ParamsHash {
    match node {
        OpNode::Source(_) => image_core::ParamsHash(0),
        OpNode::Apply(a) => a.params_hash,
    }
}

/// A source leaf's cache-key input component: its stable op id mixed
/// with the requested ROI. Same leaf + same ROI ⇒ same key ⇒ a re-pull
/// hits (M0 sources are immutable; a mutable leaf would fold a content
/// generation in here — that lands with Engine B).
fn source_identity_hash(node: &OpNode, roi: Region) -> ContentHash {
    let OpNode::Source(src) = node else {
        unreachable!()
    };
    let mut bytes = Vec::with_capacity(8 + 16);
    bytes.extend_from_slice(&src.op_id.to_le_bytes());
    bytes.extend_from_slice(&roi.x.to_le_bytes());
    bytes.extend_from_slice(&roi.y.to_le_bytes());
    bytes.extend_from_slice(&roi.w.to_le_bytes());
    bytes.extend_from_slice(&roi.h.to_le_bytes());
    ContentHash::of(&bytes)
}

/// Content hash of a materialized map: FNV-1a over each tile's coord +
/// bytes in sorted-coord order (deterministic across HashMap iteration).
fn content_hash_of(map: &TileMap) -> ContentHash {
    let mut coords: Vec<TileCoord> = map.iter().map(|(c, _)| *c).collect();
    coords.sort_unstable();
    let mut acc = Vec::new();
    for c in coords {
        acc.extend_from_slice(&c.level.to_le_bytes());
        acc.extend_from_slice(&c.x.to_le_bytes());
        acc.extend_from_slice(&c.y.to_le_bytes());
        if let Some(tile) = map.get(c) {
            if let TileData::Heap(bytes) = &tile.data {
                acc.extend_from_slice(bytes);
            }
        }
    }
    ContentHash::of(&acc)
}

/// The pixel rectangle a level-0 tile covers (256² at its grid origin).
fn tile_pixel_region(coord: TileCoord) -> Region {
    Region::new(coord.x * TILE as i32, coord.y * TILE as i32, TILE, TILE)
}

/// A tile's WORK extent: the part of the tile inside the requested ROI.
/// `None` when the tile lies fully outside (it then contributes nothing).
fn tile_work_region(coord: TileCoord, roi: Region) -> Option<Region> {
    tile_pixel_region(coord).intersect(roi)
}

/// Decode `decode_roi` from `source` and bridge its straight pixels into
/// `out` (the working buffer for the full `work` extent), placing each
/// decoded pixel at its position within `work`. Pixels of `work` outside
/// `decode_roi` are left as the caller initialized them (zeros).
///
/// M0 BRIDGE: this maps decoded U8/F32 channels straight to f16 verbatim
/// (U8 via `/255`), with NO premultiply and NO transfer/colorspace cast.
/// Correct transfer + premultiply + CMS handling is the M1 cms/cast lane
/// (BREAKAGE I-02); the working `PixelFormat` already declares
/// premultiplied-linear, so this bridge is deliberately a placeholder
/// that round-trips bit-faithfully for the conformance gradient input.
fn decode_tile_to_working(
    source: &mut dyn ImageSource,
    src_format: &PixelFormat,
    shrink: u32,
    work: Region,
    decode_roi: Region,
    out: &mut [u8],
) -> Result<(), PipelineError> {
    let bpp = src_format.bytes_per_pixel();
    let mut raw = vec![0u8; decode_roi.w as usize * decode_roi.h as usize * bpp];
    let row_stride = decode_roi.w as usize * bpp;
    {
        let mut slice = TileSliceMut {
            region: decode_roi,
            format: *src_format,
            bytes: &mut raw,
            row_stride,
        };
        // `decode_roi` is already in post-shrink coordinates (the bounds
        // and tiling were computed there), matching the `read_region`
        // contract; the decoder applies the native `shrink` itself.
        source.read_region(decode_roi, shrink, &mut slice)?;
    }

    let channels = src_format.channels;
    let depth = src_format.depth;
    // Pixel offset of `decode_roi` within `work`.
    let dx = (decode_roi.x - work.x) as usize;
    let dy = (decode_roi.y - work.y) as usize;
    for ry in 0..decode_roi.h as usize {
        for rx in 0..decode_roi.w as usize {
            let rgba = read_pixel_rgba(
                &raw,
                (ry * decode_roi.w as usize + rx) * bpp,
                channels,
                depth,
            )?;
            let wp = (dy + ry) * work.w as usize + (dx + rx);
            for (c, v) in rgba.into_iter().enumerate() {
                let bits = f16::from_f32(v).to_bits().to_le_bytes();
                let o = wp * WORKING_BPP + c * 2;
                out[o] = bits[0];
                out[o + 1] = bits[1];
            }
        }
    }
    Ok(())
}

/// Read one interleaved source pixel as straight rgba f32 (U8 normalized
/// to [0,1]). Only the formats the M0 raw bridge produces are handled;
/// richer layouts (CMYK, gray, U16) are the M1 cast lane.
fn read_pixel_rgba(
    raw: &[u8],
    off: usize,
    channels: ChannelLayout,
    depth: SampleDepth,
) -> Result<[f32; 4], PipelineError> {
    match (channels, depth) {
        (ChannelLayout::Rgba, SampleDepth::U8) => Ok([
            raw[off] as f32 / 255.0,
            raw[off + 1] as f32 / 255.0,
            raw[off + 2] as f32 / 255.0,
            raw[off + 3] as f32 / 255.0,
        ]),
        (ChannelLayout::Rgba, SampleDepth::F32) => {
            let f = |i: usize| {
                f32::from_le_bytes(raw[off + i * 4..off + i * 4 + 4].try_into().unwrap())
            };
            Ok([f(0), f(1), f(2), f(3)])
        }
        other => Err(PipelineError::Graph(format!(
            "M0 decode bridge handles RGBA U8/F32 only, got {other:?} (cast lane is M1)"
        ))),
    }
}

/// Borrow a heap tile's bytes; M0 only ever produces `Heap` tiles, so a
/// GPU/swapped tile here is an internal invariant violation.
fn heap_bytes(tile: &Tile) -> Result<&[u8], PipelineError> {
    match &tile.data {
        TileData::Heap(bytes) => Ok(bytes),
        _ => Err(PipelineError::Graph(
            "M0 pipeline materializes heap tiles only".into(),
        )),
    }
}

/// Wrap working-space bytes as a generation-0 heap tile (§5.3).
fn heap_tile(bytes: Vec<u8>) -> Arc<Tile> {
    Arc::new(Tile {
        format: PixelFormat::GPU_WORKING,
        data: TileData::Heap(Arc::from(bytes.into_boxed_slice())),
        generation: 0,
    })
}
