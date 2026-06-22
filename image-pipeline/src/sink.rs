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

//! Sinks (spec §7.3). `to_buffer` materializes a region into a `TileMap`
//! (feeds Engine B and the SDK texture-provider contract); `to_encoder`
//! streams a region into an `ImageTarget` strip-by-strip — the ONLY
//! structured GPU-readback path in the system. `to_pyramid` is M2.
//!
//! The sink is the demand origin: it partitions the requested ROI into
//! work units (`Region::tiles_at`) and pulls the graph for them. For
//! `to_buffer` the partition is implicit — `materialize_node` tiles the
//! ROI itself — so it is a thin entry point that returns the assembled
//! per-node `TileMap`. `to_encoder` makes the partition explicit: it
//! walks the ROI in tile-row STRIPS (full ROI width × ≤256px high),
//! pulls each strip through the same demand path, converts the
//! rgba16float working pixels down to the requested 8-bit format, and
//! writes the strips in top-to-bottom order.

use half::f16;
use image_codecs::{EncodedStats, ImageTarget, TargetInfo};
use image_core::{
    ChannelLayout, PixelFormat, Region, SampleDepth, Tile, TileCoord, TileData, TileMap,
    TileSliceRef, TILE,
};
use image_gpu::GpuContext;

use crate::cache::OperationCache;
use crate::node::OpNode;
use crate::schedule::{materialize_node, materialize_node_async};
use crate::{NodeId, PipelineError};

/// Bytes per pixel of the rgba16float working format (4 × f16).
const WORKING_BPP: usize = 8;

/// Materialize `node` over `roi` into a `TileMap` of rgba16float heap
/// tiles. Empty ROIs yield an empty map (no work, no tiles).
pub(crate) fn to_buffer(
    nodes: &[OpNode],
    cache: &mut OperationCache,
    node: NodeId,
    roi: Region,
    ctx: &GpuContext,
) -> Result<TileMap, PipelineError> {
    if roi.is_empty() {
        return Ok(TileMap::new(image_core::PixelFormat::GPU_WORKING));
    }
    let (map, _hash) = materialize_node(nodes, cache, node, roi, ctx)?;
    Ok(map)
}

/// [`to_buffer`]'s ASYNC twin (the wasm32/WebGPU readback lane) — keep
/// in LOCKSTEP.
pub(crate) async fn to_buffer_async(
    nodes: &[OpNode],
    cache: &mut OperationCache,
    node: NodeId,
    roi: Region,
    ctx: &GpuContext,
) -> Result<TileMap, PipelineError> {
    if roi.is_empty() {
        return Ok(TileMap::new(image_core::PixelFormat::GPU_WORKING));
    }
    let (map, _hash) = materialize_node_async(nodes, cache, node, roi, ctx).await?;
    Ok(map)
}

/// Stream `node`'s output over `roi` into `target`, strip-by-strip
/// (spec §7.3). See [`crate::Pipeline::to_encoder`] for the contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn to_encoder(
    nodes: &[OpNode],
    cache: &mut OperationCache,
    node: NodeId,
    roi: Region,
    ctx: &GpuContext,
    target: &mut dyn ImageTarget,
    fmt: PixelFormat,
) -> Result<EncodedStats, PipelineError> {
    if fmt.depth != SampleDepth::U8 {
        return Err(PipelineError::Graph(format!(
            "to_encoder is U8 only (got {:?}); the depth cast lane is M1",
            fmt.depth
        )));
    }
    let out_bpp = fmt.bytes_per_pixel();

    target
        .begin(TargetInfo {
            width: roi.w,
            height: roi.h,
            format: fmt,
            icc: None,
        })
        .map_err(PipelineError::Codec)?;

    // Tile-row strips, top to bottom. The first strip's height is the
    // distance to the next tile-grid boundary below `roi.y`, so every
    // later strip starts tile-aligned (256px tall except a short last
    // one) — the demand pull then materializes whole tile rows.
    let mut y = roi.y;
    let bottom = roi.bottom();
    while (y as i64) < bottom {
        let next_grid = next_tile_boundary(y);
        let strip_bottom = next_grid.min(bottom);
        let h = (strip_bottom - y as i64) as u32;
        let strip_roi = Region::new(roi.x, y, roi.w, h);

        let (map, _hash) = materialize_node(nodes, cache, node, strip_roi, ctx)?;
        let strip = convert_strip(&map, strip_roi, fmt, out_bpp)?;

        // The target's strip coordinates are target-LOCAL (origin 0,0 at
        // the ROI top-left); the demand pull used graph coordinates.
        let local = Region::new(0, y - roi.y, roi.w, h);
        let slice = TileSliceRef {
            region: local,
            format: fmt,
            bytes: &strip,
            row_stride: roi.w as usize * out_bpp,
        };
        target
            .write_strip(local, &slice)
            .map_err(PipelineError::Codec)?;

        y = strip_bottom as i32;
    }

    target.finish().map_err(PipelineError::Codec)
}

/// [`to_encoder`]'s ASYNC twin (the wasm32/WebGPU readback lane) — keep
/// in LOCKSTEP with the sync strip walk.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn to_encoder_async(
    nodes: &[OpNode],
    cache: &mut OperationCache,
    node: NodeId,
    roi: Region,
    ctx: &GpuContext,
    target: &mut dyn ImageTarget,
    fmt: PixelFormat,
) -> Result<EncodedStats, PipelineError> {
    if fmt.depth != SampleDepth::U8 {
        return Err(PipelineError::Graph(format!(
            "to_encoder is U8 only (got {:?}); the depth cast lane is M1",
            fmt.depth
        )));
    }
    let out_bpp = fmt.bytes_per_pixel();

    target
        .begin(TargetInfo {
            width: roi.w,
            height: roi.h,
            format: fmt,
            icc: None,
        })
        .map_err(PipelineError::Codec)?;

    let mut y = roi.y;
    let bottom = roi.bottom();
    while (y as i64) < bottom {
        let next_grid = next_tile_boundary(y);
        let strip_bottom = next_grid.min(bottom);
        let h = (strip_bottom - y as i64) as u32;
        let strip_roi = Region::new(roi.x, y, roi.w, h);

        let (map, _hash) = materialize_node_async(nodes, cache, node, strip_roi, ctx).await?;
        let strip = convert_strip(&map, strip_roi, fmt, out_bpp)?;

        let local = Region::new(0, y - roi.y, roi.w, h);
        let slice = TileSliceRef {
            region: local,
            format: fmt,
            bytes: &strip,
            row_stride: roi.w as usize * out_bpp,
        };
        target
            .write_strip(local, &slice)
            .map_err(PipelineError::Codec)?;

        y = strip_bottom as i32;
    }

    target.finish().map_err(PipelineError::Codec)
}

/// The next tile-grid row boundary strictly below `y` (so a strip that
/// starts mid-tile is clipped to the grid; later strips start aligned).
fn next_tile_boundary(y: i32) -> i64 {
    let t = TILE as i64;
    (y as i64).div_euclid(t) * t + t
}

/// Assemble `strip_roi` from the materialized `map` and convert the
/// rgba16float working pixels into a tightly packed `fmt` (U8) strip
/// buffer, dropping the working alpha for alpha-less layouts.
fn convert_strip(
    map: &TileMap,
    strip_roi: Region,
    fmt: PixelFormat,
    out_bpp: usize,
) -> Result<Vec<u8>, PipelineError> {
    let out_channels = fmt.channels.count() as usize;
    let mut out = vec![0u8; strip_roi.w as usize * strip_roi.h as usize * out_bpp];

    for coord in strip_roi.tiles_at(0) {
        let Some(work) = tile_pixel_region(coord).intersect(strip_roi) else {
            continue;
        };
        // Absent tiles read as the transparent-black background — already
        // the zero the buffer is initialized to, so skip them.
        let Some(tile) = map.get(coord) else { continue };
        let bytes = heap_bytes(tile)?;
        for ry in 0..work.h as usize {
            for rx in 0..work.w as usize {
                // Position inside the source tile's work extent.
                let sp = (ry * work.w as usize + rx) * WORKING_BPP;
                let rgba = read_working_rgba(bytes, sp);
                // Destination position inside the strip buffer.
                let ox = (work.x - strip_roi.x) as usize + rx;
                let oy = (work.y - strip_roi.y) as usize + ry;
                let op = (oy * strip_roi.w as usize + ox) * out_bpp;
                write_u8_pixel(&mut out[op..op + out_bpp], rgba, fmt.channels, out_channels);
            }
        }
    }
    Ok(out)
}

/// Read one rgba16float working texel (4 × f16) into straight f32.
fn read_working_rgba(bytes: &[u8], off: usize) -> [f32; 4] {
    let mut px = [0.0f32; 4];
    for (c, slot) in px.iter_mut().enumerate() {
        let o = off + c * 2;
        *slot = f16::from_bits(u16::from_le_bytes([bytes[o], bytes[o + 1]])).to_f32();
    }
    px
}

/// Convert a working rgba f32 pixel to the requested U8 layout. The
/// working space is premultiplied-linear; M1's CMS/cast lane (BREAKAGE
/// I-02) owns transfer + unpremultiply, so this is the verbatim
/// `clamp(x, 0, 1) * 255` round mirroring the decode-bridge policy on the
/// way back out. Alpha-less layouts drop the working alpha channel.
fn write_u8_pixel(dst: &mut [u8], rgba: [f32; 4], channels: ChannelLayout, out_channels: usize) {
    let q = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8 };
    match channels {
        ChannelLayout::Rgba => {
            dst[0] = q(rgba[0]);
            dst[1] = q(rgba[1]);
            dst[2] = q(rgba[2]);
            dst[3] = q(rgba[3]);
        }
        ChannelLayout::Gray => {
            // Luma from the RGB working channels (alpha dropped).
            dst[0] = q(rgba[0]);
        }
        ChannelLayout::GrayA => {
            dst[0] = q(rgba[0]);
            dst[1] = q(rgba[3]);
        }
        // CMYK has no working-space round-trip here (the cast/CMS lane is
        // M1); fall back to copying the leading channels so the buffer is
        // never left uninitialized.
        ChannelLayout::Cmyk | ChannelLayout::Cmyka => {
            for (i, slot) in dst.iter_mut().enumerate().take(out_channels.min(4)) {
                *slot = q(rgba[i.min(3)]);
            }
        }
    }
}

/// The pixel rectangle a level-0 tile covers (256² at its grid origin).
fn tile_pixel_region(coord: TileCoord) -> Region {
    Region::new(coord.x * TILE as i32, coord.y * TILE as i32, TILE, TILE)
}

/// Borrow a heap tile's bytes; the demand pull only produces `Heap`
/// tiles, so any other residency here is an internal invariant violation.
fn heap_bytes(tile: &Tile) -> Result<&[u8], PipelineError> {
    match &tile.data {
        TileData::Heap(bytes) => Ok(bytes),
        _ => Err(PipelineError::Graph(
            "to_encoder reads heap tiles only".into(),
        )),
    }
}
