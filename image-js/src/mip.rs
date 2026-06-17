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

//! C-6 (I-06) — the mip-level tile window path. The level-0 window
//! (`DecodedImage::tile_window_rgba8`) is pure CPU windowing; this module
//! resolves `level > 0` windows by routing through Engine B's tiled
//! buffer graph (`image_graph::BufferGraph`): the decoded image becomes a
//! per-level pyramid of rgba16float source tiles, and a requested window
//! is gathered from `(level, coord)` source reads and downconverted back
//! to straight RGBA8.
//!
//! The pyramid is a 2× box-downsample per level, built on the CPU
//! (resampling-on-ingest is inherently-CPU orchestration in this slice —
//! the GPU mip kernel is the M2 follow-up). Source-tile reads need no GPU
//! context (a passthrough source carries no kernel dispatch), so a
//! `level > 0` window resolves without `init_gpu` — the honest subset
//! that lets the editor sample downsampled tiles for zoomed-out frames.
//!
//! rgba16float is the graph's tile format (`TILE*TILE*8` bytes, 4×f16 per
//! pixel); we encode u8/255 on the way in and quantise `clamp·255+0.5` on
//! the way out (the same bridge Engine A's sink uses, mirrored here
//! because that helper is private to image-pipeline).

use half::f16;
use image_core::{Region, TileCoord, TILE};
use image_graph::{BufferGraph, SourceData};

/// A decoded image's Engine-B mip pyramid: a `BufferGraph` whose single
/// source node holds rgba16float tiles for levels 0..=`max_level`, plus
/// the per-level pixel dimensions for windowing math.
pub struct MipPyramid {
    graph: BufferGraph,
    source: image_graph::NodeId,
    /// (width, height) at each level, index = level.
    level_dims: Vec<(u32, u32)>,
}

impl MipPyramid {
    /// Build the pyramid from a straight-RGBA8 image. `max_level` is the
    /// highest mip level to materialise (0 = level-0 only); each level
    /// halves the resolution (rounded up, min 1) by a 2× box filter. A
    /// level whose dimension has collapsed to 1×1 is the natural top.
    pub fn build(width: u32, height: u32, rgba: &[u8], max_level: u8) -> Self {
        let mut graph = BufferGraph::new();
        let mut data = SourceData::new();
        let mut level_dims = Vec::new();

        // Level 0 is the source pixels verbatim; higher levels are box-
        // downsampled from the previous level's RGBA8.
        let mut cur: Vec<u8> = rgba.to_vec();
        let (mut cw, mut ch) = (width, height);
        for level in 0..=max_level {
            level_dims.push((cw, ch));
            store_level_tiles(&mut data, level, cw, ch, &cur);
            if cw <= 1 && ch <= 1 {
                break; // pyramid top reached
            }
            let (nw, nh, next) = downsample_2x(cw, ch, &cur);
            cur = next;
            cw = nw;
            ch = nh;
        }

        let source = graph.add_source(data);
        MipPyramid {
            graph,
            source,
            level_dims,
        }
    }

    /// The highest level the pyramid materialised.
    pub fn max_level(&self) -> u8 {
        (self.level_dims.len() as u8).saturating_sub(1)
    }

    /// (width, height) at `level`, or `None` if the level was not built.
    pub fn level_dims(&self, level: u8) -> Option<(u32, u32)> {
        self.level_dims.get(level as usize).copied()
    }

    /// Resolve a window `(x, y, w, h)` at `level` to tightly-packed
    /// straight RGBA8 (`w'*h'*4`, clipped to the level's extent), routing
    /// through Engine B's source tiles + the f16→u8 downconvert. Returns
    /// `(bytes, w', h')`; an empty `Vec` with `(0, 0)` when the window
    /// lies fully outside the level. No GPU context required (a source
    /// read is a passthrough — `image_graph::BufferGraph::read_source_tile`).
    pub fn window_rgba8(&self, level: u8, x: u32, y: u32, w: u32, h: u32) -> (Vec<u8>, u32, u32) {
        let Some((lw, lh)) = self.level_dims(level) else {
            return (Vec::new(), 0, 0);
        };
        let x0 = x.min(lw);
        let y0 = y.min(lh);
        let x1 = x.saturating_add(w).min(lw);
        let y1 = y.saturating_add(h).min(lh);
        if x1 <= x0 || y1 <= y0 {
            return (Vec::new(), 0, 0);
        }
        let (tw, th) = (x1 - x0, y1 - y0);
        let mut out = vec![0u8; (tw as usize) * (th as usize) * 4];

        // Which source tiles cover the clipped window? Gather each from the
        // graph, downconvert, and blit the overlapping rows into `out`.
        let region = Region::new(x0 as i32, y0 as i32, tw, th);
        for coord in region.tiles_at(level) {
            let Some(tile_f16) = self.graph.read_source_tile(self.source, coord) else {
                continue; // not a source node — unreachable for this pyramid
            };
            let tile_rgba8 = f16_tile_to_rgba8(&tile_f16);
            blit_tile_window(&tile_rgba8, coord, &mut out, tw, th, x0, y0, x1, y1);
        }
        (out, tw, th)
    }
}

/// Store a level's RGBA8 pixels as 256×256 rgba16float source tiles. Edge
/// tiles are padded with transparent black beyond the level extent (the
/// sparse-canvas rule); only the in-extent pixels are written.
fn store_level_tiles(data: &mut SourceData, level: u8, w: u32, h: u32, rgba: &[u8]) {
    let t = TILE;
    let tiles_x = w.div_ceil(t);
    let tiles_y = h.div_ceil(t);
    let stride = w as usize * 4;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut tile = vec![0u8; (TILE * TILE * 8) as usize];
            for row in 0..t {
                let sy = ty * t + row;
                if sy >= h {
                    break;
                }
                for col in 0..t {
                    let sx = tx * t + col;
                    if sx >= w {
                        break;
                    }
                    let src = sy as usize * stride + sx as usize * 4;
                    let dst = (row as usize * TILE as usize + col as usize) * 8;
                    for ch in 0..4 {
                        let v = f16::from_f32(rgba[src + ch] as f32 / 255.0);
                        let b = v.to_le_bytes();
                        tile[dst + ch * 2] = b[0];
                        tile[dst + ch * 2 + 1] = b[1];
                    }
                }
            }
            data.set_tile(
                TileCoord {
                    level,
                    x: tx as i32,
                    y: ty as i32,
                },
                tile.into_boxed_slice(),
                0,
            );
        }
    }
}

/// Box-downsample a straight-RGBA8 image by 2× (each output pixel is the
/// average of its 2×2 source block; right/bottom odd edges average the
/// available 1–2 samples). Returns `(new_w, new_h, pixels)`.
fn downsample_2x(w: u32, h: u32, rgba: &[u8]) -> (u32, u32, Vec<u8>) {
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let stride = w as usize * 4;
    let mut out = vec![0u8; nw as usize * nh as usize * 4];
    for oy in 0..nh {
        for ox in 0..nw {
            // The 2×2 source block (clamped to the source extent).
            let sx0 = (ox * 2).min(w - 1);
            let sy0 = (oy * 2).min(h - 1);
            let sx1 = (ox * 2 + 1).min(w - 1);
            let sy1 = (oy * 2 + 1).min(h - 1);
            let corners = [(sx0, sy0), (sx1, sy0), (sx0, sy1), (sx1, sy1)];
            for ch in 0..4 {
                let mut acc = 0u32;
                for &(sx, sy) in &corners {
                    acc += rgba[sy as usize * stride + sx as usize * 4 + ch] as u32;
                }
                out[(oy as usize * nw as usize + ox as usize) * 4 + ch] = (acc / 4) as u8;
            }
        }
    }
    (nw, nh, out)
}

/// Downconvert a single rgba16float tile (`TILE*TILE*8` bytes) to straight
/// RGBA8 (`TILE*TILE*4`), quantising each f16 channel as
/// `clamp(0,1)·255 + 0.5` — the same rule Engine A's sink uses.
fn f16_tile_to_rgba8(tile: &[u8]) -> Vec<u8> {
    let px = (TILE * TILE) as usize;
    let mut out = vec![0u8; px * 4];
    for i in 0..px {
        for ch in 0..4 {
            let off = i * 8 + ch * 2;
            let v = f16::from_le_bytes([tile[off], tile[off + 1]]).to_f32();
            out[i * 4 + ch] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    out
}

/// Blit a source tile's overlap into the output window. `coord` is the
/// tile's grid coord at the level; `out` is the `out_w`×`out_h` window
/// covering the level-pixel rectangle `[win_x0, win_x1) × [win_y0, win_y1)`.
/// Only pixels inside that rectangle (already clamped to the level extent
/// by the caller) are copied, so padded edge-tile pixels are skipped.
#[allow(clippy::too_many_arguments)]
fn blit_tile_window(
    tile_rgba8: &[u8],
    coord: TileCoord,
    out: &mut [u8],
    out_w: u32,
    out_h: u32,
    win_x0: u32,
    win_y0: u32,
    win_x1: u32,
    win_y1: u32,
) {
    let _ = out_h; // window height is implied by [win_y0, win_y1)
    let t = TILE as i64;
    let tile_px_x = coord.x as i64 * t;
    let tile_px_y = coord.y as i64 * t;
    for row in 0..TILE {
        let py = tile_px_y + row as i64;
        if py < win_y0 as i64 || py >= win_y1 as i64 {
            continue;
        }
        let oy = (py - win_y0 as i64) as usize;
        for col in 0..TILE {
            let pxs = tile_px_x + col as i64;
            if pxs < win_x0 as i64 || pxs >= win_x1 as i64 {
                continue;
            }
            let ox = (pxs - win_x0 as i64) as usize;
            let src = (row as usize * TILE as usize + col as usize) * 4;
            let dst = (oy * out_w as usize + ox) * 4;
            out[dst..dst + 4].copy_from_slice(&tile_rgba8[src..src + 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid-colour RGBA8 image (every pixel the same), so downsampling
    /// preserves the colour exactly and assertions are unambiguous.
    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for px in v.chunks_exact_mut(4) {
            px.copy_from_slice(&rgba);
        }
        v
    }

    // feat: image.editor.tiles (the mip-level window path).

    /// A level-1 window of a solid image resolves to the same solid colour
    /// at half the dimensions — the Engine-B source read + f16 round-trip
    /// is colour-preserving for a uniform image.
    #[test]
    fn image_editor_tiles_level1_window_of_solid_is_preserved() {
        let color = [200u8, 100, 50, 255];
        let pyr = MipPyramid::build(8, 8, &solid(8, 8, color), 3);
        assert_eq!(pyr.level_dims(0), Some((8, 8)));
        assert_eq!(pyr.level_dims(1), Some((4, 4)), "level 1 halves to 4x4");

        let (bytes, w, h) = pyr.window_rgba8(1, 0, 0, 4, 4);
        assert_eq!((w, h), (4, 4));
        assert_eq!(bytes.len(), 4 * 4 * 4);
        for px in bytes.chunks_exact(4) {
            // f16(v/255) round-trip is exact for the 8-bit codes here.
            assert_eq!(px, color, "solid colour preserved through level 1");
        }
    }

    /// A level-2 window clips to the level-2 extent (2x2 here) and the
    /// solid colour survives the two box-downsample steps.
    #[test]
    fn image_editor_tiles_level2_window_clips_and_preserves() {
        let color = [10u8, 20, 30, 255];
        let pyr = MipPyramid::build(8, 8, &solid(8, 8, color), 4);
        assert_eq!(pyr.level_dims(2), Some((2, 2)));
        // Ask for a 4x4 window at level 2; it clips to the 2x2 extent.
        let (bytes, w, h) = pyr.window_rgba8(2, 0, 0, 4, 4);
        assert_eq!((w, h), (2, 2), "level-2 extent is 2x2");
        for px in bytes.chunks_exact(4) {
            assert_eq!(px, color);
        }
    }

    /// A window fully outside the level extent is an empty miss, never a
    /// torn buffer.
    #[test]
    fn image_editor_tiles_level_window_outside_is_empty() {
        let pyr = MipPyramid::build(8, 8, &solid(8, 8, [1, 2, 3, 4]), 2);
        let (bytes, w, h) = pyr.window_rgba8(1, 100, 100, 4, 4);
        assert!(bytes.is_empty());
        assert_eq!((w, h), (0, 0));
    }

    /// A level beyond the pyramid top is an empty miss.
    #[test]
    fn image_editor_tiles_level_above_top_is_empty() {
        let pyr = MipPyramid::build(4, 4, &solid(4, 4, [9, 9, 9, 255]), 1);
        assert_eq!(pyr.max_level(), 1);
        let (bytes, _w, _h) = pyr.window_rgba8(5, 0, 0, 2, 2);
        assert!(bytes.is_empty(), "level 5 was never built");
    }

    /// A two-colour split (left half A, right half B) downsamples its
    /// boundary column to the average — proving the box filter actually
    /// blends rather than nearest-sampling.
    #[test]
    fn image_editor_tiles_box_filter_blends_a_boundary() {
        // 2x1 image: pixel0 = (0,..), pixel1 = (255,..). 2x downsample to
        // 1x1 averages to ~128.
        let img = vec![0u8, 0, 0, 255, 255, 0, 0, 255];
        let pyr = MipPyramid::build(2, 1, &img, 1);
        let (bytes, w, h) = pyr.window_rgba8(1, 0, 0, 1, 1);
        assert_eq!((w, h), (1, 1));
        // (0 + 255 + 0 + 255)/4 = 127 (the 2x2 block clamps the 1-tall
        // image's missing rows to the existing row).
        assert_eq!(bytes[0], 127, "boundary column averaged");
        assert_eq!(bytes[3], 255, "alpha preserved");
    }
}
