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

//! The PSD render-level oracle (spec §10.4 oracle 2 / §15 M1 "PSD flatten
//! pipeline → render oracle"; TEST-ONLY, like everything in this crate).
//!
//! Flattens a layered PSD into a single premultiplied-linear RGBA canvas
//! two ways and proves they agree:
//!
//! * [`flatten_reference`] — the scalar spine: decode each layer's
//!   channels, place them in the canvas, premultiply, and composite
//!   bottom-up via [`compose_ref::composite`] (the SAME blend reference
//!   the compose-kernel parity tests consume). This is the golden.
//! * [`flatten_gpu`] — the SAME flatten through Engine A
//!   ([`image_pipeline::Pipeline`]) + the `compose.*` GPU kernels: a chain
//!   of `apply2` compose nodes over per-layer full-canvas premultiplied
//!   leaves. The two agree within the compose kernels' declared f16
//!   tolerance.
//!
//! # Model (the M1 cut)
//!
//! PSD stores layers bottom-most first in a FLAT list with `lsct`
//! section-divider markers for groups (image-psd `model::layers`). The
//! oracle composites that flat list bottom-up over a transparent-black
//! backdrop, mapping each layer's blend-mode fourcc to a [`Blend`]
//! operator via [`Blend::from_psd_key`] and folding `opacity/255`.
//!
//! Deliberately OUT of scope for M1 (documented, not silently dropped):
//!
//! * **Group / divider / folder records** (`lsct` kind 1/2/3) carry no
//!   pixels — they are SKIPPED. Pass-through group blending and isolated
//!   group compositing are M2.
//! * **Clipping** (`clipping == 1`) and **layer masks** (channel ids
//!   -2/-3) are IGNORED: a clipped or masked layer composites as if it
//!   were an ordinary full-coverage layer. Correct clip-group and mask
//!   handling is M2.
//! * **Placement / tiling optimization** — [`flatten_gpu`] builds each
//!   layer as a FULL-CANVAS raw buffer CPU-side and chains whole-canvas
//!   compose dispatches. Per-layer ROI placement (compositing only a
//!   layer's own rect) is the M2 optimization; correctness here does not
//!   depend on it.
//! * **Unknown blend keys** fall back to [`Blend::Normal`] at the call
//!   site (the preservation path keeps the bytes regardless).
//! * Only **8-bit RGB\[A\]** layers are flattened (the M1 fixture corpus);
//!   16/32-bit and CMYK/Lab are the M2 cast/CMS lane.

use image_core::{
    AlphaMode, ChannelLayout, ColorSpaceRef, NamedSpace, PixelFormat, Region, SampleDepth,
    TileCoord, TileData, Transfer,
};
use image_gpu::GpuContext;
use image_kernels::families::compose::{self, ComposeParams};
use image_kernels::KernelDef;
use image_psd::model::{LayerRecord, PsdFile, SectionKind};
use std::sync::Arc;

use crate::compose_ref::{self, Blend};
use crate::Px;

/// A pixel-bearing layer reduced to what the flatten oracle needs: a
/// full-canvas PREMULTIPLIED RGBA f32 buffer (the working space), the
/// mapped blend operator, and the folded opacity. Built once, consumed by
/// both lanes so they composite identical stimulus.
struct LayerPlate {
    /// Canvas-major (`width * height`) premultiplied straight-linear RGBA.
    premul: Vec<Px>,
    blend: Blend,
    /// Layer `opacity / 255` (W3C §5.1 group opacity for a single layer).
    opacity: f32,
}

/// The source pixel format the [`flatten_gpu`] leaves declare: RGBA / F32
/// / PREMULTIPLIED / linear. F32 (not U8) is chosen on purpose — the
/// Engine A M0 decode bridge maps F32 channels to f16 VERBATIM (no `/255`
/// scale, no premultiply, no transfer cast; see
/// `image_pipeline::schedule`), so feeding already-premultiplied f32
/// canvas pixels lands them in the GPU working space bit-faithfully, which
/// is exactly what the `compose.*` kernels expect on `in0`/`in1`.
const PLATE_FMT: PixelFormat = PixelFormat {
    channels: ChannelLayout::Rgba,
    depth: SampleDepth::F32,
    alpha: AlphaMode::Premultiplied,
    transfer: Transfer::Linear,
    space: ColorSpaceRef::Named(NamedSpace::LinearSrgb),
};

/// Is this layer record a group structural marker (divider/folder) with
/// no pixels? Such records carry an `lsct` of kind 1/2/3 and are skipped
/// by the flatten (M1 cut — they hold no channels).
fn is_group_marker(layer: &LayerRecord) -> bool {
    matches!(
        layer.addl.iter().find_map(|a| a.lsct()).map(|d| d.kind),
        Some(SectionKind::OpenFolder)
            | Some(SectionKind::ClosedFolder)
            | Some(SectionKind::BoundingDivider)
    )
}

/// The straight-RGBA-f32 canvas plate for one layer: decode each modeled
/// color channel (ids 0/1/2 = R/G/B, id -1 = alpha), normalize 8-bit
/// samples to `[0,1]`, and place them in the canvas at the layer rect
/// (clipped to the canvas). Pixels outside the layer rect stay transparent
/// black. Missing alpha → opaque (1.0) inside the rect, the PSD convention
/// for a layer with no transparency channel.
fn layer_straight_canvas(file: &PsdFile, layer: &LayerRecord) -> Vec<Px> {
    let cw = file.header.width as i64;
    let ch = file.header.height as i64;
    let depth = file.header.depth;
    let container = file.container;

    // Layer rect (T/L/B/R); a degenerate (empty) rect contributes nothing.
    let lw = (layer.right - layer.left).max(0) as u32;
    let lh = (layer.bottom - layer.top).max(0) as u32;

    // Decode the modeled channels into planar 8-bit buffers, keyed by id.
    // R/G/B start black, alpha starts OPAQUE (a no-alpha layer is opaque
    // inside its rect). Masks (ids -2/-3) are ignored (M1 cut).
    let plane_len = lw as usize * lh as usize;
    let mut r = vec![0u8; plane_len];
    let mut g = vec![0u8; plane_len];
    let mut b = vec![0u8; plane_len];
    let mut a = vec![255u8; plane_len];
    let mut has_pixels = plane_len > 0;
    for (ci, ch_info) in layer.channels.iter().enumerate() {
        let dst = match ch_info.id {
            0 => &mut r,
            1 => &mut g,
            2 => &mut b,
            -1 => &mut a,
            _ => continue, // masks (-2/-3) / extra channels: M1 cut
        };
        // decode() yields rows*cols planar bytes (rows = lh, cols = lw).
        match layer.channel_data[ci].decode(container, lh, lw, depth) {
            Ok(plane) if plane.len() == plane_len => dst.copy_from_slice(&plane),
            _ => has_pixels = false,
        }
    }

    let mut canvas = vec![Px([0.0; 4]); (cw * ch) as usize];
    if !has_pixels {
        return canvas;
    }

    // Place the layer rect into the canvas, clipping to canvas bounds.
    for ly in 0..lh as i64 {
        let dy = layer.top as i64 + ly;
        if dy < 0 || dy >= ch {
            continue;
        }
        for lx in 0..lw as i64 {
            let dx = layer.left as i64 + lx;
            if dx < 0 || dx >= cw {
                continue;
            }
            let si = (ly * lw as i64 + lx) as usize;
            let di = (dy * cw + dx) as usize;
            // STRAIGHT (unpremultiplied) RGBA in [0,1]; the caller
            // premultiplies. 8-bit normalize is `/255` (matches the
            // working-space decode convention).
            canvas[di] = Px([
                r[si] as f32 / 255.0,
                g[si] as f32 / 255.0,
                b[si] as f32 / 255.0,
                a[si] as f32 / 255.0,
            ]);
        }
    }
    canvas
}

/// Premultiply a straight RGBA pixel (rgb·a, alpha through) — the
/// reference-prelude `premul4` semantics, the working-space convention.
fn premultiply(p: Px) -> Px {
    Px([p.0[0] * p.0[3], p.0[1] * p.0[3], p.0[2] * p.0[3], p.0[3]])
}

/// Reduce the PSD's flat layer list (bottom-most first) to the ordered
/// pixel plates the flatten composites, skipping group markers and
/// ignoring clipping/masks (the documented M1 cut). Built once and shared
/// by both lanes so they composite byte-identical stimulus.
fn layer_plates(file: &PsdFile) -> Vec<LayerPlate> {
    let mut plates = Vec::new();
    for layer in &file.layer_mask.layers {
        if is_group_marker(layer) {
            continue;
        }
        let straight = layer_straight_canvas(file, layer);
        let premul: Vec<Px> = straight.into_iter().map(premultiply).collect();
        // Unknown blend keys fall back to Normal (the bytes are preserved
        // by the parser regardless).
        let blend = Blend::from_psd_key(std::str::from_utf8(&layer.blend_key).unwrap_or("norm"))
            .unwrap_or(Blend::Normal);
        plates.push(LayerPlate {
            premul,
            blend,
            opacity: layer.opacity as f32 / 255.0,
        });
    }
    plates
}

/// Flatten `file` with the SCALAR reference (the golden). Returns the
/// premultiplied-linear RGBA canvas, row-major (`width * height`), built
/// bottom-up over a transparent-black backdrop with
/// [`compose_ref::composite`] per layer. This is oracle 2's reference
/// lane; [`flatten_gpu`] is checked against it within the compose kernels'
/// f16 tolerance.
pub fn flatten_reference(file: &PsdFile) -> Vec<Px> {
    let n = file.header.width as usize * file.header.height as usize;
    let mut canvas = vec![Px([0.0; 4]); n]; // transparent black backdrop
    for plate in layer_plates(file) {
        for (bd, &src) in canvas.iter_mut().zip(plate.premul.iter()) {
            *bd = compose_ref::composite(*bd, src, plate.opacity, plate.blend);
        }
    }
    canvas
}

/// The `compose.*` `KernelDef` for a [`Blend`] — joins the blend operator
/// to its GPU kernel by id (the registry dispatch key).
fn compose_kernel(blend: Blend) -> &'static KernelDef {
    match blend {
        Blend::Normal => &compose::COMPOSE_NORMAL,
        Blend::Multiply => &compose::COMPOSE_MULTIPLY,
        Blend::Screen => &compose::COMPOSE_SCREEN,
        Blend::Overlay => &compose::COMPOSE_OVERLAY,
        Blend::Darken => &compose::COMPOSE_DARKEN,
        Blend::Lighten => &compose::COMPOSE_LIGHTEN,
        Blend::ColorDodge => &compose::COMPOSE_COLOR_DODGE,
        Blend::ColorBurn => &compose::COMPOSE_COLOR_BURN,
        Blend::HardLight => &compose::COMPOSE_HARD_LIGHT,
        Blend::SoftLight => &compose::COMPOSE_SOFT_LIGHT,
        Blend::Difference => &compose::COMPOSE_DIFFERENCE,
        Blend::Exclusion => &compose::COMPOSE_EXCLUSION,
        Blend::Hue => &compose::COMPOSE_HUE,
        Blend::Saturation => &compose::COMPOSE_SATURATION,
        Blend::Color => &compose::COMPOSE_COLOR,
        Blend::Luminosity => &compose::COMPOSE_LUMINOSITY,
    }
}

/// A full-canvas plate as the interleaved f32 RGBA bytes a [`RawSource`]
/// of [`PLATE_FMT`] expects (tightly packed, `width * 16` stride).
fn plate_to_f32_bytes(plate: &[Px]) -> Vec<u8> {
    let mut out = Vec::with_capacity(plate.len() * 16);
    for p in plate {
        for c in p.0 {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    out
}

/// Flatten `file` through Engine A + the `compose.*` GPU kernels — the
/// SAME bottom-up composite as [`flatten_reference`], realized as a chain
/// of `apply2` compose nodes over per-layer full-canvas premultiplied
/// leaves. Returns the readback canvas as f32 (widened from the rgba16float
/// working tiles) so it diffs against the reference in the same units.
/// `None` when there are no pixel layers (nothing to dispatch — the caller
/// treats it as the all-transparent canvas).
///
/// The first compose runs the bottom plate over an explicit
/// transparent-black backdrop leaf; each subsequent plate composites over
/// the running result. Per-layer placement/tiling is the M2 optimization
/// (see the module docs); the M1 cut is whole-canvas dispatches, which is
/// honest and correct for the small layered fixtures.
pub fn flatten_gpu(file: &PsdFile, ctx: &GpuContext) -> Option<Vec<Px>> {
    use image_codecs::raw::RawSource;
    use image_pipeline::Pipeline;

    let w = file.header.width;
    let h = file.header.height;
    let plates = layer_plates(file);
    if plates.is_empty() || w == 0 || h == 0 {
        return None;
    }

    let mut pipe = Pipeline::new();

    // Explicit transparent-black backdrop leaf (the canvas starts clear).
    let zero = vec![Px([0.0; 4]); (w * h) as usize];
    let backdrop_src = RawSource::new(
        w,
        h,
        PLATE_FMT,
        plate_to_f32_bytes(&zero).into_boxed_slice(),
    )
    .expect("backdrop raw source");
    let mut acc = pipe.source(Box::new(backdrop_src));

    // Chain: acc = composite(acc, plate) for each plate bottom-up. `apply2`
    // wires (a = backdrop, b = source) — the compose kernel binds in0 = a,
    // in1 = b, matching `compose_ref::composite(a, b, ..)`.
    for plate in &plates {
        let src = RawSource::new(
            w,
            h,
            PLATE_FMT,
            plate_to_f32_bytes(&plate.premul).into_boxed_slice(),
        )
        .expect("layer raw source");
        let leaf = pipe.source(Box::new(src));
        let def = compose_kernel(plate.blend);
        let params = Arc::<[u8]>::from(ComposeParams::new(plate.opacity).as_bytes());
        acc = pipe.apply2(acc, leaf, def, params);
    }

    let roi = Region::new(0, 0, w, h);
    let map = pipe.to_buffer(acc, roi, ctx).expect("flatten pull");

    // Reassemble the canvas from the materialized tiles (the fixtures fit
    // a single tile, but walk the full tile grid for generality). Absent
    // tiles read as the transparent-black background.
    let mut canvas = vec![Px([0.0; 4]); (w * h) as usize];
    for coord in roi.tiles_at(0) {
        let Some(work) = tile_pixel_region(coord).intersect(roi) else {
            continue;
        };
        let Some(tile) = map.get(coord) else { continue };
        let TileData::Heap(bytes) = &tile.data else {
            panic!("flatten_gpu expects heap tiles");
        };
        for ry in 0..work.h as usize {
            for rx in 0..work.w as usize {
                let sp = (ry * work.w as usize + rx) * 8; // rgba16float
                let px = read_working_rgba(bytes, sp);
                let ox = (work.x as usize) + rx;
                let oy = (work.y as usize) + ry;
                canvas[oy * w as usize + ox] = px;
            }
        }
    }
    Some(canvas)
}

/// The pixel rectangle a level-0 tile covers (256² at its grid origin) —
/// mirrors the pipeline sink's tiling so the readback walks the same grid.
fn tile_pixel_region(coord: TileCoord) -> Region {
    use image_core::TILE;
    Region::new(coord.x * TILE as i32, coord.y * TILE as i32, TILE, TILE)
}

/// Read one rgba16float working texel (4 × f16, little-endian) into f32.
fn read_working_rgba(bytes: &[u8], off: usize) -> Px {
    use half::f16;
    let mut px = [0.0f32; 4];
    for (c, slot) in px.iter_mut().enumerate() {
        let o = off + c * 2;
        *slot = f16::from_bits(u16::from_le_bytes([bytes[o], bytes[o + 1]])).to_f32();
    }
    Px(px)
}
