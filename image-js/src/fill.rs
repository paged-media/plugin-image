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

//! FILL-THE-SELECTION — the generator family's editor reach (`gen.*`).
//!
//! # Why this is a DESTRUCTIVE step and not an adjust-chain stage
//!
//! The adjust chain ([`crate::ingest::adjust_rgba8`]) is a re-runnable
//! function of the DECODED source: the panel's params are the state and
//! every Apply recomputes from scratch. A generator is not a function of
//! the source at all — it REPLACES pixels. Threading it into the chain
//! would mean the chain carried an ordered, unbounded list of paint
//! operations (a document model this plugin does not own — the layer
//! graph is a recorded deferral). So the honest v0 is what the task
//! allows: an EXPLICIT command (`…command.fillSelection`) that applies
//! the generator ONCE into the working image and swaps the engine-held
//! source, exactly like the crop/resize commits already do. Undo is the
//! host's (the document is untouched); re-ingesting the frame restores
//! the original pixels.
//!
//! # How the selection mask is honoured (two GPU passes, no CPU blend)
//!
//! The generator kernels are `KernelClass::Generator` — they ignore
//! `in0` and derive the texel's GLOBAL coordinate from `gid + (ox, oy)`.
//! Under `Pipeline` that global coordinate would restart per TILE (the
//! params are per NODE, not per tile), so the fill does NOT ride the
//! tiled pipeline: it runs as two WHOLE-IMAGE dispatches through
//! `execute_tile_once_async` (the `resize_image` precedent — one
//! dispatch, `ox = oy = 0`, so `gid` IS the image coordinate):
//!
//! 1. **generate** — the `gen.*` kernel, mask `None` (the generator must
//!    paint everywhere; masking it here would multiply the fill by the
//!    coverage BEFORE the composite and double-darken feathered edges).
//! 2. **composite** — `compose.normal` with `in0` = the ORIGINAL image,
//!    `in1` = the generated field, opacity 1, and the selection coverage
//!    bound at `@group(2)`. The ABI's `mix(a, result, m)` then yields
//!    exactly `mix(original, generated, coverage)`: the fill lands inside
//!    the selection, feathered edges blend once, and outside the
//!    selection the original survives bit-for-bit.
//!
//! No selection ⇒ mask `None` ⇒ the whole image is filled.
//!
//! # Honest v0 scope
//!
//! * **Two stops only.** The gradient kernels take exactly `c0`/`c1`; a
//!   multi-stop ramp needs either a LUT kernel or an N-stop param block —
//!   neither exists. The panel exposes two colour wells and says so.
//! * **Geometry is derived, not dragged.** The gradient axis/centre/
//!   radius come from the SELECTION BOUNDS (the whole image when there is
//!   no selection) — see [`FillGeometry`]. There is no on-canvas gradient
//!   drag handle yet.
//! * **Noise is monochrome + opaque** (`gen.noise` emits `(v, v, v, 1)`);
//!   `amount` scales the hash amplitude, and the seed is caller-supplied
//!   so a repeat is reproducible.

use std::sync::Arc;

use half::f16;
use image_core::Region;
use image_gpu::{GpuContext, SelectionCoverage, TileInput};
use image_kernels::families::compose::{ComposeParams, COMPOSE_NORMAL};
use image_kernels::families::gen::{
    GenAngularGradientParams, GenDiamondGradientParams, GenLinearGradientParams, GenNoiseParams,
    GenRadialGradientParams, GenReflectedGradientParams, GEN_ANGULAR_GRADIENT,
    GEN_DIAMOND_GRADIENT, GEN_LINEAR_GRADIENT, GEN_NOISE, GEN_RADIAL_GRADIENT,
    GEN_REFLECTED_GRADIENT,
};

use crate::ingest::{DecodedImage, IngestError};

/// Which gradient generator a [`FillSpec::Gradient`] dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientKind {
    Linear,
    Radial,
    Angular,
    Reflected,
    Diamond,
}

impl GradientKind {
    /// Decode the wire name (mirrored by the TS `GradientKind` union).
    pub fn from_wire(s: &str) -> Option<GradientKind> {
        Some(match s {
            "linear" => GradientKind::Linear,
            "radial" => GradientKind::Radial,
            "angular" => GradientKind::Angular,
            "reflected" => GradientKind::Reflected,
            "diamond" => GradientKind::Diamond,
            _ => return None,
        })
    }

    pub fn as_wire(self) -> &'static str {
        match self {
            GradientKind::Linear => "linear",
            GradientKind::Radial => "radial",
            GradientKind::Angular => "angular",
            GradientKind::Reflected => "reflected",
            GradientKind::Diamond => "diamond",
        }
    }
}

/// What to paint into the selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillSpec {
    /// A fixed TWO-STOP gradient; colours are straight (non-premultiplied)
    /// RGBA in `[0, 1]` — they are premultiplied here for the kernel.
    Gradient {
        kind: GradientKind,
        c0: [f32; 4],
        c1: [f32; 4],
    },
    /// Deterministic monochrome noise (`hash(x, y, seed) · amount`).
    Noise { amount: f32, seed: u32 },
}

/// The pixel-space frame the gradient geometry is derived from: the
/// selection's bounding box, or the whole image when nothing is
/// selected. Kept separate so the derivation is unit-testable without a
/// GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillGeometry {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl FillGeometry {
    /// The frame for `image` under `selection` — the coverage's non-zero
    /// bounding box when there is one (and it is non-degenerate), else
    /// the full image extent.
    pub fn derive(image: &DecodedImage, selection: Option<&SelectionCoverage>) -> FillGeometry {
        let full = FillGeometry {
            x: 0.0,
            y: 0.0,
            w: image.width as f32,
            h: image.height as f32,
        };
        match selection.and_then(|c| c.bounds()) {
            Some(r) if r.w > 0 && r.h > 0 => FillGeometry {
                x: r.x as f32,
                y: r.y as f32,
                w: r.w as f32,
                h: r.h as f32,
            },
            _ => full,
        }
    }

    fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    /// Half the longer side — the radius/scale the radial and diamond
    /// ramps span (so the ramp reaches the frame's far edge).
    fn half_extent(&self) -> f32 {
        self.w.max(self.h).max(1.0) / 2.0
    }
}

/// Premultiply a straight RGBA colour (the generator params are
/// premultiplied by the family's contract).
fn premul(c: [f32; 4]) -> [f32; 4] {
    [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]]
}

/// Straight RGBA8 → the rgba16float working window (`/255`, the I-02
/// bridge the decode + resize lanes already use). Shared with the
/// straighten lane (`ingest::straighten_crop_rgba8`) — one bridge, one
/// rounding rule.
pub(crate) fn rgba8_to_f16(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() * 2);
    for &b in rgba {
        out.extend_from_slice(&f16::from_f32(b as f32 / 255.0).to_le_bytes());
    }
    out
}

/// rgba16float → straight RGBA8 (clamped + rounded, the resize lane's
/// downconvert). Shared with the straighten lane.
pub(crate) fn f16_to_rgba8(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let v = f16::from_le_bytes([pair[0], pair[1]]).to_f32();
        out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    out
}

/// Paint `spec` into `image` through `selection` (the whole image when
/// `None`) and return the new straight RGBA8. GPU-only — the generator
/// AND the composite are registered WGSL kernels; there is no CPU blend
/// path (spec §6). See the module docs for the two-pass shape.
pub async fn fill_rgba8(
    ctx: &GpuContext,
    image: &DecodedImage,
    spec: &FillSpec,
    selection: Option<Arc<SelectionCoverage>>,
) -> Result<Vec<u8>, IngestError> {
    let (w, h) = (image.width, image.height);
    if w == 0 || h == 0 {
        return Err(IngestError::Unsupported("fill on an empty image".into()));
    }
    let geom = FillGeometry::derive(image, selection.as_deref());
    let src_f16 = rgba8_to_f16(&image.rgba);

    // ── pass 1: generate (unmasked, whole image, ox = oy = 0) ────────
    let gen_f16 = match *spec {
        FillSpec::Gradient { kind, c0, c1 } => {
            let (a, b) = (premul(c0), premul(c1));
            let (cx, cy) = geom.center();
            match kind {
                GradientKind::Linear => {
                    // Left → right across the frame, vertically centred.
                    let p =
                        GenLinearGradientParams::new(0, 0, geom.x, cy, geom.x + geom.w, cy, a, b);
                    dispatch_gen(ctx, &GEN_LINEAR_GRADIENT, p.as_bytes(), &src_f16, w, h).await?
                }
                GradientKind::Radial => {
                    let p = GenRadialGradientParams::new(0, 0, cx, cy, geom.half_extent(), a, b);
                    dispatch_gen(ctx, &GEN_RADIAL_GRADIENT, p.as_bytes(), &src_f16, w, h).await?
                }
                GradientKind::Angular => {
                    let p = GenAngularGradientParams::new(0, 0, cx, cy, 0.0, a, b);
                    dispatch_gen(ctx, &GEN_ANGULAR_GRADIENT, p.as_bytes(), &src_f16, w, h).await?
                }
                GradientKind::Reflected => {
                    // Mirrored about the frame CENTRE, reaching the right edge.
                    let p =
                        GenReflectedGradientParams::new(0, 0, cx, cy, geom.x + geom.w, cy, a, b);
                    dispatch_gen(ctx, &GEN_REFLECTED_GRADIENT, p.as_bytes(), &src_f16, w, h).await?
                }
                GradientKind::Diamond => {
                    let p =
                        GenDiamondGradientParams::new(0, 0, cx, cy, 0.0, geom.half_extent(), a, b);
                    dispatch_gen(ctx, &GEN_DIAMOND_GRADIENT, p.as_bytes(), &src_f16, w, h).await?
                }
            }
        }
        FillSpec::Noise { amount, seed } => {
            let p = GenNoiseParams::new(0, 0, seed, amount);
            dispatch_gen(ctx, &GEN_NOISE, p.as_bytes(), &src_f16, w, h).await?
        }
    };

    // ── pass 2: composite the generated field over the original through
    // the selection mask (`mix(a, normal(a, b), m)`; the fill is opaque,
    // so inside the selection the result IS the generated field) ─────
    let mask = selection.map(|c| c.mask_window_f16(Region::new(0, 0, w, h)));
    let out_f16 = image_gpu::execute_tile_once_async(
        ctx,
        &COMPOSE_NORMAL,
        &[
            TileInput {
                f16_bytes: &src_f16,
            },
            TileInput {
                f16_bytes: &gen_f16,
            },
        ],
        ComposeParams::new(1.0).as_bytes(),
        mask.as_deref(),
        w,
        h,
    )
    .await
    .map_err(|e| IngestError::Pipeline(e.to_string()))?;

    Ok(f16_to_rgba8(&out_f16))
}

/// One whole-image generator dispatch. The `in0` window is the source
/// image only because the family's zero-input convention wires the
/// generators as UNARY (`inputs: 1`) — the shaders never sample it.
async fn dispatch_gen(
    ctx: &GpuContext,
    def: &'static image_kernels::KernelDef,
    params: &[u8],
    src_f16: &[u8],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, IngestError> {
    image_gpu::execute_tile_once_async(
        ctx,
        def,
        &[TileInput { f16_bytes: src_f16 }],
        params,
        None,
        w,
        h,
    )
    .await
    .map_err(|e| IngestError::Pipeline(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> DecodedImage {
        DecodedImage::from_rgba8(w, h, vec![0u8; (w * h * 4) as usize]).expect("valid")
    }

    // feat: image.editor.generate — the fill geometry falls back to the
    // whole image when nothing is selected.
    #[test]
    fn image_editor_generate_geometry_defaults_to_the_whole_image() {
        let g = FillGeometry::derive(&img(8, 4), None);
        assert_eq!(
            g,
            FillGeometry {
                x: 0.0,
                y: 0.0,
                w: 8.0,
                h: 4.0
            }
        );
    }

    #[test]
    fn image_editor_generate_geometry_follows_the_selection_bounds() {
        // Coverage selects the 2×2 block at (1, 1) of a 4×4 field.
        let mut data = vec![0u8; 16];
        for y in 1..3 {
            for x in 1..3 {
                data[y * 4 + x] = 255;
            }
        }
        let cov = SelectionCoverage::from_data(4, 4, data).expect("16 px");
        let g = FillGeometry::derive(&img(4, 4), Some(&cov));
        assert_eq!(
            g,
            FillGeometry {
                x: 1.0,
                y: 1.0,
                w: 2.0,
                h: 2.0
            }
        );
        assert_eq!(g.center(), (2.0, 2.0));
        assert_eq!(g.half_extent(), 1.0);
    }

    #[test]
    fn image_editor_generate_geometry_ignores_an_empty_selection() {
        // An explicit-but-EMPTY selection has no bounds → whole image
        // (never a zero-size gradient frame).
        let cov = SelectionCoverage::empty(4, 4);
        let g = FillGeometry::derive(&img(4, 4), Some(&cov));
        assert_eq!(g.w, 4.0);
        assert_eq!(g.h, 4.0);
    }

    #[test]
    fn image_editor_generate_gradient_kind_round_trips_its_wire_name() {
        for k in [
            GradientKind::Linear,
            GradientKind::Radial,
            GradientKind::Angular,
            GradientKind::Reflected,
            GradientKind::Diamond,
        ] {
            assert_eq!(GradientKind::from_wire(k.as_wire()), Some(k));
        }
        assert_eq!(GradientKind::from_wire("spiral"), None);
    }

    #[test]
    fn image_editor_generate_premultiplies_the_stop_colors() {
        // The generator family's contract is PREMULTIPLIED endpoint
        // colours; a half-transparent white must arrive as 0.5 rgb.
        assert_eq!(premul([1.0, 1.0, 1.0, 0.5]), [0.5, 0.5, 0.5, 0.5]);
        assert_eq!(premul([0.2, 0.4, 0.6, 1.0]), [0.2, 0.4, 0.6, 1.0]);
    }

    #[test]
    fn image_editor_generate_f16_bridge_round_trips_bytes() {
        let src = vec![0u8, 51, 128, 255];
        let back = f16_to_rgba8(&rgba8_to_f16(&src));
        assert_eq!(back, src, "the /255 → f16 → ×255 bridge is lossless at u8");
    }
}
