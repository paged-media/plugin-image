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

//! The stroke COMPOSITOR — where a stroke's coverage becomes pixels.
//!
//! GPU-ONLY (spec §6/§9), and with NO new kernel: the whole of painting
//! falls out of the frozen mask ABI plus kernels that already ship.
//! Given a window of the untouched base image and the effective stroke
//! coverage from [`crate::dab::StrokeAccumulator::mask_window_f16`]:
//!
//! ## Paint ([`PaintMode::Paint`]) — four registered dispatches
//!
//! 1. `gen.solid` → a field of the PREMULTIPLIED brush colour.
//! 2. `cast.premultiply` → the base window, alpha-associated.
//! 3. `compose.<blend>` → `in0` = the premultiplied base, `in1` = the
//!    colour field, `opacity` = 1, **mask = the effective coverage**.
//! 4. `cast.unpremultiply` → back to the straight working space.
//!
//! The identity that makes step 3 exactly right for ALL 26 blend modes:
//! the compose module computes `result = over(a, b·α)` and the ABI then
//! stores `mix(a, result, m)`. Expanding the source-over spine,
//!
//! ```text
//! mix(a, over(a, b, α), m) ≡ over(a, b, α·m)
//! ```
//!
//! — both the alpha and the three colour terms agree identically. So
//! binding the coverage as the mask composites the dab at effective
//! alpha `coverage · opacity`, with the tip's antialiased rim and any
//! feathered selection edge blending ONCE, in the right place, under
//! whichever blend mode the user picked. `image-conformance`'s
//! `brush_stroke` suite proves it on the device.
//!
//! ## Erase ([`PaintMode::Erase`]) — ONE registered dispatch
//!
//! `band.set_alpha(alpha = 0)` produces `(a.rgb, 0)`, and the ABI stores
//! `mix(a, (a.rgb, 0), m)` = `(a.rgb, a.a·(1 − m))`. In the straight
//! (non-premultiplied) working space this is precisely destination-out:
//! alpha is scaled down by the coverage and the RGB is preserved, so
//! partially erased pixels keep their colour instead of decaying toward
//! black. No premultiply bracket is needed — or wanted, since
//! `unpremultiply` maps zero alpha to zero RGB and would discard the
//! colour under a fully erased pixel.
//!
//! ## Filter strokes ([`PaintMode::Filter`], [`PaintMode::Unsharp`])
//!
//! The dodge, burn, sponge, blur and sharpen brushes deposit no paint at
//! all: the dab's coverage MASKS A KERNEL'S EFFECT on the window. A point
//! kernel (`adjust.dodge_burn`) is one masked dispatch — the ABI already
//! stores `mix(a, f(a), m)`. Blur and sharpen are the existing unsharp
//! chain: `conv.gaussian_h` → `conv.gaussian_v` → `conv.unsharp` with
//! the original window as `in0`, the blur as `in1` and the coverage as
//! the mask, so `mix(a, a + amount·(a − blur), m)`. A NEGATIVE amount of
//! −1 is exactly the blur (`a − (a − blur) = blur`), which is why the blur
//! brush needs no kernel of its own and why its mask lands on the
//! ORIGINAL window rather than on an intermediate pass. The Gaussian is
//! windowed, so the caller hands over a window with a halo of the blur's
//! radius and crops the result.
//!
//! ## The premultiply bracket, and when it is skipped
//!
//! The engine's working buffers are STRAIGHT RGBA (the decode bridge
//! maps u8 verbatim, `/255`, with no premultiply), while the compose
//! family's contract is premultiplied on both inputs. For an opaque
//! image the two coincide and nothing is at stake; over a PNG with
//! alpha — or over pixels the eraser has just made translucent — they do
//! not, and compositing straight bytes as if they were premultiplied
//! gives the wrong backdrop colour. Steps 2 and 4 are what make
//! brush-after-erase correct.
//!
//! They are also two GPU round-trips, and per-dispatch latency is what
//! painting is bounded by (~1.8 ms per dispatch on the reference Metal
//! adapter, near enough independent of the window size). So the bracket
//! is applied ONLY when the base window actually carries alpha: if every
//! texel is fully opaque then `premultiply` is the identity, provably
//! and not approximately, and skipping it takes the paint path from four
//! dispatches to two. [`window_is_opaque`] is the (cheap, CPU) test —
//! opaque is the overwhelmingly common case, since a JPEG, a PSD
//! composite and most placed photographs have no alpha at all.

use half::f16;

use image_kernels::families::arithmetic::{MathAddParams, MATH_ADD};
use image_kernels::families::band::{BandSetAlphaParams, BAND_SET_ALPHA};
use image_kernels::families::cast::{
    CastPremultiplyParams, CastUnpremultiplyParams, CAST_PREMULTIPLY, CAST_UNPREMULTIPLY,
};
use image_kernels::families::compose::ComposeParams;
use image_kernels::families::conv::{
    ConvGaussianParams, ConvUnsharpParams, CONV_GAUSSIAN_H, CONV_GAUSSIAN_V, CONV_UNSHARP,
};
use image_kernels::families::gen::{GenSolidParams, GEN_SOLID};
use image_kernels::KernelDef;

use crate::resident::{GpuBatch, Resident, TexFormat};
use crate::{GpuContext, GpuError};

/// What a stroke deposits.
///
/// The lifetime is [`PaintMode::Sample`]'s: a cloning stroke's paint
/// layer is a WINDOW of the image itself, so unlike a colour it cannot
/// be carried by value.
#[derive(Debug, Clone, Copy)]
pub enum PaintMode<'a> {
    /// Lay down `color` (STRAIGHT RGBA in `[0, 1]`) through `blend` —
    /// any of the 26 `compose.*` kernels.
    Paint {
        blend: &'static KernelDef,
        color: [f32; 4],
    },
    /// Take alpha away (destination-out in the straight working space).
    Erase,
    /// CLONE / HEAL: the paint layer is a window of pixels sampled from
    /// somewhere ELSE in the image, not a generated colour.
    ///
    /// That is the whole of the clone stamp — it needs no new kernel,
    /// because a dab has never cared where its paint layer came from.
    /// `gen.solid` is simply replaced by an uploaded window, and the
    /// existing coverage, spacing, pressure and selection masking all
    /// apply unchanged.
    ///
    /// `correction_f16` is an additive FIELD applied to the source
    /// before compositing (`math.add`), and it is the only difference
    /// between the two tools: `None` for CLONE, and for HEAL the
    /// gradient-domain correction that makes a healed patch take on the
    /// surrounding tone instead of pasting a visibly different one.
    ///
    /// A field rather than a constant, because a constant can only
    /// cancel a UNIFORM mismatch — healing across a luminance ramp with
    /// one number leaves the seam it exists to remove. It is a straight
    /// rgba16float window like the source, so applying it is one
    /// registered dispatch and no new kernel.
    Sample {
        blend: &'static KernelDef,
        /// A STRAIGHT rgba16float window, the same size as the base.
        source_f16: &'a [u8],
        correction_f16: Option<&'a [u8]>,
    },
    /// A FILTER STROKE through one single-input POINT kernel (dodge /
    /// burn / sponge): the coverage masks `kernel(base)` against the
    /// base. `params` is the kernel's parameter block.
    Filter {
        kernel: &'static KernelDef,
        params: &'a [u8],
    },
    /// A FILTER STROKE through the unsharp chain: `amount` −1 is the
    /// BLUR brush, a positive amount the SHARPEN brush. `radius` is the
    /// Gaussian's half-width, which is also the halo the caller's window
    /// must carry.
    Unsharp {
        sigma: f32,
        radius: u32,
        amount: f32,
    },
}

impl PaintMode<'_> {
    /// The kernels this mode dispatches over a window that CARRIES
    /// ALPHA, in order — the honest answer to "what actually runs on the
    /// GPU", and what the conformance suite asserts against. Over a
    /// fully opaque window the two `cast.*` steps drop out (see
    /// [`window_is_opaque`]); [`Self::kernel_ids_for`] answers per
    /// window.
    pub fn kernel_ids(&self) -> Vec<&'static str> {
        match self {
            PaintMode::Paint { blend, .. } => vec![
                GEN_SOLID.id,
                CAST_PREMULTIPLY.id,
                blend.id,
                CAST_UNPREMULTIPLY.id,
            ],
            PaintMode::Erase => vec![BAND_SET_ALPHA.id],
            PaintMode::Sample {
                blend,
                correction_f16,
                ..
            } => {
                let mut ids = Vec::with_capacity(5);
                if correction_f16.is_some() {
                    ids.push(MATH_ADD.id);
                }
                // Both windows enter the blend premultiplied.
                ids.push(CAST_PREMULTIPLY.id);
                ids.push(CAST_PREMULTIPLY.id);
                ids.push(blend.id);
                ids.push(CAST_UNPREMULTIPLY.id);
                ids
            }
            PaintMode::Filter { kernel, .. } => {
                vec![CAST_PREMULTIPLY.id, kernel.id, CAST_UNPREMULTIPLY.id]
            }
            PaintMode::Unsharp { .. } => vec![
                CAST_PREMULTIPLY.id,
                CONV_GAUSSIAN_H.id,
                CONV_GAUSSIAN_V.id,
                CONV_UNSHARP.id,
                CAST_UNPREMULTIPLY.id,
            ],
        }
    }

    /// The kernels actually dispatched for a window with the given
    /// opacity character.
    pub fn kernel_ids_for(&self, opaque_window: bool) -> Vec<&'static str> {
        match self {
            PaintMode::Paint { blend, .. } if opaque_window => vec![GEN_SOLID.id, blend.id],
            PaintMode::Sample {
                blend,
                correction_f16,
                ..
            } if opaque_window => {
                // Both windows come from the same opaque image, so the
                // premultiply bracket is the identity on each.
                let mut ids = Vec::with_capacity(2);
                if correction_f16.is_some() {
                    ids.push(MATH_ADD.id);
                }
                ids.push(blend.id);
                ids
            }
            PaintMode::Filter { kernel, .. } if opaque_window => vec![kernel.id],
            PaintMode::Unsharp { .. } if opaque_window => {
                vec![CONV_GAUSSIAN_H.id, CONV_GAUSSIAN_V.id, CONV_UNSHARP.id]
            }
            _ => self.kernel_ids(),
        }
    }

    /// The halo (px) a window must carry around the region it writes, so
    /// a windowed filter sees real neighbours instead of the window's
    /// edge. Zero for every per-texel mode.
    pub fn halo(&self) -> u32 {
        match self {
            PaintMode::Unsharp { radius, .. } => *radius,
            _ => 0,
        }
    }
}

/// Is every texel of a straight rgba16float window FULLY opaque?
///
/// When it is, `cast.premultiply` maps the window to itself exactly
/// (`rgb·1 = rgb`) and `cast.unpremultiply` undoes an identity, so the
/// bracket is a provable no-op and the compositor drops it. Alpha is
/// channel 3 of each 8-byte texel, i.e. bytes `6..8`; the comparison is
/// against exactly `1.0`, which is representable in f16, so there is no
/// tolerance to get wrong.
pub fn window_is_opaque(f16_bytes: &[u8]) -> bool {
    f16_bytes
        .chunks_exact(8)
        .all(|t| f16::from_le_bytes([t[6], t[7]]).to_f32() == 1.0)
}

/// Premultiply a straight RGBA colour — the `gen.*` family's param
/// contract (the same rule `image_js::fill` applies to gradient stops).
fn premul(c: [f32; 4]) -> [f32; 4] {
    let a = c[3].clamp(0.0, 1.0);
    [c[0] * a, c[1] * a, c[2] * a, a]
}

/// Composite one window of a stroke.
///
/// `base_f16` is the UNTOUCHED base image window as straight
/// rgba16float (`w·h·8` bytes); `mask_f16` is the effective coverage as
/// r16float (`w·h·2` bytes, from
/// [`crate::dab::StrokeAccumulator::mask_window_f16`]). Returns the
/// painted window in the same straight rgba16float layout.
///
/// Always composites from the BASE, never from the previous result:
/// re-running it for a grown coverage is idempotent, which is what lets
/// the caller re-derive only the dirty rectangle and still land on
/// exactly the pixels a from-scratch composite of the whole stroke
/// would have produced.
pub async fn composite_stroke_window(
    ctx: &GpuContext,
    mode: &PaintMode<'_>,
    base_f16: &[u8],
    mask_f16: &[u8],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, GpuError> {
    let texels = (w as usize) * (h as usize);
    if base_f16.len() != texels * 8 {
        return Err(GpuError::Kernel {
            kernel: "stroke",
            detail: format!(
                "base window is {} bytes, expected {}",
                base_f16.len(),
                texels * 8
            ),
        });
    }
    if mask_f16.len() != texels * 2 {
        return Err(GpuError::Kernel {
            kernel: "stroke",
            detail: format!(
                "mask window is {} bytes, expected {}",
                mask_f16.len(),
                texels * 2
            ),
        });
    }
    if texels == 0 {
        return Ok(Vec::new());
    }

    // Every dispatch below is recorded into ONE batch: the inputs are
    // uploaded once, intermediates stay on the device, and only the
    // result comes back — one submit, one readback per sample instead of
    // one per dispatch. Every bracket decision reads the INPUT bytes
    // (known on the CPU), never an intermediate, so the recorded chain
    // is exactly the chain the per-dispatch path ran.
    let mut b = GpuBatch::new(ctx);
    let base = b.upload(w, h, TexFormat::Rgba16Float, base_f16);
    let mask = b.upload(w, h, TexFormat::R16Float, mask_f16);
    let unary = |b: &mut GpuBatch<'_>, def, params: &[u8], src: &Resident| {
        let out = b.target(w, h);
        b.dispatch(def, &[src], params, None, &out).map(|()| out)
    };
    let result = match *mode {
        // ── erase: one dispatch, straight space, RGB preserved ───────
        PaintMode::Erase => {
            let out = b.target(w, h);
            b.dispatch(
                &BAND_SET_ALPHA,
                &[&base],
                BandSetAlphaParams::new(0.0).as_bytes(),
                Some(&mask),
                &out,
            )?;
            out
        }

        // ── clone / heal: a WINDOW is the paint layer ────────────────
        //
        // Structurally identical to `Paint` — the only change is where
        // the paint layer comes from, which is the whole reason the
        // clone stamp needed no new kernel. The correction (zero for
        // clone) runs first, so heal and clone differ by one dispatch
        // and nothing else.
        PaintMode::Sample {
            blend,
            source_f16,
            correction_f16,
        } => {
            if source_f16.len() != texels * 8 {
                return Err(GpuError::Kernel {
                    kernel: "stroke",
                    detail: format!(
                        "clone source window is {} bytes, expected {}",
                        source_f16.len(),
                        texels * 8
                    ),
                });
            }
            let src = b.upload(w, h, TexFormat::Rgba16Float, source_f16);
            // The opacity test reads the CORRECTED source when there is
            // a correction — that one is an intermediate, so heal reads
            // it back first, exactly as it always did.
            let (source, opaque) = match correction_f16 {
                Some(field) => {
                    if field.len() != texels * 8 {
                        return Err(GpuError::Kernel {
                            kernel: "stroke",
                            detail: format!(
                                "heal correction is {} bytes, expected {}",
                                field.len(),
                                texels * 8
                            ),
                        });
                    }
                    let f = b.upload(w, h, TexFormat::Rgba16Float, field);
                    let sum = b.target(w, h);
                    b.dispatch(
                        &MATH_ADD,
                        &[&src, &f],
                        MathAddParams::new().as_bytes(),
                        None,
                        &sum,
                    )?;
                    let t = b.read(&sum);
                    let corrected = std::mem::replace(&mut b, GpuBatch::new(ctx))
                        .finish_async()
                        .await?
                        .swap_remove(t.0);
                    let opaque = window_is_opaque(base_f16) && window_is_opaque(&corrected);
                    (b.upload(w, h, TexFormat::Rgba16Float, &corrected), opaque)
                }
                None => (
                    src,
                    window_is_opaque(base_f16) && window_is_opaque(source_f16),
                ),
            };
            let premul = CastPremultiplyParams::new();
            let (bp, sp) = if opaque {
                (base.clone(), source.clone())
            } else {
                (
                    unary(&mut b, &CAST_PREMULTIPLY, premul.as_bytes(), &base)?,
                    unary(&mut b, &CAST_PREMULTIPLY, premul.as_bytes(), &source)?,
                )
            };
            let composed = b.target(w, h);
            b.dispatch(
                blend,
                &[&bp, &sp],
                ComposeParams::new(1.0).as_bytes(),
                Some(&mask),
                &composed,
            )?;
            if opaque {
                composed
            } else {
                unary(
                    &mut b,
                    &CAST_UNPREMULTIPLY,
                    CastUnpremultiplyParams::new().as_bytes(),
                    &composed,
                )?
            }
        }

        // ── filter strokes: the coverage masks a kernel's effect ─────
        PaintMode::Filter { kernel, params } => {
            if kernel.inputs != 1 {
                return Err(GpuError::Kernel {
                    kernel: kernel.id,
                    detail: "a filter stroke takes a single-input kernel".into(),
                });
            }
            let opaque = window_is_opaque(base_f16);
            let bp = if opaque {
                base.clone()
            } else {
                unary(
                    &mut b,
                    &CAST_PREMULTIPLY,
                    CastPremultiplyParams::new().as_bytes(),
                    &base,
                )?
            };
            let out = b.target(w, h);
            b.dispatch(kernel, &[&bp], params, Some(&mask), &out)?;
            if opaque {
                out
            } else {
                unary(
                    &mut b,
                    &CAST_UNPREMULTIPLY,
                    CastUnpremultiplyParams::new().as_bytes(),
                    &out,
                )?
            }
        }
        PaintMode::Unsharp {
            sigma,
            radius,
            amount,
        } => {
            let opaque = window_is_opaque(base_f16);
            let bp = if opaque {
                base.clone()
            } else {
                unary(
                    &mut b,
                    &CAST_PREMULTIPLY,
                    CastPremultiplyParams::new().as_bytes(),
                    &base,
                )?
            };
            let g = ConvGaussianParams::new(sigma, radius);
            let gh = unary(&mut b, &CONV_GAUSSIAN_H, g.as_bytes(), &bp)?;
            let gv = unary(&mut b, &CONV_GAUSSIAN_V, g.as_bytes(), &gh)?;
            let out = b.target(w, h);
            b.dispatch(
                &CONV_UNSHARP,
                &[&bp, &gv],
                ConvUnsharpParams::new(amount, 0.0).as_bytes(),
                Some(&mask),
                &out,
            )?;
            if opaque {
                out
            } else {
                unary(
                    &mut b,
                    &CAST_UNPREMULTIPLY,
                    CastUnpremultiplyParams::new().as_bytes(),
                    &out,
                )?
            }
        }

        // ── paint: solid → premultiply → blend under the mask → back ─
        PaintMode::Paint { blend, color } => {
            let c = premul(color);
            let paint = unary(
                &mut b,
                &GEN_SOLID,
                GenSolidParams::new(0, 0, c[0], c[1], c[2], c[3]).as_bytes(),
                &base,
            )?;
            // The bracket is skipped over an opaque window, where it is
            // provably the identity (see the module docs).
            let opaque = window_is_opaque(base_f16);
            let bp = if opaque {
                base.clone()
            } else {
                unary(
                    &mut b,
                    &CAST_PREMULTIPLY,
                    CastPremultiplyParams::new().as_bytes(),
                    &base,
                )?
            };
            let composed = b.target(w, h);
            // Opacity rides the MASK (coverage · opacity), never this
            // param — one rule for paint and erase alike.
            b.dispatch(
                blend,
                &[&bp, &paint],
                ComposeParams::new(1.0).as_bytes(),
                Some(&mask),
                &composed,
            )?;
            if opaque {
                // The composite of an opaque backdrop with any paint
                // colour is opaque (`αo = αs + αb(1 − αs)` with `αb = 1`),
                // so unpremultiplying would be the identity too.
                composed
            } else {
                unary(
                    &mut b,
                    &CAST_UNPREMULTIPLY,
                    CastUnpremultiplyParams::new().as_bytes(),
                    &composed,
                )?
            }
        }
    };
    let t = b.read(&result);
    Ok(b.finish_async().await?.swap_remove(t.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_kernels::families::compose::{COMPOSE_MULTIPLY, COMPOSE_NORMAL};

    #[test]
    fn premultiplying_a_stop_colour_follows_the_generator_contract() {
        assert_eq!(premul([1.0, 1.0, 1.0, 0.5]), [0.5, 0.5, 0.5, 0.5]);
        assert_eq!(premul([0.2, 0.4, 0.6, 1.0]), [0.2, 0.4, 0.6, 1.0]);
        assert_eq!(premul([1.0, 1.0, 1.0, 0.0]), [0.0, 0.0, 0.0, 0.0]);
    }

    /// One rgba16float texel with the given alpha.
    fn texel(alpha: f32) -> Vec<u8> {
        let mut out = Vec::new();
        for c in [0.5f32, 0.25, 0.75, alpha] {
            out.extend_from_slice(&half::f16::from_f32(c).to_bits().to_le_bytes());
        }
        out
    }

    #[test]
    fn opacity_detection_is_exact_and_per_texel() {
        let opaque: Vec<u8> = (0..4).flat_map(|_| texel(1.0)).collect();
        assert!(window_is_opaque(&opaque));
        assert!(window_is_opaque(&[]), "an empty window is vacuously opaque");

        // A single translucent texel disqualifies the whole window …
        let mut mixed = opaque.clone();
        mixed[8 + 6..8 + 8].copy_from_slice(&half::f16::from_f32(0.999).to_bits().to_le_bytes());
        assert!(!window_is_opaque(&mixed));
        // … and so does a fully transparent one (the eraser's own output).
        let mut erased = opaque.clone();
        erased[6..8].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
        assert!(!window_is_opaque(&erased));
    }

    #[test]
    fn an_opaque_window_drops_the_premultiply_bracket() {
        let paint = PaintMode::Paint {
            blend: &COMPOSE_NORMAL,
            color: [1.0, 0.0, 0.0, 1.0],
        };
        assert_eq!(
            paint.kernel_ids_for(true),
            vec!["gen.solid", "compose.normal"],
            "two dispatches over an opaque window"
        );
        assert_eq!(
            paint.kernel_ids_for(false).len(),
            4,
            "four when alpha is live"
        );
        // Erase never brackets either way — it works in straight space.
        assert_eq!(
            PaintMode::Erase.kernel_ids_for(true),
            vec!["band.set_alpha"]
        );
        assert_eq!(
            PaintMode::Erase.kernel_ids_for(false),
            vec!["band.set_alpha"]
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn filter_strokes_name_their_chain_and_their_halo__feat__image_editor_dodge_burn() {
        use image_kernels::families::adjust::{AdjustDodgeBurnParams, ADJUST_DODGE_BURN};
        let p = AdjustDodgeBurnParams::new(0, 1, 0.5);
        let f = PaintMode::Filter {
            kernel: &ADJUST_DODGE_BURN,
            params: p.as_bytes(),
        };
        assert_eq!(f.kernel_ids_for(true), vec!["adjust.dodge_burn"]);
        assert_eq!(f.kernel_ids_for(false).len(), 3, "bracketed over alpha");
        assert_eq!(f.halo(), 0, "a point kernel needs no neighbours");
        let u = PaintMode::Unsharp {
            sigma: 2.0,
            radius: 6,
            amount: -1.0,
        };
        assert_eq!(
            u.kernel_ids_for(true),
            vec!["conv.gaussian_h", "conv.gaussian_v", "conv.unsharp"]
        );
        assert_eq!(u.halo(), 6);
    }

    #[test]
    fn paint_names_four_registered_kernels_and_erase_names_one() {
        let paint = PaintMode::Paint {
            blend: &COMPOSE_NORMAL,
            color: [1.0, 0.0, 0.0, 1.0],
        };
        assert_eq!(
            paint.kernel_ids(),
            vec![
                "gen.solid",
                "cast.premultiply",
                "compose.normal",
                "cast.unpremultiply"
            ]
        );
        let multiply = PaintMode::Paint {
            blend: &COMPOSE_MULTIPLY,
            color: [1.0, 0.0, 0.0, 1.0],
        };
        assert_eq!(multiply.kernel_ids()[2], "compose.multiply");
        assert_eq!(PaintMode::Erase.kernel_ids(), vec!["band.set_alpha"]);
    }
}
