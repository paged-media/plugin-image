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
use image_kernels::families::gen::{GenSolidParams, GEN_SOLID};
use image_kernels::KernelDef;

use crate::execute::{execute_tile_once_async, TileInput};
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
    /// `correction` is the per-channel additive term applied to the
    /// source before compositing, and it is the ONLY difference between
    /// the two tools: zero for CLONE, and the destination−source mean
    /// for HEAL, which is what makes a healed patch take on the
    /// surrounding tone instead of pasting a visibly different one.
    ///
    /// It runs as `gen.solid` → `math.add` rather than `math.add_const`
    /// because `add_const` broadcasts ONE scalar to every channel, and a
    /// heal that could only shift luminance would leave a colour cast
    /// exactly where the tool is supposed to remove one. Two registered
    /// dispatches, no new kernel.
    Sample {
        blend: &'static KernelDef,
        /// A STRAIGHT rgba16float window, the same size as the base.
        source_f16: &'a [u8],
        correction: [f32; 4],
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
                blend, correction, ..
            } => {
                let mut ids = Vec::with_capacity(6);
                if correction.iter().any(|c| *c != 0.0) {
                    ids.push(GEN_SOLID.id);
                    ids.push(MATH_ADD.id);
                }
                // Both windows enter the blend premultiplied.
                ids.push(CAST_PREMULTIPLY.id);
                ids.push(CAST_PREMULTIPLY.id);
                ids.push(blend.id);
                ids.push(CAST_UNPREMULTIPLY.id);
                ids
            }
        }
    }

    /// The kernels actually dispatched for a window with the given
    /// opacity character.
    pub fn kernel_ids_for(&self, opaque_window: bool) -> Vec<&'static str> {
        match self {
            PaintMode::Paint { blend, .. } if opaque_window => vec![GEN_SOLID.id, blend.id],
            PaintMode::Sample {
                blend, correction, ..
            } if opaque_window => {
                // Both windows come from the same opaque image, so the
                // premultiply bracket is the identity on each.
                let mut ids = Vec::with_capacity(3);
                if correction.iter().any(|c| *c != 0.0) {
                    ids.push(GEN_SOLID.id);
                    ids.push(MATH_ADD.id);
                }
                ids.push(blend.id);
                ids
            }
            _ => self.kernel_ids(),
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

    match *mode {
        // ── erase: one dispatch, straight space, RGB preserved ───────
        PaintMode::Erase => {
            execute_tile_once_async(
                ctx,
                &BAND_SET_ALPHA,
                &[TileInput {
                    f16_bytes: base_f16,
                }],
                BandSetAlphaParams::new(0.0).as_bytes(),
                Some(mask_f16),
                w,
                h,
            )
            .await
        }

        // ── clone / heal: a WINDOW is the paint layer ────────────────
        //
        // Structurally identical to `Paint` — the only change is where
        // the paint layer comes from, which is the whole reason the
        // clone stamp needed no new kernel. The correction (zero for
        // clone) runs first, so heal and clone differ by two dispatches
        // and nothing else.
        PaintMode::Sample {
            blend,
            source_f16,
            correction,
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
            let corrected = if correction.iter().any(|c| *c != 0.0) {
                // `gen.solid` builds the per-channel constant; `math.add`
                // applies it. Alpha's correction is always 0 — a heal
                // shifts tone, never transparency.
                let konst = execute_tile_once_async(
                    ctx,
                    &GEN_SOLID,
                    &[TileInput {
                        f16_bytes: source_f16,
                    }],
                    GenSolidParams::new(0, 0, correction[0], correction[1], correction[2], 0.0)
                        .as_bytes(),
                    None,
                    w,
                    h,
                )
                .await?;
                Some(
                    execute_tile_once_async(
                        ctx,
                        &MATH_ADD,
                        &[
                            TileInput {
                                f16_bytes: source_f16,
                            },
                            TileInput { f16_bytes: &konst },
                        ],
                        MathAddParams::new().as_bytes(),
                        None,
                        w,
                        h,
                    )
                    .await?,
                )
            } else {
                None
            };
            let source = corrected.as_deref().unwrap_or(source_f16);

            // Both windows come from the SAME image, so they are opaque
            // together or not at all — one test, two brackets skipped.
            let opaque = window_is_opaque(base_f16) && window_is_opaque(source);
            let (base_premul, source_premul) = if opaque {
                (None, None)
            } else {
                (
                    Some(
                        execute_tile_once_async(
                            ctx,
                            &CAST_PREMULTIPLY,
                            &[TileInput {
                                f16_bytes: base_f16,
                            }],
                            CastPremultiplyParams::new().as_bytes(),
                            None,
                            w,
                            h,
                        )
                        .await?,
                    ),
                    Some(
                        execute_tile_once_async(
                            ctx,
                            &CAST_PREMULTIPLY,
                            &[TileInput { f16_bytes: source }],
                            CastPremultiplyParams::new().as_bytes(),
                            None,
                            w,
                            h,
                        )
                        .await?,
                    ),
                )
            };

            let composed = execute_tile_once_async(
                ctx,
                blend,
                &[
                    TileInput {
                        f16_bytes: base_premul.as_deref().unwrap_or(base_f16),
                    },
                    TileInput {
                        f16_bytes: source_premul.as_deref().unwrap_or(source),
                    },
                ],
                ComposeParams::new(1.0).as_bytes(),
                Some(mask_f16),
                w,
                h,
            )
            .await?;

            if opaque {
                return Ok(composed);
            }
            execute_tile_once_async(
                ctx,
                &CAST_UNPREMULTIPLY,
                &[TileInput {
                    f16_bytes: &composed,
                }],
                CastUnpremultiplyParams::new().as_bytes(),
                None,
                w,
                h,
            )
            .await
        }

        // ── paint: solid → premultiply → blend under the mask → back ─
        PaintMode::Paint { blend, color } => {
            let c = premul(color);
            let paint = execute_tile_once_async(
                ctx,
                &GEN_SOLID,
                &[TileInput {
                    f16_bytes: base_f16,
                }],
                GenSolidParams::new(0, 0, c[0], c[1], c[2], c[3]).as_bytes(),
                None,
                w,
                h,
            )
            .await?;

            // The bracket is skipped over an opaque window, where it is
            // provably the identity — two fewer round-trips in the case
            // that dominates (see the module docs).
            let opaque = window_is_opaque(base_f16);
            let base_premul = if opaque {
                None
            } else {
                Some(
                    execute_tile_once_async(
                        ctx,
                        &CAST_PREMULTIPLY,
                        &[TileInput {
                            f16_bytes: base_f16,
                        }],
                        CastPremultiplyParams::new().as_bytes(),
                        None,
                        w,
                        h,
                    )
                    .await?,
                )
            };

            let composed = execute_tile_once_async(
                ctx,
                blend,
                &[
                    TileInput {
                        f16_bytes: base_premul.as_deref().unwrap_or(base_f16),
                    },
                    TileInput { f16_bytes: &paint },
                ],
                // Opacity rides the MASK (coverage · opacity), never this
                // param — one rule for paint and erase alike.
                ComposeParams::new(1.0).as_bytes(),
                Some(mask_f16),
                w,
                h,
            )
            .await?;

            if opaque {
                // The composite of an opaque backdrop with an opaque
                // paint colour is opaque, so unpremultiplying would be
                // the identity too. (A colour with alpha < 1 still
                // composites to alpha 1 over an opaque backdrop — the
                // source-over `αo = αs + αb(1 − αs)` with `αb = 1`.)
                return Ok(composed);
            }
            execute_tile_once_async(
                ctx,
                &CAST_UNPREMULTIPLY,
                &[TileInput {
                    f16_bytes: &composed,
                }],
                CastUnpremultiplyParams::new().as_bytes(),
                None,
                w,
                h,
            )
            .await
        }
    }
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
