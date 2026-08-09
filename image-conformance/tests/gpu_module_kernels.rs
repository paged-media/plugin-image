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

//! THE GPU BEHAVIOURAL LANE for handwritten `module: true` kernels
//! (spec §6.4 Definition of Done, the half of it the macro cannot give).
//!
//! `kernel_family!` emits a scalar twin behind `feature = "reference"`,
//! so every macro kernel gets `parity(gpu↔ref)` for free. A handwritten
//! ABI v1.1 module gets no twin: unless somebody wrote a bespoke
//! reference by hand (the adjust/compose families did), the ONLY gate on
//! it is that naga parses its WGSL — which proves it compiles, not that
//! it computes anything. This file closes that hole for ALL of them at
//! once, without 90-odd hand-written twins, by asserting properties that
//! must hold WHATEVER the implementation is:
//!
//! 1. IDENTITY — dispatched with its documented no-op parameters over a
//!    non-trivial image, a kernel must return that image within its own
//!    declared `gpu_tolerance`. A kernel whose identity does not hold is
//!    broken in a way naga cannot see, and the panel slider that starts
//!    at zero is already lying to the user.
//! 2. MASK SCOPING — the ABI's core promise. With an all-zero selection
//!    mask the output must equal the input EXACTLY, whatever the params;
//!    with a half mask the masked-out half must be untouched. A kernel
//!    that forgets `mix(a, result, m)` silently paints the whole tile
//!    and looks perfect until somebody makes a selection.
//! 3. DETERMINISM — same inputs twice, byte-identical output. The
//!    `gallery.*` and `gen.*` kernels hash the pixel coordinate plus a
//!    seed precisely so this holds; one that reached for real randomness
//!    would fail here.
//! 4. FINITENESS — every output channel finite under working params.
//!    Several kernels divide by a weight sum or take `atan2`/`length`; a
//!    NaN smuggled through poisons the identity blend downstream, since
//!    `mix(a, NaN, 0.0)` is NaN, not `a`.
//!
//! Plus two structural gates that need no adapter and therefore always
//! run: the declared `ParamsLayout` must agree with the module's own
//! WGSL `struct Params`, and EVERY handwritten module kernel must have a
//! row in [`TABLE`] — a new kernel cannot quietly avoid this lane.
//!
//! # Dispatcher choice (this is load-bearing)
//!
//! `Windowed` kernels are driven through [`execute_windowed_once`] with
//! a properly EXPANDED input window (`out + 2·radius`), which is the
//! production path (`image_graph::gather_window`). `execute_tile_once`
//! sizes `in0` at the OUTPUT dims with no halo, and the kernels that
//! hardcode `xy + (rx, ry)` — `conv.box`, the Gaussian pair, `morph.*`,
//! `rank.*` — would then read outside the texture and get zeros back.
//! That is the hazard documented in `image_kernels::abi`; running the
//! wrong lane here would manufacture a false failure for the kernels
//! that are correct and hide the halo bug in the ones that are not.
//! Everything else (`Point`, `Resample`, `Generator`) goes through
//! [`execute_tile_once`] with inputs at the output dims.
//!
//! # What the hand-written half is
//!
//! [`TABLE`] — one row per kernel carrying its identity parameters, a
//! non-identity "stress" block, and a note. That data cannot be derived:
//! only the kernel's own documentation says which slider position is the
//! no-op. Everything else in this file is derived from `KernelDef`.

use half::f16;
use image_conformance::device::test_device;
use image_conformance::quantize::f16_ulp_distance;
use image_gpu::{execute_tile_once, execute_windowed_once, GpuContext, TileInput};
use image_kernels::families::adjust::{
    AdjustGradientMapParams, AdjustLut1dParams, AdjustLut3dParams,
};
use image_kernels::{KernelClass, KernelDef, Tolerance};

/// Output edge for every dispatch. Small on purpose — 64² is plenty to
/// exercise a shader and keeps ~500 dispatches (each one a pipeline
/// build) inside a sane test runtime.
const OUT: u32 = 64;
/// `OUT / 2` as an f32 — the tile centre several identity blocks name
/// literally (a `&'static` table cannot compute it).
const HALF_OUT: f32 = 32.0;

// ─────────────────────────── the table ───────────────────────────────

/// A parameter value, tagged with the WGSL scalar type it must land in.
/// The tag is checked against the declared `ParamField::wgsl_ty`, so a
/// float written into a `u32` slot is a test failure, not a silent
/// reinterpretation of bits.
#[derive(Clone, Copy)]
enum V {
    F(f32),
    U(u32),
    I(i32),
}

/// How to produce a kernel's param block.
#[derive(Clone, Copy)]
enum P {
    /// By field NAME against the declared `ParamsLayout`. Fields absent
    /// from the list are zero — which is the identity value for most
    /// "delta" params and is always the right value for ABI pads.
    Fields(&'static [(&'static str, V)]),
    /// The whole block verbatim, for the array-valued blocks (LUTs,
    /// ramps) whose layout is not a flat run of 4-byte scalars. Built
    /// through the real typed constructor, so a param-struct change is
    /// a compile error here.
    Raw(fn() -> Vec<u8>),
}

/// What "identity" means for a kernel.
#[derive(Clone, Copy)]
enum Identity {
    /// These params are the documented no-op: output ≡ input.
    Params(P),
    /// No parameterised no-op exists, but the kernel is a neighbourhood
    /// filter that MUST preserve a constant field (a mean/median/min/max
    /// of nine identical samples is that sample). Weaker than identity,
    /// still a real behavioural claim, and checked against two different
    /// constants so "returns a fixed colour" cannot pass it.
    ConstantField,
    /// Genuinely has no identity. The reason is in the row's `note` and
    /// is printed by the lane, so an exclusion is a statement on the
    /// record rather than a silent gap.
    NotDefined,
}

struct Row {
    id: &'static str,
    identity: Identity,
    /// Non-identity params: what the mask, determinism and finiteness
    /// properties are asserted under. These must make the kernel
    /// actually DO something, or those three tests prove nothing.
    stress: P,
    note: &'static str,
}

// The three array-valued adjust blocks, via their real constructors.

fn lut1d_identity() -> Vec<u8> {
    bytemuck::bytes_of(&AdjustLut1dParams::identity()).to_vec()
}

fn lut1d_inverted() -> Vec<u8> {
    let mut t = [0u8; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = 255 - i as u8;
    }
    bytemuck::bytes_of(&AdjustLut1dParams::new(&t)).to_vec()
}

fn lut3d_identity() -> Vec<u8> {
    bytemuck::bytes_of(&AdjustLut3dParams::identity()).to_vec()
}

fn lut3d_channel_swap() -> Vec<u8> {
    bytemuck::bytes_of(&AdjustLut3dParams::from_fn(|r, g, b| [b, g, r])).to_vec()
}

fn gradient_map_teal_orange() -> Vec<u8> {
    bytemuck::bytes_of(&AdjustGradientMapParams::two_stop(
        [0.0, 0.35, 0.4],
        [1.0, 0.6, 0.2],
    ))
    .to_vec()
}

/// Every compose kernel folds `opacity` into the source FIRST, so a zero
/// opacity makes the premultiplied source `vec4(0)` and source-over
/// collapses to `term3 = (1−0)·αb·cb`, i.e. the backdrop.
const COMPOSE_IDENTITY: P = P::Fields(&[("opacity", V::F(0.0))]);
const COMPOSE_STRESS: P = P::Fields(&[("opacity", V::F(0.75))]);
const COMPOSE_NOTE: &str = "binary W3C composite; opacity 0 = the backdrop, bit-exactly";

/// The parameterless 3×3 neighbourhood filters share a row shape.
const NO_PARAMS: P = P::Fields(&[]);

/// One row per handwritten `module: true` kernel. HAND-WRITTEN DATA —
/// the identity setting comes from each kernel's own doc comment and
/// WGSL, and nothing derives it.
#[rustfmt::skip]
const TABLE: &[Row] = &[
    // ── adjust (T2 point ops; all mask-scoped via the shared template) ──
    Row { id: "adjust.exposure",
          identity: Identity::Params(P::Fields(&[("ev", V::F(0.0))])),
          stress: P::Fields(&[("ev", V::F(0.8))]),
          note: "ev = 0 ⇒ exp2(0) = 1" },
    Row { id: "adjust.brightness_contrast",
          identity: Identity::Params(P::Fields(&[("brightness", V::F(0.0)), ("contrast", V::F(1.0))])),
          stress: P::Fields(&[("brightness", V::F(0.1)), ("contrast", V::F(1.3))]),
          note: "unit contrast about 0.5, zero brightness" },
    Row { id: "adjust.levels",
          identity: Identity::Params(P::Fields(&[
              ("in_black", V::F(0.0)), ("in_white", V::F(1.0)), ("gamma", V::F(1.0)),
              ("out_black", V::F(0.0)), ("out_white", V::F(1.0))])),
          stress: P::Fields(&[
              ("in_black", V::F(0.05)), ("in_white", V::F(0.95)), ("gamma", V::F(1.4)),
              ("out_black", V::F(0.0)), ("out_white", V::F(1.0))]),
          note: "full range in and out, gamma 1" },
    Row { id: "adjust.saturation",
          identity: Identity::Params(P::Fields(&[("sat", V::F(1.0))])),
          stress: P::Fields(&[("sat", V::F(1.6))]),
          note: "sat = 1 ⇒ mix(lum, c, 1) = c" },
    Row { id: "adjust.hue_rotate",
          identity: Identity::Params(P::Fields(&[("degrees", V::F(0.0))])),
          stress: P::Fields(&[("degrees", V::F(35.0))]),
          note: "0° ⇒ the rotation matrix is the identity matrix" },
    Row { id: "adjust.white_balance",
          identity: Identity::Params(P::Fields(&[("temp", V::F(0.0)), ("tint", V::F(0.0))])),
          stress: P::Fields(&[("temp", V::F(0.15)), ("tint", V::F(-0.08))]),
          note: "zero shift ⇒ unit gains" },
    Row { id: "adjust.vibrance",
          identity: Identity::Params(P::Fields(&[])),
          stress: P::Fields(&[("vibrance", V::F(0.6)), ("saturation", V::F(0.15))]),
          note: "both sliders 0 ⇒ f = 1 ⇒ mix(lum, c, 1) = c" },
    Row { id: "adjust.color_balance",
          identity: Identity::Params(P::Fields(&[])),
          stress: P::Fields(&[
              ("sh_cr", V::F(0.1)), ("sh_mg", V::F(-0.05)), ("sh_yb", V::F(0.08)),
              ("mid_cr", V::F(-0.06)), ("mid_mg", V::F(0.04)), ("mid_yb", V::F(-0.02)),
              ("hi_cr", V::F(0.05)), ("hi_mg", V::F(0.02)), ("hi_yb", V::F(-0.1))]),
          note: "all nine deltas 0 ⇒ the luma re-add cancels" },
    Row { id: "adjust.black_white",
          identity: Identity::NotDefined,
          stress: P::Fields(&[
              ("reds", V::F(0.4)), ("yellows", V::F(0.6)), ("greens", V::F(0.4)),
              ("cyans", V::F(0.6)), ("blues", V::F(0.2)), ("magentas", V::F(0.8))]),
          note: "EXCLUDED from identity: a greyscale conversion. Every weight \
                 setting produces r = g = b; no parameterisation returns colour" },
    Row { id: "adjust.posterize",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("levels", V::F(6.0))]),
          note: "EXCLUDED from identity: quantisation to `levels` bins. \
                 `max(levels, 2)` floors the bin count, so even the minimum \
                 setting is a 2-level cut, not a pass-through" },
    Row { id: "adjust.threshold",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("threshold", V::F(0.5))]),
          note: "EXCLUDED from identity: binarises luma to 0 or alpha. Every \
                 threshold value is a cut; there is no pass-through setting" },
    Row { id: "adjust.photo_filter",
          identity: Identity::Params(P::Fields(&[
              ("fr", V::F(0.93)), ("fg", V::F(0.66)), ("fb", V::F(0.27)),
              ("density", V::F(0.0)), ("preserve", V::U(1))])),
          stress: P::Fields(&[
              ("fr", V::F(0.93)), ("fg", V::F(0.66)), ("fb", V::F(0.27)),
              ("density", V::F(0.6)), ("preserve", V::U(1))]),
          note: "density 0 ⇒ identity whatever the gel colour or preserve flag" },
    Row { id: "adjust.channel_mixer",
          identity: Identity::Params(P::Fields(&[
              ("rr", V::F(1.0)), ("gg", V::F(1.0)), ("bb", V::F(1.0))])),
          stress: P::Fields(&[
              ("rr", V::F(0.7)), ("rg", V::F(0.2)), ("rb", V::F(0.1)), ("rc", V::F(0.05)),
              ("gr", V::F(0.1)), ("gg", V::F(0.8)), ("gb", V::F(0.1)),
              ("bg", V::F(0.3)), ("bb", V::F(0.6)), ("bc", V::F(-0.02))]),
          note: "the identity matrix; off-diagonals and constants default to 0" },
    Row { id: "adjust.levels_rgb",
          identity: Identity::Params(P::Fields(&[
              ("r_in_black", V::F(0.0)), ("r_in_white", V::F(1.0)), ("r_gamma", V::F(1.0)),
              ("g_in_black", V::F(0.0)), ("g_in_white", V::F(1.0)), ("g_gamma", V::F(1.0)),
              ("b_in_black", V::F(0.0)), ("b_in_white", V::F(1.0)), ("b_gamma", V::F(1.0))])),
          stress: P::Fields(&[
              ("r_in_black", V::F(0.05)), ("r_in_white", V::F(0.95)), ("r_gamma", V::F(1.3)),
              ("g_in_black", V::F(0.0)), ("g_in_white", V::F(1.0)), ("g_gamma", V::F(0.8)),
              ("b_in_black", V::F(0.1)), ("b_in_white", V::F(0.9)), ("b_gamma", V::F(1.0))]),
          note: "per-channel full range, gamma 1" },
    Row { id: "adjust.selective_color",
          identity: Identity::Params(P::Fields(&[("range", V::U(0)), ("absolute", V::U(0))])),
          stress: P::Fields(&[
              ("range", V::U(0)), ("cyan", V::F(0.2)), ("magenta", V::F(-0.1)),
              ("yellow", V::F(0.15)), ("black", V::F(0.05)), ("absolute", V::U(0))]),
          note: "all-zero CMYK deltas are the identity for every range" },
    Row { id: "adjust.lut1d",
          identity: Identity::Params(P::Raw(lut1d_identity)),
          stress: P::Raw(lut1d_inverted),
          note: "lut[i] = i/255; the transfer is interpolated, so a linear \
                 table reconstructs the input exactly" },
    Row { id: "adjust.lut3d",
          identity: Identity::Params(P::Raw(lut3d_identity)),
          stress: P::Raw(lut3d_channel_swap),
          note: "the linear 9³ cube; trilinear interpolation of a linear \
                 lattice is the identity" },
    Row { id: "adjust.gradient_map",
          identity: Identity::NotDefined,
          stress: P::Raw(gradient_map_teal_orange),
          note: "EXCLUDED from identity: maps LUMA through a ramp, so the \
                 output is a one-dimensional recolour. Even the greyscale \
                 ramp returns luma, not the input colour" },

    // ── compose (26 binary blend modes; one shared params block) ────────
    Row { id: "compose.normal",        identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.multiply",      identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.screen",        identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.overlay",       identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.darken",        identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.lighten",       identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.color_dodge",   identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.color_burn",    identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.hard_light",    identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.soft_light",    identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.difference",    identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.exclusion",     identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.hue",           identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.saturation",    identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.color",         identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.luminosity",    identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.linear_burn",   identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.linear_dodge",  identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.vivid_light",   identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.linear_light",  identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.pin_light",     identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.hard_mix",      identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.subtract",      identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.divide",        identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.darker_color",  identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },
    Row { id: "compose.lighter_color", identity: Identity::Params(COMPOSE_IDENTITY), stress: COMPOSE_STRESS, note: COMPOSE_NOTE },

    // ── conv (windowed convolutions + one binary point op) ─────────────
    Row { id: "conv.box",
          identity: Identity::ConstantField,
          stress: NO_PARAMS,
          note: "no params at all — a fixed 3×3 mean. The constant-field \
                 property stands in for the missing identity" },
    Row { id: "conv.gaussian_h",
          identity: Identity::Params(P::Fields(&[("sigma", V::F(1.0)), ("radius", V::U(0))])),
          stress: P::Fields(&[("sigma", V::F(2.0)), ("radius", V::U(6))]),
          note: "radius 0 ⇒ one tap, weight exp(0)/exp(0) = 1" },
    Row { id: "conv.gaussian_v",
          identity: Identity::Params(P::Fields(&[("sigma", V::F(1.0)), ("radius", V::U(0))])),
          stress: P::Fields(&[("sigma", V::F(2.0)), ("radius", V::U(6))]),
          note: "radius 0 ⇒ one tap, weight exp(0)/exp(0) = 1" },
    Row { id: "conv.unsharp",
          identity: Identity::Params(P::Fields(&[("amount", V::F(0.0)), ("threshold", V::F(0.0))])),
          stress: P::Fields(&[("amount", V::F(0.8)), ("threshold", V::F(0.02))]),
          note: "binary (a = source, b = blurred); amount 0 ⇒ both branches are `a`" },
    Row { id: "conv.emboss",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("angle_deg", V::F(45.0)), ("height", V::F(1.0))]),
          note: "EXCLUDED from identity: relief is BIASED to mid-grey — \
                 height 0 yields a constant 0.5 field, not the input" },
    Row { id: "conv.find_edges",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("strength", V::F(1.0))]),
          note: "EXCLUDED from identity: an INVERTED Sobel magnitude — \
                 strength 0 yields white, not the input" },
    Row { id: "conv.motion",
          identity: Identity::Params(P::Fields(&[("angle_deg", V::F(0.0)), ("length_px", V::F(0.0))])),
          stress: P::Fields(&[("angle_deg", V::F(30.0)), ("length_px", V::F(12.0))]),
          note: "length 0 ⇒ all 17 taps land on the centre" },
    Row { id: "conv.radial",
          identity: Identity::Params(P::Fields(&[
              ("cx", V::F(0.5)), ("cy", V::F(0.5)), ("amount", V::F(0.0)), ("mode", V::U(1))])),
          stress: P::Fields(&[
              ("cx", V::F(0.5)), ("cy", V::F(0.5)), ("amount", V::F(0.15)), ("mode", V::U(0))]),
          note: "ZOOM at amount 0 ⇒ every tap is centre + rel·1.0, i.e. the \
                 pixel itself; identity is asserted on the exactly-invertible \
                 mode, spin is covered by the other three properties" },
    Row { id: "conv.lens",
          identity: Identity::Params(P::Fields(&[
              ("radius_px", V::F(0.0)), ("threshold", V::F(1.0)), ("boost", V::F(0.0))])),
          stress: P::Fields(&[
              ("radius_px", V::F(4.0)), ("threshold", V::F(0.7)), ("boost", V::F(2.0))]),
          note: "radius below half a pixel takes the explicit early return" },
    Row { id: "conv.shape",
          identity: Identity::Params(P::Fields(&[
              ("radius_px", V::F(6.0)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("radius_px", V::F(4.0)), ("amount", V::F(1.0)),
              ("shape_w", V::U(0)), ("shape_h", V::U(0))]),
          note: "binary — in1 is the shape COVERAGE bitmap, read from .r. amount 0 \
                 takes the early return, which writes the input raw, so the identity \
                 holds for ANY in1 content; that matters because this lane synthesises \
                 its second input rather than binding a real silhouette. shape_w/h 0 \
                 mean `the whole in1 texture`, so the stress block convolves with the \
                 stimulus tile and gathers a positive weight sum. Windowed AND binary, \
                 so it rides the TILE lane — execute_windowed_once rejects inputs != 1 \
                 — which is safe only because it derives its halo (RFI E-4)" },
    Row { id: "conv.bilateral",
          identity: Identity::Params(P::Fields(&[
              ("radius_px", V::F(2.0)), ("sigma_range", V::F(0.1)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("radius_px", V::F(4.0)), ("sigma_range", V::F(0.1)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, filtered, 0); also proves `filtered` is \
                 finite, since a NaN would survive the zero factor" },
    Row { id: "conv.smart_sharpen",
          identity: Identity::Params(P::Fields(&[
              ("radius_px", V::F(1.0)), ("amount", V::F(0.0)),
              ("threshold", V::F(2.0)), ("clamp_hi", V::F(1.0))])),
          stress: P::Fields(&[
              ("radius_px", V::F(2.0)), ("amount", V::F(1.2)),
              ("threshold", V::F(0.02)), ("clamp_hi", V::F(0.25))]),
          note: "channels live in [0,1] so local contrast never reaches a \
                 threshold of 2 — gate 1 returns `a` everywhere" },

    // ── gallery (12 stylisation primitives, all `amount`-blended) ───────
    Row { id: "gallery.kuwahara",
          identity: Identity::Params(P::Fields(&[("radius_px", V::F(3.0)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[("radius_px", V::F(3.0)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0)" },
    Row { id: "gallery.posterize_edges",
          identity: Identity::Params(P::Fields(&[
              ("levels", V::F(6.0)), ("edge_amount", V::F(1.0)),
              ("edge_threshold", V::F(0.1)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("levels", V::F(6.0)), ("edge_amount", V::F(1.0)),
              ("edge_threshold", V::F(0.1)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0)" },
    Row { id: "gallery.glowing_edges",
          identity: Identity::Params(P::Fields(&[
              ("intensity", V::F(1.0)), ("smoothness", V::F(0.5)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("intensity", V::F(1.0)), ("smoothness", V::F(0.5)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0)" },
    Row { id: "gallery.halftone",
          identity: Identity::Params(P::Fields(&[
              ("cell_px", V::F(6.0)), ("angle_deg", V::F(45.0)),
              ("contrast", V::F(1.0)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("cell_px", V::F(6.0)), ("angle_deg", V::F(45.0)),
              ("contrast", V::F(1.0)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0); tile origin (ox, oy) = 0" },
    Row { id: "gallery.grain",
          identity: Identity::Params(P::Fields(&[
              ("seed", V::U(7)), ("size_px", V::F(1.0)), ("mono", V::U(1)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("seed", V::U(7)), ("size_px", V::F(1.0)), ("mono", V::U(1)), ("amount", V::F(0.6))]),
          note: "amount 0 ⇒ mix(a, styled, 0); the grain hash is coordinate+seed, \
                 which the determinism property pins" },
    Row { id: "gallery.diffuse",
          identity: Identity::Params(P::Fields(&[
              ("seed", V::U(7)), ("radius_px", V::F(4.0)), ("angle_deg", V::F(0.0)),
              ("anisotropy", V::F(0.0)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("seed", V::U(7)), ("radius_px", V::F(4.0)), ("angle_deg", V::F(20.0)),
              ("anisotropy", V::F(0.5)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0)" },
    Row { id: "gallery.crosshatch",
          identity: Identity::Params(P::Fields(&[
              ("angle_deg", V::F(45.0)), ("spacing_px", V::F(6.0)),
              ("strength", V::F(1.0)), ("sets", V::U(2)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("angle_deg", V::F(45.0)), ("spacing_px", V::F(6.0)),
              ("strength", V::F(1.0)), ("sets", V::U(2)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0)" },
    Row { id: "gallery.bas_relief",
          identity: Identity::Params(P::Fields(&[
              ("angle_deg", V::F(45.0)), ("elevation_deg", V::F(30.0)),
              ("height", V::F(1.0)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("angle_deg", V::F(45.0)), ("elevation_deg", V::F(30.0)),
              ("height", V::F(1.0)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0) — the relief bias is inside `styled`" },
    Row { id: "gallery.threshold_ink",
          identity: Identity::Params(P::Fields(&[
              ("threshold", V::F(0.5)), ("softness", V::F(0.05)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("threshold", V::F(0.5)), ("softness", V::F(0.05)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0)" },
    Row { id: "gallery.stained_glass",
          identity: Identity::Params(P::Fields(&[
              ("seed", V::U(7)), ("cell_px", V::F(8.0)), ("border", V::F(0.1)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("seed", V::U(7)), ("cell_px", V::F(8.0)), ("border", V::F(0.1)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0)" },
    Row { id: "gallery.glass",
          identity: Identity::Params(P::Fields(&[
              ("seed", V::U(7)), ("scale_px", V::F(8.0)), ("distortion", V::F(4.0)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("seed", V::U(7)), ("scale_px", V::F(8.0)), ("distortion", V::F(4.0)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0)" },
    Row { id: "gallery.texturizer",
          identity: Identity::Params(P::Fields(&[
              ("seed", V::U(7)), ("kind", V::U(0)), ("scale_px", V::F(8.0)),
              ("relief", V::F(1.0)), ("angle_deg", V::F(0.0)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[
              ("seed", V::U(7)), ("kind", V::U(0)), ("scale_px", V::F(8.0)),
              ("relief", V::F(1.0)), ("angle_deg", V::F(0.0)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(a, styled, 0)" },

    // ── gen (8 generators + the binary pattern fill) ────────────────────
    Row { id: "gen.solid",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("r", V::F(0.2)), ("g", V::F(0.4)), ("b", V::F(0.6)), ("a", V::F(0.8))]),
          note: "EXCLUDED from identity: Generator — synthesises a field and \
                 never reads in0, so there is no input to preserve" },
    Row { id: "gen.checker",
          identity: Identity::NotDefined,
          stress: P::Fields(&[
              ("size", V::U(8)), ("c0a", V::F(1.0)),
              ("c1r", V::F(1.0)), ("c1g", V::F(1.0)), ("c1b", V::F(1.0)), ("c1a", V::F(1.0))]),
          note: "EXCLUDED from identity: Generator (see gen.solid)" },
    Row { id: "gen.linear_gradient",
          identity: Identity::NotDefined,
          stress: P::Fields(&[
              ("x1", V::F(63.0)), ("y1", V::F(63.0)), ("c0a", V::F(1.0)),
              ("c1r", V::F(1.0)), ("c1g", V::F(1.0)), ("c1b", V::F(1.0)), ("c1a", V::F(1.0))]),
          note: "EXCLUDED from identity: Generator (see gen.solid)" },
    Row { id: "gen.radial_gradient",
          identity: Identity::NotDefined,
          stress: P::Fields(&[
              ("cx", V::F(HALF_OUT)), ("cy", V::F(HALF_OUT)), ("radius", V::F(30.0)), ("c0a", V::F(1.0)),
              ("c1r", V::F(1.0)), ("c1g", V::F(1.0)), ("c1b", V::F(1.0)), ("c1a", V::F(1.0))]),
          note: "EXCLUDED from identity: Generator (see gen.solid)" },
    Row { id: "gen.angular_gradient",
          identity: Identity::NotDefined,
          stress: P::Fields(&[
              ("cx", V::F(HALF_OUT)), ("cy", V::F(HALF_OUT)), ("angle", V::F(0.0)), ("c0a", V::F(1.0)),
              ("c1r", V::F(1.0)), ("c1g", V::F(1.0)), ("c1b", V::F(1.0)), ("c1a", V::F(1.0))]),
          note: "EXCLUDED from identity: Generator; `atan2` at the exact centre \
                 is the finiteness property's target here" },
    Row { id: "gen.reflected_gradient",
          identity: Identity::NotDefined,
          stress: P::Fields(&[
              ("x1", V::F(63.0)), ("y1", V::F(63.0)), ("c0a", V::F(1.0)),
              ("c1r", V::F(1.0)), ("c1g", V::F(1.0)), ("c1b", V::F(1.0)), ("c1a", V::F(1.0))]),
          note: "EXCLUDED from identity: Generator (see gen.solid)" },
    Row { id: "gen.diamond_gradient",
          identity: Identity::NotDefined,
          stress: P::Fields(&[
              ("cx", V::F(HALF_OUT)), ("cy", V::F(HALF_OUT)), ("angle", V::F(0.0)),
              ("scale", V::F(30.0)), ("c0a", V::F(1.0)),
              ("c1r", V::F(1.0)), ("c1g", V::F(1.0)), ("c1b", V::F(1.0)), ("c1a", V::F(1.0))]),
          note: "EXCLUDED from identity: Generator (see gen.solid)" },
    Row { id: "gen.noise",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("seed", V::U(7)), ("amount", V::F(0.5))]),
          note: "EXCLUDED from identity: Generator. Its hash is coordinate+seed, \
                 which is exactly what the determinism property exists to pin" },
    Row { id: "gen.pattern",
          identity: Identity::Params(P::Fields(&[("scale", V::F(1.0)), ("opacity", V::F(0.0))])),
          stress: P::Fields(&[
              ("scale", V::F(1.0)), ("angle_deg", V::F(15.0)),
              ("offset_x", V::F(3.0)), ("offset_y", V::F(5.0)), ("opacity", V::F(1.0))]),
          note: "binary (in1 = the pattern tile); opacity 0 ⇒ premultiplied \
                 source is vec4(0) ⇒ source-over returns the destination. \
                 tile_w/tile_h 0 mean `the whole in1 texture`" },

    // ── geom (exact remaps + backward-mapped resamples) ────────────────
    Row { id: "geom.flip_h",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("width", V::U(OUT))]),
          note: "EXCLUDED from identity: a mirror has no neutral parameter \
                 (`width` is the extent, not a strength). Covered instead by \
                 the INVOLUTION property — flip∘flip must be the identity, \
                 which an off-by-one in `width - 1 - x` cannot survive" },
    Row { id: "geom.flip_v",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("height", V::U(OUT))]),
          note: "EXCLUDED from identity: see geom.flip_h; covered by INVOLUTION" },
    Row { id: "geom.rotate90_cw",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("src_w", V::U(OUT)), ("src_h", V::U(OUT))]),
          note: "EXCLUDED from identity: a quarter turn has no neutral \
                 parameter. Covered by the ROUND-TRIP property against \
                 geom.rotate90_ccw" },
    Row { id: "geom.rotate90_ccw",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("src_w", V::U(OUT)), ("src_h", V::U(OUT))]),
          note: "EXCLUDED from identity: see geom.rotate90_cw; covered by ROUND-TRIP" },
    Row { id: "geom.crop",
          identity: Identity::Params(P::Fields(&[("off_x", V::I(0)), ("off_y", V::I(0))])),
          stress: P::Fields(&[("off_x", V::I(5)), ("off_y", V::I(-3))]),
          note: "zero offset ⇒ one exact texel fetch at (x, y)" },
    Row { id: "geom.rotate_bilinear",
          identity: Identity::Params(P::Fields(&[
              ("cos_t", V::F(1.0)), ("sin_t", V::F(0.0)),
              ("src_cx", V::F(HALF_OUT)), ("src_cy", V::F(HALF_OUT)),
              ("dst_cx", V::F(HALF_OUT)), ("dst_cy", V::F(HALF_OUT))])),
          stress: P::Fields(&[
              ("cos_t", V::F(0.8660254)), ("sin_t", V::F(0.5)),
              ("src_cx", V::F(HALF_OUT)), ("src_cy", V::F(HALF_OUT)),
              ("dst_cx", V::F(HALF_OUT)), ("dst_cy", V::F(HALF_OUT))]),
          note: "0° about a shared centre ⇒ sx = x exactly ⇒ both mix factors 0" },
    Row { id: "geom.warp_backward",
          identity: Identity::Params(P::Fields(&[
              ("kind", V::U(0)), ("amount", V::F(0.0)),
              ("cx", V::F(HALF_OUT)), ("cy", V::F(HALF_OUT)),
              ("radius", V::F(HALF_OUT)), ("frequency", V::F(1.0))])),
          stress: P::Fields(&[
              ("kind", V::U(0)), ("amount", V::F(0.5)),
              ("cx", V::F(HALF_OUT)), ("cy", V::F(HALF_OUT)),
              ("radius", V::F(HALF_OUT)), ("frequency", V::F(4.0))]),
          note: "amount 0 is the documented identity for every warp kind" },
    Row { id: "geom.mosaic",
          identity: Identity::Params(P::Fields(&[("cell_px", V::F(1.0))])),
          stress: P::Fields(&[("cell_px", V::F(8.0))]),
          note: "cell 1 ⇒ the cell centre is the pixel itself (and cell 0 is \
                 clamped to 1 rather than dividing by zero)" },
    Row { id: "geom.offset",
          identity: Identity::Params(P::Fields(&[("dx", V::F(0.0)), ("dy", V::F(0.0)), ("edge", V::U(0))])),
          stress: P::Fields(&[("dx", V::F(5.5)), ("dy", V::F(-3.25)), ("edge", V::U(0))]),
          note: "dx = dy = 0 is the documented identity under every edge policy" },
    Row { id: "geom.move_selection",
          identity: Identity::Params(P::Fields(&[("dx", V::F(0.0)), ("dy", V::F(0.0)), ("vacate", V::U(0))])),
          stress: P::Fields(&[("dx", V::F(6.0)), ("dy", V::F(-4.0)), ("vacate", V::U(0))]),
          note: "at zero offset m_src == m_dst, so the vacate and the landing \
                 cancel for every mask value" },

    // ── morph / rank (3×3 and (2r+1)² order statistics) ─────────────────
    Row { id: "morph.dilate",
          identity: Identity::ConstantField,
          stress: NO_PARAMS,
          note: "no params — a fixed 3×3 max. Constant-field preservation \
                 stands in for the missing identity" },
    Row { id: "morph.erode",
          identity: Identity::ConstantField,
          stress: NO_PARAMS,
          note: "no params — a fixed 3×3 min (see morph.dilate)" },
    Row { id: "rank.median3",
          identity: Identity::ConstantField,
          stress: NO_PARAMS,
          note: "no params — a fixed 3×3 median (see morph.dilate)" },
    Row { id: "rank.despeckle",
          identity: Identity::Params(P::Fields(&[("edge_threshold", V::F(0.1)), ("amount", V::F(0.0))])),
          stress: P::Fields(&[("edge_threshold", V::F(0.1)), ("amount", V::F(1.0))]),
          note: "amount 0 ⇒ mix(center, …, 0)" },
    Row { id: "rank.dust_scratches",
          identity: Identity::Params(P::Fields(&[("radius", V::U(1)), ("threshold", V::F(1.0))])),
          stress: P::Fields(&[("radius", V::U(2)), ("threshold", V::F(0.1))]),
          note: "channels live in [0,1] and the substitution test is a STRICT \
                 `>`, so threshold 1 never fires" },

    // ── resample (separable rational scale) ─────────────────────────────
    Row { id: "resample.nearest",
          identity: Identity::Params(P::Fields(&[("inv_scale_x", V::F(1.0)), ("inv_scale_y", V::F(1.0))])),
          stress: P::Fields(&[("inv_scale_x", V::F(2.0)), ("inv_scale_y", V::F(2.0))]),
          note: "unit scale, zero offset ⇒ s = x ⇒ round(x) = x" },
    Row { id: "resample.mitchell",
          identity: Identity::NotDefined,
          stress: P::Fields(&[("inv_scale_x", V::F(2.0)), ("inv_scale_y", V::F(2.0))]),
          note: "EXCLUDED from identity: Mitchell–Netravali (B = C = 1/3) is an \
                 APPROXIMATING filter, not an interpolating one. At integer \
                 phase its taps are (1/18, 16/18, 1/18), so unit-scale \
                 resampling is a mild blur BY DESIGN — asserting identity here \
                 would be asserting the filter is something it is not" },
    Row { id: "resample.lanczos3",
          identity: Identity::Params(P::Fields(&[("inv_scale_x", V::F(1.0)), ("inv_scale_y", V::F(1.0))])),
          stress: P::Fields(&[("inv_scale_x", V::F(2.0)), ("inv_scale_y", V::F(2.0))]),
          note: "sinc is INTERPOLATING — zero at every non-zero integer — so \
                 unit scale really is a pass-through, unlike mitchell" },
];

/// Kernels exempt from the HALF-mask property, with the reason. The
/// all-zero-mask property still applies to them.
const HALF_MASK_EXEMPT: &[(&str, &str)] = &[(
    "geom.move_selection",
    "reads the mask at BOTH p and p−d (m_dst decides what leaves, m_src \
     decides what arrives), so content deliberately LANDS outside the \
     selection. `masked-out ⇒ untouched` is not this kernel's contract; \
     the all-zero-mask property still pins the no-op case",
)];

// ─────────────────────── param-block assembly ────────────────────────

/// Build a kernel's uniform bytes. `Fields` writes each declared field
/// at `4 * index` — valid because ABI param fields are f32/u32/i32 only
/// (all 4-byte aligned, no implicit padding), which
/// [`params_layout_matches_the_wgsl_struct`] independently confirms
/// against the module's own WGSL.
fn params_bytes(def: &KernelDef, p: &P) -> Vec<u8> {
    match p {
        P::Raw(f) => {
            let b = f();
            assert_eq!(
                b.len(),
                def.params.size,
                "{}: raw param block is {} bytes, layout says {}",
                def.id,
                b.len(),
                def.params.size
            );
            b
        }
        P::Fields(list) => {
            for (name, _) in list.iter() {
                assert!(
                    def.params.fields.iter().any(|f| f.name == *name),
                    "{}: table names param `{name}`, which the kernel does not \
                     declare (typo, or the kernel's params changed)",
                    def.id
                );
            }
            let mut out = vec![0u8; def.params.size];
            for (i, field) in def.params.fields.iter().enumerate() {
                let off = i * 4;
                assert!(
                    matches!(field.wgsl_ty, "f32" | "u32" | "i32"),
                    "{}: field `{}` is `{}` — not a flat 4-byte scalar, so this \
                     kernel needs a P::Raw builder",
                    def.id,
                    field.name,
                    field.wgsl_ty
                );
                assert!(
                    off + 4 <= out.len(),
                    "{}: declared fields overflow the {}-byte param block",
                    def.id,
                    def.params.size
                );
                let bytes = match list.iter().find(|(n, _)| *n == field.name) {
                    Some((_, V::F(x))) => {
                        assert_eq!(
                            field.wgsl_ty, "f32",
                            "{}: `{}` is not f32",
                            def.id, field.name
                        );
                        x.to_le_bytes()
                    }
                    Some((_, V::U(x))) => {
                        assert_eq!(
                            field.wgsl_ty, "u32",
                            "{}: `{}` is not u32",
                            def.id, field.name
                        );
                        x.to_le_bytes()
                    }
                    Some((_, V::I(x))) => {
                        assert_eq!(
                            field.wgsl_ty, "i32",
                            "{}: `{}` is not i32",
                            def.id, field.name
                        );
                        x.to_le_bytes()
                    }
                    // Unlisted field: zero. That is the identity value
                    // for every delta-shaped param and the required
                    // value for every ABI pad.
                    None => [0u8; 4],
                };
                out[off..off + 4].copy_from_slice(&bytes);
            }
            out
        }
    }
}

// ───────────────────────────── stimulus ──────────────────────────────

/// An rgba16float tile, tightly packed — the upload format and the
/// comparison format both.
struct Tile {
    w: u32,
    h: u32,
    bytes: Vec<u8>,
}

fn tile_from_fn(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) -> Tile {
    let mut bytes = Vec::with_capacity((w * h * 8) as usize);
    for y in 0..h {
        for x in 0..w {
            for c in f(x, y) {
                bytes.extend_from_slice(&f16::from_f32(c).to_bits().to_le_bytes());
            }
        }
    }
    Tile { w, h, bytes }
}

/// A non-trivial premultiplied stimulus: hard 8×8 blocks (so
/// neighbourhood filters have edges to chew on), two ramps, and four
/// alphas including 0.
///
/// EVERY value is a dyadic rational — k/64, k/32, 7/8, 1/8, and alphas
/// from {1, ½, ¼, 0}. So the f16 round-trip is LOSSLESS and the
/// unpremultiply division is an exponent shift, exact on both lanes.
/// That is deliberate: an identity assertion should fail because a
/// kernel is wrong, never because the stimulus was unrepresentable.
fn stimulus(w: u32, h: u32, salt: u32) -> Tile {
    const ALPHAS: [f32; 4] = [1.0, 0.5, 0.25, 0.0];
    tile_from_fn(w, h, move |x, y| {
        let al = ALPHAS[(((x / 7) + (y / 5) + salt) % 4) as usize];
        let r = ((x * 5 + salt * 13) % 64) as f32 / 64.0;
        let g = ((y * 3 + salt * 7) % 32) as f32 / 32.0;
        let b = if (x / 8 + y / 8) % 2 == 0 {
            0.875
        } else {
            0.125
        };
        [r * al, g * al, b * al, al]
    })
}

/// Two constant premultiplied fields for the `ConstantField` property.
/// TWO, because one would also be passed by a kernel that ignores its
/// input and writes a fixed colour.
const CONSTANTS: [[f32; 4]; 2] = [[0.375, 0.25, 0.125, 0.5], [0.125, 0.875, 0.375, 1.0]];

fn constant_tile(w: u32, h: u32, c: [f32; 4]) -> Tile {
    tile_from_fn(w, h, move |_, _| c)
}

/// r16float mask bytes, all zero — "nothing is selected".
fn mask_zero(w: u32, h: u32) -> Vec<u8> {
    vec![0u8; (w * h * 2) as usize]
}

/// r16float mask, 1.0 on the LEFT half and 0.0 on the right.
fn mask_left_half(w: u32, h: u32) -> Vec<u8> {
    let one = 0x3C00u16.to_le_bytes();
    let mut out = Vec::with_capacity((w * h * 2) as usize);
    for _ in 0..h {
        for x in 0..w {
            if x < w / 2 {
                out.extend_from_slice(&one);
            } else {
                out.extend_from_slice(&[0, 0]);
            }
        }
    }
    out
}

// ───────────────────────────── dispatch ──────────────────────────────

/// Input tiles at the size this kernel's class requires: a `Windowed`
/// kernel gets its window EXPANDED by the radius (the production ROI),
/// everything else gets tiles at the output dims.
/// A `Windowed` kernel that is ALSO binary rides the TILE lane, not the
/// windowed one: `execute_windowed_once` hard-rejects `inputs != 1`
/// ("windowed execution is unary (T1)"). That is not a workaround — it
/// is the live path, since `image-js::apply_binary_kernel` dispatches
/// through `execute_tile_once_async` too, and it is only safe because
/// such kernels DERIVE their halo as `(dims(in0) - dims(outp))/2`,
/// which evaluates to 0 under the tile lane. A kernel that hardcoded
/// `xy + (rx, ry)` would read shifted here (see RFI E-4).
fn rides_the_windowed_lane(def: &KernelDef) -> bool {
    matches!(def.class, KernelClass::Windowed { .. }) && def.inputs == 1
}

fn inputs_for(def: &KernelDef) -> Vec<Tile> {
    match def.class {
        KernelClass::Windowed { radius: (rx, ry) } if def.inputs == 1 => {
            vec![stimulus(
                OUT + 2 * u32::from(rx),
                OUT + 2 * u32::from(ry),
                0,
            )]
        }
        _ => (0..def.inputs.max(1))
            .map(|i| stimulus(OUT, OUT, u32::from(i)))
            .collect(),
    }
}

fn constant_inputs_for(def: &KernelDef, c: [f32; 4]) -> Vec<Tile> {
    match def.class {
        KernelClass::Windowed { radius: (rx, ry) } if def.inputs == 1 => {
            vec![constant_tile(
                OUT + 2 * u32::from(rx),
                OUT + 2 * u32::from(ry),
                c,
            )]
        }
        _ => (0..def.inputs.max(1))
            .map(|_| constant_tile(OUT, OUT, c))
            .collect(),
    }
}

/// The bytes a pass-through must reproduce: `in0` for a point kernel,
/// the CENTRE CROP of the window for a windowed one.
fn passthrough_target(inputs: &[Tile]) -> Vec<u8> {
    let t = &inputs[0];
    if t.w == OUT && t.h == OUT {
        return t.bytes.clone();
    }
    let (rx, ry) = ((t.w - OUT) / 2, (t.h - OUT) / 2);
    let mut out = Vec::with_capacity((OUT * OUT * 8) as usize);
    for y in 0..OUT {
        let row = ((y + ry) * t.w + rx) as usize * 8;
        out.extend_from_slice(&t.bytes[row..row + (OUT * 8) as usize]);
    }
    out
}

/// One dispatch on the lane the kernel's class calls for (see the module
/// docs — this choice is load-bearing for the halo-hardcoding kernels).
fn dispatch(
    ctx: &'static GpuContext,
    def: &'static KernelDef,
    inputs: &[Tile],
    params: &[u8],
    mask: Option<&[u8]>,
) -> Vec<u8> {
    let result = if rides_the_windowed_lane(def) {
        execute_windowed_once(
            ctx,
            def,
            &inputs[0].bytes,
            inputs[0].w,
            inputs[0].h,
            params,
            mask,
            OUT,
            OUT,
        )
    } else {
        let ti: Vec<TileInput<'_>> = inputs
            .iter()
            .map(|t| TileInput {
                f16_bytes: &t.bytes,
            })
            .collect();
        execute_tile_once(ctx, def, &ti, params, mask, OUT, OUT)
    };
    result.unwrap_or_else(|e| panic!("{}: dispatch failed: {e:?}", def.id))
}

// ──────────────────────────── comparison ─────────────────────────────

fn tol_ulps(def: &KernelDef) -> u32 {
    match def.gpu_tolerance {
        Tolerance::Exact => 0,
        Tolerance::ChannelEpsF16(n) => n,
        Tolerance::PerceptualDeltaE(_) => {
            panic!("{}: a ΔE tolerance has no ULP bound in this lane", def.id)
        }
    }
}

/// Worst per-channel f16 ULP divergence over the texels `keep` accepts,
/// with where it happened. NaN-vs-number reads as `u32::MAX`.
fn worst_ulp(
    got: &[u8],
    want: &[u8],
    w: u32,
    keep: impl Fn(u32, u32) -> bool,
) -> (u32, u32, u32, usize) {
    assert_eq!(got.len(), want.len(), "output/expected length mismatch");
    let mut worst = (0u32, 0u32, 0u32, 0usize);
    for i in 0..got.len() / 8 {
        let (x, y) = ((i as u32) % w, (i as u32) / w);
        if !keep(x, y) {
            continue;
        }
        for c in 0..4 {
            let g = u16::from_le_bytes([got[i * 8 + c * 2], got[i * 8 + c * 2 + 1]]);
            let e = u16::from_le_bytes([want[i * 8 + c * 2], want[i * 8 + c * 2 + 1]]);
            let d = f16_ulp_distance(e, g);
            if d > worst.0 {
                worst = (d, x, y, c);
            }
        }
    }
    worst
}

/// f16 inf/NaN: exponent field all ones.
fn nonfinite(bits: u16) -> bool {
    bits & 0x7C00 == 0x7C00
}

// ───────────────────────── table plumbing ────────────────────────────

fn module_kernels() -> Vec<&'static KernelDef> {
    let mut v: Vec<&'static KernelDef> = image_kernels::all_defined()
        .into_iter()
        .filter(|d| d.module)
        .collect();
    v.sort_unstable_by_key(|d| d.id);
    v
}

fn row_for(id: &str) -> Option<&'static Row> {
    TABLE.iter().find(|r| r.id == id)
}

/// The ABI applies the selection mask only where the module asks for it.
/// Derived from the module source rather than hand-listed, so a kernel
/// that GAINS mask handling is covered the moment it does.
fn is_mask_scoped(def: &KernelDef) -> bool {
    def.wgsl.contains("textureLoad(mask")
}

/// Loud skip. "No adapter" must never read as "passed".
fn ctx_or_skip(what: &str) -> Option<&'static GpuContext> {
    match test_device() {
        Some(c) => Some(c),
        None => {
            eprintln!("\n=================== GPU LANE SKIPPED ===================");
            eprintln!("  {what}");
            eprintln!("  NO GPU ADAPTER — this test asserted NOTHING about any");
            eprintln!("  kernel. A green result here is NOT evidence of correct");
            eprintln!("  behaviour. Run where an adapter exists (WGPU_BACKEND=");
            eprintln!("  metal|vulkan|gl) before treating the lane as a gate.");
            eprintln!("=======================================================\n");
            None
        }
    }
}

/// Report every failure at once, named — one kernel at a time would hide
/// the shape of a systemic break behind whichever kernel sorts first.
fn report(what: &str, failures: Vec<String>, checked: usize) {
    eprintln!(
        "{what}: {checked} kernel(s) checked, {} failed",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "\n{what} — {} kernel(s) FAILED:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

// ────────────────────────── structural gates ─────────────────────────
// These need no adapter, so they hold even where the GPU lane skips.

/// EVERY handwritten `module: true` kernel has a table row, and every
/// row names a live kernel. This is the gate that stops a new kernel
/// quietly slipping past the behavioural lane: adding one without a row
/// turns this test red.
#[test]
fn table_covers_every_handwritten_module_kernel() {
    let kernels = module_kernels();
    let missing: Vec<&str> = kernels
        .iter()
        .map(|d| d.id)
        .filter(|id| row_for(id).is_none())
        .collect();
    let stale: Vec<&str> = TABLE
        .iter()
        .map(|r| r.id)
        .filter(|id| !kernels.iter().any(|d| d.id == *id))
        .collect();

    let mut dup = TABLE.iter().map(|r| r.id).collect::<Vec<_>>();
    dup.sort_unstable();
    let n = dup.len();
    dup.dedup();
    assert_eq!(dup.len(), n, "duplicate id in the identity TABLE");

    assert!(
        missing.is_empty(),
        "\n{} handwritten module kernel(s) have NO row in the GPU behavioural \
         lane's TABLE:\n  {}\n\nAdd a row (identity params + stress params + a \
         note) in tests/gpu_module_kernels.rs. A `module: true` kernel has no \
         macro-generated scalar twin, so without a row the ONLY thing gating it \
         is that naga parses its WGSL.\n",
        missing.len(),
        missing.join("\n  ")
    );
    assert!(
        stale.is_empty(),
        "\nTABLE rows name kernels that no longer exist: {}\n",
        stale.join(", ")
    );
    eprintln!(
        "GPU behavioural lane covers {} module kernels",
        kernels.len()
    );
}

/// The declared `ParamsLayout` must agree with the module's own WGSL
/// `struct Params`, in ORDER. The uniform upload is raw bytes, so a
/// metadata list that has drifted from the shader silently feeds every
/// parameter into the wrong slot — and nothing else in the tree checks
/// this for handwritten modules (`min_binding_size` only checks the
/// SIZE). It is also what makes this file's by-name param assembly sound.
#[test]
fn params_layout_matches_the_wgsl_struct() {
    let mut failures = Vec::new();
    let kernels = module_kernels();
    for def in &kernels {
        let Some(wgsl_fields) = wgsl_params_fields(def.wgsl) else {
            failures.push(format!("{}: module has no `struct Params {{`", def.id));
            continue;
        };
        let declared: Vec<&str> = def.params.fields.iter().map(|f| f.name).collect();
        if wgsl_fields.len() < declared.len() {
            failures.push(format!(
                "{}: declares {} param field(s) but the WGSL struct has {}",
                def.id,
                declared.len(),
                wgsl_fields.len()
            ));
            continue;
        }
        for (i, name) in declared.iter().enumerate() {
            if wgsl_fields[i] != *name {
                failures.push(format!(
                    "{}: field {i} is `{name}` in ParamsLayout but `{}` in the \
                     WGSL struct — the uniform would land in the wrong slot",
                    def.id, wgsl_fields[i]
                ));
            }
        }
        if 4 * declared.len() > def.params.size {
            failures.push(format!(
                "{}: {} declared 4-byte fields overflow a {}-byte block",
                def.id,
                declared.len(),
                def.params.size
            ));
        }
    }
    report("params layout ↔ WGSL struct", failures, kernels.len());
}

/// Field names of the module's `struct Params`, in declaration order.
fn wgsl_params_fields(wgsl: &str) -> Option<Vec<String>> {
    const HEAD: &str = "struct Params {";
    let start = wgsl.find(HEAD)? + HEAD.len();
    let body = &wgsl[start..];
    let end = body.find('}')?;
    Some(
        body[..end]
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with("//"))
            .filter_map(|l| l.split(':').next())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    )
}

// ───────────────────────── behavioural lane ──────────────────────────

/// PROPERTY 1 — IDENTITY. At its documented no-op parameters a kernel
/// must return its input within its own declared `gpu_tolerance`.
#[test]
fn module_kernel_identity_holds_on_a_real_adapter() {
    let Some(ctx) = ctx_or_skip("identity") else {
        return;
    };
    let mut failures = Vec::new();
    let (mut checked, mut constant, mut excluded) = (0usize, 0usize, Vec::new());

    for def in module_kernels() {
        let Some(row) = row_for(def.id) else {
            failures.push(format!("{}: no TABLE row", def.id));
            continue;
        };
        let limit = tol_ulps(def);
        match &row.identity {
            Identity::NotDefined => excluded.push(def.id),
            Identity::Params(p) => {
                checked += 1;
                let inputs = inputs_for(def);
                let want = passthrough_target(&inputs);
                let got = dispatch(ctx, def, &inputs, &params_bytes(def, p), None);
                let (ulp, x, y, c) = worst_ulp(&got, &want, OUT, |_, _| true);
                if ulp > limit {
                    failures.push(format!(
                        "  {} — IDENTITY BROKEN: {ulp} f16 ULP at texel ({x},{y}) \
                         channel {c}, declared tolerance {limit}. Identity params: \
                         {}. Expected the input back verbatim.",
                        def.id, row.note
                    ));
                }
            }
            Identity::ConstantField => {
                constant += 1;
                for c0 in CONSTANTS {
                    let inputs = constant_inputs_for(def, c0);
                    let want = passthrough_target(&inputs);
                    let got = dispatch(ctx, def, &inputs, &params_bytes(def, &row.stress), None);
                    let (ulp, x, y, c) = worst_ulp(&got, &want, OUT, |_, _| true);
                    if ulp > limit {
                        failures.push(format!(
                            "  {} — CONSTANT FIELD NOT PRESERVED: {ulp} f16 ULP at \
                             texel ({x},{y}) channel {c} over the constant \
                             {c0:?}, declared tolerance {limit}.",
                            def.id
                        ));
                    }
                }
            }
        }
    }
    eprintln!(
        "identity: {checked} checked, {constant} by constant-field, \
         {} excluded ({})",
        excluded.len(),
        excluded.join(", ")
    );
    report("identity", failures, checked + constant);
}

/// PROPERTY 2a — MASK SCOPING, all-zero mask. Whatever the params, an
/// empty selection must leave the input EXACTLY alone. Exact, not within
/// tolerance: `mix(a, result, 0.0)` is `a` bit-for-bit — unless `result`
/// is NaN, in which case the zero factor does not save it.
#[test]
fn module_kernel_zero_mask_is_an_exact_no_op() {
    let Some(ctx) = ctx_or_skip("zero-mask scoping") else {
        return;
    };
    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut unscoped = Vec::new();

    for def in module_kernels() {
        let Some(row) = row_for(def.id) else {
            failures.push(format!("{}: no TABLE row", def.id));
            continue;
        };
        if !is_mask_scoped(def) {
            unscoped.push(def.id);
            continue;
        }
        checked += 1;
        let inputs = inputs_for(def);
        let want = passthrough_target(&inputs);
        let mask = mask_zero(OUT, OUT);
        let got = dispatch(
            ctx,
            def,
            &inputs,
            &params_bytes(def, &row.stress),
            Some(&mask),
        );
        let (ulp, x, y, c) = worst_ulp(&got, &want, OUT, |_, _| true);
        if ulp != 0 {
            failures.push(format!(
                "  {} — EMPTY SELECTION STILL PAINTED: {ulp} f16 ULP at texel \
                 ({x},{y}) channel {c} with an all-zero mask (expected 0 — the \
                 ABI promises `mix(a, result, 0)`). {ulp_note}",
                def.id,
                ulp_note = if ulp == u32::MAX {
                    "The maximal distance means NaN/Inf reached the blend: a \
                     non-finite `result` survives a zero mask."
                } else {
                    "The kernel is writing outside its selection."
                }
            ));
        }
    }
    eprintln!(
        "zero-mask: {checked} checked; {} kernel(s) never read the mask \
         (Resample/Generator lanes, mask reserved until M3): {}",
        unscoped.len(),
        unscoped.join(", ")
    );
    report("zero-mask scoping", failures, checked);
}

/// PROPERTY 2b — MASK SCOPING, half mask. Selected on the left, empty on
/// the right: the right half must be untouched AND the left half must
/// actually have changed, or "untouched" is a claim about a kernel that
/// does nothing at all.
#[test]
fn module_kernel_half_mask_scopes_the_effect() {
    let Some(ctx) = ctx_or_skip("half-mask scoping") else {
        return;
    };
    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut inert = Vec::new();

    for def in module_kernels() {
        let Some(row) = row_for(def.id) else {
            failures.push(format!("{}: no TABLE row", def.id));
            continue;
        };
        if !is_mask_scoped(def) {
            continue;
        }
        if let Some((_, why)) = HALF_MASK_EXEMPT.iter().find(|(id, _)| *id == def.id) {
            eprintln!("half-mask: {} EXEMPT — {why}", def.id);
            continue;
        }
        checked += 1;
        let inputs = inputs_for(def);
        let want = passthrough_target(&inputs);
        let mask = mask_left_half(OUT, OUT);
        let got = dispatch(
            ctx,
            def,
            &inputs,
            &params_bytes(def, &row.stress),
            Some(&mask),
        );
        let (ulp, x, y, c) = worst_ulp(&got, &want, OUT, |x, _| x >= OUT / 2);
        if ulp != 0 {
            failures.push(format!(
                "  {} — EFFECT LEAKED OUT OF THE SELECTION: {ulp} f16 ULP at \
                 texel ({x},{y}) channel {c}, which is in the mask-0 half.",
                def.id
            ));
        }
        // The left half must differ, or the assertion above is vacuous.
        let (moved, _, _, _) = worst_ulp(&got, &want, OUT, |x, _| x < OUT / 2);
        if moved == 0 {
            inert.push(def.id);
        }
    }
    if !inert.is_empty() {
        eprintln!(
            "half-mask NOTE: {} kernel(s) left the SELECTED half unchanged under \
             their stress params — the scoping assertion is vacuous for them and \
             their stress row should be strengthened: {}",
            inert.len(),
            inert.join(", ")
        );
    }
    report("half-mask scoping", failures, checked);
}

/// PROPERTY 3 — DETERMINISM. Two dispatches, same inputs, byte-identical
/// output. The `gallery.*`/`gen.*` kernels hash coordinate+seed exactly
/// so this holds; one that reached for real randomness fails here.
#[test]
fn module_kernel_dispatch_is_deterministic() {
    let Some(ctx) = ctx_or_skip("determinism") else {
        return;
    };
    let mut failures = Vec::new();
    let mut checked = 0usize;

    for def in module_kernels() {
        let Some(row) = row_for(def.id) else {
            failures.push(format!("{}: no TABLE row", def.id));
            continue;
        };
        checked += 1;
        let inputs = inputs_for(def);
        let params = params_bytes(def, &row.stress);
        let first = dispatch(ctx, def, &inputs, &params, None);
        let second = dispatch(ctx, def, &inputs, &params, None);
        if first != second {
            let at = first
                .iter()
                .zip(&second)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            failures.push(format!(
                "  {} — NOT DETERMINISTIC: two identical dispatches differ, first \
                 at byte {at} (texel {}, channel {}).",
                def.id,
                at / 8,
                (at % 8) / 2
            ));
        }
    }
    report("determinism", failures, checked);
}

/// PROPERTY 4 — FINITENESS. Under working params every output channel
/// must be finite. Weight-sum divisions, `atan2`, `length` and `pow` all
/// have poles; a NaN that escapes here poisons the identity blend
/// downstream, because `mix(a, NaN, 0.0)` is NaN and not `a`.
#[test]
fn module_kernel_output_is_finite() {
    let Some(ctx) = ctx_or_skip("finiteness") else {
        return;
    };
    let mut failures = Vec::new();
    let mut checked = 0usize;

    for def in module_kernels() {
        let Some(row) = row_for(def.id) else {
            failures.push(format!("{}: no TABLE row", def.id));
            continue;
        };
        checked += 1;
        let inputs = inputs_for(def);
        let got = dispatch(ctx, def, &inputs, &params_bytes(def, &row.stress), None);
        let bad = got
            .chunks_exact(2)
            .position(|b| nonfinite(u16::from_le_bytes([b[0], b[1]])));
        if let Some(i) = bad {
            let (texel, ch) = (i / 4, i % 4);
            failures.push(format!(
                "  {} — NON-FINITE OUTPUT at texel ({}, {}) channel {ch} \
                 (raw f16 0x{:04X}).",
                def.id,
                texel as u32 % OUT,
                texel as u32 / OUT,
                u16::from_le_bytes([got[i * 2], got[i * 2 + 1]])
            ));
        }
    }
    report("finiteness", failures, checked);
}

/// The exact geometric remaps have no identity parameter, so they get
/// the next-best structural claim: they are exact PERMUTATIONS of the
/// texel grid, and composing each with its inverse must reproduce the
/// input bit-for-bit. An off-by-one in `width - 1 - x` survives every
/// other property in this file and dies here.
#[test]
fn geometric_remaps_round_trip_exactly() {
    let Some(ctx) = ctx_or_skip("geometric round-trip") else {
        return;
    };
    let pairs: [(&str, &str); 3] = [
        ("geom.flip_h", "geom.flip_h"),
        ("geom.flip_v", "geom.flip_v"),
        ("geom.rotate90_cw", "geom.rotate90_ccw"),
    ];
    let mut failures = Vec::new();

    for (first_id, second_id) in pairs {
        let (Some(a), Some(b)) = (
            image_kernels::lookup(first_id),
            image_kernels::lookup(second_id),
        ) else {
            failures.push(format!("  {first_id}/{second_id}: not in the registry"));
            continue;
        };
        let (Some(ra), Some(rb)) = (row_for(first_id), row_for(second_id)) else {
            failures.push(format!("  {first_id}/{second_id}: no TABLE row"));
            continue;
        };
        let inputs = inputs_for(a);
        let once = dispatch(ctx, a, &inputs, &params_bytes(a, &ra.stress), None);
        let mid = vec![Tile {
            w: OUT,
            h: OUT,
            bytes: once,
        }];
        let twice = dispatch(ctx, b, &mid, &params_bytes(b, &rb.stress), None);
        let (ulp, x, y, c) = worst_ulp(&twice, &inputs[0].bytes, OUT, |_, _| true);
        if ulp != 0 {
            failures.push(format!(
                "  {first_id} then {second_id} — ROUND TRIP IS NOT THE IDENTITY: \
                 {ulp} f16 ULP at texel ({x},{y}) channel {c}. These are exact \
                 texel permutations; any divergence is an indexing bug.",
            ));
        }
    }
    report("geometric round-trip", failures, pairs.len());
}

// ─────────────────────── alpha consistency ───────────────────────
//
// The property that would have CAUGHT RFI E-5, added after the fact
// because none of the four above could see it.
//
// An adjustment is a statement about COLOUR. The same colour at 100%
// and at 25% opacity must therefore receive the same correction — only
// its alpha differs. A kernel that reads premultiplied `rgb` as if it
// were straight fails this, and fails it silently: identity still
// holds (zero deltas round-trip exactly), and the op is still
// deterministic, finite and mask-scoped.
//
// The buffers ARE premultiplied here: `GPU_WORKING` declares it, and as
// of the E-5 fix `apply_point_kernel` associates alpha before dispatch
// (gated on opacity, as `fill.rs` does), so the lane's stimulus and the
// shipping path finally agree. That agreement is the point — the
// original lane reported the one kernel that matched the DATA as the
// broken one, because its stimulus encoded the contract rather than
// what the dispatcher actually sent.

/// Kernels for which alpha-dependence is CORRECT, each with the reason.
/// An exemption is a claim, so it has to carry one.
const ALPHA_EXEMPT: &[(&str, &str)] = &[
    (
        "adjust.exposure",
        "a linear scale commutes with premultiplication, so it operates in \
         premultiplied space deliberately and never dissociates",
    ),
    (
        "adjust.threshold",
        "compares in premultiplied space by design — thresholding the \
         dissociated colour would binarize a near-transparent pixel on \
         evidence it does not really have",
    ),
];

#[test]
fn an_adjustment_is_the_same_at_every_alpha() {
    let Some(ctx) = ctx_or_skip("alpha consistency") else {
        return;
    };
    // One straight colour, carried at four opacities.
    const STRAIGHT: [f32; 3] = [0.75, 0.375, 0.125];
    const ALPHAS: [f32; 4] = [1.0, 0.5, 0.25, 0.125];

    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for def in module_kernels() {
        if !def.id.starts_with("adjust.") || def.inputs != 1 {
            continue;
        }
        if ALPHA_EXEMPT.iter().any(|(id, _)| *id == def.id) {
            continue;
        }
        let Some(row) = TABLE.iter().find(|r| r.id == def.id) else {
            continue;
        };
        let params = params_bytes(def, &row.stress);

        // The reference: the fully-opaque case, where premultiplied and
        // straight coincide and no kernel can get it wrong.
        let mut reference: Option<[f32; 3]> = None;
        for a in ALPHAS {
            let tile = tile_from_fn(OUT, OUT, |_, _| {
                [STRAIGHT[0] * a, STRAIGHT[1] * a, STRAIGHT[2] * a, a]
            });
            let out = dispatch(ctx, def, &[tile], &params, None);
            // Dissociate the result and compare COLOUR, not premultiplied
            // bytes — those must differ, since the alphas do.
            let px = |c: usize| f16::from_le_bytes([out[c * 2], out[c * 2 + 1]]).to_f32();
            let oa = px(3);
            if oa <= 0.0 {
                continue;
            }
            let got = [px(0) / oa, px(1) / oa, px(2) / oa];
            match reference {
                None => reference = Some(got),
                Some(want) => {
                    // Generous: f16 through a divide at alpha 1/8 loses
                    // real precision, and the failure this guards
                    // against is gross (0.25 -> 0.75), not marginal.
                    let d = (0..3).map(|i| (got[i] - want[i]).abs()).fold(0.0, f32::max);
                    if d > 0.02 {
                        failures.push(format!(
                            "{}: colour ({:.3},{:.3},{:.3}) at alpha {a} came back \
                             ({:.3},{:.3},{:.3}) but ({:.3},{:.3},{:.3}) at alpha 1 — \
                             delta {d:.3}. The adjustment depends on OPACITY, which \
                             means it is reading premultiplied rgb as straight colour.",
                            def.id,
                            STRAIGHT[0],
                            STRAIGHT[1],
                            STRAIGHT[2],
                            got[0],
                            got[1],
                            got[2],
                            want[0],
                            want[1],
                            want[2],
                        ));
                    }
                }
            }
        }
        checked += 1;
    }

    assert!(
        checked >= 15,
        "expected the whole adjust family; only checked {checked}"
    );
    assert!(failures.is_empty(), "\n{}", failures.join("\n\n"));
    eprintln!(
        "[alpha-consistency] {checked} adjust kernels agree across 4 opacities; \
         {} exempt with reasons",
        ALPHA_EXEMPT.len()
    );
}
