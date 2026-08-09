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

//! Gallery family (T1) — the PRIMITIVES behind Photoshop's Filter
//! Gallery (Artistic, Brush Strokes, Sketch, Texture, plus the gallery's
//! Distort and Stylize entries).
//!
//! The gallery presents ~47 filters. It is not 47 algorithms: it is a
//! dozen real operations, each exposed several times with different
//! defaults and a different name. Building 47 near-duplicate kernels
//! would multiply the surface to maintain without adding a capability,
//! so this file builds the OPERATIONS and lets the panel presets supply
//! the names. Each kernel's doc-comment lists the gallery filters it
//! stands behind — that mapping is the contract, and a filter with no
//! kernel listed against it is honestly not covered.
//!
//! Three conventions hold across every kernel here:
//!
//! 1. **`amount` is the last word.** Every kernel ends
//!    `mix(a, styled, amount)`, so `amount = 0.0` returns the input
//!    EXACTLY (`styled` is clamped finite everywhere, so the `0.0`
//!    factor cannot smuggle a NaN through the mix). That is the
//!    identity case each doc-comment states, and it doubles as the
//!    effect-opacity a designer actually wants.
//! 2. **Position enters through `ox`/`oy`.** Every pattern that depends
//!    on WHERE it is — the halftone lattice, the hatch phase, the glass
//!    cells, the grain — reads `ox + gid.x`, not `gid.x`. The engines
//!    dispatch tile by tile; a pattern keyed on the tile-local
//!    invocation restarts at every tile boundary and seams. `gen.noise`
//!    established this param pair; these follow it.
//! 3. **No host randomness.** Noise is a PCG hash of the (global
//!    coordinate, `seed`) pair, so the same seed reproduces the same
//!    pixels on every re-run, on every adapter, at every zoom.
//!
//! Provenance: these are standard image-processing techniques —
//! Kuwahara, Hachimura & Eiho's edge-preserving region filter (1976);
//! Voronoi / cellular texturing (Worley, SIGGRAPH 1996); classic
//! rotated-screen halftoning with the process angles from print
//! practice; value noise with a Hermite fade; bump-mapped relief from a
//! luminance heightfield; the PCG output-permutation hash (Jarzynski &
//! Olano, JCGT 2020). NO Adobe code was read or ported — this file was
//! written from the literature above and from what each named filter
//! visibly DOES, which is a fact about behaviour, not an expression.

use crate::{KernelClass, KernelDef, ParamField, ParamsLayout, Tolerance};

// ── shared WGSL fragments ──────────────────────────────────────────
//
// WGSL has no include mechanism on this lane (each `module: true`
// kernel is one self-contained string), so the helpers below are
// repeated verbatim in the shaders that need them. They are kept
// character-identical between kernels ON PURPOSE: a hash that drifts
// between two kernels would make two effects disagree about where the
// same grain sits, and that is invisible until someone stacks them.
//
//   pcg / hash21 — deterministic noise, keyed on (x, y, seed).
//   luma         — Rec. 709 luminance.
//   win_org/tap  — the windowed-input addressing rule, below.
//
// WINDOWED ADDRESSING. Under ABI v1.1 a `Windowed { radius: (rx, ry) }`
// kernel is handed `in0` as the output region EXPANDED by the radius
// (image-graph's `gather_window` builds it), so the sample matching
// output texel (x, y) sits at (x + rx, y + ry). Rather than hard-code
// the radius a second time in the shader — where it can silently fall
// out of step with the `KernelDef` — `win_org()` derives it from the
// two textures' size difference. That also degrades correctly to (0, 0)
// when a caller binds an unexpanded input, which is what a bare
// `execute_tile_once` harness does.

// ───────────────────────────── kuwahara ────────────────────────────
//
// The painterly one. Split the neighbourhood into four overlapping
// quadrants, measure the luminance VARIANCE of each, and output the
// mean colour of the calmest one. Because the winning quadrant never
// straddles an edge (a quadrant that did would be the noisy one), flat
// regions get flattened into brush-like patches while the edges between
// them stay put. Averaging the whole window instead — a box blur —
// destroys exactly the edges that make this read as paint.

/// `gallery.kuwahara` params: region `radius_px` and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryKuwaharaParams {
    pub radius_px: f32,
    pub amount: f32,
    pub _pad0: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl GalleryKuwaharaParams {
    pub fn new(radius_px: f32, amount: f32) -> Self {
        Self {
            radius_px,
            amount,
            _pad0: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Edge-preserving painterly regions (Kuwahara et al., 1976).
///
/// Serves: **Paint Daubs**, **Palette Knife**, **Watercolor**,
/// **Underpainting**, **Dry Brush**, **Smudge Stick** — they differ in
/// region size and in how much of the original is left showing, which
/// is `radius_px` and `amount`.
///
/// IDENTITY: `amount = 0.0` returns the input unchanged.
pub static GALLERY_KUWAHARA: KernelDef = KernelDef {
    id: "gallery.kuwahara",
    class: KernelClass::Windowed { radius: (8, 8) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryKuwaharaParams>(),
        fields: &[
            ParamField {
                name: "radius_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_KUWAHARA_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

// The quadrant scan is ASCENDING q with a STRICT `<` on the variance,
// so the winner is decided identically on both lanes. A tie needs two
// quadrants with bit-equal variance, which happens only where both are
// flat — and there their means agree too, so the tie is not observable
// in the output.
const GALLERY_KUWAHARA_WGSL: &str = "\
// paged.image kernel `gallery.kuwahara` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    radius_px: f32,
    amount: f32,
    _pad0: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const R_MAX : i32 = 8;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn win_org() -> vec2<i32> {
    let din = vec2<i32>(textureDimensions(in0));
    let dout = vec2<i32>(textureDimensions(outp));
    return (din - dout) / 2;
}

fn tap(c: vec2<i32>) -> vec4<f32> {
    let din = vec2<i32>(textureDimensions(in0));
    return textureLoad(in0, clamp(c, vec2<i32>(0), din - vec2<i32>(1)), 0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let c = xy + win_org();
    let a = tap(c);

    // At least 1: a zero-radius quadrant is a single pixel with zero
    // variance, which would win every time and return the input while
    // pretending to have filtered.
    let ri = i32(clamp(round(params.radius_px), 1.0, f32(R_MAX)));

    var best_var = 1.0e9;
    var best = a;
    for (var q : i32 = 0; q < 4; q = q + 1) {
        // Quadrants OVERLAP on the centre row and column — that shared
        // pixel is what keeps the four means comparable at an edge.
        var x0 = -ri;
        var x1 = 0;
        if ((q & 1) == 1) { x0 = 0; x1 = ri; }
        var y0 = -ri;
        var y1 = 0;
        if ((q & 2) == 2) { y0 = 0; y1 = ri; }

        var sum = vec4<f32>(0.0);
        var s1 = 0.0;
        var s2 = 0.0;
        var n = 0.0;
        // Loop the COMPILE-TIME max and skip: the real radius is a
        // uniform, and a loop bound that varies with it would make the
        // trip count non-uniform for no gain.
        for (var dy : i32 = -R_MAX; dy <= R_MAX; dy = dy + 1) {
            if (dy < y0 || dy > y1) { continue; }
            for (var dx : i32 = -R_MAX; dx <= R_MAX; dx = dx + 1) {
                if (dx < x0 || dx > x1) { continue; }
                let s = tap(c + vec2<i32>(dx, dy));
                let l = luma(s.rgb);
                sum = sum + s;
                s1 = s1 + l;
                s2 = s2 + l * l;
                n = n + 1.0;
            }
        }
        let inv = 1.0 / n;
        let mean_l = s1 * inv;
        // E[l²] − E[l]² can go a hair below zero on cancellation; the
        // max keeps a flat region's variance at exactly 0 rather than
        // at −1e−9, which would beat a genuinely flat neighbour.
        let v = max(s2 * inv - mean_l * mean_l, 0.0);
        if (v < best_var) {
            best_var = v;
            best = sum * inv;
        }
    }

    let styled = vec4<f32>(clamp(best.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ────────────────────────── posterize_edges ────────────────────────
//
// Two operations that always ship together in the gallery: quantise
// the colour to a few flat levels, then draw a dark outline where the
// image has an edge. Doing only the first gives Posterize (which
// already exists as `adjust.posterize`); the OUTLINE is what makes this
// read as illustration rather than as a broken gradient.

/// `gallery.posterize_edges` params: quantisation `levels`, outline
/// `edge_amount`/`edge_threshold`, and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryPosterizeEdgesParams {
    pub levels: f32,
    pub edge_amount: f32,
    pub edge_threshold: f32,
    pub amount: f32,
}

#[allow(clippy::new_without_default)]
impl GalleryPosterizeEdgesParams {
    pub fn new(levels: f32, edge_amount: f32, edge_threshold: f32, amount: f32) -> Self {
        Self {
            levels,
            edge_amount,
            edge_threshold,
            amount,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Posterised colour with darkened outlines.
///
/// Serves: **Poster Edges**, **Cutout**, **Ink Outlines**, **Torn
/// Edges** (at two levels) — the difference between them is how many
/// levels survive and how heavy the ink is.
///
/// IDENTITY: `amount = 0.0` returns the input unchanged.
pub static GALLERY_POSTERIZE_EDGES: KernelDef = KernelDef {
    id: "gallery.posterize_edges",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryPosterizeEdgesParams>(),
        fields: &[
            ParamField {
                name: "levels",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "edge_amount",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "edge_threshold",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_POSTERIZE_EDGES_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_POSTERIZE_EDGES_WGSL: &str = "\
// paged.image kernel `gallery.posterize_edges` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    levels: f32,
    edge_amount: f32,
    edge_threshold: f32,
    amount: f32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn win_org() -> vec2<i32> {
    let din = vec2<i32>(textureDimensions(in0));
    let dout = vec2<i32>(textureDimensions(outp));
    return (din - dout) / 2;
}

fn tap(c: vec2<i32>) -> vec4<f32> {
    let din = vec2<i32>(textureDimensions(in0));
    return textureLoad(in0, clamp(c, vec2<i32>(0), din - vec2<i32>(1)), 0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let c = xy + win_org();
    let a = tap(c);

    // Two levels minimum: one level has no interior step, and (L − 1)
    // in the denominator would be zero.
    let lv = max(round(params.levels), 2.0);
    let v = clamp(a.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    // min(..., L − 1) catches v == 1.0, which floors to L and would
    // otherwise post above white.
    let qi = min(floor(v * lv), vec3<f32>(lv - 1.0));
    let post = qi / (lv - 1.0);

    // Sobel on LUMINANCE here (unlike conv.find_edges, which runs per
    // channel): the outline of a poster is one ink, so a single
    // achromatic gradient is the right signal and costs a third as
    // much arithmetic.
    let tl = luma(tap(c + vec2<i32>(-1, -1)).rgb);
    let tc = luma(tap(c + vec2<i32>( 0, -1)).rgb);
    let tr = luma(tap(c + vec2<i32>( 1, -1)).rgb);
    let ml = luma(tap(c + vec2<i32>(-1,  0)).rgb);
    let mr = luma(tap(c + vec2<i32>( 1,  0)).rgb);
    let bl = luma(tap(c + vec2<i32>(-1,  1)).rgb);
    let bc = luma(tap(c + vec2<i32>( 0,  1)).rgb);
    let br = luma(tap(c + vec2<i32>( 1,  1)).rgb);
    let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    let mag = sqrt(gx * gx + gy * gy);

    // A SOFT knee, not a step: the threshold is the number a designer
    // drags, and a hard comparison makes the outline flicker in and out
    // pixel by pixel exactly while they drag it.
    let thr = max(params.edge_threshold, 0.0);
    let e = smoothstep(thr, thr + 0.05, mag) * clamp(params.edge_amount, 0.0, 1.0);

    let inked = clamp(post * (1.0 - e), vec3<f32>(0.0), vec3<f32>(1.0));
    let styled = vec4<f32>(inked, a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ─────────────────────────── glowing_edges ─────────────────────────
//
// The inverse of Find Edges: keep the gradient, throw the picture away,
// and let the edges glow on black. Run per CHANNEL so the glow inherits
// the colour of the edge that produced it — a red object on green glows
// red, which is the whole point of the filter's name and is lost if the
// gradient is taken on luminance.

/// `gallery.glowing_edges` params: edge `intensity`, the `smoothness`
/// of the glow's falloff, and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryGlowingEdgesParams {
    pub intensity: f32,
    pub smoothness: f32,
    pub amount: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl GalleryGlowingEdgesParams {
    pub fn new(intensity: f32, smoothness: f32, amount: f32) -> Self {
        Self {
            intensity,
            smoothness,
            amount,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Colourised, boosted edge magnitude on black.
///
/// Serves: **Glowing Edges**, **Neon Glow** (blend back at a partial
/// `amount` to keep the subject readable behind the glow).
///
/// IDENTITY: `amount = 0.0` returns the input unchanged.
pub static GALLERY_GLOWING_EDGES: KernelDef = KernelDef {
    id: "gallery.glowing_edges",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryGlowingEdgesParams>(),
        fields: &[
            ParamField {
                name: "intensity",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "smoothness",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_GLOWING_EDGES_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_GLOWING_EDGES_WGSL: &str = "\
// paged.image kernel `gallery.glowing_edges` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    intensity: f32,
    smoothness: f32,
    amount: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn win_org() -> vec2<i32> {
    let din = vec2<i32>(textureDimensions(in0));
    let dout = vec2<i32>(textureDimensions(outp));
    return (din - dout) / 2;
}

fn tap(c: vec2<i32>) -> vec3<f32> {
    let din = vec2<i32>(textureDimensions(in0));
    return textureLoad(in0, clamp(c, vec2<i32>(0), din - vec2<i32>(1)), 0).rgb;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let c = xy + win_org();
    let a = textureLoad(in0, clamp(c, vec2<i32>(0),
        vec2<i32>(textureDimensions(in0)) - vec2<i32>(1)), 0);

    let tl = tap(c + vec2<i32>(-1, -1));
    let tc = tap(c + vec2<i32>( 0, -1));
    let tr = tap(c + vec2<i32>( 1, -1));
    let ml = tap(c + vec2<i32>(-1,  0));
    let mr = tap(c + vec2<i32>( 1,  0));
    let bl = tap(c + vec2<i32>(-1,  1));
    let bc = tap(c + vec2<i32>( 0,  1));
    let br = tap(c + vec2<i32>( 1,  1));

    let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    let mag = sqrt(gx * gx + gy * gy);

    let g = clamp(mag * params.intensity, vec3<f32>(0.0), vec3<f32>(1.0));
    // `smoothness` interpolates toward a Hermite response: at 0 the
    // glow is linear in the gradient (every faint edge shows), at 1 the
    // weak edges are pushed down and the strong ones toward saturation,
    // which is what separates a neon tube from a grey outline.
    let s = clamp(params.smoothness, 0.0, 1.0);
    let ge = mix(g, smoothstep(vec3<f32>(0.0), vec3<f32>(1.0), g), s);

    let styled = vec4<f32>(ge, a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ───────────────────────────── halftone ────────────────────────────
//
// A print screen: cover the plane with a rotated lattice of cells and
// grow a dot in each one until its AREA matches the ink the pixel asks
// for. Area, not radius — ink coverage is what the eye integrates, so
// the radius has to be the square root of the tone or the midtones come
// out far too dark.
//
// The three channels get their own screen angles. Two screens sharing
// an angle beat against each other into moiré; offsetting them (the
// classic 15° / 75° / 0° process angles) turns that beat into the
// rosette that print has used for a century.

/// `gallery.halftone` params: tile origin, `cell_px` screen pitch,
/// screen `angle_deg`, dot-edge `contrast`, and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryHalftoneParams {
    pub ox: i32,
    pub oy: i32,
    pub cell_px: f32,
    pub angle_deg: f32,
    pub contrast: f32,
    pub amount: f32,
    pub _pad0: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl GalleryHalftoneParams {
    pub fn new(ox: i32, oy: i32, cell_px: f32, angle_deg: f32, contrast: f32, amount: f32) -> Self {
        Self {
            ox,
            oy,
            cell_px,
            angle_deg,
            contrast,
            amount,
            _pad0: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Rotated dot screen, one per channel at process angles.
///
/// Serves: **Halftone Pattern** (feed it a desaturated input for the
/// Sketch version), **Color Halftone**, **Screen** presets.
///
/// IDENTITY: `amount = 0.0` returns the input unchanged.
pub static GALLERY_HALFTONE: KernelDef = KernelDef {
    id: "gallery.halftone",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryHalftoneParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "cell_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "angle_deg",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "contrast",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_HALFTONE_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_HALFTONE_WGSL: &str = "\
// paged.image kernel `gallery.halftone` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    cell_px: f32,
    angle_deg: f32,
    contrast: f32,
    amount: f32,
    _pad0: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

// One channel through one rotated screen. Returns the channel's new
// value: 1 where the paper shows, 0 where the dot covers it.
fn screen(p: vec2<f32>, v: f32, cell: f32, ang: f32, soft: f32) -> f32 {
    let ca = cos(ang);
    let sa = sin(ang);
    let r = vec2<f32>(p.x * ca - p.y * sa, p.x * sa + p.y * ca) / cell;
    let f = fract(r) - vec2<f32>(0.5);
    let dist = length(f) * cell;

    let ink = 1.0 - clamp(v, 0.0, 1.0);
    // AREA is proportional to ink, so the radius goes as its square
    // root. 0.7071 is the cell's circumradius over its pitch: at full
    // ink the dot covers the whole cell, which is what lets the screen
    // reach solid black instead of stalling at the inscribed circle's
    // 78.5%.
    let rad = sqrt(ink) * cell * 0.70710678;
    let cov = 1.0 - smoothstep(rad - soft, rad + soft, dist);
    return 1.0 - cov;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(in0, xy, 0);

    // GLOBAL coordinates: the lattice must be continuous across tile
    // boundaries or every tile edge becomes a visible seam in the dots.
    let p = vec2<f32>(f32(params.ox + i32(gid.x)), f32(params.oy + i32(gid.y)));
    let cell = max(params.cell_px, 1.0);
    // Antialias width, in pixels. Never zero: smoothstep needs a
    // non-empty interval, and a hard dot edge crawls under motion.
    let soft = max(0.5 / max(params.contrast, 0.01), 0.001);
    let base = radians(params.angle_deg);

    let sr = screen(p, a.r, cell, base + 0.26179939, soft);
    let sg = screen(p, a.g, cell, base + 1.30899694, soft);
    let sb = screen(p, a.b, cell, base, soft);

    let styled = vec4<f32>(sr, sg, sb, a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ────────────────────────────── grain ──────────────────────────────
//
// Additive procedural noise. The only interesting decision here is
// `size_px`: hashing the raw coordinate gives 1-pixel grain that
// shimmers and disappears at any zoom below 100%. Quantising the
// coordinate FIRST makes neighbouring pixels share a sample, so the
// speckle clumps at a chosen size and survives being looked at.

/// `gallery.grain` params: tile origin, hash `seed`, grain `size_px`,
/// `mono` (1 = one sample for all channels), and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryGrainParams {
    pub ox: i32,
    pub oy: i32,
    pub seed: u32,
    pub size_px: f32,
    pub mono: u32,
    pub amount: f32,
    pub _pad0: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl GalleryGrainParams {
    pub fn new(ox: i32, oy: i32, seed: u32, size_px: f32, mono: bool, amount: f32) -> Self {
        Self {
            ox,
            oy,
            seed,
            size_px,
            mono: u32::from(mono),
            amount,
            _pad0: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Deterministic additive grain.
///
/// Serves: **Film Grain**, **Grain** (its Regular/Clumped/Speckle modes
/// are `size_px`), **Reticulation**, **Mezzotint** (at `amount` near 1
/// the clamp drives the speckle to pure black and white), **Diffuse
/// Glow** (grain plus a lifted `amount`).
///
/// IDENTITY: `amount = 0.0` returns the input unchanged — `amount` is
/// both the noise amplitude and the blend, so there is one control and
/// one identity rather than two that can disagree.
pub static GALLERY_GRAIN: KernelDef = KernelDef {
    id: "gallery.grain",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryGrainParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "seed",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "size_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "mono",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_GRAIN_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_GRAIN_WGSL: &str = "\
// paged.image kernel `gallery.grain` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    seed: u32,
    size_px: f32,
    mono: u32,
    amount: f32,
    _pad0: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

// PCG output-permutation hash (Jarzynski & Olano, JCGT 2020) — the hash
// `gen.noise` uses, repeated because WGSL has no includes here. u32
// arithmetic wraps identically everywhere, so the same (x, y, seed)
// gives the same sample on every adapter and every re-run.
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash21(x: i32, y: i32, seed: u32) -> f32 {
    let h = pcg(bitcast<u32>(x) ^ pcg(bitcast<u32>(y) ^ pcg(seed)));
    // 2^-24 exactly — the largest step an f32 mantissa represents
    // without rounding, so the [0, 1) map is exact on both lanes.
    return f32(h >> 8u) * 5.9604644775390625e-8;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(in0, xy, 0);

    let gx = params.ox + i32(gid.x);
    let gy = params.oy + i32(gid.y);
    // Quantise the coordinate BEFORE hashing: that is what gives the
    // grain a size instead of a per-pixel shimmer.
    let s = max(params.size_px, 1.0);
    let cx = i32(floor(f32(gx) / s));
    let cy = i32(floor(f32(gy) / s));

    let n0 = hash21(cx, cy, params.seed) - 0.5;
    var n = vec3<f32>(n0);
    if (params.mono == 0u) {
        // Decorrelated per channel — colour grain, as film emulsion
        // has three layers that do not agree with each other.
        n = vec3<f32>(
            n0,
            hash21(cx, cy, params.seed ^ 2654435769u) - 0.5,
            hash21(cx, cy, params.seed ^ 2246822507u) - 0.5);
    }

    let styled = vec4<f32>(clamp(a.rgb + n, vec3<f32>(0.0), vec3<f32>(1.0)), a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ───────────────────────────── diffuse ─────────────────────────────
//
// Replace each pixel with a neighbour picked at random. Isotropic, this
// is Diffuse / Spatter — the image shatters into a scatter. Bias the
// pick along one direction and the same operation becomes a STROKE:
// the scatter elongates into streaks, which is Sprayed Strokes.
//
// The pick is a hash of the pixel, not a draw from a stream: two runs
// of the same tile must produce the same scatter or the undo journal
// and the tile cache disagree about what the tile contains.

/// `gallery.diffuse` params: tile origin, hash `seed`, scatter
/// `radius_px`, stroke `angle_deg`, `anisotropy` (0 = round scatter,
/// 1 = pure streak), and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryDiffuseParams {
    pub ox: i32,
    pub oy: i32,
    pub seed: u32,
    pub radius_px: f32,
    pub angle_deg: f32,
    pub anisotropy: f32,
    pub amount: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default, clippy::too_many_arguments)]
impl GalleryDiffuseParams {
    pub fn new(
        ox: i32,
        oy: i32,
        seed: u32,
        radius_px: f32,
        angle_deg: f32,
        anisotropy: f32,
        amount: f32,
    ) -> Self {
        Self {
            ox,
            oy,
            seed,
            radius_px,
            angle_deg,
            anisotropy,
            amount,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Random local displacement, optionally biased along a stroke angle.
///
/// Serves: **Diffuse**, **Spatter**, **Sprayed Strokes**
/// (`anisotropy` near 1), **Ocean Ripple**'s scatter component,
/// **Crystallize**'s edge break-up.
///
/// IDENTITY: `amount = 0.0` returns the input unchanged;
/// `radius_px = 0.0` independently collapses the displacement to zero.
pub static GALLERY_DIFFUSE: KernelDef = KernelDef {
    id: "gallery.diffuse",
    class: KernelClass::Windowed { radius: (8, 8) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryDiffuseParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "seed",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "radius_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "angle_deg",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "anisotropy",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_DIFFUSE_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_DIFFUSE_WGSL: &str = "\
// paged.image kernel `gallery.diffuse` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    seed: u32,
    radius_px: f32,
    angle_deg: f32,
    anisotropy: f32,
    amount: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const R_MAX : f32 = 8.0;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash21(x: i32, y: i32, seed: u32) -> f32 {
    let h = pcg(bitcast<u32>(x) ^ pcg(bitcast<u32>(y) ^ pcg(seed)));
    return f32(h >> 8u) * 5.9604644775390625e-8;
}

fn win_org() -> vec2<i32> {
    let din = vec2<i32>(textureDimensions(in0));
    let dout = vec2<i32>(textureDimensions(outp));
    return (din - dout) / 2;
}

fn tap(c: vec2<i32>) -> vec4<f32> {
    let din = vec2<i32>(textureDimensions(in0));
    return textureLoad(in0, clamp(c, vec2<i32>(0), din - vec2<i32>(1)), 0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let c = xy + win_org();
    let a = tap(c);

    let gx = params.ox + i32(gid.x);
    let gy = params.oy + i32(gid.y);
    let r = clamp(params.radius_px, 0.0, R_MAX);

    let h1 = hash21(gx, gy, params.seed);
    let h2 = hash21(gx, gy, params.seed ^ 668265261u);
    let ang = h1 * 6.28318531;
    // sqrt makes the pick UNIFORM over the disc. Without it the samples
    // bunch near the centre and the scatter barely reads at any radius.
    let rad = sqrt(h2) * r;

    let sa = radians(params.angle_deg);
    let dir = vec2<f32>(cos(sa), -sin(sa));
    let per = vec2<f32>(-dir.y, dir.x);
    // Squashing the ACROSS component (not stretching the along one) is
    // what keeps the maximum displacement inside the declared window
    // radius while the scatter turns into a streak.
    let k = clamp(params.anisotropy, 0.0, 1.0);
    let v = dir * (cos(ang) * rad) + per * (sin(ang) * rad * (1.0 - k));
    let off = vec2<i32>(i32(round(v.x)), i32(round(v.y)));

    let styled = tap(c + off);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ──────────────────────────── crosshatch ───────────────────────────
//
// Tone carried by LINE WEIGHT, which is how one ink draws greys. Each
// set of parallel strokes removes some of the paper; two sets crossed
// remove more where they overlap, and the multiply is what makes the
// crossings darker than either stroke alone — an additive blend would
// make them the same darkness and the drawing would look printed rather
// than drawn.

/// `gallery.crosshatch` params: tile origin, stroke `angle_deg`,
/// `spacing_px` between strokes, `strength` (how fast the strokes
/// thicken with darkness), `sets` (1..3 stroke directions), and the
/// effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryCrosshatchParams {
    pub ox: i32,
    pub oy: i32,
    pub angle_deg: f32,
    pub spacing_px: f32,
    pub strength: f32,
    pub sets: u32,
    pub amount: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default, clippy::too_many_arguments)]
impl GalleryCrosshatchParams {
    pub fn new(
        ox: i32,
        oy: i32,
        angle_deg: f32,
        spacing_px: f32,
        strength: f32,
        sets: u32,
        amount: f32,
    ) -> Self {
        Self {
            ox,
            oy,
            angle_deg,
            spacing_px,
            strength,
            sets,
            amount,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Directional line texture modulated by luminance.
///
/// Serves: **Crosshatch** (`sets = 2`), **Angled Strokes** and
/// **Graphic Pen** (`sets = 1`), **Sumi-e** (`sets = 3`, wide
/// spacing), **Halftone Pattern**'s Line mode, **Note Paper**'s
/// fibre.
///
/// IDENTITY: `amount = 0.0` returns the input unchanged. (`strength =
/// 0.0` suppresses every stroke but leaves blank paper, which is a
/// different thing from the input.)
pub static GALLERY_CROSSHATCH: KernelDef = KernelDef {
    id: "gallery.crosshatch",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryCrosshatchParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "angle_deg",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "spacing_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "strength",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "sets",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_CROSSHATCH_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_CROSSHATCH_WGSL: &str = "\
// paged.image kernel `gallery.crosshatch` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    angle_deg: f32,
    spacing_px: f32,
    strength: f32,
    sets: u32,
    amount: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const MAX_SETS : i32 = 3;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Ink coverage of one stroke set at this point, in [0, 1].
fn hatch(p: vec2<f32>, ang: f32, spacing: f32, dens: f32) -> f32 {
    let hw = 0.5 * clamp(dens, 0.0, 1.0);
    // No density, no stroke. Falling through would hand smoothstep a
    // zero-width band and return half coverage on the line centres —
    // faint ghost strokes where the picture asked for none.
    if (hw <= 0.0) { return 0.0; }
    // Project onto the stroke NORMAL: the fractional part of that
    // projection is where we sit between two adjacent strokes.
    let n = vec2<f32>(cos(ang), sin(ang));
    let u = dot(p, n) / spacing;
    let dst = abs(fract(u) - 0.5);
    // One pixel expressed in u units — the antialias width. Without it
    // the strokes alias into dotted lines as the spacing slider moves.
    let e = max(0.5 / spacing, 0.001);
    return 1.0 - smoothstep(hw - e, hw + e, dst);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(in0, xy, 0);

    let p = vec2<f32>(f32(params.ox + i32(gid.x)), f32(params.oy + i32(gid.y)));
    let ns = i32(clamp(params.sets, 1u, 3u));
    let sp = max(params.spacing_px, 1.0);
    let l = luma(clamp(a.rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
    // Darker pixel, wider stroke — one ink, many greys.
    let dens = clamp((1.0 - l) * max(params.strength, 0.0), 0.0, 1.0);

    var paper = 1.0;
    for (var k : i32 = 0; k < MAX_SETS; k = k + 1) {
        if (k >= ns) { continue; }
        // Spread the sets over a half turn: two sets land 90 deg apart
        // (true crosshatch), three land 60 deg apart. A full turn would
        // put set 2 back on top of set 0, since a stroke has no side.
        let ang = radians(params.angle_deg) + 3.14159265 * f32(k) / f32(ns);
        // Multiplicative: crossings darken more than either stroke, the
        // way ink laid twice on paper does.
        paper = paper * (1.0 - hatch(p, ang, sp, dens));
    }

    let styled = vec4<f32>(vec3<f32>(clamp(paper, 0.0, 1.0)), a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ──────────────────────────── bas_relief ───────────────────────────
//
// Read luminance as a HEIGHTFIELD and light it from one side. The
// response is the gradient's component along the light, not the surface
// normal's dot product with it: the normal version leaves a flat region
// at `sin(elevation)` rather than at mid-grey, and relief that does not
// sit on a neutral ground reads as a tint, not as carving. `conv.emboss`
// biases by 0.5 for exactly this reason; this is the same rule with a
// real Sobel gradient and an elevation control behind it.

/// `gallery.bas_relief` params: light `angle_deg`, `elevation_deg`
/// (a high sun grazes less and flattens the relief), relief `height`,
/// and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryBasReliefParams {
    pub angle_deg: f32,
    pub elevation_deg: f32,
    pub height: f32,
    pub amount: f32,
}

#[allow(clippy::new_without_default)]
impl GalleryBasReliefParams {
    pub fn new(angle_deg: f32, elevation_deg: f32, height: f32, amount: f32) -> Self {
        Self {
            angle_deg,
            elevation_deg,
            height,
            amount,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Directional lighting of a luminance heightfield.
///
/// Serves: **Bas Relief**, **Plaster**, **Note Paper**, **Chrome**
/// (high `height`, low `elevation_deg`), **Emboss** presets.
///
/// IDENTITY: `amount = 0.0` returns the input unchanged. (`height =
/// 0.0` flattens the relief to uniform mid-grey — a valid result, not
/// the input.)
pub static GALLERY_BAS_RELIEF: KernelDef = KernelDef {
    id: "gallery.bas_relief",
    class: KernelClass::Windowed { radius: (1, 1) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryBasReliefParams>(),
        fields: &[
            ParamField {
                name: "angle_deg",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "elevation_deg",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "height",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_BAS_RELIEF_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_BAS_RELIEF_WGSL: &str = "\
// paged.image kernel `gallery.bas_relief` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    angle_deg: f32,
    elevation_deg: f32,
    height: f32,
    amount: f32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn win_org() -> vec2<i32> {
    let din = vec2<i32>(textureDimensions(in0));
    let dout = vec2<i32>(textureDimensions(outp));
    return (din - dout) / 2;
}

fn tapl(c: vec2<i32>) -> f32 {
    let din = vec2<i32>(textureDimensions(in0));
    return luma(textureLoad(in0, clamp(c, vec2<i32>(0), din - vec2<i32>(1)), 0).rgb);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let c = xy + win_org();
    let a = textureLoad(in0, clamp(c, vec2<i32>(0),
        vec2<i32>(textureDimensions(in0)) - vec2<i32>(1)), 0);

    let tl = tapl(c + vec2<i32>(-1, -1));
    let tc = tapl(c + vec2<i32>( 0, -1));
    let tr = tapl(c + vec2<i32>( 1, -1));
    let ml = tapl(c + vec2<i32>(-1,  0));
    let mr = tapl(c + vec2<i32>( 1,  0));
    let bl = tapl(c + vec2<i32>(-1,  1));
    let bc = tapl(c + vec2<i32>( 0,  1));
    let br = tapl(c + vec2<i32>( 1,  1));
    let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);

    let az = radians(params.angle_deg);
    let ldir = vec2<f32>(cos(az), -sin(az));
    // 0.125 divides out the Sobel taps' weight sum, so `height` is in
    // the same units whatever gradient operator we swap in later.
    let slope = (gx * ldir.x + gy * ldir.y) * 0.125 * params.height;
    // Elevation clamped below 90: at exactly 90 the light is overhead,
    // cos is 0, and the relief vanishes into flat grey with no way back.
    let el = cos(radians(clamp(params.elevation_deg, 0.0, 89.0)));
    // 0.5 ground: a FLAT region has zero gradient and stays mid-grey,
    // which is what makes this read as carving rather than as an edge
    // map on black.
    let sh = clamp(0.5 + slope * el, 0.0, 1.0);

    let styled = vec4<f32>(vec3<f32>(sh), a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ─────────────────────────── threshold_ink ─────────────────────────
//
// A threshold with a RAMP. `adjust.threshold` already provides the hard
// step; the whole Sketch family needs the soft one, because a hard step
// gives a jagged stair on every diagonal and there is no antialiasing
// pass afterwards to fix it. The ramp is also what keeps GPU-vs-
// reference parity meaningful: on a hard step, one ULP of luminance
// difference flips a pixel from black to white, which no per-channel
// tolerance can express.

/// `gallery.threshold_ink` params: `threshold` on luminance, ramp
/// `softness`, and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryThresholdInkParams {
    pub threshold: f32,
    pub softness: f32,
    pub amount: f32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl GalleryThresholdInkParams {
    pub fn new(threshold: f32, softness: f32, amount: f32) -> Self {
        Self {
            threshold,
            softness,
            amount,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Luminance threshold with a soft ramp — ink over paper.
///
/// Serves: **Photocopy**, **Stamp**, **Charcoal**, **Chalk &
/// Charcoal**, **Torn Edges**, **Conté Crayon**'s tonal split.
///
/// IDENTITY: `amount = 0.0` returns the input unchanged.
pub static GALLERY_THRESHOLD_INK: KernelDef = KernelDef {
    id: "gallery.threshold_ink",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryThresholdInkParams>(),
        fields: &[
            ParamField {
                name: "threshold",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "softness",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_THRESHOLD_INK_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_THRESHOLD_INK_WGSL: &str = "\
// paged.image kernel `gallery.threshold_ink` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    threshold: f32,
    softness: f32,
    amount: f32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(in0, xy, 0);

    let l = luma(clamp(a.rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
    // Floor the ramp at ~1/1024: smoothstep needs a non-empty interval,
    // and a ramp narrower than a couple of f16 steps is a hard step
    // wearing a ramp's clothes.
    let s = max(params.softness, 0.0009765625);
    let ink = smoothstep(params.threshold - s, params.threshold + s, l);

    let styled = vec4<f32>(vec3<f32>(ink), a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ─────────────────────────── stained_glass ─────────────────────────
//
// Cellular (Worley) tiling. Jitter one site inside each lattice cell,
// find the nearest site to this pixel among the 3×3 lattice
// neighbourhood, and paint the whole cell with the colour the image has
// AT that site. Taking the colour from the site rather than averaging
// the cell is what makes it read as glass CUT FROM the picture instead
// of as posterisation.
//
// The border comes free from the F2 − F1 distance: where the two
// nearest sites are equidistant we are on a cell boundary, and that
// locus is the lead came between the panes.

/// `gallery.stained_glass` params: tile origin, hash `seed`, `cell_px`
/// pane size, `border` width (in lattice units), and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryStainedGlassParams {
    pub ox: i32,
    pub oy: i32,
    pub seed: u32,
    pub cell_px: f32,
    pub border: f32,
    pub amount: f32,
    pub _pad0: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl GalleryStainedGlassParams {
    pub fn new(ox: i32, oy: i32, seed: u32, cell_px: f32, border: f32, amount: f32) -> Self {
        Self {
            ox,
            oy,
            seed,
            cell_px,
            border,
            amount,
            _pad0: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Cellular tiling with darkened borders.
///
/// Serves: **Stained Glass**, **Mosaic Tiles**, **Craquelure**
/// (thin `border`, large `cell_px`), **Patchwork**, **Crystallize**
/// (`border = 0`).
///
/// IDENTITY: `amount = 0.0` returns the input unchanged.
///
/// The window radius is 24 px against a `cell_px` capped at 16: the
/// nearest site can sit up to √2 cells away, and a window sized for the
/// average rather than the worst case reads outside its tile exactly on
/// the panes that stretch.
pub static GALLERY_STAINED_GLASS: KernelDef = KernelDef {
    id: "gallery.stained_glass",
    class: KernelClass::Windowed { radius: (24, 24) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryStainedGlassParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "seed",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "cell_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "border",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_STAINED_GLASS_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_STAINED_GLASS_WGSL: &str = "\
// paged.image kernel `gallery.stained_glass` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    seed: u32,
    cell_px: f32,
    border: f32,
    amount: f32,
    _pad0: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const CELL_MAX : f32 = 16.0;
const OFF_MAX : i32 = 24;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash21(x: i32, y: i32, seed: u32) -> f32 {
    let h = pcg(bitcast<u32>(x) ^ pcg(bitcast<u32>(y) ^ pcg(seed)));
    return f32(h >> 8u) * 5.9604644775390625e-8;
}

fn win_org() -> vec2<i32> {
    let din = vec2<i32>(textureDimensions(in0));
    let dout = vec2<i32>(textureDimensions(outp));
    return (din - dout) / 2;
}

fn tap(c: vec2<i32>) -> vec4<f32> {
    let din = vec2<i32>(textureDimensions(in0));
    return textureLoad(in0, clamp(c, vec2<i32>(0), din - vec2<i32>(1)), 0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let c = xy + win_org();
    let a = tap(c);

    let s = clamp(params.cell_px, 2.0, CELL_MAX);
    let gp = vec2<f32>(f32(params.ox + i32(gid.x)), f32(params.oy + i32(gid.y)));
    let lp = gp / s;
    let li = floor(lp);

    // F1 (nearest) and F2 (second nearest), ascending scan order so the
    // winner is decided identically on both lanes.
    var d1 = 1.0e9;
    var d2 = 1.0e9;
    var best = lp;
    for (var j : i32 = -1; j <= 1; j = j + 1) {
        for (var i : i32 = -1; i <= 1; i = i + 1) {
            let cellc = li + vec2<f32>(f32(i), f32(j));
            // Jitter the site INSIDE its own lattice cell. An unjittered
            // lattice gives squares, and squares are graph paper, not
            // glass.
            let jx = hash21(i32(cellc.x), i32(cellc.y), params.seed);
            let jy = hash21(i32(cellc.x), i32(cellc.y), params.seed ^ 3037000493u);
            let site = cellc + vec2<f32>(jx, jy);
            let dd = distance(lp, site);
            if (dd < d1) {
                d2 = d1;
                d1 = dd;
                best = site;
            } else if (dd < d2) {
                d2 = dd;
            }
        }
    }

    // Colour of the pane = the image AT the winning site.
    let sp = best * s - gp;
    let off = clamp(vec2<i32>(i32(round(sp.x)), i32(round(sp.y))),
                    vec2<i32>(-OFF_MAX), vec2<i32>(OFF_MAX));
    let pane = tap(c + off);

    // F2 − F1 is small exactly on the boundary between two panes: that
    // locus IS the lead came, and it costs nothing extra to find.
    let bw = clamp(params.border, 0.0, 1.0);
    let edge = select(0.0,
                      1.0 - smoothstep(0.0, max(bw, 0.001), (d2 - d1) * 0.5),
                      bw > 0.0);

    let styled = vec4<f32>(
        clamp(pane.rgb * (1.0 - edge), vec3<f32>(0.0), vec3<f32>(1.0)), a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ────────────────────────────── glass ──────────────────────────────
//
// Refraction through an uneven surface: build a smooth procedural
// height field, take its GRADIENT, and sample the image displaced along
// that slope. The gradient matters — displacing by the height itself
// pushes everything one way and just smears; a real surface bends light
// where it is TILTED, which is where the field is steep.

/// `gallery.glass` params: tile origin, hash `seed`, surface
/// `scale_px` (feature size), `distortion` (max displacement, px), and
/// the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryGlassParams {
    pub ox: i32,
    pub oy: i32,
    pub seed: u32,
    pub scale_px: f32,
    pub distortion: f32,
    pub amount: f32,
    pub _pad0: u32,
    pub _abi_pad: u32,
}

#[allow(clippy::new_without_default)]
impl GalleryGlassParams {
    pub fn new(ox: i32, oy: i32, seed: u32, scale_px: f32, distortion: f32, amount: f32) -> Self {
        Self {
            ox,
            oy,
            seed,
            scale_px,
            distortion,
            amount,
            _pad0: 0,
            _abi_pad: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Refraction-style displacement along a procedural surface gradient.
///
/// Serves: **Glass**, **Ocean Ripple**, **Water Paper**, **Ripple**,
/// **Diffuse Glow**'s optical softening.
///
/// IDENTITY: `amount = 0.0` returns the input unchanged;
/// `distortion = 0.0` independently collapses the displacement to zero.
pub static GALLERY_GLASS: KernelDef = KernelDef {
    id: "gallery.glass",
    class: KernelClass::Windowed { radius: (16, 16) },
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryGlassParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "seed",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "scale_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "distortion",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_GLASS_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_GLASS_WGSL: &str = "\
// paged.image kernel `gallery.glass` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    seed: u32,
    scale_px: f32,
    distortion: f32,
    amount: f32,
    _pad0: u32,
    _abi_pad: u32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

const D_MAX : f32 = 16.0;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash21(x: i32, y: i32, seed: u32) -> f32 {
    let h = pcg(bitcast<u32>(x) ^ pcg(bitcast<u32>(y) ^ pcg(seed)));
    return f32(h >> 8u) * 5.9604644775390625e-8;
}

// Value noise: hash the lattice corners, blend with a Hermite fade. The
// fade is not decoration — a linear blend leaves a visible crease along
// every lattice line, and the creases land exactly where the
// displacement is largest.
fn vnoise(p: vec2<f32>, seed: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let ix = i32(i.x);
    let iy = i32(i.y);
    let h00 = hash21(ix,     iy,     seed);
    let h10 = hash21(ix + 1, iy,     seed);
    let h01 = hash21(ix,     iy + 1, seed);
    let h11 = hash21(ix + 1, iy + 1, seed);
    return mix(mix(h00, h10, u.x), mix(h01, h11, u.x), u.y);
}

fn win_org() -> vec2<i32> {
    let din = vec2<i32>(textureDimensions(in0));
    let dout = vec2<i32>(textureDimensions(outp));
    return (din - dout) / 2;
}

fn tap(c: vec2<i32>) -> vec4<f32> {
    let din = vec2<i32>(textureDimensions(in0));
    return textureLoad(in0, clamp(c, vec2<i32>(0), din - vec2<i32>(1)), 0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let c = xy + win_org();
    let a = tap(c);

    let s = max(params.scale_px, 2.0);
    let gp = vec2<f32>(f32(params.ox + i32(gid.x)), f32(params.oy + i32(gid.y))) / s;

    // Central differences in LATTICE units: half a lattice cell either
    // way samples the slope without smoothing it into nothing.
    let e = 0.5;
    let nx = vnoise(gp + vec2<f32>(e, 0.0), params.seed)
           - vnoise(gp - vec2<f32>(e, 0.0), params.seed);
    let ny = vnoise(gp + vec2<f32>(0.0, e), params.seed)
           - vnoise(gp - vec2<f32>(0.0, e), params.seed);

    // The difference of two [0,1) samples is bounded by 1, so scaling by
    // a distortion capped at D_MAX keeps every read inside the declared
    // window — the clamp below is belt and braces, not the guarantee.
    let dm = clamp(params.distortion, 0.0, D_MAX);
    let v = vec2<f32>(nx, ny) * dm;
    let off = clamp(vec2<i32>(i32(round(v.x)), i32(round(v.y))),
                    vec2<i32>(-16), vec2<i32>(16));

    let styled = tap(c + off);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

// ──────────────────────────── texturizer ───────────────────────────
//
// A surface the picture sits ON. The texture MULTIPLIES the image
// rather than being pasted over it — an overlaid grey pattern looks
// like a screenshot of a texture layer, whereas modulating the image's
// own colour looks like it was printed on that surface.
//
// The shading is the height field's directional derivative, the same
// relief rule as `gallery.bas_relief`, so a texture and a relief lit
// from the same angle agree about which way is up.

/// `gallery.texturizer` params: tile origin, hash `seed`, pattern
/// `kind` (0 = canvas, 1 = burlap, 2 = brick), `scale_px` feature size,
/// `relief` depth, light `angle_deg`, and the effect blend.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
pub struct GalleryTexturizerParams {
    pub ox: i32,
    pub oy: i32,
    pub seed: u32,
    pub kind: u32,
    pub scale_px: f32,
    pub relief: f32,
    pub angle_deg: f32,
    pub amount: f32,
}

#[allow(clippy::new_without_default, clippy::too_many_arguments)]
impl GalleryTexturizerParams {
    pub fn new(
        ox: i32,
        oy: i32,
        seed: u32,
        kind: u32,
        scale_px: f32,
        relief: f32,
        angle_deg: f32,
        amount: f32,
    ) -> Self {
        Self {
            ox,
            oy,
            seed,
            kind,
            scale_px,
            relief,
            angle_deg,
            amount,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        ::bytemuck::bytes_of(self)
    }
}

/// Procedural surface texture modulating the image's own shading.
///
/// Serves: **Texturizer** (its Canvas / Burlap / Brick / Sandstone
/// presets are `kind` and `scale_px`), **Rough Pastels**, **Sponge**,
/// **Fresco**, **Underpainting**'s texture pass, **Grain**'s
/// Vertical/Horizontal modes.
///
/// IDENTITY: `amount = 0.0` returns the input unchanged;
/// `relief = 0.0` independently makes the shading a uniform ×1.
pub static GALLERY_TEXTURIZER: KernelDef = KernelDef {
    id: "gallery.texturizer",
    class: KernelClass::Point,
    inputs: 1,
    params: ParamsLayout {
        size: ::core::mem::size_of::<GalleryTexturizerParams>(),
        fields: &[
            ParamField {
                name: "ox",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "oy",
                wgsl_ty: "i32",
            },
            ParamField {
                name: "seed",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "kind",
                wgsl_ty: "u32",
            },
            ParamField {
                name: "scale_px",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "relief",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "angle_deg",
                wgsl_ty: "f32",
            },
            ParamField {
                name: "amount",
                wgsl_ty: "f32",
            },
        ],
    },
    wgsl: GALLERY_TEXTURIZER_WGSL,
    module: true,
    mip_exact: false,
    gpu_tolerance: Tolerance::ChannelEpsF16(4),
};

const GALLERY_TEXTURIZER_WGSL: &str = "\
// paged.image kernel `gallery.texturizer` — handwritten under ABI v1.1.
// MPL-2.0 OR LicenseRef-PMEL; (c) And The Next GmbH.

struct Params {
    ox: i32,
    oy: i32,
    seed: u32,
    kind: u32,
    scale_px: f32,
    relief: f32,
    angle_deg: f32,
    amount: f32,
}

@group(0) @binding(0) var in0 : texture_2d<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var mask : texture_2d<f32>;
@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash21(x: i32, y: i32, seed: u32) -> f32 {
    let h = pcg(bitcast<u32>(x) ^ pcg(bitcast<u32>(y) ^ pcg(seed)));
    return f32(h >> 8u) * 5.9604644775390625e-8;
}

fn vnoise(p: vec2<f32>, seed: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let ix = i32(i.x);
    let iy = i32(i.y);
    let h00 = hash21(ix,     iy,     seed);
    let h10 = hash21(ix + 1, iy,     seed);
    let h01 = hash21(ix,     iy + 1, seed);
    let h11 = hash21(ix + 1, iy + 1, seed);
    return mix(mix(h00, h10, u.x), mix(h01, h11, u.x), u.y);
}

// Surface height at p, in [0, 1].
fn height(p: vec2<f32>, kind: u32, s: f32, seed: u32) -> f32 {
    if (kind == 1u) {
        // BURLAP — cloth is two scales at once: the coarse weave and the
        // fibre inside it. One octave alone reads as fog.
        return 0.65 * vnoise(p / s, seed) + 0.35 * vnoise(p / (s * 0.5), seed ^ 1u);
    }
    if (kind == 2u) {
        // BRICK — alternate rows shift by half a brick; the mortar line
        // is the ridge. fract(row/2)*2 is 0 on even rows and 1 on odd,
        // which is the stagger with no branch.
        let row = floor(p.y / s);
        let stag = fract(row * 0.5) * 2.0;
        let u = fract(p.x / (s * 2.0) + stag * 0.5);
        let v = fract(p.y / s);
        let mu = min(u, 1.0 - u);
        let mv = min(v, 1.0 - v);
        return 1.0 - smoothstep(0.0, 0.12, min(mu, mv));
    }
    // CANVAS (default) — two orthogonal thread ridges.
    let u = 6.28318531 * p.x / s;
    let v = 6.28318531 * p.y / s;
    return 0.5 + 0.25 * (sin(u) + sin(v));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = textureDimensions(outp);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(in0, xy, 0);

    let s = max(params.scale_px, 2.0);
    let gp = vec2<f32>(f32(params.ox + i32(gid.x)), f32(params.oy + i32(gid.y)));

    // One-pixel central differences: the height field is defined in
    // pixel space, so this is the slope the light actually sees.
    let hx = height(gp + vec2<f32>(1.0, 0.0), params.kind, s, params.seed)
           - height(gp - vec2<f32>(1.0, 0.0), params.kind, s, params.seed);
    let hy = height(gp + vec2<f32>(0.0, 1.0), params.kind, s, params.seed)
           - height(gp - vec2<f32>(0.0, 1.0), params.kind, s, params.seed);

    let az = radians(params.angle_deg);
    let ldir = vec2<f32>(cos(az), -sin(az));
    // Centred on 1.0 and MULTIPLIED in: a flat surface leaves the
    // picture alone, a lit ridge brightens it, a shadowed one darkens
    // it. Adding a grey pattern instead would wash out the colour.
    let sh = clamp(1.0 + (hx * ldir.x + hy * ldir.y) * params.relief, 0.0, 2.0);

    let styled = vec4<f32>(clamp(a.rgb * sh, vec3<f32>(0.0), vec3<f32>(1.0)), a.a);
    let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));
    let m = textureLoad(mask, xy, 0).r;
    textureStore(outp, xy, mix(a, result, vec4<f32>(m)));
}
";

pub static FAMILY: &[&KernelDef] = &[
    &GALLERY_KUWAHARA,
    &GALLERY_POSTERIZE_EDGES,
    &GALLERY_GLOWING_EDGES,
    &GALLERY_HALFTONE,
    &GALLERY_GRAIN,
    &GALLERY_DIFFUSE,
    &GALLERY_CROSSHATCH,
    &GALLERY_BAS_RELIEF,
    &GALLERY_THRESHOLD_INK,
    &GALLERY_STAINED_GLASS,
    &GALLERY_GLASS,
    &GALLERY_TEXTURIZER,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Naga-validate just this family's modules. The shared
    /// `wgsl_validate` suite walks `all_defined()`, which cannot see
    /// this family until the orchestrator lands `gallery::FAMILY` in
    /// `families/mod.rs`'s `ALL_FAMILIES` — so until then THIS is the
    /// gate, and it is the same gate.
    #[test]
    fn gallery_modules_naga_validate() {
        for def in FAMILY {
            let src = crate::abi::assemble(def);
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{}: WGSL parse failed: {e}\n{src}", def.id));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{}: WGSL validation failed: {e:?}\n{src}", def.id));
        }
    }

    /// `ParamsLayout.size` is the uniform upload size AND the
    /// `min_binding_size` the pipeline enforces. A struct that grew a
    /// field without the layout following it fails at dispatch, on the
    /// GPU, in a message about buffer sizes — so check it here instead.
    #[test]
    fn gallery_params_layout_matches_the_rust_struct() {
        let pairs: &[(&str, usize)] = &[
            (
                "gallery.kuwahara",
                core::mem::size_of::<GalleryKuwaharaParams>(),
            ),
            (
                "gallery.posterize_edges",
                core::mem::size_of::<GalleryPosterizeEdgesParams>(),
            ),
            (
                "gallery.glowing_edges",
                core::mem::size_of::<GalleryGlowingEdgesParams>(),
            ),
            (
                "gallery.halftone",
                core::mem::size_of::<GalleryHalftoneParams>(),
            ),
            ("gallery.grain", core::mem::size_of::<GalleryGrainParams>()),
            (
                "gallery.diffuse",
                core::mem::size_of::<GalleryDiffuseParams>(),
            ),
            (
                "gallery.crosshatch",
                core::mem::size_of::<GalleryCrosshatchParams>(),
            ),
            (
                "gallery.bas_relief",
                core::mem::size_of::<GalleryBasReliefParams>(),
            ),
            (
                "gallery.threshold_ink",
                core::mem::size_of::<GalleryThresholdInkParams>(),
            ),
            (
                "gallery.stained_glass",
                core::mem::size_of::<GalleryStainedGlassParams>(),
            ),
            ("gallery.glass", core::mem::size_of::<GalleryGlassParams>()),
            (
                "gallery.texturizer",
                core::mem::size_of::<GalleryTexturizerParams>(),
            ),
        ];
        assert_eq!(pairs.len(), FAMILY.len(), "one pair per kernel");
        for (id, size) in pairs {
            let def = FAMILY
                .iter()
                .find(|d| d.id == *id)
                .unwrap_or_else(|| panic!("{id} is in FAMILY"));
            assert_eq!(def.params.size, *size, "{id}: ParamsLayout.size drifted");
            // 16-byte multiple: the uniform address space rounds a
            // struct's size up to 16 anyway, and a block that relies on
            // the rounding rather than stating it is a trap for the next
            // person who appends a field.
            assert_eq!(size % 16, 0, "{id}: param block must be 16-byte sized");
        }
    }

    /// Every module declares the exact ABI v1.1 binding interface and
    /// ends with the mask blend. The mask mix is how a SELECTION scopes
    /// an effect; a kernel that forgets it silently paints the whole
    /// tile and looks correct until someone selects something.
    #[test]
    fn gallery_modules_honour_the_abi_contract() {
        for def in FAMILY {
            let w = def.wgsl;
            let id = def.id;
            assert!(def.module, "{id}: gallery kernels are handwritten modules");
            assert!(def.inputs == 1, "{id}: unary");
            for needle in [
                "@group(0) @binding(0) var in0 : texture_2d<f32>;",
                "@group(1) @binding(0) var<uniform> params : Params;",
                "@group(2) @binding(0) var mask : texture_2d<f32>;",
                "@group(3) @binding(0) var outp : texture_storage_2d<rgba16float, write>;",
                "@compute @workgroup_size(16, 16, 1)",
                "let m = textureLoad(mask, xy, 0).r;",
                "textureStore(outp, xy, mix(a, result, vec4<f32>(m)));",
            ] {
                assert!(w.contains(needle), "{id}: module is missing `{needle}`");
            }
            // THE IDENTITY, machine-checked. Every doc-comment in this
            // file promises `amount = 0.0` returns the input; that
            // promise is only worth anything if the final blend is
            // literally there, so assert the line rather than trust the
            // prose. `mix(a, styled, 0.0)` is exactly `a` because every
            // `styled` above is clamped finite — a NaN would survive the
            // zero factor, which is why the clamps are not cosmetic.
            assert!(
                w.contains("let result = mix(a, styled, clamp(params.amount, 0.0, 1.0));"),
                "{id}: module has no `amount` identity blend"
            );
            // Bounds guard: without it the tail workgroup writes outside
            // the tile.
            assert!(
                w.contains("if (gid.x >= d.x || gid.y >= d.y) { return; }"),
                "{id}: module is missing the out-of-range guard"
            );
        }
    }

    /// Ids are unique, namespaced, and the family list is complete —
    /// a duplicate id would shadow a kernel in registry dispatch and the
    /// loser would simply never run.
    #[test]
    fn gallery_ids_are_unique_and_namespaced() {
        let mut ids: Vec<&str> = FAMILY.iter().map(|d| d.id).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate kernel id in gallery::FAMILY");
        for d in FAMILY {
            assert!(d.id.starts_with("gallery."), "{}: wrong namespace", d.id);
        }
    }

    /// `new()` zeroes every pad. The param bytes ARE the op-cache key
    /// (§6.1 — param identity is byte identity), so a pad left
    /// uninitialised would make two identical operations miss the cache
    /// and, worse, compare unequal in the undo journal.
    #[test]
    fn gallery_param_constructors_zero_their_pads() {
        assert_eq!(GalleryKuwaharaParams::new(3.0, 1.0)._pad0, 0);
        assert_eq!(GalleryKuwaharaParams::new(3.0, 1.0)._abi_pad, 0);
        assert_eq!(GalleryGlowingEdgesParams::new(1.0, 0.5, 1.0)._abi_pad, 0);
        assert_eq!(
            GalleryHalftoneParams::new(0, 0, 6.0, 45.0, 1.0, 1.0)._abi_pad,
            0
        );
        assert_eq!(GalleryGrainParams::new(0, 0, 7, 1.0, true, 0.5)._abi_pad, 0);
        assert_eq!(
            GalleryDiffuseParams::new(0, 0, 7, 4.0, 0.0, 0.0, 1.0)._abi_pad,
            0
        );
        assert_eq!(
            GalleryCrosshatchParams::new(0, 0, 45.0, 6.0, 1.0, 2, 1.0)._abi_pad,
            0
        );
        assert_eq!(GalleryThresholdInkParams::new(0.5, 0.05, 1.0)._abi_pad, 0);
        assert_eq!(
            GalleryStainedGlassParams::new(0, 0, 7, 8.0, 0.1, 1.0)._abi_pad,
            0
        );
        assert_eq!(GalleryGlassParams::new(0, 0, 7, 8.0, 4.0, 1.0)._abi_pad, 0);
        // `mono` is a bool at the Rust edge and a u32 on the wire —
        // check the mapping, not just the pads.
        assert_eq!(GalleryGrainParams::new(0, 0, 7, 1.0, false, 0.5).mono, 0);
        assert_eq!(GalleryGrainParams::new(0, 0, 7, 1.0, true, 0.5).mono, 1);
    }
}
