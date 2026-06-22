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

//! Scalar reference implementations of the compose.* blend/composite
//! operators — shared by the compose parity tests AND the PSD flatten
//! render-oracle (test-only, like everything in this crate). Lands with
//! the M1 fan-out compose unit; the flatten unit consumes it.
//!
//! Provenance: W3C Compositing and Blending Level 1 (public spec,
//! <https://www.w3.org/TR/compositing-1/>); Porter–Duff `over` on
//! premultiplied data is the spine (base-idea §8.4). The PSD merged-
//! composite oracle (psd_composite) joins on the `BlendMode` fourcc
//! mapping below.
//!
//! # Model (W3C §3 "Compositing and blending")
//!
//! Inputs are PREMULTIPLIED rgba: `a` = the backdrop (`Cb`·αb, αb),
//! `b` = the source layer (`Cs`·αs, αs). The layer `opacity` folds into
//! the source FIRST (W3C §5.1 group opacity for a single layer):
//! `b ← b · opacity` (the premultiplied source AND its alpha scale by
//! the same factor, so the unpremultiplied color `Cs` is unchanged).
//!
//! Then, per W3C §9 the unpremultiplied colors are blended and composited
//! source-over (W3C §11.3 "General formula", premultiplied form):
//!
//! ```text
//! Cs = unpremul(b).rgb,  Cb = unpremul(a).rgb     (zero-alpha → 0)
//! co = αs·(1 − αb)·Cs + αs·αb·B(Cb, Cs) + (1 − αs)·αb·Cb
//! αo = αs + αb·(1 − αs)
//! ```
//!
//! The output `(co, αo)` is already PREMULTIPLIED. Summation order in
//! `composite` is fixed (term1 + term2 + term3, left to right) and the
//! WGSL module mirrors it exactly (§6.3 determinism).
//!
//! The blend functions `B(Cb, Cs)` are the W3C §blending definitions.
//! Separable blends apply per channel; the non-separable four
//! (hue/saturation/color/luminosity) operate on the rgb triple via the
//! W3C §10.3 `Lum`/`ClipColor`/`SetLum`/`Sat`/`SetSat` pseudo-code,
//! transcribed identically here and in WGSL.

use image_kernels::reference_prelude::Px;

/// Unpremultiply guard mirroring `abi::unpremul4` / `reference_prelude`:
/// zero alpha maps to all-zero rgb (no Inf/NaN leak).
fn unpremul_rgb(p: Px) -> [f32; 3] {
    let alpha = p.0[3];
    if alpha == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [p.0[0] / alpha, p.0[1] / alpha, p.0[2] / alpha]
    }
}

// ---------------------------------------------------------------------
// Separable blend functions B(Cb, Cs) — W3C §blending. Each is defined
// per channel; `blend_separable` lifts a scalar `f(cb, cs)` over rgb.
// ---------------------------------------------------------------------

/// W3C: `B(Cb, Cs) = Cs`. Source-over degenerates to Porter–Duff `over`.
pub fn b_normal(_cb: f32, cs: f32) -> f32 {
    cs
}

/// W3C §"multiply": `B = Cb · Cs`.
pub fn b_multiply(cb: f32, cs: f32) -> f32 {
    cb * cs
}

/// W3C §"screen": `B = Cb + Cs − Cb·Cs`.
pub fn b_screen(cb: f32, cs: f32) -> f32 {
    cb + cs - cb * cs
}

/// W3C §"hard-light": `HardLight(Cb,Cs)` = if Cs ≤ 0.5 multiply(Cb, 2·Cs)
/// else screen(Cb, 2·Cs − 1).
pub fn b_hard_light(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        b_multiply(cb, 2.0 * cs)
    } else {
        b_screen(cb, 2.0 * cs - 1.0)
    }
}

/// W3C §"overlay": `Overlay(Cb,Cs) = HardLight(Cs, Cb)` (roles swapped).
pub fn b_overlay(cb: f32, cs: f32) -> f32 {
    b_hard_light(cs, cb)
}

/// W3C §"darken": `B = min(Cb, Cs)`.
pub fn b_darken(cb: f32, cs: f32) -> f32 {
    cb.min(cs)
}

/// W3C §"lighten": `B = max(Cb, Cs)`.
pub fn b_lighten(cb: f32, cs: f32) -> f32 {
    cb.max(cs)
}

/// W3C §"color-dodge": if Cb = 0 → 0; if Cs = 1 → 1; else
/// `min(1, Cb / (1 − Cs))`. (Branch order matches the WGSL.)
pub fn b_color_dodge(cb: f32, cs: f32) -> f32 {
    if cb == 0.0 {
        0.0
    } else if cs == 1.0 {
        1.0
    } else {
        1.0_f32.min(cb / (1.0 - cs))
    }
}

/// W3C §"color-burn": if Cb = 1 → 1; if Cs = 0 → 0; else
/// `1 − min(1, (1 − Cb) / Cs)`.
pub fn b_color_burn(cb: f32, cs: f32) -> f32 {
    if cb == 1.0 {
        1.0
    } else if cs == 0.0 {
        0.0
    } else {
        1.0 - 1.0_f32.min((1.0 - cb) / cs)
    }
}

/// W3C §"soft-light": Photoshop formula.
/// if Cs ≤ 0.5: `Cb − (1 − 2·Cs)·Cb·(1 − Cb)`
/// else:        `Cb + (2·Cs − 1)·(D(Cb) − Cb)` where
/// `D(Cb) = ((16·Cb − 12)·Cb + 4)·Cb` for Cb ≤ 0.25, else `sqrt(Cb)`.
pub fn b_soft_light(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
    } else {
        let d = if cb <= 0.25 {
            ((16.0 * cb - 12.0) * cb + 4.0) * cb
        } else {
            cb.sqrt()
        };
        cb + (2.0 * cs - 1.0) * (d - cb)
    }
}

/// W3C §"difference": `B = |Cb − Cs|`.
pub fn b_difference(cb: f32, cs: f32) -> f32 {
    (cb - cs).abs()
}

/// W3C §"exclusion": `B = Cb + Cs − 2·Cb·Cs`.
pub fn b_exclusion(cb: f32, cs: f32) -> f32 {
    cb + cs - 2.0 * cb * cs
}

/// Lift a per-channel separable blend over the rgb triple.
fn blend_separable(f: fn(f32, f32) -> f32, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    [f(cb[0], cs[0]), f(cb[1], cs[1]), f(cb[2], cs[2])]
}

// ---------------------------------------------------------------------
// Non-separable blends — W3C §10.3 pseudo-code, transcribed verbatim.
// Lum/Sat use the SAME fixed weights and branch order in WGSL.
// ---------------------------------------------------------------------

/// W3C §10.3 `Lum(C) = 0.3·R + 0.59·G + 0.11·B`.
fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

/// W3C §10.3 `ClipColor(C)`: pull out-of-gamut colors back toward the
/// luminance, preserving Lum. Channel order in min/max is r,g,b.
fn clip_color(mut c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    if n < 0.0 {
        for ch in c.iter_mut() {
            *ch = l + (*ch - l) * l / (l - n);
        }
    }
    if x > 1.0 {
        for ch in c.iter_mut() {
            *ch = l + (*ch - l) * (1.0 - l) / (x - l);
        }
    }
    c
}

/// W3C §10.3 `SetLum(C, l)`: translate C to luminance l, then clip.
fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d])
}

/// W3C §10.3 `Sat(C) = max(R,G,B) − min(R,G,B)`.
fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

/// W3C §10.3 `SetSat(C, s)`: rescale the mid channel between 0 and s and
/// set min→0, max→s, mid→interpolated; non-min/non-max channels are 0.
/// Implemented via an explicit min/mid/max sort (indices) — the SAME
/// branch order as the WGSL lane (no library sort, no NaN ambiguity over
/// the finite stimulus).
fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    // Rank the three channels into DISTINCT min/mid/max indices with a
    // total order that breaks value ties by index (`ge(i, j)` = strictly
    // greater, or equal-and-later-index). This guarantees imax, imin,
    // imid are always {0, 1, 2} in some order even when channels are
    // equal — no out-of-bounds, no W3C tie ambiguity. The WGSL lane uses
    // the identical `ge` ladder.
    let ge = |i: usize, j: usize| c[i] > c[j] || (c[i] == c[j] && i >= j);
    // imax is ≥ both others; imin is ≤ both others (others are ge to it).
    let imax = if ge(0, 1) && ge(0, 2) {
        0
    } else if ge(1, 0) && ge(1, 2) {
        1
    } else {
        2
    };
    let imin = if ge(1, 0) && ge(2, 0) {
        0
    } else if ge(0, 1) && ge(2, 1) {
        1
    } else {
        2
    };
    let imid = 3 - imax - imin; // the remaining distinct index (0+1+2 = 3)
    let mut out = [0.0f32; 3];
    let cmax = c[imax];
    let cmin = c[imin];
    if cmax > cmin {
        out[imid] = (c[imid] - cmin) * s / (cmax - cmin);
        out[imax] = s;
    } else {
        // max == min (flat color): the W3C `Cmax > Cmin` guard fails → 0.
        out[imid] = 0.0;
        out[imax] = 0.0;
    }
    out[imin] = 0.0;
    out
}

/// W3C §"hue": `SetLum(SetSat(Cs, Sat(Cb)), Lum(Cb))`.
pub fn b_hue(cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    set_lum(set_sat(cs, sat(cb)), lum(cb))
}

/// W3C §"saturation": `SetLum(SetSat(Cb, Sat(Cs)), Lum(Cb))`.
pub fn b_saturation(cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    set_lum(set_sat(cb, sat(cs)), lum(cb))
}

/// W3C §"color": `SetLum(Cs, Lum(Cb))`.
pub fn b_color(cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    set_lum(cs, lum(cb))
}

/// W3C §"luminosity": `SetLum(Cb, Lum(Cs))`.
pub fn b_luminosity(cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    set_lum(cb, lum(cs))
}

// ---------------------------------------------------------------------
// The blend dispatch — a kernel is identified by id (compose.*) and by
// the PSD `BlendMode` fourcc (the flatten unit joins on `psd_key`).
// ---------------------------------------------------------------------

/// One compose operator: its kernel id, its PSD blend-mode fourcc key,
/// and the blend `B(Cb, Cs)` it applies (separable or non-separable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blend {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl Blend {
    /// The compose.* kernel id (dispatch / registry key).
    pub fn kernel_id(self) -> &'static str {
        match self {
            Blend::Normal => "compose.normal",
            Blend::Multiply => "compose.multiply",
            Blend::Screen => "compose.screen",
            Blend::Overlay => "compose.overlay",
            Blend::Darken => "compose.darken",
            Blend::Lighten => "compose.lighten",
            Blend::ColorDodge => "compose.color_dodge",
            Blend::ColorBurn => "compose.color_burn",
            Blend::HardLight => "compose.hard_light",
            Blend::SoftLight => "compose.soft_light",
            Blend::Difference => "compose.difference",
            Blend::Exclusion => "compose.exclusion",
            Blend::Hue => "compose.hue",
            Blend::Saturation => "compose.saturation",
            Blend::Color => "compose.color",
            Blend::Luminosity => "compose.luminosity",
        }
    }

    /// The PSD layer-record `BlendMode` four-character key (Adobe PSD
    /// spec, "Blend mode key"). Separable keys are space-padded to four
    /// chars (`"mul "`, `"div "`, `"dark"`, `"lite"`, `"hue "`, `"sat "`,
    /// `"lum "`); the rest are exact four-char codes. The PSD flatten
    /// unit reads this key from each layer and looks up the operator via
    /// [`Blend::from_psd_key`].
    pub fn psd_key(self) -> &'static str {
        match self {
            Blend::Normal => "norm",
            Blend::Multiply => "mul ",
            Blend::Screen => "scrn",
            Blend::Overlay => "over",
            Blend::Darken => "dark",
            Blend::Lighten => "lite",
            Blend::ColorDodge => "div ",
            Blend::ColorBurn => "idiv",
            Blend::HardLight => "hLit",
            Blend::SoftLight => "sLit",
            Blend::Difference => "diff",
            Blend::Exclusion => "smud",
            Blend::Hue => "hue ",
            Blend::Saturation => "sat ",
            Blend::Color => "colr",
            Blend::Luminosity => "lum ",
        }
    }

    /// Reverse of [`Blend::psd_key`] — resolve a PSD blend-mode fourcc to
    /// its operator (used by the PSD flatten oracle). Unknown keys → None
    /// (the preservation path keeps the layer's bytes; flatten falls back
    /// to `Normal` at the call site).
    pub fn from_psd_key(key: &str) -> Option<Blend> {
        Some(match key {
            "norm" => Blend::Normal,
            "mul " => Blend::Multiply,
            "scrn" => Blend::Screen,
            "over" => Blend::Overlay,
            "dark" => Blend::Darken,
            "lite" => Blend::Lighten,
            "div " => Blend::ColorDodge,
            "idiv" => Blend::ColorBurn,
            "hLit" => Blend::HardLight,
            "sLit" => Blend::SoftLight,
            "diff" => Blend::Difference,
            "smud" => Blend::Exclusion,
            "hue " => Blend::Hue,
            "sat " => Blend::Saturation,
            "colr" => Blend::Color,
            "lum " => Blend::Luminosity,
            _ => return None,
        })
    }

    /// Whether this blend is non-separable (operates on the rgb triple).
    pub fn is_non_separable(self) -> bool {
        matches!(
            self,
            Blend::Hue | Blend::Saturation | Blend::Color | Blend::Luminosity
        )
    }

    /// Apply `B(Cb, Cs)` for unpremultiplied rgb triples.
    pub fn blend(self, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
        match self {
            Blend::Normal => blend_separable(b_normal, cb, cs),
            Blend::Multiply => blend_separable(b_multiply, cb, cs),
            Blend::Screen => blend_separable(b_screen, cb, cs),
            Blend::Overlay => blend_separable(b_overlay, cb, cs),
            Blend::Darken => blend_separable(b_darken, cb, cs),
            Blend::Lighten => blend_separable(b_lighten, cb, cs),
            Blend::ColorDodge => blend_separable(b_color_dodge, cb, cs),
            Blend::ColorBurn => blend_separable(b_color_burn, cb, cs),
            Blend::HardLight => blend_separable(b_hard_light, cb, cs),
            Blend::SoftLight => blend_separable(b_soft_light, cb, cs),
            Blend::Difference => blend_separable(b_difference, cb, cs),
            Blend::Exclusion => blend_separable(b_exclusion, cb, cs),
            Blend::Hue => b_hue(cb, cs),
            Blend::Saturation => b_saturation(cb, cs),
            Blend::Color => b_color(cb, cs),
            Blend::Luminosity => b_luminosity(cb, cs),
        }
    }
}

/// Composite premultiplied source `b` over premultiplied backdrop `a`
/// under `blend` with layer `opacity`, returning the PREMULTIPLIED
/// result. This is the single source of truth shared by the compose
/// parity tests and the PSD merged-composite oracle.
///
/// Steps (W3C §3 / §11.3, premultiplied form; fixed summation order):
/// 1. fold opacity into the source: `b ← b · opacity` (rgb AND alpha);
/// 2. unpremultiply both to `Cs`, `Cb` (zero-alpha → 0);
/// 3. `Bc = blend(Cb, Cs)`;
/// 4. `co = αs·(1−αb)·Cs + αs·αb·Bc + (1−αs)·αb·Cb` (left to right);
/// 5. `αo = αs + αb·(1−αs)`; output `(co, αo)`.
pub fn composite(a: Px, b: Px, opacity: f32, blend: Blend) -> Px {
    // 1. Fold layer opacity into the premultiplied source.
    let bs = Px([
        b.0[0] * opacity,
        b.0[1] * opacity,
        b.0[2] * opacity,
        b.0[3] * opacity,
    ]);

    // 2. Unpremultiply both (zero-alpha guard → 0).
    let alpha_s = bs.0[3];
    let alpha_b = a.0[3];
    let cs = unpremul_rgb(bs);
    let cb = unpremul_rgb(a);

    // 3. Blend the unpremultiplied colors.
    let blended = blend.blend(cb, cs);

    // 4/5. Source-over composite in premultiplied space; fixed term order.
    let alpha_o = alpha_s + alpha_b * (1.0 - alpha_s);
    let mut out = [0.0f32; 4];
    for ch in 0..3 {
        let term1 = alpha_s * (1.0 - alpha_b) * cs[ch];
        let term2 = alpha_s * alpha_b * blended[ch];
        let term3 = (1.0 - alpha_s) * alpha_b * cb[ch];
        out[ch] = term1 + term2 + term3;
    }
    out[3] = alpha_o;
    Px(out)
}
