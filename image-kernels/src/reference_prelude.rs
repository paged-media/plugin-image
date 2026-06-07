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

//! The Rust half of the kernel DSL (test-only; `feature = "reference"`,
//! enabled solely by image-conformance). Every helper here has the
//! exact semantics of its WGSL counterpart — `abi::WGSL_PRELUDE` and
//! the WGSL builtin set — so a `kernel_family!` body means the same
//! thing in both languages. f32, no fast-math, fixed evaluation order:
//! bit-stable across platforms by construction (spec §6.3).

/// A pixel: rgba in the working space, f32 (reference precision; f16
/// quantization is the FINAL step before diffing, spec §5.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Px(pub [f32; 4]);

impl Px {
    pub fn map(self, f: impl Fn(f32) -> f32) -> Px {
        Px([f(self.0[0]), f(self.0[1]), f(self.0[2]), f(self.0[3])])
    }

    pub fn zip(self, o: Px, f: impl Fn(f32, f32) -> f32) -> Px {
        Px([
            f(self.0[0], o.0[0]),
            f(self.0[1], o.0[1]),
            f(self.0[2], o.0[2]),
            f(self.0[3], o.0[3]),
        ])
    }
}

impl core::ops::Add for Px {
    type Output = Px;
    fn add(self, o: Px) -> Px {
        self.zip(o, |x, y| x + y)
    }
}

impl core::ops::Sub for Px {
    type Output = Px;
    fn sub(self, o: Px) -> Px {
        self.zip(o, |x, y| x - y)
    }
}

impl core::ops::Mul for Px {
    type Output = Px;
    fn mul(self, o: Px) -> Px {
        self.zip(o, |x, y| x * y)
    }
}

impl core::ops::Div for Px {
    type Output = Px;
    fn div(self, o: Px) -> Px {
        self.zip(o, |x, y| x / y)
    }
}

impl core::ops::Neg for Px {
    type Output = Px;
    fn neg(self) -> Px {
        self.map(|x| -x)
    }
}

impl core::ops::Index<usize> for Px {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        &self.0[i]
    }
}

/// Dynamic channel indexing (`a[p.channel]`) — WGSL indexes vectors
/// with u32 natively; the Rust lane mirrors it.
impl core::ops::Index<u32> for Px {
    type Output = f32;
    fn index(&self, i: u32) -> &f32 {
        &self.0[i as usize]
    }
}

/// WGSL `splat4` (see `abi::WGSL_PRELUDE`).
pub fn splat4(x: f32) -> Px {
    Px([x; 4])
}

/// WGSL `pack4`.
pub fn pack4(x: f32, y: f32, z: f32, w: f32) -> Px {
    Px([x, y, z, w])
}

fn bool4(a: Px, b: Px, f: impl Fn(f32, f32) -> bool) -> Px {
    a.zip(b, |x, y| if f(x, y) { 1.0 } else { 0.0 })
}

/// WGSL `eq4` — exact 0.0/1.0; NaN compares false (IEEE, both lanes).
pub fn eq4(a: Px, b: Px) -> Px {
    bool4(a, b, |x, y| x == y)
}

/// WGSL `ne4`. NOTE: `!=` with NaN is TRUE on both lanes (IEEE).
pub fn ne4(a: Px, b: Px) -> Px {
    bool4(a, b, |x, y| x != y)
}

/// WGSL `lt4`.
pub fn lt4(a: Px, b: Px) -> Px {
    bool4(a, b, |x, y| x < y)
}

/// WGSL `le4`.
pub fn le4(a: Px, b: Px) -> Px {
    bool4(a, b, |x, y| x <= y)
}

/// WGSL `gt4`.
pub fn gt4(a: Px, b: Px) -> Px {
    bool4(a, b, |x, y| x > y)
}

/// WGSL `ge4`.
pub fn ge4(a: Px, b: Px) -> Px {
    bool4(a, b, |x, y| x >= y)
}

fn truthy(x: f32) -> bool {
    x > 0.5
}

/// WGSL `and4` — truthiness is `> 0.5`, NaN deterministically false.
pub fn and4(a: Px, b: Px) -> Px {
    bool4(a, b, |x, y| truthy(x) && truthy(y))
}

/// WGSL `or4`.
pub fn or4(a: Px, b: Px) -> Px {
    bool4(a, b, |x, y| truthy(x) || truthy(y))
}

/// WGSL `xor4`.
pub fn xor4(a: Px, b: Px) -> Px {
    bool4(a, b, |x, y| truthy(x) != truthy(y))
}

/// WGSL `not4`.
pub fn not4(a: Px) -> Px {
    a.map(|x| if truthy(x) { 0.0 } else { 1.0 })
}

/// WGSL `sign4` — composed from comparisons so NaN → 0 on both lanes
/// (the builtin `sign` is not NaN-deterministic across drivers).
pub fn sign4(a: Px) -> Px {
    gt4(a, splat4(0.0)) - lt4(a, splat4(0.0))
}

/// WGSL `neg4`.
pub fn neg4(a: Px) -> Px {
    -a
}

/// WGSL `premul4` — rgb · a, alpha through.
pub fn premul4(c: Px) -> Px {
    Px([c.0[0] * c.0[3], c.0[1] * c.0[3], c.0[2] * c.0[3], c.0[3]])
}

/// WGSL `unpremul4` — zero alpha maps to all-zero (no Inf/NaN leak).
pub fn unpremul4(c: Px) -> Px {
    if c.0[3] == 0.0 {
        splat4(0.0)
    } else {
        Px([c.0[0] / c.0[3], c.0[1] / c.0[3], c.0[2] / c.0[3], c.0[3]])
    }
}

/// WGSL builtin `clamp` (componentwise).
pub fn clamp(x: Px, lo: Px, hi: Px) -> Px {
    // WGSL clamp(e, low, high) = min(max(e, low), high).
    min(max(x, lo), hi)
}

/// WGSL builtin `mix` (componentwise linear blend).
pub fn mix(a: Px, b: Px, t: Px) -> Px {
    // WGSL mix(e1, e2, e3) = e1 * (1 - e3) + e2 * e3.
    Px([
        a.0[0] * (1.0 - t.0[0]) + b.0[0] * t.0[0],
        a.0[1] * (1.0 - t.0[1]) + b.0[1] * t.0[1],
        a.0[2] * (1.0 - t.0[2]) + b.0[2] * t.0[2],
        a.0[3] * (1.0 - t.0[3]) + b.0[3] * t.0[3],
    ])
}

/// WGSL builtin `min` (componentwise).
pub fn min(a: Px, b: Px) -> Px {
    a.zip(b, f32::min)
}

/// WGSL builtin `max` (componentwise).
pub fn max(a: Px, b: Px) -> Px {
    a.zip(b, f32::max)
}

/// WGSL builtin `abs` (componentwise).
pub fn abs(x: Px) -> Px {
    x.map(f32::abs)
}

/// WGSL builtin `floor` (componentwise).
pub fn floor(x: Px) -> Px {
    x.map(f32::floor)
}
