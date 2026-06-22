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

//! minmax family (T0, spec §11): componentwise `min`/`max` over two
//! inputs, `clamp` to a constant `[lo, hi]` window, and the constant
//! variants `min_const`/`max_const`. All five are exact-tolerance: the
//! WGSL builtins (`min`, `max`, `clamp`) pick an existing input value
//! per channel — no arithmetic, so the f16-quantized reference and the
//! GPU agree bit-for-bit.
//!
//! Provenance: elementary pointwise algebra / no reference reading.
//! libvips-equivalent behavior verified through the differential oracle
//! harness (M0 fan-out), not by reference reading.

use crate::{KernelClass, KernelDef, Tolerance};

kernel_family! {
    /// out = min(a, b) (per channel, alpha included).
    static MATH_MIN, params MathMinParams, ref math_min {
        id: "math.min",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| min(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = max(a, b) (per channel, alpha included).
    static MATH_MAX, params MathMaxParams, ref math_max {
        id: "math.max",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| max(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = clamp(a, lo, hi) (per channel, constant window).
    static MATH_CLAMP, params MathClampParams, ref math_clamp {
        id: "math.clamp",
        class: KernelClass::Point,
        inputs: 1,
        params: { lo: f32, hi: f32 },
        eval: |a, b, p| clamp(a, splat4(p.lo), splat4(p.hi)),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = min(a, v) (per channel, constant ceiling).
    static MATH_MIN_CONST, params MathMinConstParams, ref math_min_const {
        id: "math.min_const",
        class: KernelClass::Point,
        inputs: 1,
        params: { v: f32 },
        eval: |a, b, p| min(a, splat4(p.v)),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = max(a, v) (per channel, constant floor).
    static MATH_MAX_CONST, params MathMaxConstParams, ref math_max_const {
        id: "math.max_const",
        class: KernelClass::Point,
        inputs: 1,
        params: { v: f32 },
        eval: |a, b, p| max(a, splat4(p.v)),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

pub static FAMILY: &[&KernelDef] = &[
    &MATH_MIN,
    &MATH_MAX,
    &MATH_CLAMP,
    &MATH_MIN_CONST,
    &MATH_MAX_CONST,
];
