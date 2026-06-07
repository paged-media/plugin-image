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

//! arithmetic family (T0, spec §11): elementary pointwise algebra over
//! the working-space rgba — dyadic `add`/`sub`/`mul`/`div`, the scalar
//! variants `add_const`/`mul_const`, and the unary `abs`/`sign`/`neg`.
//!
//! Provenance: elementary pointwise algebra / no reference reading.
//! libvips-equivalent behavior is verified through the differential
//! oracle harness (M0 fan-out), not by reading any reference source.

use crate::{KernelClass, KernelDef, Tolerance};

kernel_family! {
    /// out = a + b (per channel, alpha included).
    static MATH_ADD, params MathAddParams, ref math_add {
        id: "math.add",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| a + b,
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(1),
    }
}

kernel_family! {
    /// out = a - b (per channel, alpha included).
    static MATH_SUB, params MathSubParams, ref math_sub {
        id: "math.sub",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| a - b,
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(1),
    }
}

kernel_family! {
    /// out = a * b (per channel, alpha included).
    static MATH_MUL, params MathMulParams, ref math_mul {
        id: "math.mul",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| a * b,
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(1),
    }
}

kernel_family! {
    /// out = a / b (per channel, alpha included). Divisor-near-zero is
    /// the caller's domain to keep away from (parity stimulus does).
    static MATH_DIV, params MathDivParams, ref math_div {
        id: "math.div",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| a / b,
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(2),
    }
}

kernel_family! {
    /// out = a + v (scalar broadcast to every channel).
    static MATH_ADD_CONST, params MathAddConstParams, ref math_add_const {
        id: "math.add_const",
        class: KernelClass::Point,
        inputs: 1,
        params: { v: f32 },
        eval: |a, b, p| a + splat4(p.v),
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(1),
    }
}

kernel_family! {
    /// out = a * v (scalar broadcast to every channel).
    static MATH_MUL_CONST, params MathMulConstParams, ref math_mul_const {
        id: "math.mul_const",
        class: KernelClass::Point,
        inputs: 1,
        params: { v: f32 },
        eval: |a, b, p| a * splat4(p.v),
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(1),
    }
}

kernel_family! {
    /// out = |a| (per channel).
    static MATH_ABS, params MathAbsParams, ref math_abs {
        id: "math.abs",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| abs(a),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = sign(a) ∈ {-1, 0, +1} (comparison-composed, NaN → 0 on
    /// both lanes — see `sign4` in the prelude).
    static MATH_SIGN, params MathSignParams, ref math_sign {
        id: "math.sign",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| sign4(a),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = -a (per channel).
    static MATH_NEG, params MathNegParams, ref math_neg {
        id: "math.neg",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| neg4(a),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

pub static FAMILY: &[&KernelDef] = &[
    &MATH_ADD,
    &MATH_SUB,
    &MATH_MUL,
    &MATH_DIV,
    &MATH_ADD_CONST,
    &MATH_MUL_CONST,
    &MATH_ABS,
    &MATH_SIGN,
    &MATH_NEG,
];
