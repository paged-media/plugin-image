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

//! boolean family (T0, spec §11): pointwise logical `and`/`or`/`xor`/
//! `not` over the prelude truthiness rule (`> 0.5`), emitting exact
//! 0.0/1.0. Outputs land on f16-representable integers, so every op is
//! bit-exact gpu↔ref (Tolerance::Exact).
//!
//! Provenance: elementary pointwise algebra / no reference reading;
//! libvips boolean-equivalent behavior is checked through the
//! differential oracle harness (M0 fan-out), never by reference reading.

use crate::{KernelClass, KernelDef, Tolerance};

kernel_family! {
    /// out = (a > 0.5) AND (b > 0.5) → 1.0/0.0 (per channel).
    static BOOL_AND, params BoolAndParams, ref bool_and {
        id: "bool.and",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| and4(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = (a > 0.5) OR (b > 0.5) → 1.0/0.0 (per channel).
    static BOOL_OR, params BoolOrParams, ref bool_or {
        id: "bool.or",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| or4(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = (a > 0.5) XOR (b > 0.5) → 1.0/0.0 (per channel).
    static BOOL_XOR, params BoolXorParams, ref bool_xor {
        id: "bool.xor",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| xor4(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = NOT (a > 0.5) → 1.0/0.0 (per channel).
    static BOOL_NOT, params BoolNotParams, ref bool_not {
        id: "bool.not",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| not4(a),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

pub static FAMILY: &[&KernelDef] = &[&BOOL_AND, &BOOL_OR, &BOOL_XOR, &BOOL_NOT];
