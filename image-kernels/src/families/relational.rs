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

//! Relational family (T0, spec §11): the six per-channel comparisons
//! `eq`/`ne`/`lt`/`le`/`gt`/`ge` over two inputs, each emitting an exact
//! 0.0/1.0 mask. The work is entirely the shared prelude helpers
//! (`eq4`…`ge4`, `abi::WGSL_PRELUDE` ≡ `reference_prelude`); the kernel
//! body is one call apiece, so WGSL ≡ Rust is true by construction and
//! the result lands on exact f16 representables (0.0/1.0) ⇒ Exact
//! tolerance, no ULP slack.
//!
//! Provenance: elementary pointwise algebra / no reference reading;
//! libvips-equivalent relational behavior verified through the
//! differential oracle harness (vips), not by reference reading.

use crate::{KernelClass, KernelDef, Tolerance};

kernel_family! {
    /// out = (a == b) ? 1 : 0 (per channel; NaN compares false, IEEE).
    static REL_EQ, params RelEqParams, ref rel_eq {
        id: "rel.eq",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| eq4(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = (a != b) ? 1 : 0 (per channel; NaN compares true, IEEE).
    static REL_NE, params RelNeParams, ref rel_ne {
        id: "rel.ne",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| ne4(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = (a < b) ? 1 : 0 (per channel).
    static REL_LT, params RelLtParams, ref rel_lt {
        id: "rel.lt",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| lt4(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = (a <= b) ? 1 : 0 (per channel).
    static REL_LE, params RelLeParams, ref rel_le {
        id: "rel.le",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| le4(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = (a > b) ? 1 : 0 (per channel).
    static REL_GT, params RelGtParams, ref rel_gt {
        id: "rel.gt",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| gt4(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

kernel_family! {
    /// out = (a >= b) ? 1 : 0 (per channel).
    static REL_GE, params RelGeParams, ref rel_ge {
        id: "rel.ge",
        class: KernelClass::Point,
        inputs: 2,
        params: {},
        eval: |a, b, p| ge4(a, b),
        mip_exact: true,
        tolerance: Tolerance::Exact,
    }
}

pub static FAMILY: &[&KernelDef] = &[&REL_EQ, &REL_NE, &REL_LT, &REL_LE, &REL_GT, &REL_GE];
