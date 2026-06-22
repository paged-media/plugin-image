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

//! Linear family (T0): `linear` (a·x + b) and `invert` (1 − x).
//! `math.linear` is the phase-0 codegen proof — the first kernel
//! through the full dual-emission + gpu↔ref parity gate.
//!
//! Provenance: elementary pointwise algebra; libvips-equivalent
//! behavior verified through the differential oracle harness (M0
//! fan-out), not by reference reading.

use crate::{KernelClass, KernelDef, Tolerance};

kernel_family! {
    /// out = a * gain + bias (per channel, alpha included).
    static MATH_LINEAR, params MathLinearParams, ref math_linear {
        id: "math.linear",
        class: KernelClass::Point,
        inputs: 1,
        params: { gain: f32, bias: f32 },
        eval: |a, b, p| a * splat4(p.gain) + splat4(p.bias),
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(2),
    }
}

kernel_family! {
    /// out = 1 - a (photometric negate in linear working space).
    static MATH_INVERT, params MathInvertParams, ref math_invert {
        id: "math.invert",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| splat4(1.0) - a,
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(1),
    }
}

pub static FAMILY: &[&KernelDef] = &[&MATH_LINEAR, &MATH_INVERT];
