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

//! cast family (T0, spec §11) — the alpha-association casts: associate
//! colour with alpha (`premultiply`) and dissociate it (`unpremultiply`).
//! Both reduce to the orchestrator-owned prelude primitives, where the
//! divide-by-zero contract lives: `unpremul4` maps zero alpha to all-zero
//! deterministically (no Inf/NaN leak), so the cast is total.
//!
//! Provenance: elementary pointwise algebra (rgb·α and rgb/α with the
//! zero-alpha guard); GEGL-equivalent behaviour verified through the
//! differential oracle harness (M0 fan-out), not by reference reading.

use crate::{KernelClass, KernelDef, Tolerance};

kernel_family! {
    /// out = (rgb · a, a) — associate colour with alpha.
    static CAST_PREMULTIPLY, params CastPremultiplyParams, ref cast_premultiply {
        id: "cast.premultiply",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| premul4(a),
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(1),
    }
}

kernel_family! {
    /// out = (rgb / a, a); zero alpha → all-zero (no Inf/NaN leak).
    static CAST_UNPREMULTIPLY, params CastUnpremultiplyParams, ref cast_unpremultiply {
        id: "cast.unpremultiply",
        class: KernelClass::Point,
        inputs: 1,
        params: {},
        eval: |a, b, p| unpremul4(a),
        mip_exact: true,
        tolerance: Tolerance::ChannelEpsF16(2),
    }
}

pub static FAMILY: &[&KernelDef] = &[&CAST_PREMULTIPLY, &CAST_UNPREMULTIPLY];
