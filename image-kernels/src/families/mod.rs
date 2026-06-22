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

//! T0 kernel families (spec §11). One file per family; each file owns
//! its `kernel_family!` rows, its registry entries
//! (`registry/kernels.yaml`), and its conformance tests — the
//! conflict-free fan-out unit (spec §16.3 rule 5).
//!
//! `linear` was the phase-0 codegen proof; the other six land with the
//! M0 fan-out (one agent per file).

pub mod adjust;
pub mod arithmetic;
pub mod band;
pub mod boolean;
pub mod cast;
pub mod compose;
pub mod conv;
pub mod gen;
pub mod geom;
pub mod linear;
pub mod minmax;
pub mod morph;
pub mod relational;
pub mod resample;

use crate::KernelDef;

/// Every family's definition slice — `all_defined()` concatenates
/// these; the conformance gate asserts set-equality with the
/// registry-generated dispatch table.
pub static ALL_FAMILIES: &[&[&KernelDef]] = &[
    adjust::FAMILY,
    arithmetic::FAMILY,
    band::FAMILY,
    boolean::FAMILY,
    cast::FAMILY,
    compose::FAMILY,
    conv::FAMILY,
    gen::FAMILY,
    geom::FAMILY,
    linear::FAMILY,
    minmax::FAMILY,
    morph::FAMILY,
    relational::FAMILY,
    resample::FAMILY,
];
