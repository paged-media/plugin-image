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
//! M0 fan-out adds: arithmetic, relational, boolean, cast, band,
//! minmax. `linear` is the phase-0 codegen proof.

pub mod linear;

use crate::KernelDef;

/// Every family's definition slice — `all_defined()` concatenates
/// these; the conformance gate asserts set-equality with the
/// registry-generated dispatch table.
pub static ALL_FAMILIES: &[&[&KernelDef]] = &[linear::FAMILY];
