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

//! Parse orchestration: header → color mode → image resources → layer
//! & mask info → merged composite. Per-section parsers land with the
//! M0 fan-out units (U2–U6); this file owns the top-level sequencing
//! and the section length framing.

use crate::model::PsdFile;
use crate::Result;

pub fn parse(bytes: &[u8]) -> Result<PsdFile> {
    let _ = bytes;
    unimplemented!("M0 fan-out: parse orchestration (units U2–U6)")
}
