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

//! Write orchestration with the lazy-verbatim guard: nodes carrying
//! their source bytes (`Raw = Some`) re-emit them verbatim (zero-edit ⇒
//! byte-identical); `None` nodes re-encode canonically via `framed()`
//! back-patching. Lands with M0 fan-out unit U7.

use crate::model::PsdFile;
use crate::Result;

pub fn write(file: &PsdFile) -> Result<Vec<u8>> {
    let _ = file;
    unimplemented!("M0 fan-out: write orchestration (unit U7)")
}
