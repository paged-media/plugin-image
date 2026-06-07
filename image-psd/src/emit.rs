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

//! Write orchestration with the lazy-verbatim guard: nodes carrying their
//! source bytes (`Raw = Some`) re-emit them verbatim (zero-edit ⇒
//! byte-identical); `None` nodes re-encode canonically via the per-section
//! emitters (`framed()` back-patching, canonical padding).

use crate::model::PsdFile;
use crate::writer::ByteWriter;
use crate::Result;

pub fn write(file: &PsdFile) -> Result<Vec<u8>> {
    let mut w = ByteWriter::new();
    let container = file.container;

    file.header.emit(&mut w, container);
    file.color_mode.emit(&mut w);
    file.resources.emit(&mut w);
    file.layer_mask.emit(&mut w, container);
    // The merged composite closes the file — un-framed, runs to EOF.
    file.composite.emit(&mut w);

    Ok(w.into_bytes())
}
