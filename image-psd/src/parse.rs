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

//! Parse orchestration: header → color mode → image resources → layer
//! & mask info → merged composite. Per-section parsers live next to their
//! model types (`model/*.rs`); this file owns the top-level sequencing.
//!
//! Every section captures its verbatim source bytes into the model's `Raw`
//! fields, so an unmodified parse re-emits byte-identically (the §10.4
//! preservation invariant).

use crate::model::{
    ColorModeData, FileHeader, GlobalImageData, ImageResources, LayerAndMaskInfo, PsdFile,
};
use crate::reader::ByteReader;
use crate::Result;

pub fn parse(bytes: &[u8]) -> Result<PsdFile> {
    let mut r = ByteReader::new(bytes);

    let (container, header) = FileHeader::parse(&mut r)?;
    let color_mode = ColorModeData::parse(&mut r)?;
    let resources = ImageResources::parse(&mut r)?;
    let layer_mask = LayerAndMaskInfo::parse(&mut r, container)?;
    // The merged composite is the final, un-framed section: it runs to EOF.
    let composite = GlobalImageData::parse(&mut r)?;

    Ok(PsdFile {
        container,
        header,
        color_mode,
        resources,
        layer_mask,
        composite,
    })
}
