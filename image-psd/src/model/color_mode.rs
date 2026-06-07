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

//! Color mode data section — meaningful only for Indexed (palette) and
//! Duotone modes; zero-length otherwise. Opaque-verbatim by design
//! (preservation strategy 2): no editing semantics in scope.
//!
//! Provenance: Adobe Photoshop File Format specification, "Color Mode
//! Data Section".

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ColorModeData {
    /// The section payload exactly as read (without the length field).
    pub raw: Vec<u8>,
}
