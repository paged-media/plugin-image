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

//! paged.image core types (spec §5) — the M0 phase-0 interface freeze.
//!
//! This crate is a LEAF: it depends on nothing in the workspace and on
//! no engine. `TextureSlot` / `OpfsKey` are plain newtypes here; the
//! GPU pool (image-gpu) and the OPFS tier own the actual resources
//! keyed by them, so core stays engine-agnostic (spec §4 dep rule 1).
//!
//! Changes to these types after the freeze go through the orchestrator
//! as versioned amendments — never drive-by edits.

#![forbid(unsafe_code)]

mod format;
mod generation;
mod region;
mod slice;
mod tile;
mod tilemap;

pub use format::{
    AlphaMode, ChannelLayout, ColorSpaceRef, IccHash, NamedSpace, PixelFormat, SampleDepth,
    Transfer, TransferCurve,
};
pub use generation::{ContentHash, GenerationId, ParamsHash};
pub use region::Region;
pub use slice::{TileSliceMut, TileSliceRef};
pub use tile::{OpfsKey, TextureSlot, Tile, TileCoord, TileData, TILE};
pub use tilemap::{ConstantPixel, PersistentBuffer, ResidencyMeta, TileMap};
