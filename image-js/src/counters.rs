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

//! Engine work counters: the CPU-side costs the GPU counters
//! (`image_gpu::counters`) cannot see — whole-image copies, 16→8-bit
//! narrowings, layer composites, tile cuts. Budgets are written against
//! these counts, never wall-clock. Thread-local and always on, for the
//! same reason as the GPU counters: one operation's work runs on the
//! thread that asked for it (the single wasm thread in the browser).

use std::cell::Cell;

/// One operation's engine work. `*_bytes` are bytes, everything else
/// is a number of events.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EngineCounters {
    /// Whole-image buffers copied (an owned copy of a full image).
    pub whole_image_copies: u64,
    /// Bytes those copies moved.
    pub bytes_copied: u64,
    /// 16-bit buffers narrowed to 8 bits (`Pixels::to_rgba8` on a 16-bit store).
    pub depth_narrowings: u64,
    /// Bytes produced by those narrowings.
    pub narrowed_bytes: u64,
    /// Layer-stack composites requested.
    pub composites: u64,
    /// Layers blended by composites that had to fold (the one-plain-layer
    /// fast path folds nothing).
    pub layers_folded: u64,
    /// Tile windows cut for the host.
    pub tiles_cut: u64,
}

impl EngineCounters {
    /// What happened between `earlier` and `self`.
    pub fn since(&self, earlier: &EngineCounters) -> EngineCounters {
        EngineCounters {
            whole_image_copies: self
                .whole_image_copies
                .saturating_sub(earlier.whole_image_copies),
            bytes_copied: self.bytes_copied.saturating_sub(earlier.bytes_copied),
            depth_narrowings: self
                .depth_narrowings
                .saturating_sub(earlier.depth_narrowings),
            narrowed_bytes: self.narrowed_bytes.saturating_sub(earlier.narrowed_bytes),
            composites: self.composites.saturating_sub(earlier.composites),
            layers_folded: self.layers_folded.saturating_sub(earlier.layers_folded),
            tiles_cut: self.tiles_cut.saturating_sub(earlier.tiles_cut),
        }
    }
}

thread_local! {
    static COUNTERS: Cell<EngineCounters> = const {
        Cell::new(EngineCounters {
            whole_image_copies: 0,
            bytes_copied: 0,
            depth_narrowings: 0,
            narrowed_bytes: 0,
            composites: 0,
            layers_folded: 0,
            tiles_cut: 0,
        })
    };
}

/// This thread's counters since the last [`reset`].
pub fn snapshot() -> EngineCounters {
    COUNTERS.with(Cell::get)
}

/// Zero this thread's engine AND GPU counters.
pub fn reset() {
    COUNTERS.with(|c| c.set(EngineCounters::default()));
    image_gpu::counters::reset();
}

/// Run `f` and return its engine and GPU cost.
pub fn measure<R>(f: impl FnOnce() -> R) -> (R, EngineCounters, image_gpu::GpuCounters) {
    let (e0, g0) = (snapshot(), image_gpu::counters::snapshot());
    let r = f();
    (
        r,
        snapshot().since(&e0),
        image_gpu::counters::snapshot().since(&g0),
    )
}

pub(crate) fn bump(f: impl FnOnce(&mut EngineCounters)) {
    COUNTERS.with(|c| {
        let mut v = c.get();
        f(&mut v);
        c.set(v);
    });
}

/// Count one owned copy of a whole image of `bytes` bytes.
pub(crate) fn whole_copy(bytes: usize) {
    bump(|c| {
        c.whole_image_copies += 1;
        c.bytes_copied += bytes as u64;
    });
}

/// Both counter sets as one JSON object (the `perf_counters` wasm
/// export; field names are the Rust names).
pub fn to_json() -> String {
    let e = snapshot();
    let g = image_gpu::counters::snapshot();
    format!(
        "{{\"engine\":{{\"wholeImageCopies\":{},\"bytesCopied\":{},\"depthNarrowings\":{},\
         \"narrowedBytes\":{},\"composites\":{},\"layersFolded\":{},\"tilesCut\":{}}},\
         \"gpu\":{{\"shaderModules\":{},\"pipelinesBuilt\":{},\"dispatches\":{},\
         \"dispatchedTexels\":{},\"submits\":{},\"texturesCreated\":{},\"bytesUploaded\":{},\
         \"readbacks\":{},\"bytesReadBack\":{}}}}}",
        e.whole_image_copies,
        e.bytes_copied,
        e.depth_narrowings,
        e.narrowed_bytes,
        e.composites,
        e.layers_folded,
        e.tiles_cut,
        g.shader_modules,
        g.pipelines_built,
        g.dispatches,
        g.dispatched_texels,
        g.submits,
        g.textures_created,
        g.bytes_uploaded,
        g.readbacks,
        g.bytes_read_back,
    )
}
