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

//! Work counters for the GPU layer: how many shader modules, pipelines,
//! textures, dispatches, submits, uploads and readbacks an operation
//! cost. Performance budgets are written against these COUNTS, never
//! against wall-clock time, so a budget means the same thing on a
//! laptop, on a CI software adapter and in the browser.
//!
//! The counters are THREAD-LOCAL. All GPU work for one operation runs
//! on the thread that asked for it (a blocking `poll` on native, the
//! single wasm thread in the browser), so a test reads exactly its own
//! work even while the harness runs other tests in parallel. They are
//! always on: a counter bump is a thread-local add, which is noise next
//! to a single GPU call.

use std::cell::Cell;

/// One operation's GPU work. Field names say what was counted;
/// `*_bytes` are bytes, everything else is a number of calls.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GpuCounters {
    /// `create_shader_module` calls (WGSL compiled).
    pub shader_modules: u64,
    /// `create_compute_pipeline` calls.
    pub pipelines_built: u64,
    /// `dispatch_workgroups` calls.
    pub dispatches: u64,
    /// Output texels covered by those dispatches.
    pub dispatched_texels: u64,
    /// `queue.submit` calls.
    pub submits: u64,
    /// `create_texture` calls.
    pub textures_created: u64,
    /// Bytes written by `write_texture` / `write_buffer`.
    pub bytes_uploaded: u64,
    /// Buffers mapped back to the CPU.
    pub readbacks: u64,
    /// Bytes copied out of mapped buffers.
    pub bytes_read_back: u64,
}

impl GpuCounters {
    /// What happened between `earlier` and `self` (saturating, so a
    /// reset in between reads as zero rather than wrapping).
    pub fn since(&self, earlier: &GpuCounters) -> GpuCounters {
        GpuCounters {
            shader_modules: self.shader_modules.saturating_sub(earlier.shader_modules),
            pipelines_built: self.pipelines_built.saturating_sub(earlier.pipelines_built),
            dispatches: self.dispatches.saturating_sub(earlier.dispatches),
            dispatched_texels: self
                .dispatched_texels
                .saturating_sub(earlier.dispatched_texels),
            submits: self.submits.saturating_sub(earlier.submits),
            textures_created: self
                .textures_created
                .saturating_sub(earlier.textures_created),
            bytes_uploaded: self.bytes_uploaded.saturating_sub(earlier.bytes_uploaded),
            readbacks: self.readbacks.saturating_sub(earlier.readbacks),
            bytes_read_back: self.bytes_read_back.saturating_sub(earlier.bytes_read_back),
        }
    }
}

thread_local! {
    static COUNTERS: Cell<GpuCounters> = const {
        Cell::new(GpuCounters {
            shader_modules: 0,
            pipelines_built: 0,
            dispatches: 0,
            dispatched_texels: 0,
            submits: 0,
            textures_created: 0,
            bytes_uploaded: 0,
            readbacks: 0,
            bytes_read_back: 0,
        })
    };
}

/// This thread's counters since the last [`reset`].
pub fn snapshot() -> GpuCounters {
    COUNTERS.with(Cell::get)
}

/// Zero this thread's counters.
pub fn reset() {
    COUNTERS.with(|c| c.set(GpuCounters::default()));
}

/// Run `f` and return what it cost (does not disturb the running
/// totals, so measurements nest).
pub fn measure<R>(f: impl FnOnce() -> R) -> (R, GpuCounters) {
    let before = snapshot();
    let r = f();
    (r, snapshot().since(&before))
}

pub(crate) fn bump(f: impl FnOnce(&mut GpuCounters)) {
    COUNTERS.with(|c| {
        let mut v = c.get();
        f(&mut v);
        c.set(v);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn measure_reports_only_the_work_inside_it__feat__image_conformance_harness() {
        reset();
        bump(|c| c.dispatches += 2);
        let ((), inner) = measure(|| bump(|c| c.dispatches += 3));
        assert_eq!(inner.dispatches, 3);
        assert_eq!(snapshot().dispatches, 5, "the running total keeps both");
        reset();
        assert_eq!(snapshot(), GpuCounters::default());
    }

    #[test]
    #[allow(non_snake_case)]
    fn counters_are_per_thread__feat__image_conformance_harness() {
        reset();
        bump(|c| c.submits += 1);
        std::thread::spawn(|| {
            assert_eq!(snapshot().submits, 0, "another thread starts at zero");
            bump(|c| c.submits += 10);
        })
        .join()
        .expect("thread");
        assert_eq!(snapshot().submits, 1);
    }
}
