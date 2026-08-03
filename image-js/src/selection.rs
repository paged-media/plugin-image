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

//! The per-session SELECTION state behind the wasm `selection_*` doors:
//! one selection, bound to one engine-held image (its resolution fixes
//! the coverage field; its pixels feed the magic wand). All coverage
//! math lives in `image_gpu::coverage` (mask PREP — inherently CPU); the
//! mask is CONSUMED GPU-only through the adjust chain's `@group(2)`
//! r16float binding (`Pipeline::set_selection`).
//!
//! Semantics (documented on the wasm doors):
//! * "No selection" (`coverage == None`) means EVERYTHING is selected —
//!   adjustments run unmasked, the historic constant-1 behavior.
//! * A combine against "no selection" starts from the mode-appropriate
//!   implicit state: replace/add start from the shape; subtract and
//!   intersect start from FULL coverage (the Photoshop convention).
//! * `clear` returns to "no selection" (NOT to an all-zero coverage);
//!   an explicitly all-zero selection (e.g. subtract-to-empty) is kept
//!   and means "adjust applies nowhere" — the honest empty selection.
//! * Re-binding to a different image handle or resolution drops the
//!   coverage (a selection is meaningless across resolutions).

use std::sync::Arc;

use image_gpu::{CombineMode, SelectionCoverage};

/// The image a selection is bound to (dims fix the coverage field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundImage {
    pub handle: u32,
    pub width: u32,
    pub height: u32,
}

/// One session's selection: the bound image + the optional coverage +
/// a monotone revision (bumped on every mutation so the glue can cheaply
/// detect change).
#[derive(Default)]
pub struct SessionSelection {
    bound: Option<BoundImage>,
    coverage: Option<Arc<SelectionCoverage>>,
    revision: u64,
}

impl SessionSelection {
    pub fn new() -> Self {
        SessionSelection::default()
    }

    /// Bind the selection to an engine-held image. Re-binding to the
    /// SAME handle+dims keeps the coverage; anything else drops it
    /// (bumping the revision when a selection was actually lost).
    pub fn bind(&mut self, handle: u32, width: u32, height: u32) {
        let next = BoundImage {
            handle,
            width,
            height,
        };
        if self.bound == Some(next) {
            return;
        }
        self.bound = Some(next);
        if self.coverage.take().is_some() {
            self.revision += 1;
        }
    }

    pub fn bound(&self) -> Option<BoundImage> {
        self.bound
    }

    /// Re-point the selection at a NEW handle that holds the SAME
    /// extent, KEEPING the coverage. A destructive in-place edit (the
    /// generator FILL) registers its result as a new engine image at
    /// identical dimensions — the selection is still exactly as
    /// meaningful there, so dropping it would be pure friction. Returns
    /// `true` when the coverage was carried over; on any extent change
    /// it falls back to the ordinary [`Self::bind`] rule (drop) and
    /// returns `false`.
    pub fn transfer(&mut self, handle: u32, width: u32, height: u32) -> bool {
        match self.bound {
            Some(b) if b.width == width && b.height == height => {
                self.bound = Some(BoundImage {
                    handle,
                    width,
                    height,
                });
                true
            }
            _ => {
                self.bind(handle, width, height);
                false
            }
        }
    }

    /// The coverage, when an explicit selection exists.
    pub fn coverage(&self) -> Option<&Arc<SelectionCoverage>> {
        self.coverage.as_ref()
    }

    /// The coverage the ADJUST chain should mask with for `handle`:
    /// `Some` only when a selection is bound to that same image AND is
    /// non-trivial (an all-one coverage is the constant-1 default — no
    /// mask bind needed).
    pub fn mask_for(&self, handle: u32) -> Option<Arc<SelectionCoverage>> {
        let bound = self.bound?;
        if bound.handle != handle {
            return None;
        }
        let cov = self.coverage.as_ref()?;
        if cov.is_all_one() {
            return None;
        }
        Some(Arc::clone(cov))
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Fold a freshly rasterized `shape` (at the bound resolution) into
    /// the selection under `mode`. Errors when nothing is bound.
    pub fn apply_shape(
        &mut self,
        shape: SelectionCoverage,
        mode: CombineMode,
    ) -> Result<(), String> {
        let bound = self.bound.ok_or("no image bound (selection_bind first)")?;
        if shape.width() != bound.width || shape.height() != bound.height {
            return Err(format!(
                "shape {}x{} does not match the bound image {}x{}",
                shape.width(),
                shape.height(),
                bound.width,
                bound.height
            ));
        }
        let mut base = match (&self.coverage, mode) {
            // No selection yet: replace/add adopt the shape directly …
            (None, CombineMode::Replace) | (None, CombineMode::Add) => {
                self.coverage = Some(Arc::new(shape));
                self.revision += 1;
                return Ok(());
            }
            // … subtract/intersect start from the implicit FULL selection.
            (None, _) => SelectionCoverage::full(bound.width, bound.height),
            (Some(existing), _) => (**existing).clone(),
        };
        base.combine(&shape, mode);
        self.coverage = Some(Arc::new(base));
        self.revision += 1;
        Ok(())
    }

    /// Select everything explicitly (an all-one coverage — shows as a
    /// full-extent selection in the panel; the adjust chain still takes
    /// the trivial-mask fast path).
    pub fn select_all(&mut self) -> Result<(), String> {
        let bound = self.bound.ok_or("no image bound (selection_bind first)")?;
        self.coverage = Some(Arc::new(SelectionCoverage::full(bound.width, bound.height)));
        self.revision += 1;
        Ok(())
    }

    /// Back to "no selection" (everything selected implicitly).
    pub fn clear(&mut self) {
        if self.coverage.take().is_some() {
            self.revision += 1;
        }
    }

    /// Invert. With no explicit selection, "everything" inverts to the
    /// explicit EMPTY selection (all-zero coverage).
    pub fn invert(&mut self) -> Result<(), String> {
        let bound = self.bound.ok_or("no image bound (selection_bind first)")?;
        let mut cov = match self.coverage.take() {
            Some(c) => (*c).clone(),
            None => SelectionCoverage::full(bound.width, bound.height),
        };
        cov.invert();
        self.coverage = Some(Arc::new(cov));
        self.revision += 1;
        Ok(())
    }

    /// Feather the explicit selection by `sigma` px (no-op without one —
    /// "everything" has no edge to soften).
    pub fn feather(&mut self, sigma: f32) -> Result<(), String> {
        let Some(existing) = self.coverage.take() else {
            return Err("no selection to feather".into());
        };
        let mut cov = (*existing).clone();
        cov.feather(sigma);
        self.coverage = Some(Arc::new(cov));
        self.revision += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect_shape(w: u32, h: u32) -> SelectionCoverage {
        SelectionCoverage::rasterize_rect(w, h, 1.0, 1.0, 2.0, 2.0)
    }

    #[test]
    fn unbound_shape_application_errors() {
        let mut s = SessionSelection::new();
        assert!(s
            .apply_shape(rect_shape(4, 4), CombineMode::Replace)
            .is_err());
        assert!(s.select_all().is_err());
        assert!(s.invert().is_err());
    }

    #[test]
    fn replace_adopts_the_shape_and_bumps_revision() {
        let mut s = SessionSelection::new();
        s.bind(7, 4, 4);
        let r0 = s.revision();
        s.apply_shape(rect_shape(4, 4), CombineMode::Replace)
            .unwrap();
        assert!(s.revision() > r0);
        let cov = s.coverage().unwrap();
        assert_eq!(cov.coverage_at(1, 1), 255);
        assert_eq!(cov.coverage_at(0, 0), 0);
    }

    #[test]
    fn subtract_from_no_selection_starts_from_full() {
        let mut s = SessionSelection::new();
        s.bind(7, 4, 4);
        s.apply_shape(rect_shape(4, 4), CombineMode::Subtract)
            .unwrap();
        let cov = s.coverage().unwrap();
        assert_eq!(cov.coverage_at(1, 1), 0, "subtracted hole");
        assert_eq!(cov.coverage_at(0, 0), 255, "rest stays (implicit full)");
    }

    #[test]
    fn intersect_from_no_selection_is_the_shape() {
        let mut s = SessionSelection::new();
        s.bind(7, 4, 4);
        s.apply_shape(rect_shape(4, 4), CombineMode::Intersect)
            .unwrap();
        let cov = s.coverage().unwrap();
        assert_eq!(cov.coverage_at(1, 1), 255);
        assert_eq!(cov.coverage_at(0, 0), 0);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let mut s = SessionSelection::new();
        s.bind(7, 8, 8);
        assert!(s
            .apply_shape(rect_shape(4, 4), CombineMode::Replace)
            .is_err());
    }

    #[test]
    fn rebind_same_image_keeps_rebind_other_drops() {
        let mut s = SessionSelection::new();
        s.bind(7, 4, 4);
        s.apply_shape(rect_shape(4, 4), CombineMode::Replace)
            .unwrap();
        s.bind(7, 4, 4); // same image — selection survives
        assert!(s.coverage().is_some());
        s.bind(8, 4, 4); // a different handle (crop/resize swap) — drops
        assert!(s.coverage().is_none());
    }

    #[test]
    fn mask_for_gates_on_handle_and_triviality() {
        let mut s = SessionSelection::new();
        s.bind(7, 4, 4);
        assert!(s.mask_for(7).is_none(), "no selection → no mask");
        s.select_all().unwrap();
        assert!(s.mask_for(7).is_none(), "all-one is the trivial mask");
        s.apply_shape(rect_shape(4, 4), CombineMode::Replace)
            .unwrap();
        assert!(s.mask_for(7).is_some());
        assert!(s.mask_for(9).is_none(), "a different handle is unmasked");
    }

    #[test]
    fn clear_returns_to_no_selection() {
        let mut s = SessionSelection::new();
        s.bind(7, 4, 4);
        s.apply_shape(rect_shape(4, 4), CombineMode::Replace)
            .unwrap();
        let r = s.revision();
        s.clear();
        assert!(s.coverage().is_none());
        assert!(s.revision() > r);
        let r2 = s.revision();
        s.clear(); // idempotent — no revision churn
        assert_eq!(s.revision(), r2);
    }

    #[test]
    fn invert_of_no_selection_is_the_empty_selection() {
        let mut s = SessionSelection::new();
        s.bind(7, 4, 4);
        s.invert().unwrap();
        let cov = s.coverage().unwrap();
        assert!(cov.is_all_zero(), "everything inverts to nothing");
    }

    #[test]
    fn feather_requires_an_explicit_selection() {
        let mut s = SessionSelection::new();
        s.bind(7, 8, 8);
        assert!(s.feather(2.0).is_err());
        s.apply_shape(
            SelectionCoverage::rasterize_rect(8, 8, 2.0, 2.0, 4.0, 4.0),
            CombineMode::Replace,
        )
        .unwrap();
        s.feather(1.0).unwrap();
        let cov = s.coverage().unwrap();
        let edge = cov.coverage_at(1, 4);
        assert!(edge > 0 && edge < 255, "feathered edge is soft ({edge})");
    }
}
