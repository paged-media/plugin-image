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

//! A layer MASK at its bounds (ADR 464): either canvas-sized coverage, or
//! a rectangle of coverage with one value everywhere else — a PSD mask
//! record is exactly that (its rectangle and its default colour). A
//! layered import keeps masks this way, so a mock-up's masks cost their
//! rectangles, not one canvas each.
//!
//! The fold reads a mask texel by texel ([`LayerMask::at`]) and keys its
//! GPU copies by identity ([`LayerMask::same`]), so a bounded mask never
//! has to be expanded to be drawn. Paths that need canvas coverage (the
//! adjust chain, a product with another mask, an edit) ask for
//! [`LayerMask::canvas`].

use std::sync::Arc;

use image_core::Region;
use image_gpu::coverage::SelectionCoverage;

/// A layer's mask (see the module docs).
#[derive(Clone, Debug)]
pub enum LayerMask {
    Canvas(Arc<SelectionCoverage>),
    Bounded(Arc<BoundedMask>),
}

/// Coverage over `rect`, `outside` everywhere else, on a `w × h` canvas.
#[derive(Debug)]
pub struct BoundedMask {
    pub rect: Region,
    pub outside: u8,
    pub data: Vec<u8>,
    pub canvas_w: u32,
    pub canvas_h: u32,
    expanded: std::sync::OnceLock<Arc<SelectionCoverage>>,
}

impl BoundedMask {
    /// `rect` must lie inside the canvas and `data` cover it.
    pub fn new(rect: Region, outside: u8, data: Vec<u8>, w: u32, h: u32) -> Option<Self> {
        let inside = rect.x >= 0
            && rect.y >= 0
            && rect.x as u64 + u64::from(rect.w) <= u64::from(w)
            && rect.y as u64 + u64::from(rect.h) <= u64::from(h);
        (inside && data.len() == rect.w as usize * rect.h as usize).then(|| BoundedMask {
            rect,
            outside,
            data,
            canvas_w: w,
            canvas_h: h,
            expanded: std::sync::OnceLock::new(),
        })
    }

    pub fn at(&self, x: u32, y: u32) -> u8 {
        let r = self.rect;
        let (rx, ry) = (x as i64 - i64::from(r.x), y as i64 - i64::from(r.y));
        if rx < 0 || ry < 0 || rx >= i64::from(r.w) || ry >= i64::from(r.h) {
            return self.outside;
        }
        self.data[(ry * i64::from(r.w) + rx) as usize]
    }

    fn expand(&self) -> SelectionCoverage {
        let (w, h) = (self.canvas_w, self.canvas_h);
        let mut out = vec![self.outside; w as usize * h as usize];
        let r = self.rect;
        for y in 0..r.h as usize {
            let d = (r.y as usize + y) * w as usize + r.x as usize;
            out[d..d + r.w as usize]
                .copy_from_slice(&self.data[y * r.w as usize..(y + 1) * r.w as usize]);
        }
        SelectionCoverage::from_data(w, h, out).expect("canvas-sized")
    }
}

impl From<Arc<SelectionCoverage>> for LayerMask {
    fn from(c: Arc<SelectionCoverage>) -> Self {
        LayerMask::Canvas(c)
    }
}

impl LayerMask {
    /// The coverage at canvas texel (x, y).
    pub fn at(&self, x: u32, y: u32) -> u8 {
        match self {
            LayerMask::Canvas(c) => c.coverage_at(x, y),
            LayerMask::Bounded(b) => b.at(x, y),
        }
    }

    /// Canvas-sized coverage; a bounded mask is expanded once and kept.
    pub fn canvas(&self) -> &Arc<SelectionCoverage> {
        match self {
            LayerMask::Canvas(c) => c,
            LayerMask::Bounded(b) => b.expanded.get_or_init(|| Arc::new(b.expand())),
        }
    }

    /// Canvas-sized coverage WITHOUT keeping an expansion.
    pub fn to_canvas(&self) -> Arc<SelectionCoverage> {
        match self {
            LayerMask::Canvas(c) => Arc::clone(c),
            LayerMask::Bounded(b) => match b.expanded.get() {
                Some(c) => Arc::clone(c),
                None => Arc::new(b.expand()),
            },
        }
    }

    /// Is it the same mask (by identity — what the fold's caches key on)?
    pub fn same(&self, o: &LayerMask) -> bool {
        match (self, o) {
            (LayerMask::Canvas(a), LayerMask::Canvas(b)) => Arc::ptr_eq(a, b),
            (LayerMask::Bounded(a), LayerMask::Bounded(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Full coverage everywhere — a mask that changes nothing.
    pub fn is_all_one(&self) -> bool {
        match self {
            LayerMask::Canvas(c) => c.is_all_one(),
            LayerMask::Bounded(b) => {
                (b.outside == 255 || b.rect == Region::new(0, 0, b.canvas_w, b.canvas_h))
                    && b.data.iter().all(|&v| v == 255)
            }
        }
    }

    pub fn bounded(&self) -> Option<&BoundedMask> {
        match self {
            LayerMask::Bounded(b) => Some(b),
            LayerMask::Canvas(_) => None,
        }
    }

    /// Bytes the mask occupies as stored.
    pub fn stored_bytes(&self) -> usize {
        match self {
            LayerMask::Canvas(c) => c.data().len(),
            LayerMask::Bounded(b) => b.data.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bounded_mask_reads_and_expands_like_its_canvas_twin() {
        let b = BoundedMask::new(Region::new(1, 1, 2, 1), 0, vec![100, 200], 4, 3).expect("fits");
        let m = LayerMask::Bounded(Arc::new(b));
        assert_eq!(m.at(0, 0), 0);
        assert_eq!(m.at(1, 1), 100);
        assert_eq!(m.at(2, 1), 200);
        let c = m.to_canvas();
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(c.coverage_at(x, y), m.at(x, y));
            }
        }
        assert!(!m.is_all_one());
        assert!(
            Arc::ptr_eq(m.canvas(), m.clone().canvas()),
            "one cached expansion"
        );
    }
}
