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

//! `Region` (spec §5.4). All engine traffic is in regions (request ROI
//! / validity); kernels never see whole images.

use crate::tile::{TileCoord, TILE};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Region {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl Region {
    pub const fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
        Region { x, y, w, h }
    }

    pub const fn is_empty(self) -> bool {
        self.w == 0 || self.h == 0
    }

    /// Exclusive right edge. i64 to avoid overflow at i32::MAX origins.
    pub const fn right(self) -> i64 {
        self.x as i64 + self.w as i64
    }

    /// Exclusive bottom edge.
    pub const fn bottom(self) -> i64 {
        self.y as i64 + self.h as i64
    }

    pub fn intersect(self, o: Region) -> Option<Region> {
        let x = self.x.max(o.x);
        let y = self.y.max(o.y);
        let r = self.right().min(o.right());
        let b = self.bottom().min(o.bottom());
        if (x as i64) < r && (y as i64) < b {
            Some(Region {
                x,
                y,
                w: (r - x as i64) as u32,
                h: (b - y as i64) as u32,
            })
        } else {
            None
        }
    }

    /// Smallest region covering both. Empty operands are identity.
    pub fn union(self, o: Region) -> Region {
        if self.is_empty() {
            return o;
        }
        if o.is_empty() {
            return self;
        }
        let x = self.x.min(o.x);
        let y = self.y.min(o.y);
        let r = self.right().max(o.right());
        let b = self.bottom().max(o.bottom());
        Region {
            x,
            y,
            w: (r - x as i64) as u32,
            h: (b - y as i64) as u32,
        }
    }

    /// Window expansion for `KernelClass::Windowed` ROI propagation
    /// (spec §7.1): grow by the kernel radius on each side.
    pub fn expand_by(self, rx: u16, ry: u16) -> Region {
        Region {
            x: self.x.saturating_sub(rx as i32),
            y: self.y.saturating_sub(ry as i32),
            w: self.w.saturating_add(2 * rx as u32),
            h: self.h.saturating_add(2 * ry as u32),
        }
    }

    /// Tile coordinates covering this region, interpreted in the pixel
    /// space of `level` (mip-aware: the caller scales the region to the
    /// level's space first; this function does NOT rescale).
    pub fn tiles_at(self, level: u8) -> impl Iterator<Item = TileCoord> {
        let t = TILE as i64;
        let (x0, y0) = ((self.x as i64).div_euclid(t), (self.y as i64).div_euclid(t));
        // Exclusive end tile indices; empty regions yield nothing.
        let (x1, y1) = if self.is_empty() {
            (x0, y0)
        } else {
            (
                (self.right() - 1).div_euclid(t) + 1,
                (self.bottom() - 1).div_euclid(t) + 1,
            )
        };
        (y0..y1).flat_map(move |ty| {
            (x0..x1).map(move |tx| TileCoord {
                level,
                x: tx as i32,
                y: ty as i32,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_disjoint_is_none() {
        let a = Region::new(0, 0, 10, 10);
        let b = Region::new(10, 0, 5, 5); // edge-adjacent: empty
        assert_eq!(a.intersect(b), None);
    }

    #[test]
    fn intersect_overlap() {
        let a = Region::new(0, 0, 10, 10);
        let b = Region::new(5, 5, 10, 10);
        assert_eq!(a.intersect(b), Some(Region::new(5, 5, 5, 5)));
    }

    #[test]
    fn union_covers_both() {
        let a = Region::new(-5, -5, 5, 5);
        let b = Region::new(5, 5, 5, 5);
        assert_eq!(a.union(b), Region::new(-5, -5, 15, 15));
    }

    #[test]
    fn expand_negative_origin() {
        let r = Region::new(0, 0, 4, 4).expand_by(3, 1);
        assert_eq!(r, Region::new(-3, -1, 10, 6));
    }

    #[test]
    fn tiles_cover_region_across_origin() {
        // 256-tile grid: a region straddling the origin touches 4 tiles.
        let r = Region::new(-1, -1, 2, 2);
        let tiles: Vec<_> = r.tiles_at(0).collect();
        assert_eq!(tiles.len(), 4);
        assert!(tiles.contains(&TileCoord {
            level: 0,
            x: -1,
            y: -1
        }));
        assert!(tiles.contains(&TileCoord {
            level: 0,
            x: 0,
            y: 0
        }));
    }

    #[test]
    fn tiles_single() {
        let r = Region::new(0, 0, 256, 256);
        assert_eq!(r.tiles_at(2).count(), 1);
        let r = Region::new(0, 0, 257, 256);
        assert_eq!(r.tiles_at(0).count(), 2);
    }

    #[test]
    fn tiles_empty_region_none() {
        assert_eq!(Region::new(5, 5, 0, 10).tiles_at(0).count(), 0);
    }
}
