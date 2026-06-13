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

//! Crop + straighten GEOMETRY — pure view math for the on-canvas crop
//! affordance (8 handles + move + a straighten angle + aspect-ratio lock
//! presets). NO pixels, NO GPU: this is the deterministic geometry the
//! glue's crop interaction machine drives (handle hit-testing, the new
//! rectangle from a drag, the angle, and the integer pixel [`Region`] a
//! commit cuts). It lives WITH the engine because the math is reusable
//! and worth a Rust property/unit test; the TS stays thin (it forwards
//! pointer points and renders the frame the host overlay draws).
//!
//! All coordinates are in IMAGE-PIXEL space (`f32`), origin top-left, +x
//! right / +y down — the same space the engine-held buffer and the
//! [`Region`] commit speak. The straighten angle is the clockwise frame
//! rotation in DEGREES about the crop-rect centre; a commit's pixel
//! [`Region`] is the angle-0 axis-aligned cut (rotation of the content is
//! a separate resample stage the caller composes — this module owns the
//! frame geometry + the clamped integer region, not the resampler).
//!
//! Standard direct-manipulation geometry (no reference reading).

use crate::region::Region;

/// A crop rectangle in image-pixel space (top-left origin). `w`/`h` are
/// kept non-negative by the constructors and the resize math (a drag that
/// crosses an edge is normalized, never inverted).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl CropRect {
    /// Construct from a corner + size, normalizing a negative extent (the
    /// rect is stored top-left origin with non-negative `w`/`h`).
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        let (x, w) = if w < 0.0 { (x + w, -w) } else { (x, w) };
        let (y, h) = if h < 0.0 { (y + h, -h) } else { (y, h) };
        CropRect { x, y, w, h }
    }

    /// The full-image crop (the identity / reset rect for a `w`×`h` image).
    pub fn full(image_w: u32, image_h: u32) -> Self {
        CropRect {
            x: 0.0,
            y: 0.0,
            w: image_w as f32,
            h: image_h as f32,
        }
    }

    /// Exclusive right edge (`x + w`).
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// Exclusive bottom edge (`y + h`).
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    /// The eight handle anchor points, in [`Handle`] discriminant order
    /// (TL, T, TR, R, BR, B, BL, L). The panel/overlay places its grab
    /// dots here; [`hit_handle`](Self::hit_handle) inverts the mapping.
    pub fn handle_points(&self) -> [(f32, f32); 8] {
        let (l, t, r, b) = (self.x, self.y, self.right(), self.bottom());
        let (cx, cy) = self.center();
        [
            (l, t),  // TopLeft
            (cx, t), // Top
            (r, t),  // TopRight
            (r, cy), // Right
            (r, b),  // BottomRight
            (cx, b), // Bottom
            (l, b),  // BottomLeft
            (l, cy), // Left
        ]
    }

    /// Clamp the rect to the `image_w`×`image_h` extent (a move/resize can
    /// never leave the image). Returns the intersected rect; a fully-out
    /// rect collapses to a zero-size rect at the nearest in-bounds corner.
    pub fn clamp_to_image(&self, image_w: u32, image_h: u32) -> CropRect {
        let iw = image_w as f32;
        let ih = image_h as f32;
        let x0 = self.x.clamp(0.0, iw);
        let y0 = self.y.clamp(0.0, ih);
        let x1 = self.right().clamp(0.0, iw);
        let y1 = self.bottom().clamp(0.0, ih);
        CropRect {
            x: x0,
            y: y0,
            w: (x1 - x0).max(0.0),
            h: (y1 - y0).max(0.0),
        }
    }

    /// The integer pixel [`Region`] a commit cuts: the rect rounded to the
    /// nearest pixel and clamped to the image extent. `None` when the
    /// region is empty (nothing to crop). The angle-0 axis-aligned cut;
    /// straighten rotation is the caller's resample stage.
    pub fn to_region(&self, image_w: u32, image_h: u32) -> Option<Region> {
        let c = self.clamp_to_image(image_w, image_h);
        let x = c.x.round().clamp(0.0, image_w as f32) as i32;
        let y = c.y.round().clamp(0.0, image_h as f32) as i32;
        let r = c.right().round().clamp(0.0, image_w as f32) as i32;
        let b = c.bottom().round().clamp(0.0, image_h as f32) as i32;
        if r <= x || b <= y {
            return None;
        }
        Some(Region::new(x, y, (r - x) as u32, (b - y) as u32))
    }
}

/// The 8 resize grips + the body MOVE target. Discriminant order matches
/// [`CropRect::handle_points`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handle {
    TopLeft = 0,
    Top = 1,
    TopRight = 2,
    Right = 3,
    BottomRight = 4,
    Bottom = 5,
    BottomLeft = 6,
    Left = 7,
    /// The rect body — a drag here MOVES the whole crop frame.
    Move = 8,
}

impl Handle {
    /// Does this handle move the left edge?
    fn moves_left(self) -> bool {
        matches!(self, Handle::TopLeft | Handle::Left | Handle::BottomLeft)
    }
    fn moves_right(self) -> bool {
        matches!(self, Handle::TopRight | Handle::Right | Handle::BottomRight)
    }
    fn moves_top(self) -> bool {
        matches!(self, Handle::TopLeft | Handle::Top | Handle::TopRight)
    }
    fn moves_bottom(self) -> bool {
        matches!(
            self,
            Handle::BottomLeft | Handle::Bottom | Handle::BottomRight
        )
    }
}

/// Aspect-ratio lock presets. `Ratio(w, h)` constrains a resize to `w:h`;
/// `Original` is the placed image's own ratio (the caller passes it as a
/// `Ratio`); `Free` is unconstrained; `Square` is `1:1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AspectLock {
    Free,
    Square,
    /// `w:h` — a positive ratio (`w/h`); a non-positive component falls
    /// back to free (defensive).
    Ratio(f32, f32),
}

impl AspectLock {
    /// The locked `w/h` ratio, or `None` for [`AspectLock::Free`] (and for
    /// a degenerate ratio, which is treated as free).
    pub fn ratio(self) -> Option<f32> {
        match self {
            AspectLock::Free => None,
            AspectLock::Square => Some(1.0),
            AspectLock::Ratio(w, h) if w > 0.0 && h > 0.0 => Some(w / h),
            AspectLock::Ratio(..) => None,
        }
    }
}

/// Hit-test the crop chrome at `point` (image-px): the nearest grab handle
/// within `tol` pixels, else [`Handle::Move`] when the point is inside the
/// rect body, else `None` (outside — a drag there starts a fresh marquee).
/// `tol` is the grab radius the caller derives from the screen tolerance
/// (`host.viewport.pxToPt`), so the grip stays a constant screen size.
pub fn hit_handle(rect: &CropRect, point: (f32, f32), tol: f32) -> Option<Handle> {
    let tol2 = tol * tol;
    let mut best: Option<(Handle, f32)> = None;
    for (i, (hx, hy)) in rect.handle_points().iter().enumerate() {
        let dx = point.0 - hx;
        let dy = point.1 - hy;
        let d2 = dx * dx + dy * dy;
        if d2 <= tol2 && best.map(|(_, bd)| d2 < bd).unwrap_or(true) {
            best = Some((HANDLE_BY_INDEX[i], d2));
        }
    }
    if let Some((h, _)) = best {
        return Some(h);
    }
    // Inside the body (with the same tolerance margin) → move.
    if point.0 >= rect.x - tol
        && point.0 <= rect.right() + tol
        && point.1 >= rect.y - tol
        && point.1 <= rect.bottom() + tol
    {
        return Some(Handle::Move);
    }
    None
}

/// Index → [`Handle`] in [`CropRect::handle_points`] order.
const HANDLE_BY_INDEX: [Handle; 8] = [
    Handle::TopLeft,
    Handle::Top,
    Handle::TopRight,
    Handle::Right,
    Handle::BottomRight,
    Handle::Bottom,
    Handle::BottomLeft,
    Handle::Left,
];

/// Smallest crop extent (px) a resize will produce — prevents a degenerate
/// zero/inverted rect when a grip is dragged past the opposite edge.
const MIN_EXTENT: f32 = 1.0;

/// Apply a pointer drag from `start` to `point` (both image-px) to `rect`
/// at `handle`, with the `aspect` lock and image-extent clamp. Returns the
/// new rect. [`Handle::Move`] translates; an edge/corner grip resizes
/// about the OPPOSITE anchor; a non-free aspect locks `w/h` (corner grips
/// drive the dominant axis; edge grips drive their own axis and derive the
/// other). The rect never inverts (a grip dragged past its anchor is
/// clamped to [`MIN_EXTENT`]) and is finally clamped to the image.
pub fn apply_drag(
    rect: &CropRect,
    handle: Handle,
    start: (f32, f32),
    point: (f32, f32),
    aspect: AspectLock,
    image_w: u32,
    image_h: u32,
) -> CropRect {
    let dx = point.0 - start.0;
    let dy = point.1 - start.1;

    if handle == Handle::Move {
        let moved = CropRect {
            x: rect.x + dx,
            y: rect.y + dy,
            w: rect.w,
            h: rect.h,
        };
        return clamp_move(&moved, image_w, image_h);
    }

    // Resize: hold the OPPOSITE edges fixed, move the grabbed ones.
    let mut l = rect.x;
    let mut t = rect.y;
    let mut r = rect.right();
    let mut b = rect.bottom();
    if handle.moves_left() {
        l = (rect.x + dx).min(r - MIN_EXTENT);
    }
    if handle.moves_right() {
        r = (rect.right() + dx).max(l + MIN_EXTENT);
    }
    if handle.moves_top() {
        t = (rect.y + dy).min(b - MIN_EXTENT);
    }
    if handle.moves_bottom() {
        b = (rect.bottom() + dy).max(t + MIN_EXTENT);
    }

    let out = CropRect {
        x: l,
        y: t,
        w: r - l,
        h: b - t,
    };
    match aspect.ratio() {
        // Aspect-locked: lock the ratio, then fit RATIO-PRESERVINGLY into
        // the image (a plain per-axis clamp would clip one side and break
        // the ratio). The anchor (the fixed corner) stays put.
        Some(ratio) => fit_aspect_to_image(
            &lock_aspect(&out, handle, ratio),
            handle,
            ratio,
            image_w,
            image_h,
        ),
        // Free: a plain per-axis image clamp is correct.
        None => out.clamp_to_image(image_w, image_h),
    }
}

/// Fit an aspect-LOCKED rect inside the image WITHOUT breaking its ratio:
/// when the rect exceeds the image on either axis, shrink both axes by the
/// same factor (so `w/h` is preserved) and re-anchor at the resize
/// handle's fixed corner. Then snap any out-of-bounds origin back in.
fn fit_aspect_to_image(
    rect: &CropRect,
    handle: Handle,
    ratio: f32,
    image_w: u32,
    image_h: u32,
) -> CropRect {
    let iw = image_w as f32;
    let ih = image_h as f32;
    // Largest ratio-true (w, h) that fits the image.
    let max_w = iw.min(ih * ratio);
    let max_h = max_w / ratio;
    let mut w = rect.w.min(max_w).max(MIN_EXTENT);
    let mut h = rect.h.min(max_h).max(MIN_EXTENT);
    // Re-tie w/h after the per-axis min so the ratio is exact.
    if (w / h - ratio).abs() > 1e-4 {
        if w / ratio <= max_h {
            h = w / ratio;
        } else {
            w = h * ratio;
        }
    }
    // Anchor at the handle's fixed corner, then nudge fully in-bounds.
    let (anchor_x, anchor_y) = anchor_of(rect, handle);
    let mut x = if anchor_x <= rect.x {
        anchor_x
    } else {
        anchor_x - w
    };
    let mut y = if anchor_y <= rect.y {
        anchor_y
    } else {
        anchor_y - h
    };
    x = x.clamp(0.0, (iw - w).max(0.0));
    y = y.clamp(0.0, (ih - h).max(0.0));
    CropRect { x, y, w, h }
}

/// Move-clamp: translate the rect fully back inside the image (preserving
/// size where it fits; the larger-than-image axis pins to 0).
fn clamp_move(rect: &CropRect, image_w: u32, image_h: u32) -> CropRect {
    let iw = image_w as f32;
    let ih = image_h as f32;
    let x = if rect.w >= iw {
        0.0
    } else {
        rect.x.clamp(0.0, iw - rect.w)
    };
    let y = if rect.h >= ih {
        0.0
    } else {
        rect.y.clamp(0.0, ih - rect.h)
    };
    CropRect {
        x,
        y,
        w: rect.w.min(iw),
        h: rect.h.min(ih),
    }
}

/// Re-impose `ratio = w/h` on a resized rect, anchored so the moved
/// corner/edge stays put. Corner grips pick the axis with the larger
/// change as the driver; edge grips drive their own axis. The anchor is
/// the fixed (opposite) corner of the grabbed handle.
fn lock_aspect(rect: &CropRect, handle: Handle, ratio: f32) -> CropRect {
    let (anchor_x, anchor_y) = anchor_of(rect, handle);

    // Decide the driving axis. Horizontal-only edges (Left/Right) drive w;
    // vertical-only edges (Top/Bottom) drive h; corners pick the larger.
    let drive_w = match handle {
        Handle::Left | Handle::Right => true,
        Handle::Top | Handle::Bottom => false,
        _ => rect.w >= rect.h * ratio, // corner: keep the bigger move
    };

    let (mut w, mut h) = if drive_w {
        (rect.w, rect.w / ratio)
    } else {
        (rect.h * ratio, rect.h)
    };
    w = w.max(MIN_EXTENT);
    h = h.max(MIN_EXTENT);

    // Grow away from the anchor (which is the side NOT moving).
    let x = if anchor_x <= rect.x {
        anchor_x
    } else {
        anchor_x - w
    };
    let y = if anchor_y <= rect.y {
        anchor_y
    } else {
        anchor_y - h
    };
    CropRect { x, y, w, h }
}

/// The fixed anchor corner for a resize handle: the corner diagonally
/// OPPOSITE a corner grip; the opposite edge midpoint reference for an
/// edge grip (returned as the relevant fixed corner for aspect growth).
fn anchor_of(rect: &CropRect, handle: Handle) -> (f32, f32) {
    let (l, t, r, b) = (rect.x, rect.y, rect.right(), rect.bottom());
    match handle {
        Handle::TopLeft => (r, b),
        Handle::TopRight => (l, b),
        Handle::BottomRight => (l, t),
        Handle::BottomLeft => (r, t),
        Handle::Top => (l, b),
        Handle::Bottom => (l, t),
        Handle::Left => (r, t),
        Handle::Right => (l, t),
        Handle::Move => (l, t),
    }
}

/// Normalize a straighten angle (degrees) to the canonical `(-180, 180]`
/// range. The crop frame rotates clockwise by this amount about its
/// centre; the panel's straighten slider lives in a small `[-45, 45]`
/// window but the math is general.
pub fn normalize_angle(degrees: f32) -> f32 {
    let mut a = degrees % 360.0;
    if a > 180.0 {
        a -= 360.0;
    } else if a <= -180.0 {
        a += 360.0;
    }
    a
}

/// Rotate `point` about `center` by `degrees` CLOCKWISE (image-space +y
/// down, so a positive angle turns the +x axis toward +y). Used to map the
/// straightened crop FRAME corners into image space for the overlay
/// polyline and to test the angle math; the actual content rotation is a
/// resample stage the caller composes.
pub fn rotate_point(point: (f32, f32), center: (f32, f32), degrees: f32) -> (f32, f32) {
    let rad = degrees.to_radians();
    let (s, c) = rad.sin_cos();
    let px = point.0 - center.0;
    let py = point.1 - center.1;
    (center.0 + px * c - py * s, center.1 + px * s + py * c)
}

/// The four corners of the crop FRAME, rotated by the straighten `degrees`
/// about the rect centre (TL, TR, BR, BL order — a closed polyline the
/// host overlay draws). Pixel-space; the overlay caller maps them to
/// page-local pt.
pub fn frame_corners(rect: &CropRect, degrees: f32) -> [(f32, f32); 4] {
    let center = rect.center();
    let (l, t, r, b) = (rect.x, rect.y, rect.right(), rect.bottom());
    let corners = [(l, t), (r, t), (r, b), (l, b)];
    [
        rotate_point(corners[0], center, degrees),
        rotate_point(corners[1], center, degrees),
        rotate_point(corners[2], center, degrees),
        rotate_point(corners[3], center, degrees),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // feat: image.editor.crop — pure crop + straighten geometry. The
    // naming carries the feature tag until the state feature_test macro
    // ships (CLAUDE.md test-tag convention).

    #[test]
    fn image_editor_crop_new_normalizes_negative_extent() {
        // A drag up-left produces a negative w/h; new() stores top-left.
        let r = CropRect::new(50.0, 40.0, -30.0, -20.0);
        assert_eq!(r, CropRect::new(20.0, 20.0, 30.0, 20.0));
    }

    #[test]
    fn image_editor_crop_full_is_image_extent() {
        let r = CropRect::full(800, 600);
        assert_eq!((r.x, r.y, r.w, r.h), (0.0, 0.0, 800.0, 600.0));
    }

    #[test]
    fn image_editor_crop_handle_points_in_order() {
        let r = CropRect::new(0.0, 0.0, 100.0, 80.0);
        let p = r.handle_points();
        assert_eq!(p[Handle::TopLeft as usize], (0.0, 0.0));
        assert_eq!(p[Handle::TopRight as usize], (100.0, 0.0));
        assert_eq!(p[Handle::BottomRight as usize], (100.0, 80.0));
        assert_eq!(p[Handle::Top as usize], (50.0, 0.0));
        assert_eq!(p[Handle::Left as usize], (0.0, 40.0));
    }

    #[test]
    fn image_editor_crop_hit_handle_grabs_nearest_grip() {
        let r = CropRect::new(0.0, 0.0, 100.0, 80.0);
        // Near the top-right corner within tolerance.
        assert_eq!(hit_handle(&r, (98.0, 2.0), 6.0), Some(Handle::TopRight));
        // Mid-body → move.
        assert_eq!(hit_handle(&r, (50.0, 40.0), 6.0), Some(Handle::Move));
        // Far outside → nothing.
        assert_eq!(hit_handle(&r, (300.0, 300.0), 6.0), None);
    }

    #[test]
    fn image_editor_crop_move_drag_translates_and_clamps() {
        let r = CropRect::new(10.0, 10.0, 40.0, 40.0);
        // Drag right+down by (20, 20).
        let moved = apply_drag(
            &r,
            Handle::Move,
            (0.0, 0.0),
            (20.0, 20.0),
            AspectLock::Free,
            100,
            100,
        );
        assert_eq!(moved, CropRect::new(30.0, 30.0, 40.0, 40.0));
        // Drag far right: pinned to the right image edge (x = 100-40).
        let pinned = apply_drag(
            &r,
            Handle::Move,
            (0.0, 0.0),
            (500.0, 0.0),
            AspectLock::Free,
            100,
            100,
        );
        assert_eq!(pinned.x, 60.0);
        assert_eq!(pinned.w, 40.0);
    }

    #[test]
    fn image_editor_crop_resize_corner_holds_opposite_anchor() {
        let r = CropRect::new(20.0, 20.0, 60.0, 60.0); // BR = (80,80)
                                                       // Drag TopLeft in by (10,10): TL→(30,30), BR stays (80,80).
        let out = apply_drag(
            &r,
            Handle::TopLeft,
            (20.0, 20.0),
            (30.0, 30.0),
            AspectLock::Free,
            200,
            200,
        );
        assert_eq!(out.x, 30.0);
        assert_eq!(out.y, 30.0);
        assert_eq!(out.right(), 80.0);
        assert_eq!(out.bottom(), 80.0);
    }

    #[test]
    fn image_editor_crop_resize_never_inverts() {
        let r = CropRect::new(0.0, 0.0, 50.0, 50.0);
        // Drag the right edge far past the left edge: clamps to MIN_EXTENT.
        let out = apply_drag(
            &r,
            Handle::Right,
            (50.0, 25.0),
            (-200.0, 25.0),
            AspectLock::Free,
            100,
            100,
        );
        assert!(out.w >= MIN_EXTENT, "w never inverts: {}", out.w);
        assert_eq!(out.w, MIN_EXTENT);
    }

    #[test]
    fn image_editor_crop_square_aspect_locks_ratio() {
        let r = CropRect::new(0.0, 0.0, 40.0, 40.0);
        // Drag BottomRight to make it wide; 1:1 forces h == w.
        let out = apply_drag(
            &r,
            Handle::BottomRight,
            (40.0, 40.0),
            (140.0, 60.0),
            AspectLock::Square,
            400,
            400,
        );
        assert!(
            (out.w - out.h).abs() < 1e-3,
            "square: w==h, got {}x{}",
            out.w,
            out.h
        );
    }

    #[test]
    fn image_editor_crop_aspect_fits_image_preserving_ratio() {
        // A 2×1 image with a 1:1 lock: the square must FIT (1×1), ratio
        // preserved — not a per-axis clip that would leave 2×1.
        let r = CropRect::full(2, 1);
        let out = apply_drag(
            &r,
            Handle::BottomRight,
            (2.0, 1.0),
            (2.0, 1.0),
            AspectLock::Square,
            2,
            1,
        );
        assert!(
            (out.w - out.h).abs() < 1e-3,
            "ratio preserved: {}x{}",
            out.w,
            out.h
        );
        assert!(out.w <= 2.0 + 1e-3 && out.h <= 1.0 + 1e-3, "fits the image");
        assert!((out.w - 1.0).abs() < 1e-3, "largest 1:1 in 2×1 is 1×1");
    }

    #[test]
    fn image_editor_crop_ratio_lock_holds_w_over_h() {
        let r = CropRect::new(0.0, 0.0, 30.0, 30.0);
        // 16:9 on a width-driving edge grip.
        let out = apply_drag(
            &r,
            Handle::Right,
            (30.0, 15.0),
            (130.0, 15.0),
            AspectLock::Ratio(16.0, 9.0),
            600,
            600,
        );
        let ratio = out.w / out.h;
        assert!(
            (ratio - 16.0 / 9.0).abs() < 1e-2,
            "ratio held ~16:9, got {ratio}"
        );
    }

    #[test]
    fn image_editor_crop_to_region_rounds_and_clamps() {
        let r = CropRect::new(10.4, 20.6, 30.2, 40.9);
        // x→10, y→21, right=40.6→41, bottom=61.5→62 → 31×41.
        let region = r.to_region(100, 100).expect("non-empty");
        assert_eq!(region.x, 10);
        assert_eq!(region.y, 21);
        assert_eq!(region.w, 31);
        assert_eq!(region.h, 41);
    }

    #[test]
    fn image_editor_crop_to_region_empty_is_none() {
        // A rect fully off the image → None.
        let r = CropRect::new(200.0, 200.0, 50.0, 50.0);
        assert_eq!(r.to_region(100, 100), None);
        // Zero-size → None.
        assert_eq!(
            CropRect::new(10.0, 10.0, 0.0, 0.0).to_region(100, 100),
            None
        );
    }

    #[test]
    fn image_editor_crop_clamp_to_image_intersects() {
        let r = CropRect::new(-10.0, -10.0, 50.0, 50.0);
        let c = r.clamp_to_image(100, 100);
        assert_eq!((c.x, c.y), (0.0, 0.0));
        assert_eq!((c.right(), c.bottom()), (40.0, 40.0));
    }

    #[test]
    fn image_editor_crop_normalize_angle_wraps() {
        assert_eq!(normalize_angle(0.0), 0.0);
        assert_eq!(normalize_angle(190.0), -170.0);
        assert_eq!(normalize_angle(-190.0), 170.0);
        assert_eq!(normalize_angle(45.0), 45.0);
        assert_eq!(normalize_angle(360.0), 0.0);
    }

    #[test]
    fn image_editor_crop_rotate_point_90_cw() {
        // +y-down clockwise: (1,0) about origin by 90° → (0,1).
        let (x, y) = rotate_point((1.0, 0.0), (0.0, 0.0), 90.0);
        assert!((x - 0.0).abs() < 1e-5, "x≈0, got {x}");
        assert!((y - 1.0).abs() < 1e-5, "y≈1, got {y}");
    }

    #[test]
    fn image_editor_crop_frame_corners_identity_at_zero() {
        let r = CropRect::new(10.0, 20.0, 100.0, 60.0);
        let c = frame_corners(&r, 0.0);
        assert_eq!(c[0], (10.0, 20.0)); // TL
        assert_eq!(c[2], (110.0, 80.0)); // BR
    }

    #[test]
    fn image_editor_crop_frame_corners_rotate_about_center() {
        let r = CropRect::new(0.0, 0.0, 100.0, 100.0); // center (50,50)
        let c = frame_corners(&r, 90.0);
        // A 90° CW rotation about center keeps the corner SET (the square
        // is symmetric): TL→ where TR was, etc. Centroid is preserved.
        let cx: f32 = c.iter().map(|p| p.0).sum::<f32>() / 4.0;
        let cy: f32 = c.iter().map(|p| p.1).sum::<f32>() / 4.0;
        assert!((cx - 50.0).abs() < 1e-4 && (cy - 50.0).abs() < 1e-4);
    }
}
