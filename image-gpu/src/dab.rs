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

//! DAB + STROKE coverage — the painting half of the mask model.
//!
//! A brush stroke is, at bottom, a coverage field: a round tip with a
//! hardness falloff ([`BrushTip`]) stamped at interpolated positions
//! along the pointer path ([`plan_segment`]) into an accumulation buffer
//! ([`StrokeAccumulator`]). The accumulated coverage is then the ABI's
//! `@group(2)` mask for ONE composite dispatch (`crate::stroke`), so a
//! dab is composited by exactly the same `out = mix(a, result, mask)`
//! rule that `coverage.rs` uses for selections.
//!
//! HONESTY NOTE (GPU-only constitution, spec §6) — identical in kind to
//! the note at the head of [`crate::coverage`]: everything in this module
//! is mask *preparation*. No image pixel is read or written here; the
//! output is a coverage field. Its CONSUMPTION is GPU-only — every
//! painted texel is produced by a registered WGSL kernel dispatch in
//! [`crate::stroke`]. A GPU-side dab rasterizer (stamping into an
//! r16float accumulation texture, so the coverage never leaves the
//! device) is the natural Stage-B follow-up; it would replace this
//! module's internals without changing [`StrokeAccumulator`]'s contract.
//!
//! # The model, and why
//!
//! * **Tip.** Round, with `hardness` naming the fraction of the radius
//!   that is fully opaque; the remainder is a `smoothstep` falloff to
//!   zero at the radius. `smoothstep` (not linear) because a linear ramp
//!   leaves a visible crease at the plateau join on overlapping dabs.
//! * **Antialiasing.** The falloff band is widened to at least
//!   [`AA_BAND_PX`] so that even `hardness == 1` antialiases (the band
//!   becomes a 1 px ramp just inside the rim). The PENCIL opts out
//!   (`antialias: false`) and takes binary coverage — a hard, aliased
//!   edge is the pencil's defining trait, not a defect.
//! * **Sub-pixel position.** Coverage is evaluated against pixel CENTRES
//!   (`x + 0.5`), so a dab centre moving by a third of a pixel changes
//!   the coverage field. Stroke interpolation therefore does not need to
//!   snap to the pixel grid.
//! * **Spacing.** Dabs are emitted every `spacing · diameter` px of ARC
//!   LENGTH, with the leftover distance carried across pointer samples
//!   ([`StrokeWalk`]) so a fast drag paints a continuous, evenly-spaced
//!   stroke instead of one dot per pointer event. Spacing is a fraction
//!   of the DIAMETER (the Photoshop convention; 0.25 default).
//! * **Determinism (spec §6.3).** Pure f32 arithmetic, no randomness, no
//!   time or event-rate dependence: the same sample list always produces
//!   the same coverage, which is what makes a stroke replayable from a
//!   recorded action or a script.

use image_core::Region;

use crate::coverage::SelectionCoverage;
use crate::selection::SelectionMask;

/// Minimum width (px) of a tip's antialiasing band. At `hardness == 1`
/// the plateau would otherwise reach the rim and the edge would alias;
/// widening the band to one pixel yields the "hard round" brush — a
/// crisp edge that is still antialiased.
pub const AA_BAND_PX: f32 = 1.0;

/// Floor on the dab spacing (px of arc length). Guards the interpolation
/// walk against a zero/absurd spacing request; 0.5 px is already denser
/// than any visible stroke needs.
pub const MIN_SPACING_PX: f32 = 0.5;

/// Cap on the dabs one pointer segment may emit. A pathological jump
/// (a pointer teleport across a large canvas with a tiny tip) is
/// DECIMATED rather than allowed to stall the frame — the walk stops
/// emitting, it never loops forever. At the 0.5 px floor this still
/// covers a 4096 px jump at full density.
pub const MAX_DABS_PER_SEGMENT: usize = 8192;

/// Which stroke property the pointer's pressure drives.
///
/// The default is [`PressureTarget::Both`] — the Photoshop "size +
/// opacity" pen preset: light pressure paints a thinner AND fainter
/// mark, which is what people expect a pen to do. `Size` alone gives a
/// calligraphic feel; `Opacity` alone an airbrush feel; `None` makes the
/// stroke pressure-independent (what a mouse wants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PressureTarget {
    /// Pressure is ignored — every dab is full size at full flow.
    None,
    /// Pressure scales the tip diameter only.
    Size,
    /// Pressure scales the per-dab flow only.
    Opacity,
    /// Pressure scales BOTH the diameter and the flow (the default).
    #[default]
    Both,
}

impl PressureTarget {
    /// Decode the wire name (mirrored by the TS `PressureTarget` union).
    pub fn from_wire(s: &str) -> Option<PressureTarget> {
        Some(match s {
            "none" => PressureTarget::None,
            "size" => PressureTarget::Size,
            "opacity" => PressureTarget::Opacity,
            "both" => PressureTarget::Both,
            _ => return None,
        })
    }

    pub fn as_wire(self) -> &'static str {
        match self {
            PressureTarget::None => "none",
            PressureTarget::Size => "size",
            PressureTarget::Opacity => "opacity",
            PressureTarget::Both => "both",
        }
    }

    /// The diameter multiplier for `pressure`.
    pub fn size_scale(self, pressure: f32) -> f32 {
        match self {
            PressureTarget::Size | PressureTarget::Both => pressure.clamp(0.0, 1.0),
            _ => 1.0,
        }
    }

    /// The flow multiplier for `pressure`.
    pub fn flow_scale(self, pressure: f32) -> f32 {
        match self {
            PressureTarget::Opacity | PressureTarget::Both => pressure.clamp(0.0, 1.0),
            _ => 1.0,
        }
    }
}

/// A round brush tip: diameter in IMAGE pixels, `hardness` in `[0, 1]`
/// (the fully-opaque fraction of the radius), and whether the rim is
/// antialiased.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrushTip {
    pub diameter: f32,
    pub hardness: f32,
    /// `false` = the PENCIL: binary coverage, hard aliased edge.
    pub antialias: bool,
}

impl BrushTip {
    /// A soft round tip (the brush).
    pub fn soft(diameter: f32, hardness: f32) -> Self {
        BrushTip {
            diameter,
            hardness,
            antialias: true,
        }
    }

    /// A hard, ALIASED tip (the pencil) — `hardness` is irrelevant.
    pub fn hard(diameter: f32) -> Self {
        BrushTip {
            diameter,
            hardness: 1.0,
            antialias: false,
        }
    }

    pub fn radius(&self) -> f32 {
        (self.diameter * 0.5).max(0.0)
    }

    /// Coverage in `[0, 1]` at distance `d` px from the dab centre.
    ///
    /// Antialiased: `1` inside `hardness · r`, `smoothstep` down to `0`
    /// at `r`, with the transition band widened to at least
    /// [`AA_BAND_PX`]. Aliased (pencil): `1` inside `r`, else `0`.
    ///
    /// A sub-pixel tip (`r < AA_BAND_PX`) never reaches full coverage
    /// even at its centre. That is deliberate and correct: a
    /// half-pixel-wide mark should not paint a whole pixel solid.
    pub fn coverage_at_distance(&self, d: f32) -> f32 {
        let r = self.radius();
        if r <= 0.0 || !d.is_finite() {
            return 0.0;
        }
        if !self.antialias {
            return if d <= r { 1.0 } else { 0.0 };
        }
        let inner = r * self.hardness.clamp(0.0, 1.0);
        let band = (r - inner).max(AA_BAND_PX);
        let t = ((r - d) / band).clamp(0.0, 1.0);
        // smoothstep — C¹ at both ends, so overlapping dabs join without
        // the crease a linear ramp leaves at the plateau.
        t * t * (3.0 - 2.0 * t)
    }

    /// Coverage of the pixel whose integer coords are `(px, py)` for a
    /// dab centred at `(cx, cy)` in image px. Evaluated at the pixel
    /// CENTRE — this is what makes the tip sub-pixel positionable.
    pub fn coverage_at_pixel(&self, cx: f32, cy: f32, px: u32, py: u32) -> f32 {
        let dx = (px as f32 + 0.5) - cx;
        let dy = (py as f32 + 0.5) - cy;
        self.coverage_at_distance((dx * dx + dy * dy).sqrt())
    }

    /// The dab's integer bounding box clipped to a `w`×`h` field, or
    /// `None` when it falls entirely outside. The box is the radius
    /// inflated by the AA band so the falloff's outer edge is included.
    pub fn dab_bounds(&self, cx: f32, cy: f32, w: u32, h: u32) -> Option<Region> {
        let r = self.radius();
        if r <= 0.0 || !cx.is_finite() || !cy.is_finite() || w == 0 || h == 0 {
            return None;
        }
        let reach = r + 1.0;
        let x0 = (cx - reach).floor().max(0.0) as i64;
        let y0 = (cy - reach).floor().max(0.0) as i64;
        let x1 = (cx + reach).ceil().min(w as f32) as i64;
        let y1 = (cy + reach).ceil().min(h as f32) as i64;
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some(Region::new(
            x0 as i32,
            y0 as i32,
            (x1 - x0) as u32,
            (y1 - y0) as u32,
        ))
    }
}

/// One pointer sample: position in IMAGE px plus normalized pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeSample {
    pub x: f32,
    pub y: f32,
    /// `0..1`. See [`PressureTarget`]; a mouse has no real pressure and
    /// the caller normalizes it to 1 (documented at the wasm door).
    pub pressure: f32,
}

impl StrokeSample {
    pub fn new(x: f32, y: f32, pressure: f32) -> Self {
        StrokeSample { x, y, pressure }
    }
}

/// One planned dab: an interpolated position + the pressure there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dab {
    pub x: f32,
    pub y: f32,
    pub pressure: f32,
}

/// The arc-length bookkeeping that makes spacing continuous ACROSS
/// pointer samples. `residual` is the distance already travelled since
/// the last emitted dab; carrying it is the difference between an evenly
/// spaced stroke and one that clumps at every pointer event.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrokeWalk {
    residual: f32,
}

impl StrokeWalk {
    pub fn new() -> Self {
        StrokeWalk::default()
    }

    /// Distance travelled since the last emitted dab.
    pub fn residual(&self) -> f32 {
        self.residual
    }
}

/// Plan the dabs for the segment `from → to` at `step` px of arc length,
/// appending them to `out` and advancing `walk`'s residual.
///
/// The first dab of a segment lands `step − residual` px in, so spacing
/// is measured along the WHOLE stroke, not restarted per segment.
/// Pressure interpolates linearly along the segment.
pub fn plan_segment(
    walk: &mut StrokeWalk,
    from: StrokeSample,
    to: StrokeSample,
    step: f32,
    out: &mut Vec<Dab>,
) {
    let step = if step.is_finite() {
        step.max(MIN_SPACING_PX)
    } else {
        MIN_SPACING_PX
    };
    let (dx, dy) = (to.x - from.x, to.y - from.y);
    let len = (dx * dx + dy * dy).sqrt();
    if !len.is_finite() {
        return;
    }
    if len <= 0.0 {
        // A repeated sample adds no arc length — and must NOT re-stamp
        // (that would double-deposit flow at a stationary pointer).
        return;
    }
    let inv = 1.0 / len;
    let mut travelled = (step - walk.residual).max(0.0);
    let mut last_emitted: Option<f32> = None;
    let mut emitted = 0usize;
    while travelled <= len && emitted < MAX_DABS_PER_SEGMENT {
        let t = travelled * inv;
        out.push(Dab {
            x: from.x + dx * t,
            y: from.y + dy * t,
            pressure: from.pressure + (to.pressure - from.pressure) * t,
        });
        last_emitted = Some(travelled);
        emitted += 1;
        travelled += step;
    }
    walk.residual = match last_emitted {
        // Decimated: re-anchor at the segment end so the NEXT segment
        // starts a fresh step rather than inheriting a bogus residual.
        Some(_) if emitted == MAX_DABS_PER_SEGMENT => 0.0,
        Some(d) => len - d,
        None => walk.residual + len,
    };
}

/// The per-stroke coverage accumulation buffer: one `f32` in `[0, 1]`
/// per image pixel, plus the dirty/stroke bookkeeping the compositor
/// needs.
///
/// # Why an accumulator and not per-dab compositing
///
/// Dabs within one stroke OVERLAP (spacing is a quarter of the diameter
/// by default). Compositing each dab separately would let the overlap
/// build past the brush's opacity — the classic "dark blobs at every
/// dab" artifact. Accumulating coverage first and compositing ONCE per
/// dirty region gives Photoshop's semantics exactly:
///
/// * `flow` is how much each dab deposits into the accumulator
///   (`acc ← acc + flow·dab·(1 − acc)` — an over-composite, so repeated
///   dabs approach but never exceed full coverage);
/// * `opacity` is the ceiling the whole stroke composites at, applied
///   ONCE when the accumulator is lowered to the ABI mask.
///
/// It is also the cheap shape: N dabs cost N cheap CPU stamps and ONE
/// GPU composite of their union, not N GPU composites.
///
/// MEMORY: 4 bytes per image pixel for the life of the stroke (a
/// 4000×3000 image ⇒ 48 MB), released on commit/cancel. A tiled/sparse
/// accumulator is the obvious refinement; it is not needed for
/// correctness and is recorded here rather than pretended away.
#[derive(Debug, Clone)]
pub struct StrokeAccumulator {
    width: u32,
    height: u32,
    acc: Vec<f32>,
    /// Union of the dab boxes stamped since the last [`Self::take_dirty`].
    dirty: Option<Region>,
    /// Union of every dab box in the stroke so far.
    stroke: Option<Region>,
    dabs: u64,
}

impl StrokeAccumulator {
    pub fn new(width: u32, height: u32) -> Self {
        StrokeAccumulator {
            width,
            height,
            acc: vec![0.0; (width as usize) * (height as usize)],
            dirty: None,
            stroke: None,
            dabs: 0,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Dabs stamped so far (the honest stroke readout).
    pub fn dab_count(&self) -> u64 {
        self.dabs
    }

    /// Accumulated coverage at `(x, y)`; `0` outside the field.
    pub fn value_at(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.acc[(y * self.width + x) as usize]
    }

    /// Nothing stamped yet?
    pub fn is_empty(&self) -> bool {
        self.stroke.is_none()
    }

    /// Union of the dab boxes stamped since the last [`Self::take_dirty`].
    pub fn dirty(&self) -> Option<Region> {
        self.dirty
    }

    /// The whole stroke's bounding box so far.
    pub fn stroke_bounds(&self) -> Option<Region> {
        self.stroke
    }

    /// Take (and clear) the pending dirty region — the window the
    /// compositor must re-derive from the untouched base image.
    pub fn take_dirty(&mut self) -> Option<Region> {
        self.dirty.take()
    }

    /// Stamp one dab of `tip` at `(cx, cy)` depositing `flow` (`0..1`).
    /// Returns `true` when it changed anything.
    ///
    /// `acc ← acc + flow·dab·(1 − acc)`: an over-composite, so a soft
    /// brush builds up smoothly toward 1 and never overshoots it.
    pub fn stamp(&mut self, tip: &BrushTip, cx: f32, cy: f32, flow: f32) -> bool {
        let flow = flow.clamp(0.0, 1.0);
        if flow <= 0.0 {
            return false;
        }
        let Some(bounds) = tip.dab_bounds(cx, cy, self.width, self.height) else {
            return false;
        };
        let mut touched = false;
        for y in bounds.y as u32..bounds.bottom() as u32 {
            for x in bounds.x as u32..bounds.right() as u32 {
                let c = tip.coverage_at_pixel(cx, cy, x, y);
                if c <= 0.0 {
                    continue;
                }
                let i = (y * self.width + x) as usize;
                let prev = self.acc[i];
                let next = (prev + flow * c * (1.0 - prev)).min(1.0);
                if next > prev {
                    self.acc[i] = next;
                    touched = true;
                }
            }
        }
        if touched {
            self.dirty = Some(match self.dirty {
                Some(d) => d.union(bounds),
                None => bounds,
            });
            self.stroke = Some(match self.stroke {
                Some(s) => s.union(bounds),
                None => bounds,
            });
        }
        self.dabs += 1;
        touched
    }

    /// The EFFECTIVE coverage at `(x, y)`: the accumulated stroke
    /// coverage scaled by the brush `opacity` and CLIPPED by the
    /// session's selection.
    ///
    /// This product is the whole of the selection-clipping story: with a
    /// selection bound, coverage outside it is exactly `0`, so the ABI's
    /// `mix(a, result, 0)` returns the backdrop unchanged and the brush
    /// cannot paint outside the selection.
    pub fn effective_at(
        &self,
        x: u32,
        y: u32,
        opacity: f32,
        selection: Option<&SelectionCoverage>,
    ) -> f32 {
        let base = self.value_at(x, y) * opacity.clamp(0.0, 1.0);
        match selection {
            Some(sel) => base * (sel.coverage_at(x, y) as f32 / 255.0),
            None => base,
        }
    }

    /// Lower a WINDOW of the effective coverage to the ABI's r16float
    /// mask bytes — the `mask` argument of `execute_tile_once`. Routed
    /// through [`SelectionMask`] so the f16 quantization is identical to
    /// the selection lane's ([`SelectionCoverage::mask_window_f16`]);
    /// one lowering point, one rounding rule.
    pub fn mask_window_f16(
        &self,
        region: Region,
        opacity: f32,
        selection: Option<&SelectionCoverage>,
    ) -> Vec<u8> {
        SelectionMask::from_fn(region.w, region.h, |x, y| {
            let ix = region.x + x as i32;
            let iy = region.y + y as i32;
            if ix < 0 || iy < 0 {
                return 0.0;
            }
            self.effective_at(ix as u32, iy as u32, opacity, selection)
        })
        .into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── tip falloff ──────────────────────────────────────────────────

    #[test]
    fn soft_tip_is_solid_inside_the_hardness_plateau_and_zero_outside() {
        let tip = BrushTip::soft(20.0, 0.5); // r = 10, plateau to 5
        assert_eq!(tip.coverage_at_distance(0.0), 1.0, "centre");
        assert_eq!(tip.coverage_at_distance(5.0), 1.0, "plateau rim");
        assert_eq!(tip.coverage_at_distance(10.0), 0.0, "outer rim");
        assert_eq!(tip.coverage_at_distance(11.0), 0.0, "outside");
        let mid = tip.coverage_at_distance(7.5);
        assert!((mid - 0.5).abs() < 1e-6, "smoothstep midpoint {mid}");
    }

    #[test]
    fn soft_tip_falloff_is_monotone_non_increasing() {
        let tip = BrushTip::soft(16.0, 0.2);
        let mut prev = f32::INFINITY;
        for i in 0..=160 {
            let c = tip.coverage_at_distance(i as f32 * 0.1);
            assert!(c <= prev + 1e-6, "non-monotone at d={}", i as f32 * 0.1);
            prev = c;
        }
    }

    #[test]
    fn a_fully_hard_tip_still_antialiases_over_a_one_pixel_band() {
        // hardness 1 would collapse the band to zero width; AA_BAND_PX
        // keeps a 1 px ramp just inside the rim.
        let tip = BrushTip::soft(20.0, 1.0);
        assert_eq!(tip.coverage_at_distance(9.0), 1.0);
        assert_eq!(tip.coverage_at_distance(10.0), 0.0);
        let rim = tip.coverage_at_distance(9.5);
        assert!(rim > 0.0 && rim < 1.0, "rim coverage {rim} should be soft");
    }

    #[test]
    fn the_pencil_tip_is_binary_and_aliased() {
        let tip = BrushTip::hard(20.0);
        assert_eq!(tip.coverage_at_distance(0.0), 1.0);
        assert_eq!(tip.coverage_at_distance(9.99), 1.0, "no rim ramp");
        assert_eq!(tip.coverage_at_distance(10.0), 1.0, "inclusive rim");
        assert_eq!(tip.coverage_at_distance(10.01), 0.0);
        // Every sampled coverage is exactly 0 or 1 — that IS the pencil.
        for i in 0..200 {
            let c = tip.coverage_at_distance(i as f32 * 0.1);
            assert!(c == 0.0 || c == 1.0, "pencil emitted {c}");
        }
    }

    #[test]
    fn a_sub_pixel_tip_never_paints_a_whole_pixel_solid() {
        let tip = BrushTip::soft(1.0, 1.0); // r = 0.5, band = 1.0
        assert!(tip.coverage_at_distance(0.0) < 1.0);
        assert!(tip.coverage_at_distance(0.0) > 0.0);
    }

    #[test]
    fn a_degenerate_tip_covers_nothing() {
        assert_eq!(BrushTip::soft(0.0, 1.0).coverage_at_distance(0.0), 0.0);
        assert!(BrushTip::soft(0.0, 1.0)
            .dab_bounds(4.0, 4.0, 8, 8)
            .is_none());
    }

    // ── sub-pixel positioning ────────────────────────────────────────

    #[test]
    fn a_sub_pixel_shift_changes_the_coverage_field() {
        let tip = BrushTip::soft(6.0, 0.0);
        let a = tip.coverage_at_pixel(4.0, 4.0, 2, 4);
        let b = tip.coverage_at_pixel(4.33, 4.0, 2, 4);
        assert_ne!(a, b, "a third-pixel move must move the coverage");
    }

    #[test]
    fn dab_bounds_clip_to_the_field_and_reject_a_miss() {
        let tip = BrushTip::soft(8.0, 0.5); // r = 4, reach = 5
        let b = tip.dab_bounds(1.0, 1.0, 16, 16).expect("overlaps");
        assert_eq!((b.x, b.y), (0, 0), "clipped at the top-left corner");
        assert!(b.right() <= 16 && b.bottom() <= 16);
        assert!(tip.dab_bounds(100.0, 100.0, 16, 16).is_none(), "far miss");
    }

    // ── spacing walk ─────────────────────────────────────────────────

    #[test]
    fn a_straight_drag_emits_evenly_spaced_dabs() {
        let mut walk = StrokeWalk::new();
        let mut out = Vec::new();
        plan_segment(
            &mut walk,
            StrokeSample::new(0.0, 0.0, 1.0),
            StrokeSample::new(100.0, 0.0, 1.0),
            10.0,
            &mut out,
        );
        // First dab a full step in (10), last at 100 ⇒ 10 dabs.
        assert_eq!(out.len(), 10);
        for (i, d) in out.iter().enumerate() {
            assert!((d.x - (i as f32 + 1.0) * 10.0).abs() < 1e-4, "dab {i}");
            assert_eq!(d.y, 0.0);
        }
        assert!(walk.residual().abs() < 1e-4, "landed exactly on the end");
    }

    #[test]
    fn spacing_is_continuous_across_pointer_samples() {
        // The SAME path delivered as one long segment and as three short
        // ones must produce the same dab positions — that is the whole
        // point of carrying the residual, and it is what makes a fast
        // drag (few, long segments) match a slow one (many, short).
        let step = 7.0;
        let mut one = Vec::new();
        let mut w1 = StrokeWalk::new();
        plan_segment(
            &mut w1,
            StrokeSample::new(0.0, 0.0, 1.0),
            StrokeSample::new(90.0, 0.0, 1.0),
            step,
            &mut one,
        );

        let mut many = Vec::new();
        let mut w2 = StrokeWalk::new();
        for (a, b) in [(0.0, 13.0), (13.0, 55.0), (55.0, 90.0)] {
            plan_segment(
                &mut w2,
                StrokeSample::new(a, 0.0, 1.0),
                StrokeSample::new(b, 0.0, 1.0),
                step,
                &mut many,
            );
        }
        assert_eq!(one.len(), many.len(), "same dab count");
        for (a, b) in one.iter().zip(many.iter()) {
            assert!((a.x - b.x).abs() < 1e-3, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn a_segment_shorter_than_the_spacing_emits_nothing_but_banks_it() {
        let mut walk = StrokeWalk::new();
        let mut out = Vec::new();
        for _ in 0..4 {
            plan_segment(
                &mut walk,
                StrokeSample::new(0.0, 0.0, 1.0),
                StrokeSample::new(3.0, 0.0, 1.0),
                10.0,
                &mut out,
            );
        }
        // 4 × 3 px = 12 px banked ⇒ exactly one dab fires on the 4th.
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn pressure_interpolates_along_the_segment() {
        let mut walk = StrokeWalk::new();
        let mut out = Vec::new();
        plan_segment(
            &mut walk,
            StrokeSample::new(0.0, 0.0, 0.0),
            StrokeSample::new(100.0, 0.0, 1.0),
            25.0,
            &mut out,
        );
        assert_eq!(out.len(), 4);
        for (i, d) in out.iter().enumerate() {
            let want = (i as f32 + 1.0) * 0.25;
            assert!((d.pressure - want).abs() < 1e-4, "dab {i}: {d:?}");
        }
    }

    #[test]
    fn a_zero_length_segment_never_restamps() {
        let mut walk = StrokeWalk::new();
        let mut out = Vec::new();
        for _ in 0..50 {
            plan_segment(
                &mut walk,
                StrokeSample::new(5.0, 5.0, 1.0),
                StrokeSample::new(5.0, 5.0, 1.0),
                4.0,
                &mut out,
            );
        }
        assert!(out.is_empty(), "a stationary pointer deposits nothing");
    }

    #[test]
    fn a_pathological_jump_is_decimated_not_infinite() {
        let mut walk = StrokeWalk::new();
        let mut out = Vec::new();
        plan_segment(
            &mut walk,
            StrokeSample::new(0.0, 0.0, 1.0),
            StrokeSample::new(1.0e7, 0.0, 1.0),
            0.0, // clamps to MIN_SPACING_PX
            &mut out,
        );
        assert_eq!(out.len(), MAX_DABS_PER_SEGMENT);
        assert_eq!(walk.residual(), 0.0, "re-anchored after decimation");
    }

    #[test]
    fn non_finite_samples_are_ignored() {
        let mut walk = StrokeWalk::new();
        let mut out = Vec::new();
        plan_segment(
            &mut walk,
            StrokeSample::new(0.0, 0.0, 1.0),
            StrokeSample::new(f32::NAN, 0.0, 1.0),
            4.0,
            &mut out,
        );
        assert!(out.is_empty());
    }

    // ── pressure mapping ─────────────────────────────────────────────

    #[test]
    fn pressure_targets_scale_the_documented_properties() {
        assert_eq!(PressureTarget::None.size_scale(0.25), 1.0);
        assert_eq!(PressureTarget::None.flow_scale(0.25), 1.0);
        assert_eq!(PressureTarget::Size.size_scale(0.25), 0.25);
        assert_eq!(PressureTarget::Size.flow_scale(0.25), 1.0);
        assert_eq!(PressureTarget::Opacity.size_scale(0.25), 1.0);
        assert_eq!(PressureTarget::Opacity.flow_scale(0.25), 0.25);
        assert_eq!(PressureTarget::Both.size_scale(0.25), 0.25);
        assert_eq!(PressureTarget::Both.flow_scale(0.25), 0.25);
        assert_eq!(PressureTarget::default(), PressureTarget::Both);
    }

    #[test]
    fn pressure_target_round_trips_its_wire_name() {
        for t in [
            PressureTarget::None,
            PressureTarget::Size,
            PressureTarget::Opacity,
            PressureTarget::Both,
        ] {
            assert_eq!(PressureTarget::from_wire(t.as_wire()), Some(t));
        }
        assert_eq!(PressureTarget::from_wire("tilt"), None);
    }

    // ── accumulation ─────────────────────────────────────────────────

    #[test]
    fn a_single_full_flow_dab_saturates_its_centre() {
        let mut acc = StrokeAccumulator::new(16, 16);
        assert!(acc.is_empty());
        assert!(acc.stamp(&BrushTip::soft(8.0, 0.6), 8.0, 8.0, 1.0));
        assert!(!acc.is_empty());
        assert_eq!(acc.dab_count(), 1);
        assert!((acc.value_at(8, 8) - 1.0).abs() < 1e-5, "centre saturates");
        assert_eq!(acc.value_at(0, 0), 0.0, "far corner untouched");
    }

    #[test]
    fn overlapping_dabs_build_up_but_never_exceed_one() {
        let mut acc = StrokeAccumulator::new(16, 16);
        let tip = BrushTip::soft(8.0, 1.0);
        let mut prev = 0.0;
        for _ in 0..40 {
            acc.stamp(&tip, 8.0, 8.0, 0.2);
            let v = acc.value_at(8, 8);
            assert!(v >= prev, "build-up must be monotone");
            assert!(v <= 1.0 + 1e-6, "coverage overshot: {v}");
            prev = v;
        }
        assert!(
            prev > 0.99,
            "40 dabs at flow 0.2 should approach 1 ({prev})"
        );
    }

    #[test]
    fn a_zero_flow_dab_is_a_no_op() {
        let mut acc = StrokeAccumulator::new(8, 8);
        assert!(!acc.stamp(&BrushTip::soft(4.0, 1.0), 4.0, 4.0, 0.0));
        assert!(acc.is_empty());
        assert!(acc.dirty().is_none());
    }

    #[test]
    fn dirty_accumulates_until_taken_and_stroke_bounds_never_shrink() {
        let mut acc = StrokeAccumulator::new(64, 64);
        let tip = BrushTip::soft(4.0, 1.0);
        acc.stamp(&tip, 8.0, 8.0, 1.0);
        acc.stamp(&tip, 40.0, 8.0, 1.0);
        let d = acc.take_dirty().expect("dirty");
        assert!(d.x <= 5 && d.right() >= 42, "union of both dabs: {d:?}");
        assert!(acc.take_dirty().is_none(), "taking clears");
        let s0 = acc.stroke_bounds().expect("stroke");
        acc.stamp(&tip, 8.0, 40.0, 1.0);
        let s1 = acc.stroke_bounds().expect("stroke");
        assert!(s1.bottom() > s0.bottom(), "stroke bounds grow");
        assert!(acc.take_dirty().is_some(), "the third dab re-dirtied");
    }

    // ── effective coverage / selection clipping ──────────────────────

    #[test]
    fn opacity_scales_the_effective_coverage() {
        let mut acc = StrokeAccumulator::new(8, 8);
        acc.stamp(&BrushTip::soft(6.0, 1.0), 4.0, 4.0, 1.0);
        assert!((acc.effective_at(4, 4, 1.0, None) - 1.0).abs() < 1e-5);
        assert!((acc.effective_at(4, 4, 0.25, None) - 0.25).abs() < 1e-5);
        assert_eq!(acc.effective_at(4, 4, 0.0, None), 0.0);
    }

    #[test]
    fn a_selection_clips_the_effective_coverage_to_exactly_zero_outside() {
        // A dab that straddles a selection edge: inside the selection the
        // coverage survives, outside it is EXACTLY 0 — which is what makes
        // `mix(a, result, 0)` return the backdrop bit-for-bit.
        let mut acc = StrokeAccumulator::new(16, 16);
        acc.stamp(&BrushTip::soft(12.0, 1.0), 8.0, 8.0, 1.0);
        let sel = SelectionCoverage::rasterize_rect(16, 16, 0.0, 0.0, 8.0, 16.0);
        for y in 0..16 {
            for x in 0..16 {
                let eff = acc.effective_at(x, y, 1.0, Some(&sel));
                if x >= 8 {
                    assert_eq!(eff, 0.0, "({x},{y}) is outside the selection");
                }
            }
        }
        assert!(
            acc.effective_at(4, 8, 1.0, Some(&sel)) > 0.0,
            "inside the selection the dab still paints"
        );
    }

    #[test]
    fn a_feathered_selection_edge_scales_the_dab_smoothly() {
        let mut acc = StrokeAccumulator::new(16, 16);
        acc.stamp(&BrushTip::soft(16.0, 1.0), 8.0, 8.0, 1.0);
        let mut sel = SelectionCoverage::rasterize_rect(16, 16, 0.0, 0.0, 8.0, 16.0);
        sel.feather(2.0);
        let eff = acc.effective_at(8, 8, 1.0, Some(&sel));
        assert!(eff > 0.0 && eff < 1.0, "feathered seam scales: {eff}");
    }

    // ── mask lowering ────────────────────────────────────────────────

    #[test]
    fn the_mask_window_matches_the_effective_coverage_and_zero_pads() {
        let mut acc = StrokeAccumulator::new(8, 8);
        acc.stamp(&BrushTip::soft(6.0, 1.0), 4.0, 4.0, 1.0);
        let bytes = acc.mask_window_f16(Region::new(2, 2, 4, 4), 0.5, None);
        let m = SelectionMask::from_bytes(4, 4, bytes).expect("4x4");
        let want = acc.effective_at(4, 4, 0.5, None);
        assert!((m.weight_at(2, 2) - want).abs() < 1e-3, "f16 round-trip");

        // A window off the field reads 0 (nothing to paint there).
        let bytes = acc.mask_window_f16(Region::new(20, 20, 2, 2), 1.0, None);
        let m = SelectionMask::from_bytes(2, 2, bytes).expect("2x2");
        assert_eq!(m.weight_at(0, 0), 0.0);
        assert_eq!(m.weight_at(1, 1), 0.0);
    }

    #[test]
    fn the_mask_window_carries_the_selection_clip() {
        let mut acc = StrokeAccumulator::new(16, 16);
        acc.stamp(&BrushTip::soft(12.0, 1.0), 8.0, 8.0, 1.0);
        let sel = SelectionCoverage::rasterize_rect(16, 16, 0.0, 0.0, 8.0, 16.0);
        let bytes = acc.mask_window_f16(Region::new(0, 0, 16, 16), 1.0, Some(&sel));
        let m = SelectionMask::from_bytes(16, 16, bytes).expect("16x16");
        for y in 0..16 {
            for x in 8..16 {
                assert_eq!(m.weight_at(x, y), 0.0, "({x},{y}) clipped to zero");
            }
        }
    }

    // ── determinism ──────────────────────────────────────────────────

    #[test]
    fn the_same_samples_always_produce_the_same_coverage() {
        // Actions and scripts replay strokes; identical input samples must
        // give identical pixels, so the coverage field must be a pure
        // function of the sample list.
        let samples = [
            StrokeSample::new(2.0, 3.0, 0.4),
            StrokeSample::new(17.5, 9.25, 0.9),
            StrokeSample::new(30.0, 28.75, 0.2),
            StrokeSample::new(5.5, 30.0, 1.0),
        ];
        let run = || {
            let tip = BrushTip::soft(9.0, 0.35);
            let mut acc = StrokeAccumulator::new(48, 48);
            let mut walk = StrokeWalk::new();
            let mut dabs = Vec::new();
            acc.stamp(&tip, samples[0].x, samples[0].y, 0.5 * samples[0].pressure);
            for pair in samples.windows(2) {
                dabs.clear();
                plan_segment(&mut walk, pair[0], pair[1], 9.0 * 0.25, &mut dabs);
                for d in &dabs {
                    let t = BrushTip::soft(9.0 * d.pressure, 0.35);
                    acc.stamp(&t, d.x, d.y, 0.5 * d.pressure);
                }
            }
            acc
        };
        let a = run();
        let b = run();
        assert_eq!(a.dab_count(), b.dab_count());
        for y in 0..48 {
            for x in 0..48 {
                assert_eq!(a.value_at(x, y), b.value_at(x, y), "({x},{y})");
            }
        }
        assert!(a.dab_count() > 10, "the fixture actually painted");
    }
}
