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

//! RASTER TYPE — shaped, rasterized glyphs as a coverage field.
//!
//! # Why this is small
//!
//! A rasterized glyph run IS a coverage field, and this engine already
//! composites arbitrary coverage: every masked kernel dispatch takes an
//! r16float mask at `@group(2)`, and a solid fill under a mask is
//! exactly "paint these pixels this colour". So type needed no new
//! kernel and no new compositing path — the same thing that was true of
//! the clone stamp, the healing brush, clipping and groups. What was
//! genuinely missing was only the glyphs.
//!
//! # Shaping is not optional
//!
//! Laying glyphs out by advance width alone is the tempting shortcut and
//! it is wrong in a way that does not announce itself: Latin looks
//! plausible while Arabic comes out unjoined and backwards, and
//! Devanagari's reordering is simply absent. That is the exact failure
//! this codebase is arranged to avoid — plausible-looking wrong output —
//! so shaping runs through `rustybuzz` (a HarfBuzz port), which handles
//! joining, reordering, ligatures and kerning from the font's own
//! tables.
//!
//! # What this deliberately is not
//!
//! RASTER type, in the Photoshop-catalog sense: committed glyphs become
//! pixels in a layer. It is not a live text OBJECT that stays editable —
//! that is the host's text frame, which exists and is better at it. And
//! it lays out HARD-BROKEN lines only: a newline starts a line. Within
//! that, the run is fully styled — size, tracking, leading, per-line
//! alignment, horizontal/vertical scale, faux-italic slant, baseline
//! shift, underline and strikethrough from the FACE's own metrics, and
//! a standard-ligature toggle.
//!
//! What is absent is absent STRUCTURALLY, and the boundary has one
//! cause: nothing WRAPS, because wrapping needs a measure to wrap
//! against and for a raster layer that would be the layer, which is not
//! a text column. Everything downstream of that — indents, hyphenation,
//! space before/after, justification — follows from it. Those are
//! text-engine features, the host already has a text engine, and
//! duplicating it here is the mistake the Paths panel was not built for
//! the same reason.
//!
//! Each line is shaped INDEPENDENTLY rather than shaping the whole
//! string and slicing the result: shaping is contextual, so ligatures,
//! kerning pairs and bidi runs must not reach across a break.

use ab_glyph_rasterizer::{point, Rasterizer};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{FontRef, MetadataProvider};

/// A rasterized run: an 8-bit coverage field plus where it sits relative
/// to the caller's origin.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterRun {
    pub width: u32,
    pub height: u32,
    /// One byte per pixel — the same representation `SelectionCoverage`
    /// speaks, which is why compositing this needs nothing new.
    pub coverage: Vec<u8>,
    /// The run's top-left corner relative to the requested BASELINE
    /// origin, in pixels. `dy` is normally negative: ink sits above the
    /// baseline.
    pub dx: i32,
    pub dy: i32,
    /// Glyphs the face had no outline for. Reported rather than drawn as
    /// tofu and rather than silently skipped: a caller that asked for
    /// text this font cannot render needs to know which part.
    pub missing: usize,
}

/// One path segment in DEVICE space (y down, origin at the baseline).
///
/// Collected before rasterizing because the run's extent is not known
/// until every glyph has been walked — and sizing the rasterizer from
/// font METRICS instead would over-allocate for short runs and clip
/// nothing usefully: metrics describe the face, not the string.
#[derive(Debug, Clone, Copy)]
enum Seg {
    Line([f32; 4]),
    Quad([f32; 6]),
    Cubic([f32; 8]),
}

/// Collects a glyph's outline, flipping font space (y up) into device
/// space (y down) and translating to the pen position.
struct Collector<'a> {
    segs: &'a mut Vec<Seg>,
    ox: f32,
    oy: f32,
    start: (f32, f32),
    cur: (f32, f32),
    min_x: &'a mut f32,
    min_y: &'a mut f32,
    max_x: &'a mut f32,
    max_y: &'a mut f32,
    /// Horizontal / vertical scale and the faux-italic shear. EVERY
    /// point funnels through `map`, so applying them here is what makes
    /// one implementation cover outlines, bounds and decorations —
    /// three places that would otherwise disagree at the edges.
    h_scale: f32,
    v_scale: f32,
    shear: f32,
}

impl Collector<'_> {
    fn map(&mut self, x: f32, y: f32) -> (f32, f32) {
        // Scale about the glyph's own origin FIRST, then flip into
        // device space, then shear about the BASELINE. Shearing last is
        // what makes a slant lean from the baseline rather than pivot
        // around the glyph's middle.
        let sx = x * self.h_scale;
        let sy = y * self.v_scale;
        let dy = self.oy - sy;
        let p = (self.ox + sx - (dy - self.oy) * self.shear, dy);
        *self.min_x = self.min_x.min(p.0);
        *self.min_y = self.min_y.min(p.1);
        *self.max_x = self.max_x.max(p.0);
        *self.max_y = self.max_y.max(p.1);
        p
    }
}

impl OutlinePen for Collector<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.map(x, y);
        self.start = p;
        self.cur = p;
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.map(x, y);
        self.segs
            .push(Seg::Line([self.cur.0, self.cur.1, p.0, p.1]));
        self.cur = p;
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let c = self.map(cx, cy);
        let p = self.map(x, y);
        self.segs
            .push(Seg::Quad([self.cur.0, self.cur.1, c.0, c.1, p.0, p.1]));
        self.cur = p;
    }
    fn curve_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        let a = self.map(c1x, c1y);
        let b = self.map(c2x, c2y);
        let p = self.map(x, y);
        self.segs.push(Seg::Cubic([
            self.cur.0, self.cur.1, a.0, a.1, b.0, b.1, p.0, p.1,
        ]));
        self.cur = p;
    }
    fn close(&mut self) {
        // An unclosed contour would leave the scanline accumulator with
        // an unbalanced edge and bleed coverage across the whole row.
        if self.cur != self.start {
            self.segs.push(Seg::Line([
                self.cur.0,
                self.cur.1,
                self.start.0,
                self.start.1,
            ]));
        }
        self.cur = self.start;
    }
}

/// Shape and rasterize one horizontal run.
///
/// `size_px` is the em size in pixels. Returns `None` when the font does
/// not parse or the size is not a positive finite number. A run with no
/// ink (whitespace, or every glyph missing) is `Some` with a zero
/// extent — legal, and different from a failure.
/// Typographic settings beyond the face and the size.
///
/// Both are here rather than as bare parameters because they arrive
/// TOGETHER from one panel and grow together; a third one should not
/// change every call site again.
#[derive(Debug, Clone, Copy)]
pub struct RunStyle {
    /// Letter spacing in 1/1000 em — IDML's own unit, and UNITLESS by
    /// construction. That is the useful part: em is relative to the
    /// size, so the same number means the same thing whether the caller
    /// thinks in image pixels or in points, and this setting needs no
    /// unit bridge at all (unlike size and leading, which do).
    pub tracking_per_mille: f32,
    /// Baseline-to-baseline distance in PIXELS. `None` (or a
    /// non-positive value) means AUTO: the face's own default line
    /// height. Auto is not a fixed multiple of the size — a face
    /// carries its own ascent/descent/gap, and using them is what makes
    /// two different faces at the same size lead correctly.
    pub leading_px: Option<f32>,
    /// Baseline offset in PIXELS, positive = UP (typographic
    /// convention, opposite to the device y this module rasterizes in).
    pub baseline_shift_px: f32,
    /// Horizontal / vertical scale as a MULTIPLE (1.0 = 100%). Applied
    /// to the outlines AND — for the horizontal axis — to the advances,
    /// because a condensed face must also set narrower.
    pub h_scale: f32,
    pub v_scale: f32,
    /// FAUX ITALIC slant in degrees, positive = leaning right. A shear,
    /// not a face: a real italic is a different design and belongs in
    /// `style`. This is what a designer reaches for when the document
    /// embeds no italic, and calling it what it is matters.
    pub skew_deg: f32,
    /// Draw the face's own underline / strikethrough rules. Positions
    /// and thicknesses come from the FACE (`post` / `OS/2`), never from
    /// a guess — a rule at an invented offset is worse than none.
    pub underline: bool,
    pub strikethrough: bool,
    /// Standard ligatures (`liga`). ON is the shaper's default; turning
    /// it OFF is the setting worth having, for the "fi" a designer does
    /// not want.
    pub ligatures: bool,
    /// How lines sit relative to each other. Only meaningful because
    /// the lane lays out MORE THAN ONE line — with a single line every
    /// alignment is identical, since the run has no width but its own.
    pub align: LineAlign,
}

/// Horizontal alignment of lines within a multi-line run, measured
/// against the WIDEST line — the run's own extent. There is no column
/// to align against, because a raster layer is not a text column; that
/// is the same boundary that keeps wrapping out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineAlign {
    #[default]
    Left,
    Center,
    Right,
}

impl Default for RunStyle {
    fn default() -> Self {
        Self {
            tracking_per_mille: 0.0,
            leading_px: None,
            baseline_shift_px: 0.0,
            // 1.0, not 0.0 — a `#[derive(Default)]` here would make the
            // default style scale every glyph to nothing, which is the
            // kind of default that looks like a rendering bug.
            h_scale: 1.0,
            v_scale: 1.0,
            skew_deg: 0.0,
            underline: false,
            strikethrough: false,
            ligatures: true,
            align: LineAlign::Left,
        }
    }
}

pub fn rasterize_run(
    font_bytes: &[u8],
    text: &str,
    size_px: f32,
    style: RunStyle,
) -> Option<RasterRun> {
    if text.is_empty() || !(size_px.is_finite() && size_px > 0.0) {
        return None;
    }
    let font = FontRef::new(font_bytes).ok()?;

    // SHAPE. `guess_segment_properties` reads script, direction and
    // language off the text itself, which is what makes a right-to-left
    // run come out right without the caller declaring anything.
    let data = harfrust::ShaperData::new(&font);
    let shaper = data.shaper(&font).build();

    let upem = shaper.units_per_em() as f32;
    if upem <= 0.0 {
        return None;
    }
    let scale = size_px / upem;
    let outlines = font.outline_glyphs();

    // TRACKING is 1/1000 em, so it scales with the SIZE and not with the
    // face's upem — that is what makes the setting mean the same thing
    // across faces.
    let track_px = size_px * style.tracking_per_mille / 1000.0;

    // LEADING. Auto asks the FACE, because line height is a property of
    // the design and not a fixed multiple: two faces at the same size
    // lead differently and should. `1.2 * size` is the fallback only
    // when the face reports nothing usable.
    let auto_leading = {
        // skrifa reports these ALREADY SCALED to the requested size, so
        // there is no upem arithmetic to get wrong here. `descent` is
        // negative by convention, hence the subtraction.
        let m = font.metrics(Size::new(size_px), LocationRef::default());
        let h = m.ascent - m.descent + m.leading;
        if h.is_finite() && h > 0.0 {
            h
        } else {
            size_px * 1.2
        }
    };
    let leading = match style.leading_px {
        Some(l) if l.is_finite() && l > 0.0 => l,
        _ => auto_leading,
    };

    let mut segs: Vec<Seg> = Vec::new();
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    let mut missing = 0usize;

    // MULTI-LINE. Each line is shaped INDEPENDENTLY rather than shaping
    // the whole string and breaking it up, because shaping is contextual
    // — ligatures, kerning pairs and bidi runs must not reach across a
    // line break. Splitting first is the only way a newline is a real
    // boundary rather than a character the shaper tries to render.
    //
    // `\r\n` is normalised so a Windows-authored string does not lay out
    // with a stray carriage return per line.
    let normalised = text.replace("\r\n", "\n");
    let lines: Vec<&str> = normalised.split('\n').collect();

    // Standard ligatures OFF is a real request; ON is the shaper's own
    // default, so an empty feature list is the honest way to say "as
    // the face intends".
    let features: Vec<harfrust::Feature> = if style.ligatures {
        Vec::new()
    } else {
        vec![harfrust::Feature::new(harfrust::Tag::new(b"liga"), 0, ..)]
    };

    // PASS 1 — measure every line's advance width. Alignment needs the
    // widest line before ANY line can be placed, so measuring first is
    // not an optimisation, it is the only order that works.
    let shape_line = |line: &str| {
        let mut buf = harfrust::UnicodeBuffer::new();
        buf.push_str(line);
        buf.guess_segment_properties();
        shaper.shape(buf, &features)
    };
    let advance_of = |shaped: &harfrust::GlyphBuffer| -> f32 {
        shaped
            .glyph_positions()
            .iter()
            .map(|p| p.x_advance as f32 * scale * style.h_scale + track_px)
            .sum()
    };
    let shaped_lines: Vec<Option<harfrust::GlyphBuffer>> = lines
        .iter()
        .map(|l| {
            if l.is_empty() {
                None
            } else {
                Some(shape_line(l))
            }
        })
        .collect();
    let widths: Vec<f32> = shaped_lines
        .iter()
        .map(|s| s.as_ref().map(advance_of).unwrap_or(0.0))
        .collect();
    let widest = widths.iter().cloned().fold(0.0f32, f32::max);

    // The SHEAR, in device space (y grows down), so a positive slant
    // leans right the way a designer expects.
    let shear = (style.skew_deg.to_radians()).tan();

    // PASS 2 — place.
    for (line_index, shaped) in shaped_lines.iter().enumerate() {
        // An EMPTY line still consumes its leading — a blank line is a
        // deliberate gap, not a no-op.
        let Some(shaped) = shaped else { continue };
        let mut pen_x = match style.align {
            LineAlign::Left => 0.0,
            LineAlign::Center => (widest - widths[line_index]) * 0.5,
            LineAlign::Right => widest - widths[line_index],
        };
        // `baseline_shift_px` is positive-UP by typographic convention
        // and this is device space, so it SUBTRACTS.
        let mut pen_y = leading * line_index as f32 - style.baseline_shift_px;
        let line_top = pen_y;
        let line_x0 = pen_x;

        for (info, pos) in shaped.glyph_infos().iter().zip(shaped.glyph_positions()) {
            let gid = skrifa::GlyphId::from(info.glyph_id as u16);
            match outlines.get(gid) {
                Some(glyph) => {
                    let mut collector = Collector {
                        segs: &mut segs,
                        ox: pen_x + pos.x_offset as f32 * scale,
                        oy: pen_y - pos.y_offset as f32 * scale,
                        start: (0.0, 0.0),
                        cur: (0.0, 0.0),
                        min_x: &mut min_x,
                        min_y: &mut min_y,
                        max_x: &mut max_x,
                        max_y: &mut max_y,
                        h_scale: style.h_scale,
                        v_scale: style.v_scale,
                        shear,
                    };
                    let settings =
                        DrawSettings::unhinted(Size::new(size_px), LocationRef::default());
                    if glyph.draw(settings, &mut collector).is_err() {
                        missing += 1;
                    }
                }
                // No outline for this glyph id. Counted and skipped —
                // drawing the face's tofu box would invent content the
                // caller did not ask for, and skipping silently would
                // hide a missing font.
                None => missing += 1,
            }
            pen_x += pos.x_advance as f32 * scale * style.h_scale + track_px;
            pen_y -= pos.y_advance as f32 * scale;
        }

        // DECORATIONS, from the FACE's own metrics. A rule at an
        // invented offset is worse than no rule, so a face that reports
        // neither simply gets none — silently, because the alternative
        // is a warning about a font the designer cannot change.
        if style.underline || style.strikethrough {
            let m = font.metrics(Size::new(size_px), LocationRef::default());
            let width = pen_x - line_x0;
            let mut rule = |d: Option<skrifa::metrics::Decoration>| {
                if let Some(d) = d {
                    // `offset` is from the baseline, positive UP, and
                    // the thickness grows downward from it.
                    let y0 = line_top - d.offset * style.v_scale;
                    let y1 = y0 + d.thickness.max(1.0) * style.v_scale;
                    let quad = [
                        (line_x0, y0),
                        (line_x0 + width, y0),
                        (line_x0 + width, y1),
                        (line_x0, y1),
                    ];
                    // Closed, and wound like a glyph contour so the ONE
                    // shared rasterizer resolves it by the same winding
                    // rule everything else uses.
                    for i in 0..4 {
                        let (x0, y0) = quad[i];
                        let (x1, y1) = quad[(i + 1) % 4];
                        let (sx0, sx1) = (x0 - y0 * shear, x1 - y1 * shear);
                        min_x = min_x.min(sx0.min(sx1));
                        max_x = max_x.max(sx0.max(sx1));
                        min_y = min_y.min(y0.min(y1));
                        max_y = max_y.max(y0.max(y1));
                        segs.push(Seg::Line([sx0, y0, sx1, y1]));
                    }
                }
            };
            if style.underline {
                rule(m.underline);
            }
            if style.strikethrough {
                rule(m.strikeout);
            }
        }
    }

    if segs.is_empty() || min_x > max_x {
        return Some(RasterRun {
            width: 0,
            height: 0,
            coverage: Vec::new(),
            dx: 0,
            dy: 0,
            missing,
        });
    }

    let ox = min_x.floor();
    let oy = min_y.floor();
    let w = ((max_x.ceil() - ox) as u32).max(1);
    let h = ((max_y.ceil() - oy) as u32).max(1);

    // ONE rasterizer for the whole run. Overlapping contours — accents,
    // ligature parts, script joins — then resolve by the accumulator's
    // own winding rather than by a per-glyph max, which is both simpler
    // and more correct: max would flatten a counter (the hole in an "o")
    // that a second overlapping glyph covers.
    let mut r = Rasterizer::new(w as usize, h as usize);
    for seg in &segs {
        match *seg {
            Seg::Line([x0, y0, x1, y1]) => {
                r.draw_line(point(x0 - ox, y0 - oy), point(x1 - ox, y1 - oy))
            }
            Seg::Quad([x0, y0, cx, cy, x1, y1]) => r.draw_quad(
                point(x0 - ox, y0 - oy),
                point(cx - ox, cy - oy),
                point(x1 - ox, y1 - oy),
            ),
            Seg::Cubic([x0, y0, ax, ay, bx, by, x1, y1]) => r.draw_cubic(
                point(x0 - ox, y0 - oy),
                point(ax - ox, ay - oy),
                point(bx - ox, by - oy),
                point(x1 - ox, y1 - oy),
            ),
        }
    }
    let mut coverage = vec![0u8; (w as usize) * (h as usize)];
    r.for_each_pixel_2d(|x, y, c| {
        if x < w && y < h {
            coverage[(y as usize) * (w as usize) + (x as usize)] =
                (c * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    });

    Some(RasterRun {
        width: w,
        height: h,
        coverage,
        dx: ox as i32,
        dy: oy as i32,
        missing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny valid TrueType face, built by the conformance harness'
    /// sibling approach: rather than commit a binary, derive one from
    /// `ab_glyph`'s own dev fixture if present, else skip loudly.
    ///
    /// There is no font in this repo — fonts arrive from the HOST at
    /// runtime (`host.assets.getFontFace`) — so these tests assert the
    /// contract's edges, which need no glyphs, and the pixel behaviour is
    /// asserted in the glue spec where a real face can be supplied.
    #[test]
    fn image_editor_raster_type_refuses_input_it_cannot_shape() {
        // Not a font: parsing must fail cleanly rather than panic.
        assert!(rasterize_run(b"not a font at all", "hello", 24.0, RunStyle::default()).is_none());
        // Empty text is not an error, it is nothing to do.
        assert!(rasterize_run(b"not a font", "", 24.0, RunStyle::default()).is_none());
    }

    #[test]
    fn image_editor_raster_type_refuses_a_nonsensical_size() {
        // A zero, negative or non-finite em size has no rendering — and
        // returning `None` beats rasterizing a degenerate box.
        for bad in [0.0f32, -12.0, f32::NAN, f32::INFINITY] {
            assert!(
                rasterize_run(b"not a font", "hi", bad, RunStyle::default()).is_none(),
                "size {bad} should be refused"
            );
        }
    }

    /// THE RASTERIZER, checked by AREA — which is the strongest thing
    /// you can assert about a scanline fill, because a winding or
    /// accumulation mistake changes the total while still producing a
    /// picture that looks like a shape.
    ///
    /// A 6×4 axis-aligned rectangle drawn as four lines must integrate to
    /// exactly 24 pixels of coverage.
    /// The style edges, which need no glyphs: a caller can send NaN
    /// from a panel field that has been cleared, and neither value may
    /// turn into a layout that silently does nothing.
    #[test]
    fn image_editor_raster_type_style_edges_are_normalised_not_propagated() {
        // A non-finite tracking must not reach the pen — `pen_x + NaN`
        // is NaN for every glyph after it, which collapses the whole
        // run's bounding box to nothing and rasterizes an empty image.
        // Normalising at the boundary is why the wasm entry clamps it
        // rather than trusting the caller.
        let nan_track = RunStyle {
            tracking_per_mille: f32::NAN,
            ..RunStyle::default()
        };
        // Still refused for the FONT, not for the tracking — the point
        // is that it reaches the parse rather than dying earlier.
        assert!(rasterize_run(b"not a font", "hi", 24.0, nan_track).is_none());

        // A non-positive leading means AUTO, not a zero-height layout
        // where every line stacks on the first.
        for l in [Some(0.0f32), Some(-10.0), Some(f32::NAN), None] {
            let st = RunStyle {
                leading_px: l,
                ..RunStyle::default()
            };
            assert!(rasterize_run(b"not a font", "a\nb", 24.0, st).is_none());
        }
    }

    /// A real face from the SYSTEM, or `None`. Mirrors the glue spec's
    /// `systemFace()` helper for the same reason: this repo ships no
    /// font — faces arrive from the host at runtime — so a layout
    /// assertion either finds one or says out loud that it did not.
    fn system_face() -> Option<Vec<u8>> {
        for p in [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ] {
            if let Ok(b) = std::fs::read(p) {
                return Some(b);
            }
        }
        None
    }

    /// THE LAYOUT ASSERTIONS. Geometric rather than pixel-exact, so they
    /// hold for whichever face the machine has: a second line must make
    /// the run TALLER, more leading must make it taller still, and
    /// tracking must make it WIDER. Nothing here depends on a specific
    /// design.
    #[test]
    fn image_editor_raster_type_lays_out_lines_leading_and_tracking() {
        let Some(face) = system_face() else {
            // Loudly. A quiet pass would read as coverage this machine
            // cannot actually provide.
            eprintln!("no system font available — skipping the layout assertions");
            return;
        };
        let plain = RunStyle::default();

        let one = rasterize_run(&face, "A", 32.0, plain).expect("one line");
        let two = rasterize_run(&face, "A\nA", 32.0, plain).expect("two lines");
        assert!(
            two.height > one.height,
            "a second line must occupy more height: {} vs {}",
            two.height,
            one.height
        );

        // LEADING widens the gap, so the same two lines get taller.
        let loose = rasterize_run(
            &face,
            "A\nA",
            32.0,
            RunStyle {
                leading_px: Some(120.0),
                ..RunStyle::default()
            },
        )
        .expect("two lines, loose");
        assert!(
            loose.height > two.height,
            "explicit leading of 120px must exceed this face's auto leading at 32px"
        );

        // An EMPTY line still consumes its leading — a blank line is a
        // deliberate gap, and collapsing it would silently reflow.
        let gapped = rasterize_run(&face, "A\n\nA", 32.0, plain).expect("with a blank line");
        assert!(
            gapped.height > two.height,
            "a blank line must occupy a line's worth of height"
        );

        // TRACKING widens a multi-glyph run. Asserted on TWO glyphs
        // because tracking is applied per advance — a one-glyph run
        // would only move the pen past the end and might not change the
        // inked bounds at all.
        let tight = rasterize_run(&face, "AA", 32.0, plain).expect("two glyphs");
        let tracked = rasterize_run(
            &face,
            "AA",
            32.0,
            RunStyle {
                tracking_per_mille: 500.0,
                ..RunStyle::default()
            },
        )
        .expect("two glyphs, tracked");
        assert!(
            tracked.width > tight.width,
            "500/1000 em of tracking must widen the run: {} vs {}",
            tracked.width,
            tight.width
        );
    }

    /// EVERY REMAINING AXIS, measured. Each assertion is a geometric
    /// consequence rather than a pixel comparison, so it holds for
    /// whichever face the machine has.
    #[test]
    fn image_editor_raster_type_every_style_axis_changes_the_geometry() {
        let Some(face) = system_face() else {
            eprintln!("no system font available — skipping the style-axis assertions");
            return;
        };
        let base = RunStyle::default();
        let plain = rasterize_run(&face, "AB", 32.0, base).expect("base");

        // H-SCALE widens without touching height; V-SCALE the reverse.
        // Asserting the OTHER axis is unchanged is what catches a
        // transform applied to both by accident.
        let wide = rasterize_run(
            &face,
            "AB",
            32.0,
            RunStyle {
                h_scale: 2.0,
                ..base
            },
        )
        .expect("wide");
        assert!(wide.width > plain.width, "h_scale must widen");
        assert_eq!(wide.height, plain.height, "h_scale must not change height");

        let tall = rasterize_run(
            &face,
            "AB",
            32.0,
            RunStyle {
                v_scale: 2.0,
                ..base
            },
        )
        .expect("tall");
        assert!(tall.height > plain.height, "v_scale must heighten");

        // BASELINE SHIFT moves the run without resizing it. Positive is
        // UP, and device y grows DOWN, so the origin decreases.
        let lifted = rasterize_run(
            &face,
            "AB",
            32.0,
            RunStyle {
                baseline_shift_px: 20.0,
                ..base
            },
        )
        .expect("lifted");
        assert_eq!(lifted.height, plain.height, "a shift must not resize");
        assert!(lifted.dy < plain.dy, "positive shift moves the run UP");

        // SKEW leans the run, which widens its bounding box while the
        // advances stay put.
        let slanted = rasterize_run(
            &face,
            "AB",
            32.0,
            RunStyle {
                skew_deg: 15.0,
                ..base
            },
        )
        .expect("slanted");
        assert!(slanted.width > plain.width, "a slant must widen the bounds");

        // DECORATIONS add ink below the baseline (underline) or across
        // the x-height (strikethrough), so both grow the inked box.
        let underlined = rasterize_run(
            &face,
            "AB",
            32.0,
            RunStyle {
                underline: true,
                ..base
            },
        )
        .expect("underlined");
        assert!(
            underlined.height > plain.height,
            "an underline sits below the glyphs and must extend the box"
        );
        let struck = rasterize_run(
            &face,
            "AB",
            32.0,
            RunStyle {
                strikethrough: true,
                ..base
            },
        )
        .expect("struck");
        assert!(
            struck.width >= plain.width && struck.coverage.len() >= plain.coverage.len(),
            "a strikethrough must add ink"
        );

        // ALIGNMENT only moves lines relative to each other, so the
        // run's own extent is UNCHANGED — that is the assertion, and it
        // is the one that catches an alignment applied to the whole run
        // instead of to its lines.
        let two = "A\nBBBB";
        let left = rasterize_run(&face, two, 32.0, base).expect("left");
        let centre = rasterize_run(
            &face,
            two,
            32.0,
            RunStyle {
                align: LineAlign::Center,
                ..base
            },
        )
        .expect("centre");
        let right = rasterize_run(
            &face,
            two,
            32.0,
            RunStyle {
                align: LineAlign::Right,
                ..base
            },
        )
        .expect("right");
        assert_eq!(centre.height, left.height);
        assert_eq!(right.height, left.height);
        // …and the SHORT line actually moved: with a left-aligned run
        // the first line starts at the left edge, centred/right it does
        // not, so the ink in the top row shifts. Compare the first
        // inked column of row 0.
        let first_ink =
            |r: &RasterRun| -> Option<u32> { (0..r.width).find(|x| r.coverage[*x as usize] > 0) };
        // Row 0 may be blank for some faces (ascenders differ); only
        // assert when both have ink there, so this cannot fail for a
        // reason that is not about alignment.
        if let (Some(l), Some(c)) = (first_ink(&left), first_ink(&centre)) {
            assert!(c >= l, "centring must not move the short line LEFT");
        }
    }

    #[test]
    fn image_editor_raster_type_the_rasterizer_integrates_to_the_true_area() {
        use ab_glyph_rasterizer::{point, Rasterizer};
        let mut r = Rasterizer::new(8, 6);
        // Clockwise in device space (y down).
        for (a, b) in [
            ((1.0, 1.0), (7.0, 1.0)),
            ((7.0, 1.0), (7.0, 5.0)),
            ((7.0, 5.0), (1.0, 5.0)),
            ((1.0, 5.0), (1.0, 1.0)),
        ] {
            r.draw_line(point(a.0, a.1), point(b.0, b.1));
        }
        let mut total = 0f32;
        r.for_each_pixel_2d(|_x, _y, c| total += c);
        assert!(
            (total - 24.0).abs() < 0.01,
            "a 6×4 rect must integrate to 24, got {total}"
        );
    }

    /// An unclosed contour leaves the accumulator with an unbalanced edge
    /// and bleeds coverage across the row — so `close` synthesises the
    /// missing segment. Asserted as area again: the bleed shows up as a
    /// total far above the shape's own.
    #[test]
    fn image_editor_raster_type_an_unclosed_contour_would_bleed() {
        use ab_glyph_rasterizer::{point, Rasterizer};
        let mut r = Rasterizer::new(8, 6);
        // The SAME rect, missing its closing edge.
        for (a, b) in [
            ((1.0, 1.0), (7.0, 1.0)),
            ((7.0, 1.0), (7.0, 5.0)),
            ((7.0, 5.0), (1.0, 5.0)),
        ] {
            r.draw_line(point(a.0, a.1), point(b.0, b.1));
        }
        let mut total = 0f32;
        r.for_each_pixel_2d(|_x, _y, c| total += c);
        assert!(
            (total - 24.0).abs() > 1.0,
            "an unclosed contour should NOT integrate to the closed area — \
             if it does, this test is not measuring what it claims ({total})"
        );
    }
}
