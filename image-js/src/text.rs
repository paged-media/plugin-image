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
//! it lays out HARD-BROKEN lines only: a newline starts a line, and
//! leading, tracking and size are settable — but nothing WRAPS, there is
//! no paragraph layout and no styles. The boundary is deliberate and it
//! is not effort: wrapping needs a measure to wrap against, which for a
//! raster layer is the layer, and a layer is not a text column. Those
//! are text-engine features, the host already has a text engine, and
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
}

impl Collector<'_> {
    fn map(&mut self, x: f32, y: f32) -> (f32, f32) {
        let p = (self.ox + x, self.oy - y);
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
#[derive(Debug, Clone, Copy, Default)]
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
    for (line_index, line) in normalised.split('\n').enumerate() {
        // The pen returns to x=0 for every line and steps down by the
        // leading. An EMPTY line still consumes one — a blank line is a
        // deliberate gap, not a no-op.
        let mut pen_x = 0f32;
        let mut pen_y = leading * line_index as f32;
        if line.is_empty() {
            continue;
        }

        let mut buf = harfrust::UnicodeBuffer::new();
        buf.push_str(line);
        buf.guess_segment_properties();
        let shaped = shaper.shape(buf, &[]);

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
            pen_x += pos.x_advance as f32 * scale + track_px;
            pen_y -= pos.y_advance as f32 * scale;
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
            leading_px: None,
        };
        // Still refused for the FONT, not for the tracking — the point
        // is that it reaches the parse rather than dying earlier.
        assert!(rasterize_run(b"not a font", "hi", 24.0, nan_track).is_none());

        // A non-positive leading means AUTO, not a zero-height layout
        // where every line stacks on the first.
        for l in [Some(0.0f32), Some(-10.0), Some(f32::NAN), None] {
            let st = RunStyle {
                tracking_per_mille: 0.0,
                leading_px: l,
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
                tracking_per_mille: 0.0,
                leading_px: Some(120.0),
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
                leading_px: None,
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
