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

// ADR-023 phase D — paged.image answers the HOST's Character panel for
// RASTER TYPE, while the `rasterImage` context is active and the type
// tool is the one in hand.
//
// WHY THIS ONE MATTERS BEYOND paged.image: it is the first provider
// anywhere with a NON-EMPTY `writablePaths`.
//
//   paged.sheet proved the read side of the VALUE axis and declared
//   `writablePaths: []` — honestly, because the sheet engine has no
//   cell-style write API at all. So the host's property-WRITE lane
//   (`writeSelectionProperty` / `useSelectionPathWritable` in
//   packages/shell/src/catalog/binding-providers.tsx) has been shipped
//   and never executed against a real provider.
//
//   Raster type's settings ARE writable — they are ordinary session
//   state the panel already mutates — so this provider runs that code
//   for the first time. Any defect in it surfaces here or nowhere.
//
// WHAT IT SERVES: the ONE character path that means the same thing
// in both worlds — the font family.
// Everything else on the host Character and Paragraph panels is
// declared and answered `absent`, NOT declined — the distinction is the
// whole point of the seam. A decline means "not mine, ask core", and
// core would then answer with whatever text caret the user last touched
// somewhere else in the document, displaying a paragraph's leading over
// a raster frame. `absent` means "mine, and it has no such value", which
// the host renders as a blank control.
//
// WHAT IT DECLINES: everything, when the type tool is not active. That
// is not a hedge — outside the type tool there is no run being set, so
// there is no value to claim, and core (an honestly empty panel inside a
// raster frame with no text selection) is the truthful answer.

import type {
  BindingProvider,
  BindingRead,
  BindingWrite,
  PropertyPath,
  Value,
} from "@paged-media/plugin-api";

import type { ImageSession } from "../session";

/**
 * The paths a shaped glyph run genuinely has, in CORE's meaning of
 * them. Exactly one, and the reason the list is that short is the most
 * interesting thing in this file — see `characterFontSize` below.
 */
export const VALUED_PATHS: readonly PropertyPath[] = [
  "characterFontFamily",
  // C-25-adjacent, and the resolution of THE UNIT QUESTION this file
  // used to leave open (2026-08-09). The reading taken: this field
  // means "the size unit of whatever you are editing", surfaced in the
  // panel's own unit — POINTS — with the plugin doing the conversion.
  //
  // Why that reading and not "points, absolutely": raster type IS
  // typography, and a Character panel that goes blank over it is the
  // context-sensitivity failure ADR 024 exists to prevent. The
  // conversion is honest because the number it needs is real and
  // already in production — `submitLayer` computes points-per-image-
  // pixel to lay the composite out, and `state.ptPerPx` is that exact
  // value, cached at the point of use so panel and pixels cannot
  // disagree.
  //
  // NO CONTRACT CHANGE. The seam does not grow a declared unit per
  // path — the third option in the old note — because the governing
  // rule puts every other content type in the document-units family,
  // so that axis would have a population of one. If a second
  // intrinsic-space content type ever appears, revisit; until then a
  // per-path unit is a generalisation with one instance.
  //
  // FIT-DEPENDENCE IS THE FEATURE. Scaling the frame changes the type's
  // physical size, so the reported points change with it. That is a
  // fact about placed rasters — Photoshop models exactly it — not an
  // artifact of the conversion.
  "characterFontSize",
  // TRACKING (2026-08-09, with multi-line). The one setting on this
  // panel that needs NO unit bridge: `characterTracking` binds to a
  // bare numeric scrub and IDML measures it in 1/1000 em, which is
  // relative to the size — so the number means the same thing whether
  // the caller thinks in image pixels or points. It crosses unchanged.
  "characterTracking",
  // LEADING (2026-08-09). Converts exactly like the size, through
  // `ptPerPx`, because it binds to the same length widget.
  //
  // It is only meaningful now that the type lane lays out MORE THAN ONE
  // LINE — before that it would have been a control with nothing to
  // act on, which is the kind of live-looking dead surface the brand
  // honesty rule forbids.
  "characterLeading",
  // STYLE (2026-08-09). The provider's own note used to park this on
  // "when the face lookup grows a style axis" — and the axis was
  // already there: `AssetSurface.getFontFace(family, style?)` has taken
  // an optional style all along, and the adapter, the editor's asset
  // source and the `requestFontFaceBytes` wire payload all carry it.
  // Nothing needed adding anywhere; the type lane simply never asked.
  //
  // What this serves is the REQUEST, not the resolution — the host owns
  // face matching, so a request for Bold against a document that embeds
  // no bold face resolves to what it has, and the session says so in
  // its status line rather than letting the drift pass silently.
  "characterFontStyle",
  // THE TRANSFORM AND DECORATION AXES (2026-08-09). Each is a real
  // property of a rasterized run, and each needed only geometry the
  // shaper already had: a scale, a shear, an offset, a rule from the
  // face's own metrics, a shaper feature toggle.
  "characterBaselineShift",
  "characterHorizontalScale",
  "characterVerticalScale",
  "characterSkew",
  "characterUnderline",
  "characterStrikethru",
  "characterLigatures",
  // POSITION is superscript/subscript, and it is a DERIVED setting —
  // a baseline shift plus a scale, not a mode of its own. Serving it
  // means the panel's control works; the two axes it moves stay
  // independently settable, which is what a designer wants when the
  // preset is not quite right.
  "characterPosition",
  // ALIGNMENT is meaningful ONLY because the lane lays out more than
  // one line: lines align against the WIDEST line, which is the run's
  // own extent. There is no column to align against — a raster layer
  // is not a text column, the same boundary that keeps wrapping out.
  "paragraphJustification",
  // CASE is a string transform, applied before shaping. Upper/lower
  // only — SMALL CAPS is a font feature (`smcp`) and a face without it
  // would silently render full caps, so it is refused on write rather
  // than faked by scaling capitals.
  "characterCase",
];

/**
 * Paths this provider OWNS and has nothing to say about.
 *
 * Every one of these is real on a core story run and meaningless on a
 * single rasterized run: there is no second line to lead, no paragraph
 * to align, no style to inherit from. Owning them is what stops the
 * host asking core and painting a foreign answer into a panel the user
 * believes is describing their type.
 *
 * `characterFontStyle` sits here rather than in VALUED_PATHS on
 * purpose. The type lane takes a FAMILY and asks the host for that
 * face; it never resolves a style axis (Bold / Italic) of its own, so
 * claiming to know one would be inventing a value. When the face lookup
 * grows a style axis this row moves up and nothing else changes.
 */
export const ABSENT_PATHS: readonly PropertyPath[] = [
  // `characterFontSize` MOVED to VALUED_PATHS on 2026-08-09 — the
  // product question that kept it here was answered. See the note
  // there for the reading taken and why it needed no contract change.
  //
  // Every path below is real on a core story run and meaningless on a
  // SINGLE rasterized run: there is no second line to lead, no
  // paragraph to align, no style to inherit. Several (tracking, case,
  // underline) would become real if the type lane grew past one run on
  // one line — they are absent because the FEATURE is absent, not
  // because of any unit or binding problem.
  "characterKerningMethod",
  "paragraphLeftIndent",
  "paragraphRightIndent",
  "paragraphFirstLineIndent",
  "paragraphSpaceBefore",
  "paragraphSpaceAfter",
  "paragraphHyphenation",
  "paragraphKeepLinesTogether",
];

export const SERVED_PATHS: readonly PropertyPath[] = [
  ...VALUED_PATHS,
  ...ABSENT_PATHS,
];

/** The tool that makes this provider's answers meaningful. */
const TYPE_TOOL_ID = "media.paged.image.tool.type";

export interface TextBindingProvider {
  provider: BindingProvider;
}

export function makeTextBindingProvider(
  session: ImageSession,
  activeToolId: () => string | null,
): TextBindingProvider {
  const typeToolInHand = () => activeToolId() === TYPE_TOOL_ID;

  const provider: BindingProvider = {
    provides: {
      paths: SERVED_PATHS,
      // THE FIRST NON-EMPTY writablePaths ANYWHERE. Only the path that
      // has a value is writable; the `absent` ones are owned and
      // unwritable, so the host renders them read-only instead of
      // offering a control whose commit would land nowhere.
      writablePaths: VALUED_PATHS,
    },

    readProperty(request): BindingRead {
      if (request.target.kind !== "selection") {
        // The type run is a SELECTION-scoped notion in this plugin's
        // own realm — an element- or row-scoped read belongs to whoever
        // owns that addressing.
        return { kind: "decline", reason: "selection-scoped provider" };
      }
      if (!typeToolInHand()) {
        return {
          kind: "decline",
          reason: "the type tool is not active — no run is being set",
        };
      }
      if (!session.state().source) {
        return { kind: "decline", reason: "no raster image is ingested" };
      }
      const st = session.state();
      const t = st.type;
      switch (request.path) {
        case "characterFontFamily":
          return { kind: "value", value: t.family as unknown as Value };
        case "characterFontSize": {
          const ptPerPx = st.ptPerPx;
          if (ptPerPx === null || !(ptPerPx > 0)) {
            // Nothing has been composited, so there is no layout to
            // convert against. ABSENT, not a guess at 1:1 — a wrong
            // number in a size field is worse than a blank one, because
            // the user cannot tell it is wrong.
            return {
              kind: "absent",
              reason:
                "no composite yet — points per pixel is unknown until the image is laid out",
            };
          }
          return {
            kind: "value",
            value: (t.sizePx * ptPerPx) as unknown as Value,
          };
        }
        case "characterFontStyle":
          // Null is UNSET — the family's default face — and reads as
          // absent rather than as a guessed "Regular". The plugin has
          // not resolved anything yet, and naming a face it did not
          // choose would be inventing the value this provider exists to
          // avoid inventing.
          return t.style === null
            ? { kind: "absent", reason: "no style set — the family's default face" }
            : { kind: "value", value: t.style as unknown as Value };
        case "characterHorizontalScale":
          return { kind: "value", value: t.hScalePct as unknown as Value };
        case "characterVerticalScale":
          return { kind: "value", value: t.vScalePct as unknown as Value };
        case "characterSkew":
          return { kind: "value", value: t.skewDeg as unknown as Value };
        case "characterUnderline":
          return { kind: "value", value: t.underline as unknown as Value };
        case "characterStrikethru":
          return { kind: "value", value: t.strikethrough as unknown as Value };
        case "characterLigatures":
          return { kind: "value", value: t.ligatures as unknown as Value };
        case "characterCase":
          return { kind: "value", value: t.textCase as unknown as Value };
        case "paragraphJustification":
          return { kind: "value", value: t.align as unknown as Value };
        case "characterBaselineShift": {
          // A LENGTH, so it converts like size and leading.
          const ptPerPx = st.ptPerPx;
          if (ptPerPx === null || !(ptPerPx > 0)) {
            return {
              kind: "absent",
              reason: "no composite yet — points per pixel is unknown",
            };
          }
          return {
            kind: "value",
            value: (t.baselineShiftPx * ptPerPx) as unknown as Value,
          };
        }
        case "characterPosition":
          // DERIVED, and derived on the way OUT as well as in: the
          // preset is whatever the two underlying axes currently say,
          // so a designer who nudges the shift by hand sees the
          // position control fall back to "normal" rather than keep
          // claiming a preset that no longer describes the run.
          return {
            kind: "value",
            value: (t.baselineShiftPx > 0 && t.vScalePct < 100
              ? "superscript"
              : t.baselineShiftPx < 0 && t.vScalePct < 100
                ? "subscript"
                : "normal") as unknown as Value,
          };
        case "characterTracking":
          // No conversion, and that is the point — 1/1000 em is
          // size-relative, so it is already the panel's number.
          return {
            kind: "value",
            value: t.trackingPerMille as unknown as Value,
          };
        case "characterLeading": {
          if (t.leadingPx === null) {
            // AUTO. The face decides, and the plugin does not know the
            // resolved number without shaping — so `absent` (a blank
            // control) rather than a computed value the user did not
            // set and cannot round-trip.
            return {
              kind: "absent",
              reason: "leading is AUTO — the face's own line height",
            };
          }
          const ptPerPx = st.ptPerPx;
          if (ptPerPx === null || !(ptPerPx > 0)) {
            return {
              kind: "absent",
              reason: "no composite yet — points per pixel is unknown",
            };
          }
          return {
            kind: "value",
            value: (t.leadingPx * ptPerPx) as unknown as Value,
          };
        }
        default:
          if (ABSENT_PATHS.includes(request.path)) {
            return {
              kind: "absent",
              reason: `a rasterized glyph run has no "${request.path}"`,
            };
          }
          return { kind: "decline", reason: "not a path this provider owns" };
      }
    },

    writeProperty(request): BindingWrite {
      if (request.target.kind !== "selection") {
        return { kind: "decline", reason: "selection-scoped provider" };
      }
      if (!typeToolInHand()) {
        return {
          kind: "decline",
          reason: "the type tool is not active — nothing to write to",
        };
      }
      // Re-checked here and not assumed from the read. A panel can hold
      // a stale value across a deselect, and committing it would set
      // type settings for an image that is no longer open.
      if (!session.state().source) {
        return { kind: "decline", reason: "no raster image is ingested" };
      }
      switch (request.path) {
        case "characterFontFamily": {
          const family = String(request.value ?? "").trim();
          if (!family) {
            // A refusal is a resolved value. An empty family would make
            // the next paint ask the host for a face called "" and get
            // the honest "no such face" — a failure one step removed
            // from its cause, which is the worst place to put one.
            return { kind: "decline", reason: "a face name cannot be empty" };
          }
          session.setType({ family });
          return ok();
        }
        case "characterFontSize": {
          const pt = Number(request.value);
          if (!Number.isFinite(pt) || pt <= 0) {
            return { kind: "decline", reason: "a type size must be positive" };
          }
          const ptPerPx = session.state().ptPerPx;
          if (ptPerPx === null || !(ptPerPx > 0)) {
            // Symmetric with the read: without a layout there is no
            // conversion, and writing an unconverted point count as a
            // pixel count would silently resize type by the frame's
            // scale factor.
            return {
              kind: "decline",
              reason:
                "no composite yet — points per pixel is unknown until the image is laid out",
            };
          }
          // Round, because sizePx feeds a rasterizer that will floor it
          // anyway; rounding here keeps the value the panel reads back
          // equal to the one it committed, within a pixel.
          session.setType({ sizePx: Math.max(1, Math.round(pt / ptPerPx)) });
          return ok();
        }
        case "characterFontStyle": {
          const style = String(request.value ?? "").trim();
          // An emptied field means "the family's default face", which is
          // the only way back once a style has been typed — same shape
          // as clearing leading back to AUTO.
          session.setType({ style: style === "" ? null : style });
          return ok();
        }
        case "characterHorizontalScale":
        case "characterVerticalScale": {
          const pct = Number(request.value);
          // Zero or negative would rasterize nothing, which reads as a
          // broken renderer rather than a rejected input.
          if (!Number.isFinite(pct) || pct <= 0) {
            return { kind: "decline", reason: "scale must be a positive percent" };
          }
          session.setType(
            request.path === "characterHorizontalScale"
              ? { hScalePct: pct }
              : { vScalePct: pct },
          );
          return ok();
        }
        case "characterSkew": {
          const deg = Number(request.value);
          if (!Number.isFinite(deg) || Math.abs(deg) >= 90) {
            // At ±90° the shear is infinite and the run degenerates to
            // a line; refusing names the limit instead of rendering
            // nothing.
            return { kind: "decline", reason: "slant must be between -90 and 90" };
          }
          session.setType({ skewDeg: deg });
          return ok();
        }
        case "characterUnderline":
          session.setType({ underline: Boolean(request.value) });
          return ok();
        case "characterStrikethru":
          session.setType({ strikethrough: Boolean(request.value) });
          return ok();
        case "characterLigatures":
          session.setType({ ligatures: Boolean(request.value) });
          return ok();
        case "characterCase": {
          const v = String(request.value ?? "").toLowerCase();
          if (v === "smallcaps" || v === "small caps") {
            // REFUSED, not faked. Small caps is a font FEATURE; a face
            // without `smcp` would silently render full caps, and
            // scaling capitals to imitate it is a forgery a designer
            // would not choose knowingly.
            return {
              kind: "decline",
              reason:
                "small caps needs the face's own smcp feature — scaled capitals would be a forgery",
            };
          }
          if (v !== "none" && v !== "upper" && v !== "lower") {
            return { kind: "decline", reason: `unknown case "${v}"` };
          }
          session.setType({ textCase: v });
          return ok();
        }
        case "paragraphJustification": {
          const v = String(request.value ?? "").toLowerCase();
          if (v !== "left" && v !== "center" && v !== "right") {
            // JUSTIFIED is refused for a structural reason, not an
            // unbuilt one: justification stretches lines to a MEASURE,
            // and a raster run has no measure — it is as wide as its
            // widest line.
            return {
              kind: "decline",
              reason:
                v === "justify"
                  ? "a raster run has no measure to justify against"
                  : `unknown alignment "${v}"`,
            };
          }
          session.setType({ align: v });
          return ok();
        }
        case "characterBaselineShift": {
          const pt = Number(request.value);
          const ptPerPx = session.state().ptPerPx;
          if (ptPerPx === null || !(ptPerPx > 0)) {
            return {
              kind: "decline",
              reason: "no composite yet — points per pixel is unknown",
            };
          }
          if (!Number.isFinite(pt)) {
            return { kind: "decline", reason: "baseline shift must be a number" };
          }
          // NOT clamped to positive: a negative shift is a subscript.
          session.setType({ baselineShiftPx: Math.round(pt / ptPerPx) });
          return ok();
        }
        case "characterPosition": {
          const v = String(request.value ?? "").toLowerCase();
          const st = session.state();
          const em = st.type.sizePx;
          // The presets are the classic ones and they are written to
          // the two REAL axes, so the result stays hand-editable
          // afterwards rather than locking into a mode.
          if (v === "superscript") {
            session.setType({
              baselineShiftPx: Math.round(em * 0.33),
              hScalePct: 58,
              vScalePct: 58,
            });
            return ok();
          }
          if (v === "subscript") {
            session.setType({
              baselineShiftPx: -Math.round(em * 0.12),
              hScalePct: 58,
              vScalePct: 58,
            });
            return ok();
          }
          if (v === "normal" || v === "none") {
            session.setType({ baselineShiftPx: 0, hScalePct: 100, vScalePct: 100 });
            return ok();
          }
          return { kind: "decline", reason: `unknown position "${v}"` };
        }
        case "characterTracking": {
          const per = Number(request.value);
          if (!Number.isFinite(per)) {
            return { kind: "decline", reason: "tracking must be a number" };
          }
          // NEGATIVE IS LEGAL and deliberately not clamped — tightening
          // is half of what tracking is for.
          session.setType({ trackingPerMille: per });
          return ok();
        }
        case "characterLeading": {
          const pt = Number(request.value);
          const ptPerPx = session.state().ptPerPx;
          if (ptPerPx === null || !(ptPerPx > 0)) {
            return {
              kind: "decline",
              reason: "no composite yet — points per pixel is unknown",
            };
          }
          if (!Number.isFinite(pt) || pt <= 0) {
            // A cleared field means AUTO rather than an error: it is the
            // only way back to the face's own line height once a number
            // has been typed.
            session.setType({ leadingPx: null });
            return ok();
          }
          session.setType({ leadingPx: Math.max(1, Math.round(pt / ptPerPx)) });
          return ok();
        }
        default:
          // An `absent` path is OWNED but unwritable, and the host
          // already renders it read-only from `writablePaths`. Reaching
          // here means the host routed a write it declared it would
          // not — decline rather than silently accept, so the
          // contradiction is visible instead of absorbed.
          return {
            kind: "decline",
            reason: `"${request.path}" is not writable on a raster run`,
          };
      }
    },
  };

  return { provider };
}

/** A property write that landed. No element created, no document page
 *  touched — the type settings live in the plugin's session, and
 *  naming a page here would make the host re-render one that did not
 *  change. */
function ok(): BindingWrite {
  return {
    kind: "applied",
    outcome: { applied: true, createdId: null, pageIds: [] },
  };
}
