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
  "characterFontStyle",
  "characterLeading",
  "characterTracking",
  "characterKerningMethod",
  "characterBaselineShift",
  "characterHorizontalScale",
  "characterVerticalScale",
  "characterSkew",
  "characterCase",
  "characterPosition",
  "characterUnderline",
  "characterStrikethru",
  "characterLigatures",
  "paragraphJustification",
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
