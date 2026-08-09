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

// The selection INTERACTION machine — host-agnostic, thin over the
// engine's selection doors (the crop-machine pattern). It holds ONLY the
// in-flight gesture (the drag anchor / lasso point trail); the selection
// itself is engine state (image-js `selection_*`: a u8 coverage field,
// combined engine-side under replace/add/subtract/intersect). The caller
// (the selection tools' gestures) drives pointer points ALREADY mapped to
// image-pixel space and renders `overlayOutline()` through
// `host.overlay.setToolPreview`; `end()` / `wand()` commit the shape into
// the engine selection.
//
// Marching-ants fidelity (v0, stated honestly): the committed-selection
// outline is the coverage BOUNDS rectangle (or the lasso's own polygon
// while drawing) rendered as a dashed preview path — a coarse outline,
// not a traced mask contour. A true iso-contour tracer over the coverage
// bytes is a follow-up; the overlay door consumes the same polyline
// either way.

import type { ImageEngine, SelectionMode, SelectionStats } from "./engine";

export type SelectionShapeKind = "rect" | "ellipse" | "lasso" | "polygon";

/** Minimum drag extent (image px) below which a marquee is treated as a
 *  no-op click (avoids committing invisible slivers). */
const MIN_DRAG_PX = 1;

/** The default magic-wand tolerance (of 255, per channel): 32 — the v0
 *  fixed default (documented on the wand tool; a tool-option slider is a
 *  follow-up). */
export const WAND_TOLERANCE_DEFAULT = 32;

/** The wand's v0 contiguity: connected flood (the Photoshop default).
 *  The non-contiguous global threshold is reachable through the engine
 *  door; no tool toggle yet (stated in the tool docs). */
export const WAND_CONTIGUOUS_DEFAULT = true;

export interface SelectionMachine {
  /** Begin a marquee/lasso gesture at `point` (image px). */
  begin(
    kind: SelectionShapeKind,
    point: [number, number],
    mode: SelectionMode,
  ): void;
  /** Extend the active gesture (drag corner / append lasso vertex). */
  update(point: [number, number]): void;
  /** Commit the active gesture into the engine selection (lasso closes
   *  last→first). Returns true when a shape was actually applied. */
  end(): boolean;
  /** One-click magic wand commit at `point` (image px, rounded). */
  wand(point: [number, number], mode: SelectionMode): boolean;
  /** POLYGONAL LASSO — place a vertex. The first call starts the
   *  gesture; later ones append. Unlike the freehand lasso this is
   *  CLICK-driven, so the gesture spans many clicks and only ends when
   *  the caller commits or cancels. */
  polygonVertex(
    point: [number, number],
    mode: SelectionMode,
  ): void;
  /** Remove the last placed vertex (backspace / delete). Returns false
   *  when there was nothing to remove, so a caller can distinguish
   *  "undid a vertex" from "the gesture is already empty" — which is
   *  what lets backspace on an empty polygon fall through to whatever
   *  else backspace means. */
  polygonUndoVertex(): boolean;
  /** Close and commit the polygon (enter / double-click). Returns false
   *  for fewer than three vertices — two points bound no area, and
   *  committing an empty selection would silently deselect. */
  polygonCommit(): boolean;
  /** Abandon the polygon without committing (Esc). */
  polygonCancel(): void;
  /** True while a polygon is being placed, so a tool can route keys it
   *  would otherwise ignore. */
  placingPolygon(): boolean;
  /** The polygon's FIRST vertex, so a tool can offer click-to-close
   *  without keeping its own copy of the trail — two copies of the same
   *  list is how they drift. Null when no polygon is in progress. */
  polygonFirstVertex(): [number, number] | null;
  dragging(): boolean;
  /** The IN-GESTURE outline (image-px polyline, closed): the live
   *  marquee rect / ellipse approximation / lasso trail. Null when idle. */
  gestureOutline(): Array<[number, number]> | null;
  /** The COMMITTED selection outline: the coverage bounds rectangle
   *  (coarse v0 — see the header note). Null when no selection. */
  committedOutline(): Array<[number, number]> | null;
  /** The engine's selection readout (null = no explicit selection). */
  stats(): SelectionStats | null;
}

/** Ellipse approximation vertex count for the overlay polyline. */
const ELLIPSE_SEGMENTS = 32;

export function createSelectionMachine(
  engine: ImageEngine,
  /** Called after every committed change (the session refreshes its
   *  readout + notifies the panel). */
  onCommit: () => void = () => {},
): SelectionMachine {
  let active: {
    kind: SelectionShapeKind;
    mode: SelectionMode;
    anchor: [number, number];
    current: [number, number];
    /** Lasso trail (kind === "lasso" only), anchor included. */
    trail: Array<[number, number]>;
  } | null = null;

  const rectCorners = (
    a: [number, number],
    b: [number, number],
  ): Array<[number, number]> => {
    const x0 = Math.min(a[0], b[0]);
    const y0 = Math.min(a[1], b[1]);
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    return [
      [x0, y0],
      [x1, y0],
      [x1, y1],
      [x0, y1],
    ];
  };

  const ellipsePoints = (
    a: [number, number],
    b: [number, number],
  ): Array<[number, number]> => {
    const cx = (a[0] + b[0]) / 2;
    const cy = (a[1] + b[1]) / 2;
    const rx = Math.abs(b[0] - a[0]) / 2;
    const ry = Math.abs(b[1] - a[1]) / 2;
    const pts: Array<[number, number]> = [];
    for (let i = 0; i < ELLIPSE_SEGMENTS; i++) {
      const t = (i / ELLIPSE_SEGMENTS) * Math.PI * 2;
      pts.push([cx + rx * Math.cos(t), cy + ry * Math.sin(t)]);
    }
    return pts;
  };

  return {
    begin(kind, point, mode) {
      active = {
        kind,
        mode,
        anchor: point,
        current: point,
        trail: [point],
      };
    },

    update(point) {
      if (!active) return;
      active.current = point;
      if (active.kind === "lasso") active.trail.push(point);
    },

    end() {
      if (!active) return false;
      // A POLYGON does not end on pointer-up — it spans clicks and ends
      // only at `polygonCommit` / `polygonCancel`. Returning early here
      // is what keeps a click from committing a one-vertex shape.
      if (active.kind === "polygon") return false;
      const { kind, mode, anchor, current, trail } = active;
      active = null;
      try {
        if (kind === "lasso") {
          // Close on release (last → first is implicit engine-side).
          if (trail.length < 3) return false;
          engine.selectionSetPolygon(trail, mode);
        } else {
          const x = Math.min(anchor[0], current[0]);
          const y = Math.min(anchor[1], current[1]);
          const w = Math.abs(current[0] - anchor[0]);
          const h = Math.abs(current[1] - anchor[1]);
          if (w < MIN_DRAG_PX || h < MIN_DRAG_PX) return false;
          if (kind === "rect") {
            engine.selectionSetRect(x, y, w, h, mode);
          } else {
            engine.selectionSetEllipse(
              x + w / 2,
              y + h / 2,
              w / 2,
              h / 2,
              mode,
            );
          }
        }
      } catch {
        // Engine rejection (nothing bound, degenerate shape) — an honest
        // no-op; the session status stays authoritative.
        return false;
      }
      onCommit();
      return true;
    },

    polygonVertex(point, mode) {
      if (!active || active.kind !== "polygon") {
        active = {
          kind: "polygon",
          mode,
          anchor: point,
          current: point,
          trail: [point],
        };
        return;
      }
      // The MODE is taken from the FIRST vertex, not the last: a
      // designer holding shift to add, then releasing it to place the
      // remaining vertices, means one additive selection — not a
      // gesture whose meaning flips halfway through.
      active.current = point;
      active.trail.push(point);
    },

    polygonUndoVertex() {
      if (!active || active.kind !== "polygon" || active.trail.length === 0) {
        return false;
      }
      active.trail.pop();
      if (active.trail.length === 0) {
        active = null;
        return true;
      }
      active.current = active.trail[active.trail.length - 1];
      return true;
    },

    polygonCommit() {
      if (!active || active.kind !== "polygon") return false;
      const { mode, trail } = active;
      active = null;
      // Three is the floor: two points bound no area, and committing
      // that would REPLACE the selection with nothing — a silent
      // deselect the user did not ask for.
      if (trail.length < 3) return false;
      try {
        engine.selectionSetPolygon(trail, mode);
      } catch {
        return false;
      }
      onCommit();
      return true;
    },

    polygonCancel() {
      if (active && active.kind === "polygon") active = null;
    },

    placingPolygon: () => active !== null && active.kind === "polygon",

    polygonFirstVertex: () =>
      active && active.kind === "polygon" && active.trail.length > 0
        ? active.trail[0]
        : null,

    wand(point, mode) {
      try {
        engine.selectionMagicWand(
          Math.max(0, Math.round(point[0])),
          Math.max(0, Math.round(point[1])),
          WAND_TOLERANCE_DEFAULT,
          WAND_CONTIGUOUS_DEFAULT,
          mode,
        );
      } catch {
        return false;
      }
      onCommit();
      return true;
    },

    dragging: () => active !== null,

    gestureOutline() {
      if (!active) return null;
      if (active.kind === "lasso" || active.kind === "polygon") {
        return active.trail.length >= 2 ? [...active.trail] : null;
      }
      if (
        active.anchor[0] === active.current[0] &&
        active.anchor[1] === active.current[1]
      ) {
        return null;
      }
      return active.kind === "rect"
        ? rectCorners(active.anchor, active.current)
        : ellipsePoints(active.anchor, active.current);
    },

    committedOutline() {
      const s = engine.selectionStats();
      if (!s || s.w <= 0 || s.h <= 0) return null;
      // v0 coarse outline: the coverage bounding box (see header note).
      return [
        [s.x, s.y],
        [s.x + s.w, s.y],
        [s.x + s.w, s.y + s.h],
        [s.x, s.y + s.h],
      ];
    },

    stats: () => engine.selectionStats(),
  };
}

/** Map gesture modifiers to the combine mode — the tool convention
 *  (documented on every selection tool): shift = add, alt = subtract,
 *  shift+alt = intersect, none = replace. Exported for the tools AND the
 *  tests (the one place the convention lives). */
export function modeFromModifiers(mods: {
  shift: boolean;
  alt: boolean;
}): SelectionMode {
  if (mods.shift && mods.alt) return "intersect";
  if (mods.shift) return "add";
  if (mods.alt) return "subtract";
  return "replace";
}
