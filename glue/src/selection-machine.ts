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

export type SelectionShapeKind = "rect" | "ellipse" | "lasso";

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
  begin(kind: SelectionShapeKind, point: [number, number], mode: SelectionMode): void;
  /** Extend the active gesture (drag corner / append lasso vertex). */
  update(point: [number, number]): void;
  /** Commit the active gesture into the engine selection (lasso closes
   *  last→first). Returns true when a shape was actually applied. */
  end(): boolean;
  /** One-click magic wand commit at `point` (image px, rounded). */
  wand(point: [number, number], mode: SelectionMode): boolean;
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
            engine.selectionSetEllipse(x + w / 2, y + h / 2, w / 2, h / 2, mode);
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
      if (active.kind === "lasso") {
        return active.trail.length >= 2 ? [...active.trail] : null;
      }
      if (active.anchor[0] === active.current[0] && active.anchor[1] === active.current[1]) {
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
export function modeFromModifiers(mods: { shift: boolean; alt: boolean }): SelectionMode {
  if (mods.shift && mods.alt) return "intersect";
  if (mods.shift) return "add";
  if (mods.alt) return "subtract";
  return "replace";
}
