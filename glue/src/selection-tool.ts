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

// The selection TOOLS' gestures — the thin host adapters that turn canvas
// pointer events into the selection machine's image-px points and render
// the marching-ants preview through the LIVE host.overlay door (the
// crop-tool architecture, one gesture factory per tool kind). The
// geometry mapping is the SAME aspect-fit transform the crop tool and
// the session's Apply use (page-local pt ↔ image px over the composited
// frame's content box); the selection itself lives engine-side.
//
// MODIFIERS (the tool convention, read from CanvasPointerEvent.modifiers
// — the host gesture surface DOES carry them): shift = add, alt =
// subtract, shift+alt = intersect, none = replace.
//
// MARCHING ANTS (v0 fidelity, stated honestly): the outline renders as a
// DASHED ToolPreviewPath (the host's dashed preview vocabulary) — a
// static dashed outline, not an animated crawl; and the committed
// outline is the coverage BOUNDS box (or the live gesture's own shape),
// not a traced mask contour. Both refinements are overlay-only
// follow-ups; the engine coverage is already exact.

import type {
  BundleHost,
  CanvasPointerEvent,
  GestureHandler,
  PagedEditor,
  ToolPreviewPath,
} from "@paged-media/plugin-api";

import type { ImageSession } from "./session";
import type { SelectionShapeKind } from "./selection-machine";
import { modeFromModifiers } from "./selection-machine";
import {
  imageToPage as toPage,
  pageToImage as toImage,
  resolveFrameFit,
  type FitTransform,
} from "./frame-fit";

/** Wrap a closed image-px polyline as a DASHED ToolPreviewPath (straight
 *  segments: handles collapse onto the anchors) — the marching-ants
 *  overlay signal. */
function dashedPath(
  pageId: string,
  points: Array<[number, number]>,
): ToolPreviewPath {
  return {
    pageId,
    anchors: points.map((p) => ({ anchor: p, left: p, right: p })),
    close: true,
    dashed: true,
  };
}

/** One gesture factory for all four selection tools; `kind` picks the
 *  shape ("wand" commits on click instead of drag). */
/** Screen-space radius (CSS px) within which a click on the polygon's
 *  FIRST vertex closes it. Converted to image px per gesture through
 *  the fit scale — a fixed image-px radius is unhittable zoomed out. */
const CLOSE_TOLERANCE_PX = 8;

export function makeSelectionGesture(
  host: BundleHost,
  session: ImageSession,
  kind: SelectionShapeKind | "wand",
): GestureHandler {
  let fit: FitTransform | null = null;

  const imageToPage = (p: [number, number]): [number, number] => toPage(fit, p);
  const pageToImage = (p: [number, number]): [number, number] =>
    toImage(fit, p);

  /** Publish the marching-ants preview: the live gesture outline while
   *  dragging, else the committed selection outline; null clears. */
  const renderOverlay = () => {
    const machine = session.selectionMachine();
    if (!machine || !fit) {
      host.overlay.setToolPreview(null);
      return;
    }
    const outline = machine.gestureOutline() ?? machine.committedOutline();
    if (!outline || outline.length < 2) {
      host.overlay.setToolPreview(null);
      return;
    }
    host.overlay.setToolPreview(
      dashedPath(fit.pageId, outline.map(imageToPage)),
    );
  };

  /** Resolve the composited frame's fit transform (async; cached until
   *  deactivate — the crop tool's pattern). */
  const ensureFit = async () => {
    fit = await resolveFrameFit(host, session.state().source, "selection tool");
    renderOverlay();
  };

  return {
    onActivate(_paged: PagedEditor) {
      void ensureFit();
    },
    onDeactivate() {
      host.overlay.setToolPreview(null);
      fit = null;
    },
    onPointerDown(e: CanvasPointerEvent) {
      const machine = session.selectionMachine();
      if (!machine || !e.pagePoint || !fit) return;
      const mode = modeFromModifiers(e.modifiers);
      const point = pageToImage(e.pagePoint);
      if (kind === "wand") {
        // Click tool: commit immediately (tolerance = the documented v0
        // default; see selection-machine.ts).
        if (machine.wand(point, mode)) session.refreshSelection();
      } else if (kind === "polygon") {
        // POLYGONAL LASSO is click-driven, not drag-driven: each press
        // places a vertex and the gesture spans many of them.
        //
        // CLOSING is by clicking the FIRST vertex again, not by double-
        // click. `CanvasPointerEvent` carries no `detail`, so a double
        // click would have to be reconstructed from timestamps and
        // proximity — a heuristic with a tuning constant that is wrong
        // for somebody. Click-the-start is the other standard polygon
        // close, it is unambiguous, and it needs no timing at all.
        // Enter closes too (see onKey), which covers the case where the
        // first vertex is off screen.
        const first = machine.polygonFirstVertex();
        if (first && machine.placingPolygon()) {
          const dx = point[0] - first[0];
          const dy = point[1] - first[1];
          // Tolerance in IMAGE px scaled by the view: a fixed px radius
          // would be unhittable when zoomed out and huge when zoomed in.
          const tol = CLOSE_TOLERANCE_PX / Math.max(fit.scale, 1e-6);
          if (dx * dx + dy * dy <= tol * tol) {
            if (machine.polygonCommit()) session.refreshSelection();
            renderOverlay();
            return;
          }
        }
        machine.polygonVertex(point, mode);
      } else {
        machine.begin(kind, point, mode);
      }
      renderOverlay();
    },
    onKey(e: KeyboardEvent) {
      const machine = session.selectionMachine();
      if (!machine || kind !== "polygon" || !machine.placingPolygon()) return;
      if (e.key === "Enter") {
        if (machine.polygonCommit()) session.refreshSelection();
      } else if (e.key === "Backspace" || e.key === "Delete") {
        // Undo the last vertex. Consuming the key ONLY while a polygon
        // is in progress (the guard above) is what leaves Backspace
        // meaning what it usually means the rest of the time.
        machine.polygonUndoVertex();
      } else if (e.key === "Escape") {
        machine.polygonCancel();
      } else {
        return;
      }
      e.preventDefault();
      renderOverlay();
    },
    onPointerMove(e: CanvasPointerEvent) {
      const machine = session.selectionMachine();
      if (!machine || !e.pagePoint || kind === "wand") return;
      // A polygon does not track the pointer between clicks — its
      // outline is the vertices placed so far, and rubber-banding to
      // the cursor would need a preview vertex the machine would then
      // have to distinguish from a real one.
      if (kind === "polygon") return;
      if (!machine.dragging()) return;
      machine.update(pageToImage(e.pagePoint));
      renderOverlay();
    },
    onPointerUp() {
      const machine = session.selectionMachine();
      if (kind === "polygon") return; // ends at commit/cancel, not here
      if (machine && kind !== "wand" && machine.dragging()) {
        if (machine.end()) session.refreshSelection();
      }
      renderOverlay();
    },
  };
}
