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
      } else {
        machine.begin(kind, point, mode);
      }
      renderOverlay();
    },
    onPointerMove(e: CanvasPointerEvent) {
      const machine = session.selectionMachine();
      if (!machine || !e.pagePoint || kind === "wand") return;
      if (!machine.dragging()) return;
      machine.update(pageToImage(e.pagePoint));
      renderOverlay();
    },
    onPointerUp() {
      const machine = session.selectionMachine();
      if (machine && kind !== "wand" && machine.dragging()) {
        if (machine.end()) session.refreshSelection();
      }
      renderOverlay();
    },
  };
}
