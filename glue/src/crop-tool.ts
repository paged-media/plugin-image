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

// The crop TOOL gesture — the thin host adapter that turns canvas pointer
// events into the crop machine's image-px drags and renders the crop frame
// through the LIVE host.overlay door (setToolPreview). It owns NO geometry:
// the rect/handle/angle math is the machine's (image_core::crop on the
// wasm surface); this file only maps page-local pt ↔ image px against the
// composited frame's content box (the same aspect-fit transform the
// session's Apply uses) and forwards points.
//
// Coordinate mapping. The placed image is aspect-fit + centered inside the
// frame content box (box = the frame's page-local bounds). image-px →
// page-local pt is `box_origin + (centering offset) + image_px * scale`;
// the inverse maps a canvas point back to image px for the machine.

import type {
  BundleHost,
  CanvasPointerEvent,
  GestureHandler,
  PagedEditor,
  ToolPreviewPolyline,
} from "@paged-media/plugin-api";

import type { ImageSession } from "./session";
import {
  imageToPage as toPage,
  pageToImage as toImage,
  resolveFrameFit,
  type FitTransform,
} from "./frame-fit";

/** The crop tool's gesture. Activates over the composited frame: reads its
 *  geometry, maps pointer points to image px, drives the session's crop
 *  machine, and publishes the crop frame as a closed overlay polyline. */
export function makeCropGesture(
  host: BundleHost,
  session: ImageSession,
): GestureHandler {
  let fit: FitTransform | null = null;

  const imageToPage = (p: [number, number]): [number, number] => toPage(fit, p);
  const pageToImage = (p: [number, number]): [number, number] =>
    toImage(fit, p);

  /** Push the crop frame polyline to the overlay (or clear it). */
  const renderOverlay = () => {
    const machine = session.cropMachine();
    if (!machine || !fit) {
      host.overlay.setToolPreview(null);
      return;
    }
    const corners = machine.overlayPolyline().map(imageToPage);
    const shape: ToolPreviewPolyline = {
      pageId: fit.pageId,
      points: corners,
      close: true,
    };
    host.overlay.setToolPreview(shape);
  };

  /** Resolve the composited frame's fit transform (async; cached until the
   *  gesture deactivates or the source changes). */
  const ensureFit = async () => {
    fit = await resolveFrameFit(host, session.state().source, "crop tool");
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
      const machine = session.cropMachine();
      if (!machine || !e.pagePoint || !fit) return;
      machine.pointerDown(pageToImage(e.pagePoint));
      renderOverlay();
    },
    onPointerMove(e: CanvasPointerEvent) {
      const machine = session.cropMachine();
      if (!machine || !e.pagePoint) return;
      machine.pointerMove(pageToImage(e.pagePoint));
      renderOverlay();
    },
    onPointerUp() {
      session.cropMachine()?.pointerUp();
      renderOverlay();
    },
  };
}
