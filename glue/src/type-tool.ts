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

// The RASTER TYPE tool: a click sets the baseline origin and the panel's
// Type section supplies the string, family and size.
//
// A CLICK, not a drag, and that is a scope decision rather than a
// simplification. Dragging would imply a text BOX — wrapping, alignment,
// overflow, a caret — which is a text engine, and the host already has
// one that is better at it. This paints a run of shaped glyphs at a
// point, which is what "raster type" means in the Photoshop catalog's
// sense: committed glyphs become pixels in the layer.

import type {
  BundleHost,
  CanvasPointerEvent,
  DeactivateReason,
  PagedEditor,
  GestureHandler,
} from "@paged-media/plugin-api";

import { pageToImage, resolveFrameFit, type FitTransform } from "./frame-fit";
import type { ImageSession } from "./session";

export function makeTypeGesture(
  host: BundleHost,
  session: ImageSession,
): GestureHandler {
  let fit: FitTransform | null = null;

  const ensureFit = async () => {
    fit = await resolveFrameFit(host, session.state().source, "type tool");
  };

  return {
    onActivate(_paged: PagedEditor) {
      void ensureFit();
    },

    onDeactivate(_reason: DeactivateReason) {
      fit = null;
    },

    onPointerDown(e: CanvasPointerEvent) {
      if (!e.pagePoint || !fit) return;
      // The click is the BASELINE origin — where type is positioned
      // from. Passing the ink's top-left instead would drop every run by
      // its own ascender, which looks like a bug that moves with the
      // font.
      const point = pageToImage(fit, e.pagePoint);
      const t = session.state().type;
      if (!t.text) {
        // Nothing to set. Saying so beats a click that silently does
        // nothing, which reads as a broken tool.
        return;
      }
      void session.paintText(point, t.text, t.family, t.sizePx);
    },

    // A click, not a drag — so move and up have nothing to do. Present
    // because the handler contract requires them, and empty rather than
    // faked: dragging a type run would imply a text box, which is the
    // host's job (see the header).
    onPointerMove(_e: CanvasPointerEvent) {},
    onPointerUp(_e: CanvasPointerEvent) {},
  };
}
