// The PATCH tool: drag the selection onto the area to copy FROM (Photoshop's
// "Source" mode). On release the selected region is replaced by the region
// at the drag offset, healed so it takes on its surroundings' tone
// (`patch_selection`). While dragging, the selection's box follows the
// pointer as the preview. A drag shorter than a pixel patches nothing, and
// with no selection the tool says so instead of doing nothing silently.

import type { BundleHost, CanvasPointerEvent, GestureHandler } from "@paged-media/plugin-api";

import { imageToPage, pageToImage, resolveFrameFit, type FitTransform } from "./frame-fit";
import type { ImageSession } from "./session";

/** The source offset a drag from `a` to `b` names (image px), or null for
 *  a drag too short to mean anything. */
export function patchOffset(a: [number, number], b: [number, number]): [number, number] | null {
  const dx = Math.round(b[0] - a[0]);
  const dy = Math.round(b[1] - a[1]);
  return dx === 0 && dy === 0 ? null : [dx, dy];
}

export function makePatchGesture(host: BundleHost, session: ImageSession): GestureHandler {
  let fit: FitTransform | null = null;
  let from: [number, number] | null = null;
  let to: [number, number] | null = null;

  const preview = () => {
    const sel = session.state().selection;
    if (!fit || !from || !to || !sel) return host.overlay.setToolPreview(null);
    const [dx, dy] = [to[0] - from[0], to[1] - from[1]];
    const c: [number, number][] = [
      [sel.x + dx, sel.y + dy],
      [sel.x + sel.w + dx, sel.y + dy],
      [sel.x + sel.w + dx, sel.y + sel.h + dy],
      [sel.x + dx, sel.y + sel.h + dy],
    ];
    host.overlay.setToolPreview({
      pageId: fit.pageId,
      points: c.map((p) => imageToPage(fit!, p)),
      close: true,
    });
  };

  return {
    onActivate() {
      void resolveFrameFit(host, session.state().source, "patch tool").then((f) => {
        fit = f;
      });
    },
    onDeactivate() {
      host.overlay.setToolPreview(null);
      fit = null;
      from = to = null;
    },
    onPointerDown(e: CanvasPointerEvent) {
      if (!fit || !e.pagePoint || !session.state().source) return;
      from = to = pageToImage(fit, e.pagePoint);
      preview();
    },
    onPointerMove(e: CanvasPointerEvent) {
      if (!fit || !from || !e.pagePoint) return;
      to = pageToImage(fit, e.pagePoint);
      preview();
    },
    onPointerUp() {
      const [a, b] = [from, to];
      from = to = null;
      host.overlay.setToolPreview(null);
      const off = a && b ? patchOffset(a, b) : null;
      if (off) void session.patchSelection(off[0], off[1]);
    },
  };
}
