// The RED EYE tool: drag a box around the eye; on release, the red pupil
// inside the ellipse that fits the box turns dark (only pixels that are
// actually red change — see `selection::red_eye_coverage`). A click with
// no drag uses a box of 24 image pixels around the point.

import type { BundleHost, CanvasPointerEvent, GestureHandler } from "@paged-media/plugin-api";

import { imageToPage, pageToImage, resolveFrameFit, type FitTransform } from "./frame-fit";
import type { ImageSession } from "./session";

const CLICK_BOX = 24;

/** The box a drag from `a` to `b` describes (a click: CLICK_BOX around `a`). */
export function redEyeBox(a: [number, number], b: [number, number]) {
  if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 2) {
    return { x: a[0] - CLICK_BOX / 2, y: a[1] - CLICK_BOX / 2, w: CLICK_BOX, h: CLICK_BOX };
  }
  return {
    x: Math.min(a[0], b[0]),
    y: Math.min(a[1], b[1]),
    w: Math.abs(b[0] - a[0]),
    h: Math.abs(b[1] - a[1]),
  };
}

export function makeRedEyeGesture(host: BundleHost, session: ImageSession): GestureHandler {
  let fit: FitTransform | null = null;
  let from: [number, number] | null = null;
  let to: [number, number] | null = null;
  const preview = () => {
    if (!fit || !from || !to) return host.overlay.setToolPreview(null);
    const b = redEyeBox(from, to);
    const c: [number, number][] = [
      [b.x, b.y],
      [b.x + b.w, b.y],
      [b.x + b.w, b.y + b.h],
      [b.x, b.y + b.h],
    ];
    host.overlay.setToolPreview({ pageId: fit.pageId, points: c.map((p) => imageToPage(fit, p)), close: true });
  };
  return {
    onActivate() {
      void resolveFrameFit(host, session.state().source, "red eye tool").then((f) => {
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
      if (a && b) void session.applyRedEye(redEyeBox(a, b));
    },
  };
}
