// The GRADIENT TOOL: drag to set where a foreground → background
// gradient starts and ends; release fills the selection (or the image).
// A drag shorter than a pixel is a click and fills nothing. Shift
// constrains the line to 45° steps, as in Photoshop.

import type { BundleHost, CanvasPointerEvent, GestureHandler } from "@paged-media/plugin-api";

import { imageToPage, pageToImage, resolveFrameFit, type FitTransform } from "./frame-fit";
import type { ImageSession } from "./session";

/** Snap the end point to the nearest 45° direction from the start. */
export function constrain45(from: [number, number], to: [number, number]): [number, number] {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy);
  const step = Math.PI / 4;
  const a = Math.round(Math.atan2(dy, dx) / step) * step;
  return [from[0] + Math.cos(a) * len, from[1] + Math.sin(a) * len];
}

export function makeGradientGesture(host: BundleHost, session: ImageSession): GestureHandler {
  let fit: FitTransform | null = null;
  let from: [number, number] | null = null;
  let to: [number, number] | null = null;

  const preview = () => {
    if (!fit || !from || !to) {
      host.overlay.setToolPreview(null);
      return;
    }
    host.overlay.setToolPreview({
      pageId: fit.pageId,
      points: [imageToPage(fit, from), imageToPage(fit, to)],
      close: false,
    });
  };

  return {
    onActivate() {
      void resolveFrameFit(host, session.state().source, "gradient tool").then((f) => {
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
      from = pageToImage(fit, e.pagePoint);
      to = from;
      preview();
    },
    onPointerMove(e: CanvasPointerEvent) {
      if (!fit || !from || !e.pagePoint) return;
      const p = pageToImage(fit, e.pagePoint);
      to = e.modifiers.shift ? constrain45(from, p) : p;
      preview();
    },
    onPointerUp() {
      const [a, b] = [from, to];
      from = to = null;
      host.overlay.setToolPreview(null);
      if (!a || !b || Math.hypot(b[0] - a[0], b[1] - a[1]) < 1) return;
      void session.fillGradientLine(a, b);
    },
  };
}
