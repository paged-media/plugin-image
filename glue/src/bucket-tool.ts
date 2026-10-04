// The PAINT BUCKET: a click floods the area around the clicked pixel
// that is within the wand tolerance (connected, unless contiguous is
// off), inside the selection, with the foreground colour. One click, one
// journaled edit.

import type { BundleHost, CanvasPointerEvent, GestureHandler } from "@paged-media/plugin-api";

import { pageToImage, resolveFrameFit, type FitTransform } from "./frame-fit";
import type { ImageSession } from "./session";

export function makeBucketGesture(host: BundleHost, session: ImageSession): GestureHandler {
  let fit: FitTransform | null = null;
  return {
    onActivate() {
      void resolveFrameFit(host, session.state().source, "paint bucket").then((f) => {
        fit = f;
      });
    },
    onDeactivate() {
      fit = null;
    },
    onPointerDown(e: CanvasPointerEvent) {
      if (!fit || !e.pagePoint) return;
      const at = pageToImage(fit, e.pagePoint);
      // Alt-click samples, as with every paint tool.
      if (e.modifiers.alt) session.sampleColor(at, 3);
      else void session.bucketFill(at);
    },
    onPointerMove() {},
    onPointerUp() {},
  };
}
