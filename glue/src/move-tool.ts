// The raster MOVE tool. Drag moves the selected pixels (Alt: a copy, the
// source keeps its pixels) or, with no selection, the whole active
// layer; the arrow keys nudge by 1 px, Shift+arrow by 10. Moves land in
// WHOLE pixels, which the engine applies bit-exactly (no resampling), so
// any number of moves never softens the image.
//
// During the drag the tool previews the moved bounds as an overlay
// rectangle; the pixels move once, on release. This is the host Selection
// tool's division of labour turned inward: that one moves the FRAME on
// the page, this one moves what is inside it.

import type {
  BundleHost,
  CanvasPointerEvent,
  GestureHandler,
} from "@paged-media/plugin-api";

import {
  imageToPage,
  pageToImage,
  resolveFrameFit,
  type FitTransform,
} from "./frame-fit";
import { EdgePolicy, VacateMode, type ImageSession } from "./session";

/** Move by whole pixels: the selected pixels when there is a selection,
 *  else the active layer. Exported for the tool's spec. */
export function movePixels(
  session: ImageSession,
  dx: number,
  dy: number,
  copy: boolean,
): Promise<boolean> {
  const sel = session.state().selection;
  if (dx === 0 && dy === 0) return Promise.resolve(false);
  return sel && sel.w > 0
    ? session.moveSelection(dx, dy, copy ? VacateMode.Copy : VacateMode.Transparent)
    : session.offsetLayer(dx, dy, EdgePolicy.Transparent);
}

/** Arrow key → whole-pixel step (Shift ×10), or null for any other key. */
export function nudgeFor(key: string, shift: boolean): [number, number] | null {
  const k = shift ? 10 : 1;
  switch (key) {
    case "ArrowLeft":
      return [-k, 0];
    case "ArrowRight":
      return [k, 0];
    case "ArrowUp":
      return [0, -k];
    case "ArrowDown":
      return [0, k];
    default:
      return null;
  }
}

export function makeMoveGesture(host: BundleHost, session: ImageSession): GestureHandler {
  let fit: FitTransform | null = null;
  let start: [number, number] | null = null;
  let delta: [number, number] = [0, 0];

  /** The bounds being moved, in image px. */
  const bounds = (): { x: number; y: number; w: number; h: number } | null => {
    const s = session.state();
    if (!s.source) return null;
    return s.selection && s.selection.w > 0
      ? s.selection
      : { x: 0, y: 0, w: s.source.width, h: s.source.height };
  };

  const preview = () => {
    const b = bounds();
    if (!fit || !b || !start) {
      host.overlay.setToolPreview(null);
      return;
    }
    const [dx, dy] = delta;
    const corners: [number, number][] = [
      [b.x + dx, b.y + dy],
      [b.x + b.w + dx, b.y + dy],
      [b.x + b.w + dx, b.y + b.h + dy],
      [b.x + dx, b.y + b.h + dy],
    ];
    host.overlay.setToolPreview({
      pageId: fit.pageId,
      points: corners.map((p) => imageToPage(fit, p)),
      close: true,
    });
  };

  return {
    onActivate() {
      void resolveFrameFit(host, session.state().source, "move tool").then((f) => {
        fit = f;
      });
    },
    onDeactivate() {
      host.overlay.setToolPreview(null);
      fit = null;
      start = null;
    },
    onPointerDown(e: CanvasPointerEvent) {
      if (!fit || !e.pagePoint || !session.state().source) return;
      start = pageToImage(fit, e.pagePoint);
      delta = [0, 0];
      preview();
    },
    onPointerMove(e: CanvasPointerEvent) {
      if (!fit || !start || !e.pagePoint) return;
      const p = pageToImage(fit, e.pagePoint);
      delta = [Math.round(p[0] - start[0]), Math.round(p[1] - start[1])];
      preview();
    },
    onPointerUp(e: CanvasPointerEvent) {
      if (!start) return;
      const [dx, dy] = delta;
      start = null;
      host.overlay.setToolPreview(null);
      void movePixels(session, dx, dy, e.modifiers.alt);
    },
    onKey(e: KeyboardEvent) {
      if (e.type !== "keydown") return;
      const step = nudgeFor(e.key, e.shiftKey);
      if (!step) return;
      e.preventDefault();
      void movePixels(session, step[0], step[1], false);
    },
  };
}
