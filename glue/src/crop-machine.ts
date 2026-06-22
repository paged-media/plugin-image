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

// The crop + straighten INTERACTION machine — host-agnostic, thin over
// the engine's pure crop geometry (`image_core::crop`, exposed on the
// wasm surface). It holds the live crop rect + straighten angle + aspect
// lock + the in-flight drag, and turns pointer points (already page-
// resolved + image-pixel-mapped by the caller) into the new rect via the
// Rust geometry — the TS NEVER re-implements the handle/rect/angle math
// (it lives in image-core, property-tested). The caller (the crop tool's
// gesture + the panel) drives this and renders `overlayPolyline()` through
// `host.overlay.setToolPreview`; `commit()` cuts the rect via the engine
// crop lane and re-composites in-frame through the existing Stage-A path.
//
// Coordinates everywhere are IMAGE-PIXEL space (origin top-left). The
// caller maps page-local pt ↔ image px using the composited frame's
// content box (the same aspect-fit transform `session.apply` uses).

import type {
  AspectLock,
  CropHandle,
  CropRect,
  ImageEngine,
} from "./engine";

/** The aspect-ratio lock presets the panel offers. `free` is
 *  unconstrained; `original` resolves to the source image's own ratio;
 *  the rest are fixed `w:h`. */
export type AspectPreset =
  | "free"
  | "original"
  | "1:1"
  | "3:2"
  | "4:3"
  | "16:9";

/** The fixed `w:h` ratios for the non-source presets (`free`/`original`
 *  resolve dynamically). */
const PRESET_RATIO: Record<Exclude<AspectPreset, "free" | "original">, [number, number]> = {
  "1:1": [1, 1],
  "3:2": [3, 2],
  "4:3": [4, 3],
  "16:9": [16, 9],
};

export interface CropState {
  /** The live crop rect in image px. */
  rect: CropRect;
  /** Straighten angle in degrees (frame rotation about the rect centre). */
  angle: number;
  preset: AspectPreset;
  /** A drag is in progress on `handle`. -1 / null = idle. */
  dragging: boolean;
}

export interface CropMachine {
  state(): CropState;
  /** Reset the rect to the full image extent + identity angle/preset (for
   *  a freshly ingested image of `imageW`×`imageH`). */
  reset(imageW: number, imageH: number): void;
  setPreset(preset: AspectPreset): void;
  setAngle(degrees: number): void;
  /** Hit-test at an image-px point (the cursor/grip the chrome shows). */
  hitTest(point: [number, number]): CropHandle;
  /** Begin a drag at `point` — grabs the handle there (a miss is ignored,
   *  returns false so the caller can fall through to a marquee). */
  pointerDown(point: [number, number]): boolean;
  /** Continue the active drag to `point` (no-op when idle). */
  pointerMove(point: [number, number]): void;
  /** End the active drag. */
  pointerUp(): void;
  /** The crop-frame outline as a closed polyline of image-px points
   *  (TL, TR, BR, BL), rotated by the straighten angle — the overlay
   *  signal the caller passes to `host.overlay.setToolPreview`. */
  overlayPolyline(): Array<[number, number]>;
  /** Commit the crop: cut the rect (axis-aligned, angle-0) out of the
   *  source image via the engine and return the NEW image handle/info.
   *  Throws (engine message) on an empty rect. The straighten-angle
   *  resample is not part of this axis-aligned cut — the honest subset. */
  commit(engine: ImageEngine, sourceHandle: number): {
    handle: number;
    width: number;
    height: number;
  };
}

/** Resolve a preset to the wasm aspect lock given the source dimensions
 *  (so `original` follows the image's own ratio). */
function presetLock(preset: AspectPreset, imageW: number, imageH: number): AspectLock {
  if (preset === "free") return null;
  if (preset === "original") return { w: imageW, h: imageH };
  const [w, h] = PRESET_RATIO[preset];
  return { w, h };
}

export function createCropMachine(
  engine: ImageEngine,
  imageW: number,
  imageH: number,
  /** Grip grab radius in image px (the caller derives it from the screen
   *  tolerance via host.viewport so the grip stays a constant size). */
  grabTol = 8,
): CropMachine {
  let imgW = imageW;
  let imgH = imageH;
  const state: CropState = {
    rect: { x: 0, y: 0, w: imgW, h: imgH },
    angle: 0,
    preset: "free",
    dragging: false,
  };
  let activeHandle: CropHandle = -1;
  let dragStart: [number, number] = [0, 0];
  let rectAtStart: CropRect = state.rect;

  const lock = () => presetLock(state.preset, imgW, imgH);

  return {
    state: () => state,

    reset(w, h) {
      imgW = w;
      imgH = h;
      state.rect = { x: 0, y: 0, w, h };
      state.angle = 0;
      state.preset = "free";
      state.dragging = false;
      activeHandle = -1;
    },

    setPreset(preset) {
      state.preset = preset;
      // Re-impose the ratio immediately by replaying a zero-delta resize
      // from the bottom-right grip (handle 4) so the rect snaps to the new
      // lock without waiting for a drag.
      const l = lock();
      if (l) {
        const br = { x: state.rect.x, y: state.rect.y, w: state.rect.w, h: state.rect.h };
        state.rect = engine.cropApplyDrag(
          br,
          4, // BottomRight
          [br.x + br.w, br.y + br.h],
          [br.x + br.w, br.y + br.h],
          l,
          imgW,
          imgH,
        );
      }
    },

    setAngle(degrees) {
      state.angle = degrees;
    },

    hitTest(point) {
      return engine.cropHitHandle(state.rect, point, grabTol);
    },

    pointerDown(point) {
      const handle = engine.cropHitHandle(state.rect, point, grabTol);
      if (handle < 0) return false;
      activeHandle = handle;
      dragStart = point;
      rectAtStart = { ...state.rect };
      state.dragging = true;
      return true;
    },

    pointerMove(point) {
      if (!state.dragging || activeHandle < 0) return;
      state.rect = engine.cropApplyDrag(
        rectAtStart,
        activeHandle,
        dragStart,
        point,
        lock(),
        imgW,
        imgH,
      );
    },

    pointerUp() {
      state.dragging = false;
      activeHandle = -1;
    },

    overlayPolyline() {
      return engine.cropFrameCorners(state.rect, state.angle);
    },

    commit(eng, sourceHandle) {
      const r = state.rect;
      return eng.crop(sourceHandle, {
        x: Math.round(r.x),
        y: Math.round(r.y),
        w: Math.round(r.w),
        h: Math.round(r.h),
      });
    },
  };
}
