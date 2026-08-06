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

// The PAINT INTERACTION machine — host-agnostic and ENGINE-FREE (the
// crop / selection machine pattern, minus the engine: those two call
// wasm geometry doors, this one only sequences pointer samples, so it
// unit-tests with no wasm at all).
//
// What it owns is the SAMPLING POLICY, and nothing else:
//
//  * pressure NORMALIZATION (see `normalizePressure`) — the one place
//    the "a mouse is not a half-pressure pen" rule lives;
//  * the FIFO of samples still owed to the engine. `brush_stroke_extend`
//    is async (a GPU dispatch plus a whole-image readback), and pointer
//    moves arrive faster than it resolves. The machine QUEUES them and
//    the tool drains the queue in order, so no sample is ever dropped:
//    the painted path is the pointer's path, not a subsample of it. Only
//    intermediate PREVIEW frames are skipped, which is invisible;
//  * the end-of-stroke handshake: `up()` marks the stroke closed, and
//    `drained()` tells the tool when the last queued sample has been
//    handed over and it is time to commit.
//
// The stroke's PIXELS are engine state (`crate::stroke::StrokeSession`
// holds the base snapshot + the coverage accumulator behind the
// `brush_stroke_*` doors); the machine holds no image data and cannot
// undo anything — `cancel()` here only drops queued samples, and the
// engine-side `brush_stroke_cancel` is what restores the pixels.
//
// Coordinates are IMAGE-PIXEL space throughout (the caller maps
// page-local pt ↔ image px against the composited frame's content box —
// `frame-fit.ts`, the same transform crop and selection use).

/** Pointer device class (Pointer Events `pointerType`, as the host
 *  forwards it on `CanvasPointerEvent`). */
export type BrushPointerType = "mouse" | "pen" | "touch";

/** One pointer sample as the engine's `brush_stroke_extend` takes it. */
export interface BrushSample {
  /** Image-pixel x. */
  x: number;
  /** Image-pixel y. */
  y: number;
  /** Normalized pressure in [0, 1] — see [`normalizePressure`]. */
  pressure: number;
}

/** The pressure a NON-PEN device paints at.
 *
 *  Pointer Events report a constant `0.5` for a mouse or a finger while
 *  a button is held — a device-neutral placeholder, not a measurement.
 *  Passing it through would make every mouse stroke permanently
 *  half-size and half-flow under the default `both` pressure target, so
 *  the glue substitutes full pressure. (The engine door takes whatever
 *  it is given verbatim, so a recorded stroke replays identically — the
 *  normalization is deliberately on THIS side of the boundary.) */
export const NON_PEN_PRESSURE = 1;

/** Normalize a `CanvasPointerEvent.pressure` for the engine.
 *
 *  A PEN's reading is a real measurement and passes through clamped to
 *  [0, 1]; everything else (mouse, touch, and any synthetic event
 *  without a pressure) paints at [`NON_PEN_PRESSURE`]. */
export function normalizePressure(
  pressure: number,
  pointerType: BrushPointerType,
): number {
  if (pointerType !== "pen" || !Number.isFinite(pressure)) {
    return NON_PEN_PRESSURE;
  }
  return Math.min(1, Math.max(0, pressure));
}

/** Vertices in the brush-tip cursor outline. */
export const TIP_SEGMENTS = 24;

/** The brush-tip cursor ring as a closed image-px polyline: the tool's
 *  overlay affordance ("this is the size that will land"), published
 *  through `host.overlay.setToolPreview`.
 *
 *  It is the NOMINAL tip — the diameter the panel is set to. A pen's
 *  pressure scales the dab that actually lands (`PressureTarget`), and
 *  the ring deliberately does not chase it: a cursor that shrank as you
 *  pressed would stop being a size reference. */
export function tipOutline(
  center: [number, number],
  diameter: number,
  segments: number = TIP_SEGMENTS,
): Array<[number, number]> {
  const r = Math.max(diameter, 0) / 2;
  const points: Array<[number, number]> = [];
  for (let i = 0; i < segments; i++) {
    const t = (i / segments) * Math.PI * 2;
    points.push([center[0] + r * Math.cos(t), center[1] + r * Math.sin(t)]);
  }
  return points;
}

export interface BrushMachineState {
  /** Between `down()` and `up()` — the pointer is laying down samples. */
  drawing: boolean;
  /** `up()` has been seen; the stroke closes once the queue drains. */
  ended: boolean;
  /** Samples ACCEPTED this stroke (the replay record's length). */
  sampleCount: number;
  /** Samples queued but not yet handed to the engine by `next()`. */
  queued: number;
}

export interface BrushMachine {
  state(): BrushMachineState;
  /** Pointer down at an image-px point: opens a stroke and queues its
   *  first sample. False when a stroke is already open (defensive — a
   *  second `down` never silently restarts one). */
  down(
    point: [number, number],
    pressure: number,
    pointerType: BrushPointerType,
  ): boolean;
  /** Pointer move: queues a sample. False when no stroke is open, when
   *  the point is not finite, or when the sample is an EXACT duplicate
   *  of the last accepted one — a repeat cannot move the spacing walk
   *  or change a dab, so sending it would buy a GPU round-trip and a
   *  whole-image readback for a provable no-op. */
  move(
    point: [number, number],
    pressure: number,
    pointerType: BrushPointerType,
  ): boolean;
  /** Pointer up: no further samples. The stroke is finished once every
   *  queued sample has been taken (see [`BrushMachine.drained`]). */
  up(): void;
  /** Take the next queued sample, or null when the queue is empty. */
  next(): BrushSample | null;
  /** `up()` has been seen and the queue is empty — the tool's signal to
   *  commit the stroke engine-side. */
  drained(): boolean;
  /** Drop the in-flight stroke's queued samples and close it.
   *
   *  This is bookkeeping only: the engine-held pixels are restored by
   *  `brush_stroke_cancel`, which the caller invokes. It is also how the
   *  tool resets after a successful COMMIT — the machine keeps no state
   *  that a commit would have to unwind. */
  cancel(): void;
  /** Every sample accepted this stroke, in order — the replay record
   *  (the engine's own stroke is a pure function of this list plus the
   *  frozen params, which is what makes a recorded stroke reproducible). */
  samples(): readonly BrushSample[];
}

export function createBrushMachine(): BrushMachine {
  let drawing = false;
  let ended = false;
  let queue: BrushSample[] = [];
  let accepted: BrushSample[] = [];
  let last: BrushSample | null = null;

  const push = (
    point: [number, number],
    pressure: number,
    pointerType: BrushPointerType,
  ): BrushSample | null => {
    if (!Number.isFinite(point[0]) || !Number.isFinite(point[1])) return null;
    const sample: BrushSample = {
      x: point[0],
      y: point[1],
      pressure: normalizePressure(pressure, pointerType),
    };
    queue.push(sample);
    accepted.push(sample);
    last = sample;
    return sample;
  };

  return {
    state: () => ({
      drawing,
      ended,
      sampleCount: accepted.length,
      queued: queue.length,
    }),

    down(point, pressure, pointerType) {
      if (drawing) return false;
      queue = [];
      accepted = [];
      last = null;
      ended = false;
      if (!push(point, pressure, pointerType)) return false;
      drawing = true;
      return true;
    },

    move(point, pressure, pointerType) {
      if (!drawing) return false;
      const p = normalizePressure(pressure, pointerType);
      if (
        last &&
        last.x === point[0] &&
        last.y === point[1] &&
        last.pressure === p
      ) {
        return false;
      }
      return push(point, pressure, pointerType) !== null;
    },

    up() {
      if (!drawing) return;
      drawing = false;
      ended = true;
    },

    next() {
      return queue.shift() ?? null;
    },

    drained() {
      return ended && queue.length === 0;
    },

    cancel() {
      drawing = false;
      ended = false;
      queue = [];
      accepted = [];
      last = null;
    },

    samples: () => accepted,
  };
}
