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

// The SELECTION lane end-to-end over the REAL engine wasm in Node (the
// crop.spec pattern): the session's selection surface (selectAll /
// deselect / invert / feather), the selection machine's marquee / lasso /
// wand commits (engine-side coverage; combine modes), the modifier→mode
// convention, the marching-ants gesture (fit-transform mapping + dashed
// overlay preview), and the handle-swap drop on a crop commit. The MASKED
// ADJUST itself is GPU-only and Node has no navigator.gpu — that proof
// lives in Rust (image-conformance selection_pipeline.rs + the mask-ABI
// suite selection_mask.rs); these specs pin the GLUE.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type {
  CanvasPointerEvent,
  ElementGeometryItem,
  PluginManifest,
} from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { bootEngine, type ImageEngine } from "../src/engine";
import { createImageSession } from "../src/session";
import { makeSelectionGesture } from "../src/selection-tool";
import {
  createSelectionMachine,
  modeFromModifiers,
  WAND_CONTIGUOUS_DEFAULT,
  WAND_TOLERANCE_DEFAULT,
} from "../src/selection-machine";
import {
  makeFakeEditor,
  mapBacking,
  psdBytes,
  shellStub,
  silentConsole,
} from "./helpers";

function makeHost(fake: ReturnType<typeof makeFakeEditor>) {
  return createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
}

function geomFor(
  id: string,
  bounds: [number, number, number, number],
): ElementGeometryItem {
  return { id: { kind: "rectangle", id } as never, pageId: "pg1", bounds };
}

/** Ingest the 2×1 PSD fixture (pixels (10,30,50) and (20,40,60)). */
async function ingest() {
  const fake = makeFakeEditor();
  fake.placed.set("u1", psdBytes());
  fake.geometry.set("u1", geomFor("u1", [0, 0, 100, 200]));
  fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
  const handle = makeHost(fake);
  const session = createImageSession(handle.host);
  expect(await session.ingestSelection()).toBe(true);
  return { fake, handle, session, machine: session.selectionMachine()! };
}

/** A minimal canvas pointer event at a page-local point. */
function ev(
  pagePoint: [number, number],
  mods: Partial<CanvasPointerEvent["modifiers"]> = {},
): CanvasPointerEvent {
  return {
    pageId: "pg1",
    pagePoint,
    docPoint: pagePoint,
    modifiers: { shift: false, alt: false, cmd: false, ctrl: false, ...mods },
    maxDelta: 0,
    button: 0,
    target: null,
    pressure: 0.5,
    tiltX: 0,
    tiltY: 0,
    pointerType: "mouse",
  };
}

describe("the modifier → combine-mode convention", () => {
  it("maps shift/alt/shift+alt/none as documented", () => {
    expect(modeFromModifiers({ shift: false, alt: false })).toBe("replace");
    expect(modeFromModifiers({ shift: true, alt: false })).toBe("add");
    expect(modeFromModifiers({ shift: false, alt: true })).toBe("subtract");
    expect(modeFromModifiers({ shift: true, alt: true })).toBe("intersect");
  });
});

describe("the session selection surface (real engine wasm)", () => {
  it("starts with no selection; selectAll covers the full extent", async () => {
    const { handle, session } = await ingest();
    expect(session.state().selection).toBeNull();
    expect(session.selectAll()).toBe(true);
    const s = session.state().selection!;
    expect([s.x, s.y, s.w, s.h]).toEqual([0, 0, 2, 1]);
    expect(s.coverage).toBeCloseTo(1, 5);
    session.dispose();
    handle.dispose();
  });

  it("POLYGONAL lasso places, undoes and closes vertices — and refuses a degenerate shape", async () => {
    // The polygonal lasso is CLICK-driven where the freehand lasso is
    // drag-driven, so its whole state model is different: the gesture
    // spans many events and ends only on an explicit commit. These
    // assertions are about that lifecycle, not about coverage maths —
    // the polygon door itself was already proven by the lasso.
    const { handle, session, machine } = await ingest();

    // Idle: nothing in progress, so keys must not be consumed.
    expect(machine.placingPolygon()).toBe(false);
    expect(machine.polygonUndoVertex()).toBe(false);
    expect(machine.polygonFirstVertex()).toBeNull();

    machine.polygonVertex([0, 0], "replace");
    expect(machine.placingPolygon(), "the first click starts it").toBe(true);
    expect(machine.polygonFirstVertex()).toEqual([0, 0]);

    // TWO vertices bound no area, so a commit there must REFUSE rather
    // than replace the selection with nothing — a silent deselect the
    // user did not ask for.
    machine.polygonVertex([2, 0], "replace");
    expect(machine.polygonCommit(), "two vertices is not a polygon").toBe(
      false,
    );

    // Backspace removes the LAST vertex, and removing the only one ends
    // the gesture rather than leaving an empty in-progress polygon.
    machine.polygonVertex([0, 0], "replace");
    machine.polygonVertex([2, 0], "replace");
    expect(machine.polygonUndoVertex()).toBe(true);
    expect(machine.polygonFirstVertex()).toEqual([0, 0]);
    expect(machine.polygonUndoVertex()).toBe(true);
    expect(machine.placingPolygon(), "the last undo ends it").toBe(false);

    // Three vertices commit for real.
    machine.polygonVertex([0, 0], "replace");
    machine.polygonVertex([2, 0], "replace");
    machine.polygonVertex([2, 1], "replace");
    expect(machine.polygonCommit()).toBe(true);
    expect(machine.placingPolygon(), "commit ends the gesture").toBe(false);
    expect(session.state().selection, "a real selection landed").not.toBeNull();

    // Esc abandons without touching the committed selection.
    const before = session.state().selection;
    machine.polygonVertex([0, 0], "replace");
    machine.polygonCancel();
    expect(machine.placingPolygon()).toBe(false);
    expect(session.state().selection).toEqual(before);

    session.dispose();
    handle.dispose();
  });

  it("a POINTER-UP never commits a polygon — it spans clicks", async () => {
    // The freehand lasso commits on release; the polygonal one must
    // not, or the first click would commit a one-vertex shape. This is
    // the assertion that keeps the two lifecycles apart.
    const { handle, session, machine } = await ingest();
    machine.polygonVertex([0, 0], "replace");
    machine.polygonVertex([2, 0], "replace");
    machine.polygonVertex([2, 1], "replace");
    expect(machine.end(), "end() is a no-op for a polygon").toBe(false);
    expect(machine.placingPolygon(), "still placing after end()").toBe(true);
    expect(session.state().selection).toBeNull();
    session.dispose();
    handle.dispose();
  });

  it("deselect returns to 'no selection'", async () => {
    const { handle, session } = await ingest();
    session.selectAll();
    expect(session.state().selection).not.toBeNull();
    expect(session.deselect()).toBe(true);
    expect(session.state().selection).toBeNull();
    session.dispose();
    handle.dispose();
  });

  it("a marquee-rect commit selects the rect; invert flips it", async () => {
    const { handle, session, machine } = await ingest();
    // Select the LEFT pixel via the machine (image-px points).
    machine.begin("rect", [0, 0], "replace");
    machine.update([1, 1]);
    expect(machine.end()).toBe(true);
    let s = session.state().selection!;
    expect([s.x, s.y, s.w, s.h]).toEqual([0, 0, 1, 1]);
    expect(s.coverage).toBeCloseTo(0.5, 5);

    expect(session.invertSelection()).toBe(true);
    s = session.state().selection!;
    expect([s.x, s.y, s.w, s.h]).toEqual([1, 0, 1, 1]);
    session.dispose();
    handle.dispose();
  });

  it("subtract mode removes from the selection (shift/alt algebra engine-side)", async () => {
    const { handle, session, machine } = await ingest();
    session.selectAll();
    machine.begin("rect", [0, 0], "subtract");
    machine.update([1, 1]);
    expect(machine.end()).toBe(true);
    const s = session.state().selection!;
    expect(s.x).toBe(1);
    expect(s.w).toBe(1);
    expect(s.coverage).toBeCloseTo(0.5, 5);
    session.dispose();
    handle.dispose();
  });

  it("feather needs a selection, then softens the coverage", async () => {
    const { handle, session, machine } = await ingest();
    expect(session.featherSelection()).toBe(false);
    expect(session.state().status).toMatch(/Feather failed/);

    machine.begin("rect", [0, 0], "replace");
    machine.update([1, 1]);
    machine.end();
    expect(session.featherSelection(0.5)).toBe(true);
    const s = session.state().selection!;
    // A feathered hard pixel: total coverage drops below the hard 0.5
    // (mass bleeds past the canvas edge) and spreads to the neighbor.
    expect(s.coverage).toBeGreaterThan(0);
    expect(s.coverage).toBeLessThan(0.5);
    expect(s.w).toBeGreaterThanOrEqual(1);
    session.dispose();
    handle.dispose();
  });

  it("an ellipse marquee commits AA coverage between 0 and 1", async () => {
    const { handle, session, machine } = await ingest();
    machine.begin("ellipse", [0, 0], "replace");
    machine.update([2, 1]);
    expect(machine.end()).toBe(true);
    const s = session.state().selection!;
    expect(s.coverage).toBeGreaterThan(0.2);
    expect(s.coverage).toBeLessThan(1);
    session.dispose();
    handle.dispose();
  });

  it("a lasso trail closes on release and selects its polygon", async () => {
    const { handle, session, machine } = await ingest();
    machine.begin("lasso", [0, 0], "replace");
    machine.update([2, 0]);
    machine.update([2, 1]);
    expect(machine.end()).toBe(true); // closes 2,1 → 0,0
    const s = session.state().selection!;
    // The right triangle covers ≈ half of the 2×1 field.
    expect(s.coverage).toBeGreaterThan(0.3);
    expect(s.coverage).toBeLessThan(0.7);
    session.dispose();
    handle.dispose();
  });

  it("the magic wand selects by color distance (v0 tolerance 32 spans both close pixels)", async () => {
    const { handle, session, machine } = await ingest();
    // Fixture pixels (10,30,50) vs (20,40,60): Chebyshev distance 10 —
    // inside the documented v0 default tolerance (32/255 per channel).
    expect(machine.wand([0, 0], "replace")).toBe(true);
    const s = session.state().selection!;
    expect(s.coverage).toBeCloseTo(1, 5);
    session.dispose();
    handle.dispose();
  });

  it("a crop commit swaps the engine handle and drops the stale selection", async () => {
    const { handle, session, machine } = await ingest();
    machine.begin("rect", [0, 0], "replace");
    machine.update([1, 1]);
    machine.end();
    expect(session.state().selection).not.toBeNull();

    // Crop to the left pixel — a NEW engine handle; the old-resolution
    // selection would be meaningless, so it drops (stated semantics).
    const crop = session.cropMachine()!;
    crop.pointerDown([2, 0.5]);
    crop.pointerMove([1, 0.5]);
    crop.pointerUp();
    expect(await session.commitCrop()).toBe(true);
    expect(session.state().selection).toBeNull();
    session.dispose();
    handle.dispose();
  });
});

// ── QUICK SELECTION over the real engine ─────────────────────────────
//
// The growth RULE is proven pixel-by-pixel in quick-select.spec.ts. What
// is proven here is the seam: that the grown coverage reaches the ENGINE
// selection through the channel door, under every combine mode, and that
// what lands is indistinguishable from what the magic wand lands — the
// masked-adjust pipeline reads one coverage plane at `@group(2)` and
// must not be able to tell the two tools apart.

const QDARK: [number, number, number, number] = [60, 60, 60, 255];
const QLIGHT: [number, number, number, number] = [180, 180, 180, 255];

/** Register a synthetic image in the REAL engine, bind the selection to
 *  it, and hand back a machine wired to it as its quick-select source. */
async function quickRig(
  width: number,
  height: number,
  at: (x: number, y: number) => [number, number, number, number],
  radius = 4,
) {
  const engine: ImageEngine = await bootEngine();
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = at(x, y);
      const o = (y * width + x) * 4;
      rgba[o] = p[0];
      rgba[o + 1] = p[1];
      rgba[o + 2] = p[2];
      rgba[o + 3] = p[3];
    }
  }
  const info = engine.ingestRgba8(width, height, rgba);
  engine.selectionBind(info.handle);
  let commits = 0;
  const machine = createSelectionMachine(
    engine,
    () => {
      commits++;
    },
    () => ({ handle: info.handle, width, height }),
    { radius },
  );
  return { engine, machine, handle: info.handle, commits: () => commits };
}

/** 32×16, DARK left of x = 16, LIGHT from it. */
const qSplit = (x: number) => (x < 16 ? QDARK : QLIGHT);

describe("quick selection reaching the engine selection", () => {
  it("a paint-to-grow stroke commits the grown region and stops at the edge", async () => {
    const { engine, machine, commits } = await quickRig(32, 16, qSplit);

    machine.begin("quick", [6, 8], "replace");
    // Live, before any engine write: the gesture already knows what it
    // grew, and the preview outline is that region's box.
    expect(machine.quickProgress()!.count).toBe(16 * 16);
    expect(machine.quickProgress()!.capped).toBe(false);
    expect(machine.gestureOutline()).toEqual([
      [0, 0],
      [16, 0],
      [16, 16],
      [0, 16],
    ]);
    expect(engine.selectionStats(), "nothing committed mid-drag").toBeNull();

    machine.update([8, 10]); // still inside the dark half
    expect(machine.end()).toBe(true);
    expect(commits()).toBe(1);

    const s = engine.selectionStats()!;
    expect([s.x, s.y, s.w, s.h]).toEqual([0, 0, 16, 16]);
    expect(s.coverage).toBeCloseTo(0.5, 5);

    // Per-pixel on the coverage plane the kernels read: the boundary held.
    const cov = engine.selectionCoverageBytes();
    expect(cov.length).toBe(32 * 16);
    for (let y = 0; y < 16; y++) {
      expect(cov[y * 32 + 15]).toBe(255);
      expect(cov[y * 32 + 16]).toBe(0);
    }
  });

  it("a drag into a differently-coloured region extends what commits", async () => {
    const { engine, machine } = await quickRig(32, 16, qSplit);
    machine.begin("quick", [6, 8], "replace");
    machine.update([26, 8]); // a dab in the LIGHT half
    expect(machine.quickProgress()!.count).toBe(32 * 16);
    expect(machine.end()).toBe(true);
    const s = engine.selectionStats()!;
    expect([s.x, s.y, s.w, s.h]).toEqual([0, 0, 32, 16]);
    expect(s.coverage).toBeCloseTo(1, 5);
  });

  it("produces the SAME coverage plane as the magic wand for the same input", async () => {
    // An 8×8 split image on which both tools have the same right answer
    // (the wand's 32 tolerance and the quick gate's 16 floor both refuse
    // a 120-level step). If quick selection built its mask any other way
    // — per-pixel rects, a threshold, a different combine fold — these
    // two planes would differ.
    const at = (x: number) => (x < 4 ? QDARK : QLIGHT);
    const { engine, machine } = await quickRig(8, 8, at, 2);

    engine.selectionMagicWand(
      1,
      4,
      WAND_TOLERANCE_DEFAULT,
      WAND_CONTIGUOUS_DEFAULT,
      "replace",
    );
    const wandBytes = Uint8Array.from(engine.selectionCoverageBytes());
    const wandStats = engine.selectionStats()!;

    engine.selectionClear();
    machine.begin("quick", [1, 4], "replace");
    expect(machine.end()).toBe(true);
    const quickBytes = engine.selectionCoverageBytes();
    const quickStats = engine.selectionStats()!;

    expect(quickBytes).toBeInstanceOf(Uint8Array);
    expect(quickBytes.length).toBe(wandBytes.length);
    expect(Array.from(quickBytes)).toEqual(Array.from(wandBytes));
    expect([quickStats.x, quickStats.y, quickStats.w, quickStats.h]).toEqual([
      wandStats.x,
      wandStats.y,
      wandStats.w,
      wandStats.h,
    ]);
    expect(quickStats.coverage).toBeCloseTo(wandStats.coverage, 6);
    expect(quickStats.coverage).toBeCloseTo(0.5, 5);
  });

  it("honours the combine modes — alt SUBTRACTS, shift+alt INTERSECTS", async () => {
    // The documented convention is `modeFromModifiers`; quick selection
    // adds no vocabulary of its own, it just carries the mode through.
    const { engine, machine } = await quickRig(32, 16, qSplit);

    engine.selectionSelectAll();
    machine.begin("quick", [6, 8], modeFromModifiers({ shift: false, alt: true }));
    expect(machine.end()).toBe(true);
    let s = engine.selectionStats()!;
    expect(
      [s.x, s.y, s.w, s.h],
      "alt removed the dark half, leaving the light one",
    ).toEqual([16, 0, 16, 16]);

    // INTERSECT is the reason the commit goes through the channel door as
    // ONE plane: intersecting per dab (or per run) would narrow against
    // each piece in turn and leave nothing.
    engine.selectionClear();
    engine.selectionSetRect(0, 0, 32, 8, "replace"); // the top half
    machine.begin("quick", [6, 8], modeFromModifiers({ shift: true, alt: true }));
    machine.update([8, 10]);
    expect(machine.end()).toBe(true);
    s = engine.selectionStats()!;
    expect([s.x, s.y, s.w, s.h], "top ∩ dark = the top-left quadrant").toEqual([
      0, 0, 16, 8,
    ]);
    expect(s.coverage).toBeCloseTo(0.25, 5);
  });

  it("commits nothing when nothing grew, and never silently deselects", async () => {
    const { engine, machine } = await quickRig(32, 16, qSplit);
    engine.selectionSetRect(0, 0, 32, 8, "replace");
    const before = engine.selectionStats()!;
    // A press entirely outside the image paints no evidence.
    machine.begin("quick", [-40, -40], "replace");
    expect(machine.quickProgress()!.count).toBe(0);
    expect(machine.end(), "an empty stroke refuses to commit").toBe(false);
    const after = engine.selectionStats()!;
    expect([after.x, after.y, after.w, after.h]).toEqual([
      before.x,
      before.y,
      before.w,
      before.h,
    ]);
  });

  it("WITHOUT a pixel source the gesture is an honest no-op", async () => {
    // This is exactly how the session builds the machine today (two
    // args). Quick selection must degrade to "nothing happened" rather
    // than throw or fake a selection.
    const { engine } = await quickRig(32, 16, qSplit);
    engine.selectionClear();
    const machine = createSelectionMachine(engine, () => {});
    machine.begin("quick", [6, 8], "replace");
    expect(machine.quickProgress()).toBeNull();
    expect(machine.gestureOutline()).toBeNull();
    machine.update([8, 8]);
    expect(machine.end()).toBe(false);
    expect(engine.selectionStats()).toBeNull();
  });
});

describe("the selection gesture (fit transform + marching-ants overlay)", () => {
  it("maps page pt → image px through the frame box and publishes a dashed preview", async () => {
    const { fake, handle, session } = await ingest();
    const gesture = makeSelectionGesture(handle.host, session, "rect");
    gesture.onActivate(undefined as never);
    await new Promise((r) => setTimeout(r, 0)); // ensureFit resolves

    // The frame box is [top 0, left 0, bottom 100, right 200] and the
    // image is 2×1 → aspect-fit scale 100, origin (0,0). Dragging page
    // (0,0)→(100,100) is image (0,0)→(1,1): the LEFT pixel.
    gesture.onPointerDown(ev([0, 0]));
    gesture.onPointerMove(ev([100, 100]));
    gesture.onPointerUp(ev([100, 100]));

    const s = session.state().selection!;
    expect([s.x, s.y, s.w, s.h]).toEqual([0, 0, 1, 1]);

    // The overlay carried a DASHED closed path (the marching-ants v0
    // vocabulary) in page-local pt.
    const last = fake.overlayShapes.at(-1) as {
      pageId: string;
      anchors: Array<{ anchor: [number, number] }>;
      close?: boolean;
      dashed?: boolean;
    };
    expect(last).toBeTruthy();
    expect(last.pageId).toBe("pg1");
    expect(last.dashed).toBe(true);
    expect(last.close).toBe(true);
    // The committed outline is the bounds box back in page pt (×100).
    expect(last.anchors[0].anchor).toEqual([0, 0]);
    expect(last.anchors[2].anchor).toEqual([100, 100]);
    session.dispose();
    handle.dispose();
  });

  it("shift-drag adds instead of replacing (modifiers reach the engine)", async () => {
    const { handle, session } = await ingest();
    const gesture = makeSelectionGesture(handle.host, session, "rect");
    gesture.onActivate(undefined as never);
    await new Promise((r) => setTimeout(r, 0));

    // Replace-select the left pixel, then SHIFT-add the right pixel.
    gesture.onPointerDown(ev([0, 0]));
    gesture.onPointerMove(ev([100, 100]));
    gesture.onPointerUp(ev([100, 100]));
    gesture.onPointerDown(ev([100, 0], { shift: true }));
    gesture.onPointerMove(ev([200, 100], { shift: true }));
    gesture.onPointerUp(ev([200, 100], { shift: true }));

    const s = session.state().selection!;
    expect([s.x, s.y, s.w, s.h]).toEqual([0, 0, 2, 1]);
    expect(s.coverage).toBeCloseTo(1, 5);
    session.dispose();
    handle.dispose();
  });

  it("deactivate clears the overlay preview", async () => {
    const { fake, handle, session } = await ingest();
    const gesture = makeSelectionGesture(handle.host, session, "rect");
    gesture.onActivate(undefined as never);
    await new Promise((r) => setTimeout(r, 0));
    gesture.onDeactivate("switch");
    expect(fake.overlayShapes.at(-1)).toBeNull();
    session.dispose();
    handle.dispose();
  });
});
