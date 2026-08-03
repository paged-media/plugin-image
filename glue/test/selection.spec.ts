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

import { createImageSession } from "../src/session";
import { makeSelectionGesture } from "../src/selection-tool";
import { modeFromModifiers } from "../src/selection-machine";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";

function makeHost(fake: ReturnType<typeof makeFakeEditor>) {
  return createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
}

function geomFor(id: string, bounds: [number, number, number, number]): ElementGeometryItem {
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
