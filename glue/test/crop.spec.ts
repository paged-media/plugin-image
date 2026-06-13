// Phase 6 — the crop interaction machine + the levels/curves/white-balance
// readouts, end to end over the REAL engine wasm in Node. The crop GEOMETRY
// is image_core::crop on the wasm surface (property-tested in Rust); these
// pin the GLUE: the histogram readout, the curve LUT build, the crop
// machine's drag/aspect/commit, and the session wiring (ingest computes the
// histogram + builds the machine; commitCrop swaps the engine source).
// Node has no navigator.gpu, so GPU adjustments degrade honestly — these
// exercise the pure (CPU/geometry) lanes that DON'T need WebGPU.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { ElementGeometryItem, PluginManifest } from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { createImageSession } from "../src/session";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";

function makeHost(fake: ReturnType<typeof makeFakeEditor>) {
  return createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
}

/** A frame geometry box (page-local pt bounds) for the crop tool's read. */
function geomFor(id: string, bounds: [number, number, number, number]): ElementGeometryItem {
  return { id: { kind: "rectangle", id } as never, pageId: "pg1", bounds };
}

describe("the histogram readout (real engine wasm)", () => {
  it("computes a 4×256 histogram on ingest and sums to the pixel count", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u1", psdBytes()); // 2×1 PSD
    fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    expect(await session.ingestSelection()).toBe(true);
    const hist = session.state().histogram;
    expect(hist).not.toBeNull();
    // 2 pixels → every channel's bins sum to 2.
    const sum = (a: Uint32Array) => a.reduce((x, y) => x + y, 0);
    expect(sum(hist!.r)).toBe(2);
    expect(sum(hist!.g)).toBe(2);
    expect(sum(hist!.b)).toBe(2);
    expect(sum(hist!.luma)).toBe(2);

    session.dispose();
    handle.dispose();
  });
});

describe("the curve LUT build (real engine wasm)", () => {
  it("the identity curve clears the LUT; a non-identity curve sets 256 bytes", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u1", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);
    await session.ingestSelection();

    session.setCurvePoints([
      [0, 0],
      [1, 1],
    ]);
    expect(session.state().params.curveLut).toBeNull();

    session.setCurvePoints([
      [0, 0.2],
      [1, 0.8],
    ]);
    const lut = session.state().params.curveLut;
    expect(lut).not.toBeNull();
    expect(lut!.length).toBe(256);
    // Raised black point → lut[0] ≈ round(0.2*255) = 51.
    expect(lut![0]).toBe(51);

    session.dispose();
    handle.dispose();
  });
});

describe("the crop machine (real engine geometry over wasm)", () => {
  async function ingestAndGetMachine() {
    const fake = makeFakeEditor();
    fake.placed.set("u1", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);
    await session.ingestSelection();
    return { fake, handle, session, machine: session.cropMachine()! };
  }

  it("starts at the full image extent", async () => {
    const { handle, session, machine } = await ingestAndGetMachine();
    expect(machine.state().rect).toEqual({ x: 0, y: 0, w: 2, h: 1 });
    session.dispose();
    handle.dispose();
  });

  it("hit-tests the corner grips and misses far outside", async () => {
    const { handle, session, machine } = await ingestAndGetMachine();
    // The top-left grip is at (0,0).
    expect(machine.hitTest([0, 0])).toBe(0); // Handle::TopLeft
    // The bottom-right grip is at (2,1).
    expect(machine.hitTest([2, 1])).toBe(4); // Handle::BottomRight
    // Far outside the (grip-tolerance-padded) body → a miss (-1). On this
    // 2×1 image the default grab tolerance is large in image px, so the
    // probe must be well beyond it.
    expect(machine.hitTest([100, 100])).toBe(-1);
    session.dispose();
    handle.dispose();
  });

  it("an aspect preset re-imposes the ratio, fitting the image", async () => {
    const { handle, session, machine } = await ingestAndGetMachine();
    machine.setPreset("1:1");
    const r = machine.state().rect;
    // 1:1 on a 2×1 image → a square that FITS (1×1), ratio preserved (not a
    // per-axis clip that would leave 2×1).
    expect(Math.abs(r.w - r.h)).toBeLessThan(1e-3);
    expect(r.w).toBeLessThanOrEqual(2 + 1e-3);
    expect(r.h).toBeLessThanOrEqual(1 + 1e-3);
    session.dispose();
    handle.dispose();
  });

  it("the overlay polyline has four corners (closed crop frame)", async () => {
    const { handle, session, machine } = await ingestAndGetMachine();
    const poly = machine.overlayPolyline();
    expect(poly).toHaveLength(4);
    // Angle 0 → axis-aligned: corner 0 is the rect TL.
    expect(poly[0]).toEqual([0, 0]);
    session.dispose();
    handle.dispose();
  });
});

describe("the crop commit (real engine wasm)", () => {
  it("commits a sub-rect, swaps the engine source, and refreshes the histogram", async () => {
    const fake = makeFakeEditor();
    // A 2×1 PSD; crop to the left 1×1 pixel.
    fake.placed.set("u1", psdBytes());
    fake.geometry.set("u1", geomFor("u1", [0, 0, 100, 200]));
    fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);
    await session.ingestSelection();

    const machine = session.cropMachine()!;
    // Drag the right edge (handle 3) in from x=2 to x=1 → a 1×1 crop.
    machine.pointerDown([2, 0.5]);
    machine.pointerMove([1, 0.5]);
    machine.pointerUp();
    expect(Math.round(machine.state().rect.w)).toBe(1);

    expect(await session.commitCrop()).toBe(true);
    const src = session.state().source!;
    expect(src.width).toBe(1);
    expect(src.height).toBe(1);
    // The histogram followed the cropped pixels (1 pixel → totals of 1).
    const hist = session.state().histogram!;
    expect(hist.r.reduce((a, b) => a + b, 0)).toBe(1);

    session.dispose();
    handle.dispose();
  });

  it("commitCrop with nothing ingested fails honestly (no machine)", async () => {
    const fake = makeFakeEditor();
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    // No ingest → no crop machine, no source.
    expect(session.cropMachine()).toBeNull();
    expect(await session.commitCrop()).toBe(false);
    expect(session.state().status).toMatch(/Nothing to crop/);

    session.dispose();
    handle.dispose();
  });

  it("a full-rect commit re-cuts the same extent (identity crop)", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u1", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);
    await session.ingestSelection();

    // The default rect is the full image; committing it yields the same dims.
    expect(await session.commitCrop()).toBe(true);
    expect(session.state().source!.width).toBe(2);
    expect(session.state().source!.height).toBe(1);

    session.dispose();
    handle.dispose();
  });
});
