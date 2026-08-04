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

// The PAINT lane's glue: the pure sampling machine (no engine, no host),
// the engine facade's brush wire, the tool gesture's begin → extend* →
// commit pump over a stubbed session, and the panel's Brush section —
// including the SCOPE NOTE, pinned so it cannot be edited away silently.
//
// What is NOT here: painted pixels. Every dab composite is a WGSL
// dispatch and Node has no navigator.gpu, so the pixel proofs live in
// Rust (image-js/src/stroke.rs's device tests + image-conformance's
// brush_stroke suite). What IS here over the real wasm: the honest
// GPU-less decline, and the blend list coming from the kernel registry.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type {
  CanvasPointerEvent,
  ElementGeometryItem,
  PluginManifest,
} from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import {
  createBrushMachine,
  normalizePressure,
  tipOutline,
  NON_PEN_PRESSURE,
  TIP_SEGMENTS,
  type BrushSample,
} from "../src/brush-machine";
import { makeBrushGesture } from "../src/brush-tool";
import { DEFAULT_BRUSH_PARAMS, wrapEngine, type ImageWasmModule } from "../src/engine";
import { createImageSession, type ImageSession } from "../src/session";
import { BRUSH_SCOPE_NOTE, BrushSection } from "../src/panels/image-panel";
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

/** A minimal canvas pointer event at a page-local point. */
function ev(
  pagePoint: [number, number] | null,
  over: Partial<CanvasPointerEvent> = {},
): CanvasPointerEvent {
  return {
    pageId: "pg1",
    pagePoint,
    docPoint: pagePoint ?? [0, 0],
    modifiers: { shift: false, alt: false, cmd: false, ctrl: false },
    maxDelta: 0,
    button: 0,
    target: null,
    pressure: 0.5,
    tiltX: 0,
    tiltY: 0,
    pointerType: "mouse",
    ...over,
  };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

// ── the pure machine ─────────────────────────────────────────────────

describe("pressure normalization (the one place the rule lives)", () => {
  it("substitutes full pressure for every non-pen device", () => {
    // Pointer Events report a constant 0.5 for a held mouse/finger — a
    // placeholder, not a measurement. Passing it through would make
    // every mouse stroke permanently half-size under `both`.
    expect(normalizePressure(0.5, "mouse")).toBe(NON_PEN_PRESSURE);
    expect(normalizePressure(0.5, "touch")).toBe(1);
    expect(normalizePressure(0, "mouse")).toBe(1);
  });

  it("passes a pen's own reading through, clamped to [0,1]", () => {
    expect(normalizePressure(0.37, "pen")).toBeCloseTo(0.37, 6);
    expect(normalizePressure(1.5, "pen")).toBe(1);
    expect(normalizePressure(-1, "pen")).toBe(0);
    expect(normalizePressure(Number.NaN, "pen")).toBe(1);
  });
});

describe("the brush machine (pure — no engine, no host)", () => {
  it("down opens a stroke and queues its first sample", () => {
    const m = createBrushMachine();
    expect(m.state().drawing).toBe(false);
    expect(m.down([4, 5], 0.5, "mouse")).toBe(true);
    expect(m.state()).toEqual({ drawing: true, ended: false, sampleCount: 1, queued: 1 });
    expect(m.next()).toEqual({ x: 4, y: 5, pressure: 1 });
    expect(m.next()).toBeNull();
  });

  it("a second down never silently restarts an open stroke", () => {
    const m = createBrushMachine();
    m.down([0, 0], 0.5, "mouse");
    expect(m.down([9, 9], 0.5, "mouse")).toBe(false);
    expect(m.state().sampleCount).toBe(1);
  });

  it("a move before down is ignored", () => {
    const m = createBrushMachine();
    expect(m.move([1, 1], 0.5, "mouse")).toBe(false);
    expect(m.state().queued).toBe(0);
  });

  it("moves queue in order and drain FIFO — no sample is dropped", () => {
    const m = createBrushMachine();
    m.down([0, 0], 0.5, "mouse");
    for (let i = 1; i <= 4; i++) m.move([i, i * 2], 0.5, "mouse");
    expect(m.state().queued).toBe(5);
    const drained: BrushSample[] = [];
    for (let s = m.next(); s; s = m.next()) drained.push(s);
    expect(drained.map((s) => [s.x, s.y])).toEqual([
      [0, 0],
      [1, 2],
      [2, 4],
      [3, 6],
      [4, 8],
    ]);
  });

  it("an EXACT duplicate is refused (a provable engine no-op), a pressure change is not", () => {
    const m = createBrushMachine();
    m.down([2, 2], 0.4, "pen");
    expect(m.move([2, 2], 0.4, "pen")).toBe(false);
    expect(m.move([2, 2], 0.9, "pen")).toBe(true);
    expect(m.move([2.0001, 2], 0.9, "pen")).toBe(true);
    expect(m.state().sampleCount).toBe(3);
  });

  it("non-finite points are refused", () => {
    const m = createBrushMachine();
    expect(m.down([Number.NaN, 0], 0.5, "mouse")).toBe(false);
    expect(m.state().drawing).toBe(false);
    m.down([0, 0], 0.5, "mouse");
    expect(m.move([Number.POSITIVE_INFINITY, 1], 0.5, "mouse")).toBe(false);
  });

  it("drained() is the commit handshake: up() plus an empty queue", () => {
    const m = createBrushMachine();
    m.down([0, 0], 0.5, "mouse");
    m.move([1, 0], 0.5, "mouse");
    expect(m.drained()).toBe(false); // not up yet
    m.up();
    expect(m.state()).toMatchObject({ drawing: false, ended: true, queued: 2 });
    expect(m.drained()).toBe(false); // samples still owed to the engine
    m.next();
    expect(m.drained()).toBe(false);
    m.next();
    expect(m.drained()).toBe(true);
  });

  it("up() without a stroke is a no-op", () => {
    const m = createBrushMachine();
    m.up();
    expect(m.drained()).toBe(false);
    expect(m.state().ended).toBe(false);
  });

  it("cancel drops the queue, the record and the stroke", () => {
    const m = createBrushMachine();
    m.down([0, 0], 0.5, "mouse");
    m.move([5, 5], 0.5, "mouse");
    m.cancel();
    expect(m.state()).toEqual({ drawing: false, ended: false, sampleCount: 0, queued: 0 });
    expect(m.next()).toBeNull();
    expect(m.drained()).toBe(false);
    // …and the machine is reusable for the next stroke.
    expect(m.down([1, 1], 0.5, "mouse")).toBe(true);
  });

  it("samples() is the replay record, with pressures already normalized", () => {
    const m = createBrushMachine();
    m.down([0, 0], 0.5, "mouse");
    m.move([3, 4], 0.25, "pen");
    expect(m.samples()).toEqual([
      { x: 0, y: 0, pressure: 1 },
      { x: 3, y: 4, pressure: 0.25 },
    ]);
  });

  it("the tip ring is a closed circle of the NOMINAL diameter", () => {
    const ring = tipOutline([10, 10], 8);
    expect(ring).toHaveLength(TIP_SEGMENTS);
    expect(ring[0]).toEqual([14, 10]); // +radius on x
    for (const [x, y] of ring) {
      expect(Math.hypot(x - 10, y - 10)).toBeCloseTo(4, 6);
    }
    // Degenerate sizes collapse to the centre instead of throwing.
    expect(tipOutline([1, 2], 0, 3)).toEqual([
      [1, 2],
      [1, 2],
      [1, 2],
    ]);
  });
});

// ── the engine facade's brush wire ───────────────────────────────────

describe("the engine facade's brush doors", () => {
  /** Records the snake_case calls a fake wasm module receives. */
  function fakeWasm() {
    const calls: Array<{ fn: string; args: unknown[] }> = [];
    let stats: Float64Array = new Float64Array([7, 1, 2, 30, 40]);
    const wasm = {
      brush_stroke_begin: (...args: unknown[]) => {
        calls.push({ fn: "begin", args });
      },
      brush_stroke_extend: async (...args: unknown[]) => {
        calls.push({ fn: "extend", args });
        return new Uint8Array([1, 2, 3, 4]);
      },
      // Async since the layer lane: the commit writes into the active
      // layer and re-composites the stack before it answers.
      brush_stroke_commit: async () => ({ handle: 9, width: 4, height: 2, free() {} }),
      brush_stroke_cancel: () => {
        calls.push({ fn: "cancel", args: [] });
      },
      brush_stroke_active: () => true,
      brush_stroke_stats: () => stats,
      brush_blend_modes: () => "normal\nmultiply\nscreen",
    } as unknown as ImageWasmModule;
    return {
      engine: wrapEngine(wasm),
      calls,
      setStats: (s: Float64Array) => {
        stats = s;
      },
    };
  }

  it("begin forwards the frozen params in the door's argument order", () => {
    const { engine, calls } = fakeWasm();
    engine.brushBegin(3, "pencil", {
      ...DEFAULT_BRUSH_PARAMS,
      size: 12,
      hardness: 0.25,
      opacity: 0.8,
      flow: 0.6,
      spacing: 0.1,
      blend: "multiply",
      color: [1, 0, 0.5, 1],
      pressureTarget: "size",
    });
    expect(calls[0].fn).toBe("begin");
    const [handle, tool, size, hardness, opacity, flow, spacing, blend, color, target] =
      calls[0].args;
    expect([handle, tool, size, hardness, opacity, flow, spacing, blend, target]).toEqual([
      3,
      "pencil",
      12,
      0.25,
      0.8,
      0.6,
      0.1,
      "multiply",
      "size",
    ]);
    expect(Array.from(color as Float32Array)).toEqual([1, 0, 0.5, 1]);
  });

  it("commit hands back the engine handle and frees the wrapper", async () => {
    const { engine } = fakeWasm();
    expect(await engine.brushExtend(1, 2, 0.5)).toEqual(new Uint8Array([1, 2, 3, 4]));
    expect(await engine.brushCommit()).toEqual({ handle: 9, width: 4, height: 2 });
    expect(engine.brushActive()).toBe(true);
  });

  it("stats map the flat readout, and an EMPTY one means 'nothing landed yet'", () => {
    const { engine, setStats } = fakeWasm();
    expect(engine.brushStats()).toEqual({ dabs: 7, x: 1, y: 2, w: 30, h: 40 });
    setStats(new Float64Array([]));
    expect(engine.brushStats()).toBeNull();
  });

  it("the blend list is the engine's newline-separated registry, not a TS list", () => {
    const { engine } = fakeWasm();
    expect(engine.brushBlendModes()).toEqual(["normal", "multiply", "screen"]);
  });
});

// ── the tool gesture (the drain pump) ────────────────────────────────

/** A stubbed session that records the brush calls the tool makes. The
 *  engine half is proven in Rust; what matters here is the SEQUENCE. */
function stubSession(begin: () => Promise<boolean> = async () => true) {
  const machine = createBrushMachine();
  const calls: string[] = [];
  const session = {
    state: () => ({
      source: { elementId: "u1", width: 2, height: 1 },
      brush: DEFAULT_BRUSH_PARAMS,
    }),
    brushMachine: () => machine,
    brushBegin: async (tool: string) => {
      const ok = await begin();
      calls.push(ok ? `begin:${tool}` : `begin-refused:${tool}`);
      return ok;
    },
    brushExtend: async (s: BrushSample) => {
      calls.push(`extend:${s.x},${s.y},${s.pressure}`);
      return true;
    },
    brushCommit: async () => {
      calls.push("commit");
      return true;
    },
    brushCancel: () => calls.push("cancel"),
  } as unknown as ImageSession;
  return { session, calls, machine };
}

async function gestureFixture(begin?: () => Promise<boolean>) {
  const fake = makeFakeEditor();
  // Frame box [top 0, left 0, bottom 100, right 200] with a 2×1 image ⇒
  // aspect-fit scale 100, origin (0,0): page (100,100) is image (1,1).
  fake.geometry.set("u1", geomFor("u1", [0, 0, 100, 200]));
  const handle = makeHost(fake);
  const { session, calls, machine } = stubSession(begin);
  const gesture = makeBrushGesture(handle.host, session, "brush");
  gesture.onActivate(undefined as never);
  await tick();
  return { fake, handle, gesture, calls, machine };
}

describe("the paint gesture", () => {
  it("runs begin → extend per sample → commit on release", async () => {
    const { handle, gesture, calls } = await gestureFixture();
    gesture.onPointerDown(ev([0, 0]));
    await tick();
    gesture.onPointerMove(ev([100, 100]));
    await tick();
    gesture.onPointerMove(ev([200, 0]));
    await tick();
    gesture.onPointerUp(ev([200, 0]));
    await tick();
    expect(calls).toEqual([
      "begin:brush",
      "extend:0,0,1", // page → image px through the frame's fit transform
      "extend:1,1,1",
      "extend:2,0,1",
      "commit",
    ]);
    handle.dispose();
  });

  it("a fast drag delivered before the engine catches up loses NO sample", async () => {
    // Every pointer event fires before a single await resolves — the
    // machine queues them and the pump drains them in order, so the
    // painted path is the pointer's path, not a subsample of it.
    const { handle, gesture, calls } = await gestureFixture();
    gesture.onPointerDown(ev([0, 0]));
    gesture.onPointerMove(ev([50, 50]));
    gesture.onPointerMove(ev([100, 100]));
    gesture.onPointerMove(ev([150, 50]));
    gesture.onPointerUp(ev([150, 50]));
    await tick();
    await tick();
    expect(calls).toEqual([
      "begin:brush",
      "extend:0,0,1",
      "extend:0.5,0.5,1",
      "extend:1,1,1",
      "extend:1.5,0.5,1",
      "commit",
    ]);
    handle.dispose();
  });

  it("a pen's pressure reaches the engine; a mouse's placeholder does not", async () => {
    const { handle, gesture, calls } = await gestureFixture();
    gesture.onPointerDown(ev([0, 0], { pointerType: "pen", pressure: 0.25 }));
    await tick();
    gesture.onPointerMove(ev([100, 100], { pointerType: "pen", pressure: 0.75 }));
    await tick();
    gesture.onPointerUp(ev([100, 100], { pointerType: "pen", pressure: 0.75 }));
    await tick();
    expect(calls).toEqual([
      "begin:brush",
      "extend:0,0,0.25",
      "extend:1,1,0.75",
      "commit",
    ]);
    handle.dispose();
  });

  it("a refused begin (no GPU, nothing ingested) paints nothing at all", async () => {
    const { handle, gesture, calls, machine } = await gestureFixture(async () => false);
    gesture.onPointerDown(ev([0, 0]));
    await tick();
    gesture.onPointerMove(ev([100, 100]));
    await tick();
    gesture.onPointerUp(ev([100, 100]));
    await tick();
    expect(calls).toEqual(["begin-refused:brush"]);
    expect(machine.state()).toMatchObject({ drawing: false, queued: 0 });
    handle.dispose();
  });

  it("publishes the brush-tip ring in page pt and clears it off-page", async () => {
    const { fake, handle, gesture } = await gestureFixture();
    gesture.onPointerMove(ev([100, 50]));
    const ring = fake.overlayShapes.at(-1) as {
      pageId: string;
      points: Array<[number, number]>;
      close?: boolean;
    };
    expect(ring.pageId).toBe("pg1");
    expect(ring.close).toBe(true);
    expect(ring.points).toHaveLength(TIP_SEGMENTS);
    // 24 px tip at scale 100 ⇒ a 1200 pt radius ring around (100, 50).
    for (const [x, y] of ring.points) {
      expect(Math.hypot(x - 100, y - 50)).toBeCloseTo(1200, 3);
    }
    gesture.onPointerMove(ev(null));
    expect(fake.overlayShapes.at(-1)).toBeNull();
    handle.dispose();
  });

  it("deactivating mid-stroke cancels it and clears the overlay", async () => {
    const { fake, handle, gesture, calls } = await gestureFixture();
    gesture.onPointerDown(ev([0, 0]));
    await tick();
    gesture.onDeactivate("switch");
    expect(calls).toContain("cancel");
    expect(fake.overlayShapes.at(-1)).toBeNull();
    handle.dispose();
  });
});

// ── the session, over the REAL engine wasm ───────────────────────────

describe("the paint session (real engine wasm, no GPU in Node)", () => {
  async function ingest() {
    const fake = makeFakeEditor();
    fake.placed.set("u1", psdBytes());
    fake.geometry.set("u1", geomFor("u1", [0, 0, 100, 200]));
    fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);
    expect(await session.ingestSelection()).toBe(true);
    return { fake, handle, session };
  }

  it("starts from the engine's documented v0 defaults", async () => {
    const { handle, session } = await ingest();
    expect(session.state().brush).toEqual({
      size: 24,
      hardness: 0.5,
      opacity: 1,
      flow: 1,
      spacing: 0.25,
      blend: "normal",
      color: [0, 0, 0, 1],
      pressureTarget: "both",
    });
    session.dispose();
    handle.dispose();
  });

  it("setBrushParams merges without disturbing the rest", async () => {
    const { handle, session } = await ingest();
    session.setBrushParams({ size: 64, blend: "multiply" });
    expect(session.state().brush).toMatchObject({
      size: 64,
      blend: "multiply",
      hardness: 0.5,
      color: [0, 0, 0, 1],
    });
    session.dispose();
    handle.dispose();
  });

  it("the blend list comes from the compose.* kernel registry (26 modes)", async () => {
    // Reading it is pure CPU, so it populates even with no GPU device —
    // and it is the registry's list, which is what stops the panel's
    // picker from ever drifting from the kernels that exist.
    const { handle, session } = await ingest();
    const modes = session.state().blendModes;
    expect(modes).toHaveLength(26);
    expect(modes).toContain("normal");
    expect(modes).toContain("multiply");
    expect(modes).toEqual([...modes].sort());
    session.dispose();
    handle.dispose();
  });

  it("declines to paint without a WebGPU device, and says so", async () => {
    const { handle, session } = await ingest();
    expect(session.state().gpu).toBe(false); // Node has no navigator.gpu
    expect(await session.brushBegin("brush")).toBe(false);
    expect(session.state().status).toMatch(/GPU-only/);
    expect(session.state().strokeActive).toBe(false);
    session.dispose();
    handle.dispose();
  });

  it("builds a paint machine per ingested source and drops it on dispose", async () => {
    const { handle, session } = await ingest();
    expect(session.brushMachine()).not.toBeNull();
    session.dispose();
    expect(session.brushMachine()).toBeNull();
    handle.dispose();
  });
});

// ── the panel's Brush section + the scope note ───────────────────────

/** Render a pure element tree to its text, executing the (hook-free)
 *  function components it contains. No react-dom needed. */
function textOf(node: unknown, out: string[] = []): string[] {
  if (node === null || node === undefined || typeof node === "boolean") return out;
  if (typeof node === "string" || typeof node === "number") {
    out.push(String(node));
    return out;
  }
  if (Array.isArray(node)) {
    for (const n of node) textOf(n, out);
    return out;
  }
  const el = node as { type?: unknown; props?: Record<string, unknown> };
  if (typeof el.type === "function") {
    return textOf((el.type as (p: unknown) => unknown)(el.props), out);
  }
  if (el.props) textOf(el.props.children, out);
  return out;
}

describe("the panel's Brush section", () => {
  const render = (over: Partial<Parameters<typeof BrushSection>[0]> = {}) =>
    textOf(
      BrushSection({
        brush: DEFAULT_BRUSH_PARAMS,
        blendModes: ["normal", "multiply"],
        strokeActive: false,
        strokeStats: null,
        masked: false,
        gpu: true,
        disabled: false,
        onChange: () => {},
        ...over,
      }),
    ).join(" ");

  it("offers the frozen-at-begin parameters", () => {
    const text = render();
    for (const label of [
      "Size (px)",
      "Hardness",
      "Opacity",
      "Flow",
      "Spacing (× diameter)",
      "Blend",
      "Colour",
      "Pen pressure drives",
    ]) {
      expect(text).toContain(label);
    }
  });

  it("populates the blend picker from the engine list it is handed", () => {
    expect(render({ blendModes: ["normal", "vivid-light"] })).toContain("vivid-light");
    // …and says so honestly when the engine has not booted, rather than
    // showing a plausible-looking hardcoded list.
    expect(render({ blendModes: [] })).toContain("engine not booted");
  });

  it("names the selection clip and the in-flight stroke readout", () => {
    expect(render({ masked: true })).toContain("clipped to the selection");
    expect(
      render({ strokeActive: true, strokeStats: { dabs: 12, x: 3, y: 4, w: 20, h: 30 } }),
    ).toContain("12 dabs · 3,4 20×30");
  });

  it("states the GPU-only rule when there is no device", () => {
    expect(render({ gpu: false })).toContain("Painting is GPU-only");
    expect(render({ gpu: true })).not.toContain("Painting is GPU-only");
  });

  it("RENDERS the scope note (an honesty note nobody reads is not one)", () => {
    expect(render()).toContain(BRUSH_SCOPE_NOTE);
  });

  it("the scope note states what the layer graph did and did NOT change", () => {
    // What changed: a stroke lands in the ACTIVE LAYER and is undoable.
    expect(BRUSH_SCOPE_NOTE).toContain("A stroke paints the ACTIVE LAYER");
    expect(BRUSH_SCOPE_NOTE).toContain("not the flattened image");
    expect(BRUSH_SCOPE_NOTE).toContain("UNDOABLE");
    expect(BRUSH_SCOPE_NOTE).toContain("tiles it touched are journaled");
    // …but the undo is BOUNDED, and the bound is a number, not a mood.
    expect(BRUSH_SCOPE_NOTE).toContain("BOUNDED");
    expect(BRUSH_SCOPE_NOTE).toContain("32 steps or 256 MB");
    expect(BRUSH_SCOPE_NOTE).toContain("become permanent");
    // What did NOT change: paint still covers what is under it in ITS
    // layer, and layer structure is not journaled at all.
    expect(BRUSH_SCOPE_NOTE).toContain("destructive INTO its own layer");
    expect(BRUSH_SCOPE_NOTE).toContain("what it covered is gone");
    expect(BRUSH_SCOPE_NOTE).toContain("STRUCTURE");
    expect(BRUSH_SCOPE_NOTE).toContain("is not journaled");
    // …and the document is still never touched.
    expect(BRUSH_SCOPE_NOTE).toContain(
      "The document and the source file are never touched",
    );
    // The in-flight escape hatch is real and is named.
    expect(BRUSH_SCOPE_NOTE).toContain("can be abandoned");
    // The live preview's honest seam — now the whole stack, not one buffer.
    expect(BRUSH_SCOPE_NOTE).toContain("the whole stack");
    expect(BRUSH_SCOPE_NOTE).toContain("the adjustment chain re-runs on release");
    // Words that would make it a nicer lie: undo is bounded and layer
    // pixels are still overwritten, so neither of these may appear.
    expect(BRUSH_SCOPE_NOTE).not.toContain("non-destructive");
    expect(BRUSH_SCOPE_NOTE).not.toContain("unlimited");
  });

  it("the note no longer claims the thing the layer graph disproved", () => {
    // The caveat this work exists to retire. If someone reverts the
    // layer graph without reverting the note, this fails.
    expect(BRUSH_SCOPE_NOTE).not.toContain("There is no layer graph");
    expect(BRUSH_SCOPE_NOTE).not.toContain("SINGLE engine-held image");
    expect(BRUSH_SCOPE_NOTE).not.toContain("no undo for a stroke");
    expect(BRUSH_SCOPE_NOTE).not.toContain("only restore it has");
  });
});
