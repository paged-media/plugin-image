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
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

// THE LAYER GRAPH's glue: the engine facade's layer wire, the session's
// stack lifecycle over the REAL image-js wasm (no navigator.gpu in Node,
// which is exactly the interesting case — a one-layer document and an
// empty added layer both fold trivially and need no device), and the
// panel's Layers palette including its SCOPE NOTE, pinned so it cannot
// be edited away silently.
//
// What is NOT here: composited pixels. Every blend is a WGSL dispatch,
// so the pixel proofs live in Rust (image-js/src/layers.rs's device
// tests) and the undo proofs in image-graph's journal suite.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import {
  EMPTY_LAYER_STACK,
  wrapEngine,
  type ImageWasmModule,
} from "../src/engine";
import { createImageSession } from "../src/session";
import { LAYERS_SCOPE_NOTE, LayersSection } from "../src/panels/image-panel";
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

// ── the engine facade's layer wire ───────────────────────────────────

describe("the engine facade's layer doors", () => {
  function fakeWasm() {
    const calls: Array<{ fn: string; args: unknown[] }> = [];
    let list =
      '{"active":1,"layers":[' +
      '{"index":0,"id":1,"name":"Background","visible":true,"locked":false,"opacity":1,"blend":"normal"},' +
      '{"index":1,"id":2,"name":"Pa\\"int","visible":false,"locked":true,"opacity":0.5,"blend":"multiply"}' +
      "]}";
    let history = "null";
    const engine = wrapEngine({
      layers_list: () => list,
      layers_history: () => history,
      layers_open: (...args: unknown[]) => calls.push({ fn: "open", args }),
      layers_add: (...args: unknown[]) => {
        calls.push({ fn: "add", args });
        return 1;
      },
      layers_set_blend: (...args: unknown[]) =>
        calls.push({ fn: "blend", args }),
      layers_undo: async () => "Paint",
    } as unknown as ImageWasmModule);
    return {
      engine,
      calls,
      setList: (v: string) => {
        list = v;
      },
      setHistory: (v: string) => {
        history = v;
      },
    };
  }

  it("maps the stack JSON, escaping and all, BOTTOM-first", () => {
    const { engine } = fakeWasm();
    const s = engine.layers();
    expect(s.active).toBe(1);
    expect(s.layers).toHaveLength(2);
    expect(s.layers[0]).toEqual({
      index: 0,
      id: 1,
      name: "Background",
      visible: true,
      locked: false,
      opacity: 1,
      blend: "normal",
    });
    // The engine escapes names into the JSON; a quote survives intact.
    expect(s.layers[1].name).toBe('Pa"int');
    expect(s.layers[1]).toMatchObject({
      visible: false,
      locked: true,
      opacity: 0.5,
    });
  });

  it("an empty stack reads as the shared EMPTY constant, never undefined", () => {
    const { engine, setList } = fakeWasm();
    setList('{"active":-1,"layers":[]}');
    expect(engine.layers()).toBe(EMPTY_LAYER_STACK);
  });

  it("the history readout is null before a stack is open, and typed after", () => {
    const { engine, setHistory } = fakeWasm();
    expect(engine.layersHistory()).toBeNull();
    setHistory(
      '{"canUndo":true,"canRedo":false,"depth":2,"redoDepth":0,"bytes":1024,' +
        '"maxBytes":268435456,"maxEntries":32,"dropped":3,"generation":7,' +
        '"undoLabel":"Paint","redoLabel":null}',
    );
    expect(engine.layersHistory()).toEqual({
      canUndo: true,
      canRedo: false,
      depth: 2,
      redoDepth: 0,
      bytes: 1024,
      maxBytes: 268435456,
      maxEntries: 32,
      dropped: 3,
      generation: 7,
      undoLabel: "Paint",
      redoLabel: null,
    });
  });

  it("forwards the mutations verbatim", async () => {
    const { engine, calls } = fakeWasm();
    engine.layersOpen(4);
    engine.layerAdd("Paint");
    engine.layerSetBlend(1, "multiply");
    expect(calls).toEqual([
      { fn: "open", args: [4] },
      { fn: "add", args: ["Paint"] },
      { fn: "blend", args: [1, "multiply"] },
    ]);
    expect(await engine.layersUndo()).toBe("Paint");
  });
});

// ── the session, over the REAL engine wasm ───────────────────────────

describe("the layer session (real engine wasm, no GPU in Node)", () => {
  async function ingest() {
    const fake = makeFakeEditor();
    fake.placed.set("u1", psdBytes());
    fake.geometry.set("u1", {
      id: { kind: "rectangle", id: "u1" } as never,
      pageId: "pg1",
      bounds: [0, 0, 100, 200],
    });
    fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);
    expect(await session.ingestSelection()).toBe(true);
    return { fake, handle, session };
  }

  it("an ingest opens a one-layer stack — so painting always lands in a layer", async () => {
    const { handle, session } = await ingest();
    const s = session.state();
    expect(s.layers.layers).toHaveLength(1);
    expect(s.layers.active).toBe(0);
    expect(s.layers.layers[0]).toMatchObject({
      name: "Background",
      visible: true,
      locked: false,
      opacity: 1,
      blend: "normal",
    });
    // …and its history starts empty, with the BOUND already stated.
    expect(s.history).toMatchObject({
      canUndo: false,
      canRedo: false,
      depth: 0,
      dropped: 0,
      maxEntries: 32,
      maxBytes: 256 * 1024 * 1024,
    });
    handle.dispose();
  });

  it("adding an empty layer needs no GPU (the fold skips it, exactly)", async () => {
    // A fully transparent layer is the identity for every blend mode, so
    // the composite stays trivial — which is what keeps "add a layer"
    // working in a realm with no device instead of failing at it.
    const { handle, session } = await ingest();
    expect(await session.addLayer("Paint")).toBe(true);
    const s = session.state();
    expect(s.layers.layers.map((l) => l.name)).toEqual(["Background", "Paint"]);
    expect(s.layers.active).toBe(1);
    expect(s.status).toContain("Paint");
    expect(s.gpu).toBe(false);
    handle.dispose();
  });

  it("layer properties round-trip through the engine, and a bad blend is refused", async () => {
    const { handle, session } = await ingest();
    await session.addLayer("Paint");
    expect(await session.setLayerOpacity(1, 0.25)).toBe(true);
    expect(session.state().layers.layers[1].opacity).toBeCloseTo(0.25, 5);
    expect(await session.setLayerBlend(1, "multiply")).toBe(true);
    expect(session.state().layers.layers[1].blend).toBe("multiply");
    expect(session.setLayerLocked(1, true)).toBe(true);
    expect(session.state().layers.layers[1].locked).toBe(true);
    // An unregistered mode is a clean refusal carrying the engine's own
    // reason — never a silent fall back to normal.
    expect(await session.setLayerBlend(1, "dissolve")).toBe(false);
    expect(session.state().status).toContain("dissolve");
    expect(session.state().layers.layers[1].blend).toBe("multiply");
    handle.dispose();
  });

  it("the last layer cannot be removed, and a removal says it is not undoable", async () => {
    const { handle, session } = await ingest();
    expect(await session.removeLayer(0)).toBe(false);
    expect(session.state().status).toContain("only layer");
    await session.addLayer("Paint");
    expect(await session.removeLayer(1)).toBe(true);
    expect(session.state().layers.layers).toHaveLength(1);
    expect(session.state().status).toContain("undo history was CLEARED");
    expect(session.state().history).toMatchObject({ canUndo: false, depth: 0 });
    handle.dispose();
  });

  it("reordering moves the layer and carries the active selection with it", async () => {
    const { handle, session } = await ingest();
    await session.addLayer("Paint");
    expect(await session.reorderLayer(1, 0)).toBe(true);
    const s = session.state();
    expect(s.layers.layers.map((l) => l.name)).toEqual(["Paint", "Background"]);
    expect(s.layers.active).toBe(0);
    handle.dispose();
  });

  it("undo with an empty journal says so instead of pretending", async () => {
    const { handle, session } = await ingest();
    expect(await session.undo()).toBe(false);
    expect(session.state().status).toContain("Nothing to undo");
    expect(await session.redo()).toBe(false);
    expect(session.state().status).toContain("Nothing to redo");
    handle.dispose();
  });

  it("baking at identity is refused with the engine's reason", async () => {
    const { handle, session } = await ingest();
    expect(await session.bakeAdjustToLayer()).toBe(false);
    expect(session.state().status).toContain("identity");
    handle.dispose();
  });

  it("a PSD with no layer records keeps the flatten AND says why", async () => {
    // The test PSD is a flat composite with no layer section, so the
    // layered import declines — and the panel is told, rather than the
    // flatten happening silently.
    const { handle, session } = await ingest();
    expect(session.state().layersNote).toContain("declined");
    expect(session.state().layersNote).toContain("no layer records");
    expect(session.state().layers.layers).toHaveLength(1);
    handle.dispose();
  });

  it("disposing closes the stack with the source", async () => {
    const { handle, session } = await ingest();
    session.dispose();
    expect(session.state().layers).toBe(EMPTY_LAYER_STACK);
    expect(session.state().history).toBeNull();
    handle.dispose();
  });
});

// ── the panel's Layers palette ───────────────────────────────────────

/** Collect the visible text of a React element tree WITHOUT a DOM (the
 *  brush spec's walker, widened to include `title` strings — a
 *  disabled-because note lives there and is still something the UI
 *  says). Pure components only; hooks are not run. */
function textOf(node: unknown, out: string[] = []): string[] {
  if (node === null || node === undefined || typeof node === "boolean")
    return out;
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
  if (el.props) {
    if (typeof el.props.title === "string") out.push(el.props.title);
    textOf(el.props.children, out);
  }
  return out;
}

describe("the panel's Layers section", () => {
  const layer = (over: Partial<import("../src/engine").LayerInfo> = {}) => ({
    index: 0,
    id: 1,
    name: "Background",
    visible: true,
    locked: false,
    opacity: 1,
    blend: "normal",
    kind: "pixels" as const,
    hasMask: false,
    maskEnabled: true,
    clipped: false,
    ...over,
  });

  const history = (
    over: Partial<import("../src/engine").LayerHistory> = {},
  ) => ({
    canUndo: true,
    canRedo: false,
    depth: 2,
    redoDepth: 0,
    bytes: 12 * 1024 * 1024,
    maxBytes: 256 * 1024 * 1024,
    maxEntries: 32,
    dropped: 0,
    generation: 2,
    undoLabel: "Paint",
    redoLabel: null,
    undoSteps: ["Paint"],
    redoSteps: [],
    ...over,
  });

  const render = (over: Partial<Parameters<typeof LayersSection>[0]> = {}) =>
    textOf(
      LayersSection({
        layers: [
          layer(),
          layer({
            index: 1,
            id: 2,
            name: "Paint",
            blend: "multiply",
            opacity: 0.5,
          }),
        ],
        active: 1,
        history: history(),
        blendModes: ["normal", "multiply", "screen"],
        layersNote: null,
        gpu: true,
        disabled: false,
        onSelect: () => {},
        onAdd: () => {},
        onDuplicate: () => {},
        onRemove: () => {},
        onMove: () => {},
        onVisible: () => {},
        onOpacity: () => {},
        onBlend: () => {},
        onLock: () => {},
        onUndoTo: () => {},
        onRedoTo: () => {},
        onAddAdjustment: () => {},
        onMaskFromSelection: () => {},
        onMaskToggle: () => {},
        onClip: () => {},
        onMaskClear: () => {},
        onBake: () => {},
        onUndo: () => {},
        onRedo: () => {},
        ...over,
      }),
    )
      .join(" ")
      // The walker yields fragment-per-child, so collapse the seams the
      // JSX interpolation leaves behind.
      .replace(/\s+/g, " ");

  it("lists every layer and marks the active one", () => {
    const text = render();
    expect(text).toContain("Layers (2)");
    expect(text).toContain("Background");
    expect(text).toContain("Paint");
    // The active row is marked in the UI, not only in the props.
    expect(text).toContain("Paint ●");
  });

  it("populates each row's blend picker from the engine's registry list", () => {
    expect(render()).toContain("screen");
    // With no engine list the row still shows its OWN mode rather than a
    // plausible-looking hardcoded one.
    expect(render({ blendModes: [] })).toContain("multiply");
  });

  it("refuses to offer removal of the only layer", () => {
    const one = render({ layers: [layer()], active: 0 });
    expect(one).toContain("A document keeps at least one layer");
    expect(render()).toContain("NOT undoable");
  });

  it("states the history BOUND, not just the depth", () => {
    const text = render();
    expect(text).toContain("2 undo / 0 redo");
    expect(text).toContain("12.0 MB of 256.0 MB");
    expect(text).toContain("32 steps");
    expect(text).toContain("Nothing has fallen past the bound yet");
  });

  it("says out loud when the bound has actually dropped edits", () => {
    const text = render({ history: history({ dropped: 4 }) });
    expect(text).toContain("4 older edits fell past that bound");
    expect(text).toContain("now permanent");
  });

  it("states the GPU-only rule for a multi-layer composite", () => {
    expect(render({ gpu: false })).toContain(
      "Compositing more than one layer is GPU-only",
    );
    expect(render({ gpu: false, layers: [layer()] })).not.toContain(
      "Compositing more than one layer is GPU-only",
    );
  });

  it("surfaces how the stack was opened (the PSD lane's honest one-liner)", () => {
    expect(
      render({ layersNote: "Layered PSD import declined — groups" }),
    ).toContain("Layered PSD import declined");
  });

  it("RENDERS the scope note (an honesty note nobody reads is not one)", () => {
    expect(render()).toContain(LAYERS_SCOPE_NOTE);
  });

  it("the scope note names every thing a layer here is NOT", () => {
    expect(LAYERS_SCOPE_NOTE).toContain("canvas-extent PIXEL layers");
    expect(LAYERS_SCOPE_NOTE).toContain("compose.* kernels");
    // The four modeling gaps, named.
    expect(LAYERS_SCOPE_NOTE).toContain("no groups");
    expect(LAYERS_SCOPE_NOTE).toContain("no clipping masks");
    expect(LAYERS_SCOPE_NOTE).toContain("no per-layer masks");
    expect(LAYERS_SCOPE_NOTE).toContain("no adjustment layers");
    // What undo does and does NOT cover.
    expect(LAYERS_SCOPE_NOTE).toContain("Undo covers PIXEL edits only");
    expect(LAYERS_SCOPE_NOTE).toContain("removing a layer clears the history");
    // The consequences people would otherwise discover by losing work.
    expect(LAYERS_SCOPE_NOTE).toContain("FLATTENS");
    expect(LAYERS_SCOPE_NOTE).toContain("undo history goes with it");
    expect(LAYERS_SCOPE_NOTE).toContain(
      "Export and save-back write the FLATTENED composite, not the layers",
    );
    // The PSD lane's exact condition.
    expect(LAYERS_SCOPE_NOTE).toContain("flat, unclipped, unmasked, 8-bit RGB");
    expect(LAYERS_SCOPE_NOTE).toContain("says why");
  });
});

// ── clipping (2026-08-06) ────────────────────────────────────────────

describe("the layer row's clip toggle", () => {
  const layer = (over: Partial<import("../src/engine").LayerInfo> = {}) => ({
    index: 0,
    id: 1,
    name: "Background",
    visible: true,
    locked: false,
    opacity: 1,
    blend: "normal",
    kind: "pixels" as const,
    hasMask: false,
    maskEnabled: true,
    clipped: false,
    ...over,
  });
  const history = () => ({
    canUndo: false,
    canRedo: false,
    depth: 0,
    redoDepth: 0,
    bytes: 0,
    maxBytes: 1024,
    maxEntries: 8,
    dropped: 0,
    generation: 0,
    undoLabel: null,
    redoLabel: null,
    undoSteps: [],
    redoSteps: [],
  });
  const render = (over: Partial<Parameters<typeof LayersSection>[0]> = {}) =>
    textOf(
      LayersSection({
        layers: [layer(), layer({ index: 1, id: 2, name: "Adjust" })],
        active: 1,
        history: history(),
        blendModes: ["normal"],
        layersNote: null,
        gpu: true,
        disabled: false,
        onSelect: () => {},
        onAdd: () => {},
        onDuplicate: () => {},
        onRemove: () => {},
        onMove: () => {},
        onVisible: () => {},
        onOpacity: () => {},
        onBlend: () => {},
        onLock: () => {},
        onUndoTo: () => {},
        onRedoTo: () => {},
        onAddAdjustment: () => {},
        onMaskFromSelection: () => {},
        onMaskToggle: () => {},
        onClip: () => {},
        onMaskClear: () => {},
        onBake: () => {},
        onUndo: () => {},
        onRedo: () => {},
        ...over,
      }),
    )
      .join(" ")
      .replace(/\s+/g, " ");

  it("renders a clip control on every row", () => {
    // The glyph is the affordance; its presence is what a spec can see
    // without a DOM.
    expect(render()).toContain("⌐");
  });

  it("explains what clipping IS in the row's own title", () => {
    // A designer meeting this control for the first time should not have
    // to know Photoshop to understand it — and the smart-filter framing
    // is the reason it exists at all.
    const tree = LayersSection({
      layers: [layer(), layer({ index: 1, id: 2, name: "Adjust" })],
      active: 1,
      history: history(),
      blendModes: ["normal"],
      layersNote: null,
      gpu: true,
      disabled: false,
      onSelect: () => {},
      onAdd: () => {},
      onDuplicate: () => {},
      onRemove: () => {},
      onMove: () => {},
      onVisible: () => {},
      onOpacity: () => {},
      onBlend: () => {},
      onLock: () => {},
      onUndoTo: () => {},
      onRedoTo: () => {},
      onAddAdjustment: () => {},
      onMaskFromSelection: () => {},
      onMaskToggle: () => {},
      onClip: () => {},
      onMaskClear: () => {},
      onBake: () => {},
      onUndo: () => {},
      onRedo: () => {},
    });
    const titles = collectTitles(tree);
    expect(titles.some((t) => t.includes("smart filter"))).toBe(true);
    expect(titles.some((t) => t.includes("Clip to the layer below"))).toBe(
      true,
    );
  });
});

/** Every `title` prop in a pure element tree. */
function collectTitles(node: unknown, out: string[] = []): string[] {
  if (!node || typeof node !== "object") return out;
  if (Array.isArray(node)) {
    for (const n of node) collectTitles(n, out);
    return out;
  }
  const el = node as { type?: unknown; props?: Record<string, unknown> };
  if (typeof el.type === "function") {
    return collectTitles((el.type as (p: unknown) => unknown)(el.props), out);
  }
  if (typeof el.props?.title === "string") out.push(el.props.title);
  if (el.props) collectTitles(el.props.children, out);
  return out;
}
