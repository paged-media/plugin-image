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

// PAINTING ON LAYER MASKS — the glue over the REAL image-js wasm: Add
// Layer Mask (reveal all / hide all), the Pixels/Mask edit target, its
// one-undo-list step, and the panel toggle.
//
// What is NOT here: a painted mask. A stroke is a WGSL dispatch and Node
// has no GPU, so the stroke-into-mask proofs (eraser reveals, preview ==
// commit, one undo restores) live in Rust (image-js/src/layers.rs,
// `…__feat__image_editor_mask_painting`).

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import { EMPTY_LAYER_STACK } from "../src/engine";
import { LayersSection } from "../src/panels/image-panel";
import { createImageSession } from "../src/session";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";

async function ingest() {
  const fake = makeFakeEditor();
  const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
  const session = createImageSession(handle.host);
  expect(await session.importBytes("t.psd", psdBytes())).toBe(true);
  return { handle, session };
}

describe("layer masks as an edit target", () => {
  it("a stack starts on pixels", async () => {
    const { handle, session } = await ingest();
    expect(session.state().layers.editTarget).toBe("pixels");
    expect(EMPTY_LAYER_STACK.editTarget).toBe("pixels");
    session.dispose();
    handle.dispose();
  });

  it("Add layer mask (reveal all) attaches a mask and targets it; one undo removes it", async () => {
    const { handle, session } = await ingest();
    await session.addLayer("Paint");
    // An empty layer folds away, so this composites without a GPU.
    expect(await session.addLayerMask(1, true)).toBe(true);
    let s = session.state();
    expect(s.layers.layers[1].hasMask).toBe(true);
    expect(s.layers.active).toBe(1);
    expect(s.layers.editTarget).toBe("mask");
    expect(s.status).toMatch(/reveal-all/);
    expect(s.history?.undoLabel).toBe("Add layer mask (reveal all)");
    // A second mask is refused with the engine's reason.
    expect(await session.addLayerMask(1, false)).toBe(false);
    expect(session.state().status).toMatch(/already has a mask/);
    expect(await session.undo()).toBe(true);
    s = session.state();
    expect(s.layers.layers[1].hasMask).toBe(false);
    expect(s.layers.editTarget).toBe("pixels");
    session.dispose();
    handle.dispose();
  });

  it("hide all is the other form", async () => {
    const { handle, session } = await ingest();
    await session.addLayer("Paint");
    expect(await session.addLayerMask(1, false)).toBe(true);
    expect(session.state().history?.undoLabel).toBe("Add layer mask (hide all)");
    expect(session.state().status).toMatch(/hide-all/);
    session.dispose();
    handle.dispose();
  });

  it("the target toggles between pixels and mask, and a maskless layer refuses the mask", async () => {
    const { handle, session } = await ingest();
    await session.addLayer("Paint");
    expect(session.setEditTarget(1, "mask")).toBe(false);
    expect(session.state().status).toMatch(/no mask/);
    await session.addLayerMask(1, true);
    expect(session.setEditTarget(1, "pixels")).toBe(true);
    expect(session.state().layers.editTarget).toBe("pixels");
    expect(session.setEditTarget(1, "mask")).toBe(true);
    expect(session.state().layers.editTarget).toBe("mask");
    // Choosing another layer puts the target back on pixels.
    expect(session.setActiveLayer(0)).toBe(true);
    expect(session.state().layers.editTarget).toBe("pixels");
    // Choosing a target is not an undo step.
    expect(session.state().history?.undoLabel).toBe("Add layer mask (reveal all)");
    session.dispose();
    handle.dispose();
  });

  it("the Add-layer-mask commands are in the manifest and the menu", async () => {
    const { MENU_ENTRIES } = await import("../src/menu");
    const cmds = (manifestJson as PluginManifest).contributes?.commands ?? [];
    for (const suffix of ["addLayerMask", "addLayerMaskHideAll"]) {
      expect(cmds).toContain(`media.paged.image.command.${suffix}`);
      expect(MENU_ENTRIES.some(([, s]) => s === suffix)).toBe(true);
    }
  });
});

describe("the panel's Pixels/Mask toggle", () => {
  const base = {
    history: null,
    blendModes: ["normal"],
    layersNote: null,
    gpu: true,
    disabled: false,
    groups: [],
  };
  const noop = () => {};
  const handlers = {
    onSelect: noop,
    onAdd: noop,
    onDuplicate: noop,
    onRemove: noop,
    onMove: noop,
    onVisible: noop,
    onOpacity: noop,
    onBlend: noop,
    onLock: noop,
    onAddAdjustment: noop,
    onMaskFromSelection: noop,
    onMaskToggle: noop,
    onClip: noop,
    onGroup: noop,
    onUngroup: noop,
    onGroupVisible: noop,
    onGroupOpacity: noop,
    onGroupPassThrough: noop,
    onMaskClear: noop,
    onUndoTo: noop,
    onRedoTo: noop,
    onBake: noop,
    onUndo: noop,
    onRedo: noop,
    onAddMask: noop,
    onEditTarget: noop,
  };
  const layer = (index: number, hasMask: boolean) => ({
    index,
    id: index + 1,
    name: `L${index}`,
    visible: true,
    locked: false,
    opacity: 1,
    blend: "normal",
    hasMask,
    maskEnabled: true,
    clipped: false,
    group: null,
    kind: "pixels" as const,
  });

  /** Every element's props in a React tree, without a DOM (pure
   *  components only; hooks are not run). */
  function propsOf(node: unknown, out: Record<string, unknown>[] = []) {
    if (node === null || node === undefined || typeof node !== "object") return out;
    if (Array.isArray(node)) {
      for (const n of node) propsOf(n, out);
      return out;
    }
    const el = node as { type?: unknown; props?: Record<string, unknown> };
    if (typeof el.type === "function") {
      return propsOf((el.type as (p: unknown) => unknown)(el.props), out);
    }
    if (el.props) {
      out.push(el.props);
      propsOf(el.props.children, out);
    }
    return out;
  }

  it("a masked layer shows the M toggle carrying the target; a maskless one offers Add mask", () => {
    const calls: unknown[][] = [];
    const all = propsOf(
      LayersSection({
        ...base,
        ...handlers,
        onAddMask: (i, reveal) => calls.push(["add", i, reveal]),
        onEditTarget: (i, t) => calls.push(["target", i, t]),
        layers: [layer(0, false), layer(1, true)],
        active: 1,
        editTarget: "mask",
      }),
    );
    const toggle = all.find((p) => p["data-image-layer-edit-target"] === 1);
    expect(toggle?.["data-target"]).toBe("mask");
    expect(all.some((p) => p["data-image-layer-edit-target"] === 0)).toBe(false);
    const add = all.find((p) => p["data-image-layer-mask-new"] === 0);
    expect(add).toBeDefined();
    expect(all.some((p) => p["data-image-layer-mask-new"] === 1)).toBe(false);
    // Clicking the toggle on the mask target goes back to pixels; Add
    // mask is reveal-all, Alt-click hide-all.
    (toggle?.onClick as () => void)();
    (add?.onClick as (e: { altKey: boolean }) => void)({ altKey: false });
    (add?.onClick as (e: { altKey: boolean }) => void)({ altKey: true });
    expect(calls).toEqual([
      ["target", 1, "pixels"],
      ["add", 0, true],
      ["add", 0, false],
    ]);
  });
});
