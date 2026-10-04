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

// Image ▸ Rotate / Flip / Canvas Size and Select ▸ Modify over the real
// engine wasm (both are CPU work for a one-layer stack), and the GPU-only
// verbs' honest decline in Node.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import { createImageSession } from "../src/session";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";

function open() {
  const fake = makeFakeEditor();
  const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
  return { handle, session: createImageSession(handle.host) };
}

describe("Image ▸ Rotate / Flip / Canvas Size", () => {
  it("rotating a 2×1 image makes it 1×2, and the pixels follow", async () => {
    const { handle, session } = open();
    expect(await session.importBytes("r.psd", psdBytes())).toBe(true);
    expect(await session.canvasOp("rotate-cw")).toBe(true);
    expect(session.state().source).toMatchObject({ width: 1, height: 2 });
    expect(session.state().status).toMatch(/history cleared/);
    // psdBytes(): left (10,30,50), right (20,40,60); clockwise puts left on top.
    expect(session.sampleColor([0, 0])!.map((v) => Math.round(v * 255))).toEqual([10, 30, 50, 255]);
    expect(session.sampleColor([0, 1])!.map((v) => Math.round(v * 255))).toEqual([20, 40, 60, 255]);
    session.dispose();
    handle.dispose();
  });

  it("flip swaps left and right; Canvas Size adds transparent area by the anchor", async () => {
    const { handle, session } = open();
    expect(await session.importBytes("f.psd", psdBytes())).toBe(true);
    expect(await session.canvasOp("flip-h")).toBe(true);
    expect(session.sampleColor([0, 0])!.map((v) => Math.round(v * 255))).toEqual([20, 40, 60, 255]);
    expect(
      await session.canvasOp("canvas", { width: 4, height: 1, anchorX: 2, anchorY: 1 }),
    ).toBe(true);
    expect(session.state().source).toMatchObject({ width: 4, height: 1 });
    expect(session.sampleColor([0, 0])![3]).toBe(0); // added area: transparent
    expect(session.sampleColor([2, 0])!.map((v) => Math.round(v * 255))).toEqual([20, 40, 60, 255]);
    session.dispose();
    handle.dispose();
  });

  it("an unknown op is refused with the engine's reason", async () => {
    const { handle, session } = open();
    expect(await session.importBytes("x.psd", psdBytes())).toBe(true);
    expect(await session.canvasOp("skew" as never)).toBe(false);
    expect(session.state().status).toMatch(/unknown canvas operation/);
    session.dispose();
    handle.dispose();
  });
});

describe("Select ▸ Modify", () => {
  it("needs a selection, then expands it", async () => {
    const { handle, session } = open();
    expect(await session.importBytes("m.psd", psdBytes())).toBe(true);
    expect(session.modifySelection("expand", 1)).toBe(false);
    expect(session.state().status).toMatch(/no selection/);
    session.selectionMachine()!.begin("rect", [0, 0], "replace");
    session.selectionMachine()!.update([1, 1]);
    session.selectionMachine()!.end();
    expect(session.state().selection?.w).toBe(1);
    expect(session.modifySelection("expand", 1)).toBe(true);
    expect(session.state().selection?.w).toBe(2);
    session.dispose();
    handle.dispose();
  });
});

describe("GPU-only verbs decline honestly in Node", () => {
  it("median, maximum, lookup, fill and bucket say they need a GPU", async () => {
    const { handle, session } = open();
    expect(await session.importBytes("g.psd", psdBytes())).toBe(true);
    for (const run of [
      () => session.applyMedian(),
      () => session.applyMorph("max"),
      () => session.fillForeground(),
      () => session.bucketFill([0, 0]),
    ]) {
      expect(await run()).toBe(false);
      expect(session.state().status).toMatch(/GPU-only/);
    }
    // A malformed lookup is refused before any GPU work.
    expect(await session.applyColorLookup("nonsense", "bad.cube")).toBe(false);
    expect(session.state().status).toMatch(/^bad\.cube: \.cube line 1/);
    session.dispose();
    handle.dispose();
  });
});
