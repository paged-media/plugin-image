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

// The protocol-66 doors are used where the host offers them and never
// assumed: each accessor answers null (or the older lane) on a host
// without the capability, and the bytes of a commit reach the frame the
// same either way. Hand-built hosts — the pinned SDK predates the doors.

import { describe, expect, it, vi } from "vitest";

import type { BundleHost } from "@paged-media/plugin-api";

import {
  enterEditContext,
  hostColorPicker,
  imageScene,
  mutateWithBytes,
  onWillSave,
  partsDelete,
  spliceBytes,
  toolSettings,
} from "../src/host66";
import { collectGarbage, recordPath } from "../src/session-store";

function fakeHost(caps: string[], extra: Record<string, unknown> = {}): BundleHost {
  const mutate = vi.fn(async () => ({ applied: true }));
  return {
    supports: (c: string) => caps.includes(c),
    document: { mutate, ...((extra.document as object) ?? {}) },
    parts: { ...((extra.parts as object) ?? {}) },
    shell: { ...((extra.shell as object) ?? {}) },
    widgets: { ...((extra.widgets as object) ?? {}) },
    ...(extra.tools ? { tools: extra.tools } : {}),
  } as unknown as BundleHost;
}

const batch = () =>
  ({
    op: "batch",
    args: {
      ops: [
        { op: "replaceImageBytes", args: { elementId: "u1", bytes: [] } },
        { op: "setPluginMetadata", args: { elementId: { kind: "rectangle", id: "u1" }, key: "k", value: "v" } },
      ],
    },
  }) as never;

describe("protocol-66 doors, feature-detected", () => {
  it("splices the bytes into the first replaceImageBytes only", () => {
    const m = {
      op: "batch",
      args: {
        ops: [
          { op: "replaceImageBytes", args: { elementId: "a", bytes: [] } },
          { op: "replaceImageBytes", args: { elementId: "b", bytes: [] } },
        ],
      },
    } as never;
    const out = spliceBytes(m, new Uint8Array([1, 2, 3])) as unknown as {
      args: { ops: Array<{ args: { bytes: number[] } }> };
    };
    expect(out.args.ops[0].args.bytes).toEqual([1, 2, 3]);
    expect(out.args.ops[1].args.bytes).toEqual([]);
  });

  it("sends image bytes as bytes on a host with the binary mutation lane", async () => {
    const binary = vi.fn(async () => ({ applied: true }));
    const host = fakeHost(["document.mutateBinary@1"], { document: { mutateWithBytes: binary } });
    const png = new Uint8Array([0x89, 0x50]);
    await mutateWithBytes(host, batch(), png);
    expect(binary).toHaveBeenCalledWith(batch(), png);
    expect(host.document.mutate).not.toHaveBeenCalled();
  });

  it("falls back to the JSON lane with the bytes spliced in", async () => {
    const host = fakeHost([]);
    await mutateWithBytes(host, batch(), new Uint8Array([7, 8]));
    const sent = (host.document.mutate as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.args.ops[0].args.bytes).toEqual([7, 8]);
  });

  it("answers null for every door an older host lacks", async () => {
    const host = fakeHost([]);
    expect(imageScene({ submit: vi.fn(), clear: vi.fn() } as never)).toBeNull();
    expect(partsDelete(host)).toBeNull();
    expect(onWillSave(host, () => {})).toBeNull();
    expect(toolSettings(host)).toBeNull();
    expect(hostColorPicker(host)).toBeNull();
    expect(await enterEditContext(host, "rasterImage", { kind: "rectangle", id: "u1" })).toBe(false);
  });

  it("uses each door when the host offers it", async () => {
    const del = vi.fn(async () => true);
    const enter = vi.fn(async () => true);
    const willSave = vi.fn(() => ({ dispose() {} }));
    const Picker = () => null;
    const tools = { settings: () => ({}), onDidChangeSettings: () => ({ dispose() {} }) };
    const host = fakeHost(
      [
        "storage.parts@2",
        "shell.enterEditContext@1",
        "document.onWillSave@1",
        "widgets.colorPicker@1",
        "tools.settings@1",
      ],
      {
        parts: { delete: del },
        shell: { enterEditContext: enter },
        document: { onWillSave: willSave },
        widgets: { ColorPicker: Picker },
        tools,
      },
    );
    expect(await partsDelete(host)!("px/a.bin")).toBe(true);
    expect(del).toHaveBeenCalledWith("px/a.bin");
    expect(await enterEditContext(host, "rasterImage", { kind: "rectangle", id: "u1" })).toBe(true);
    expect(onWillSave(host, () => {})).not.toBeNull();
    expect(hostColorPicker(host)).toBe(Picker);
    expect(toolSettings(host)).toBe(tools);
    const surface = { submit: vi.fn(), clear: vi.fn(), submitImage: vi.fn(), submitImageTiles: vi.fn() };
    expect(imageScene(surface as never)).toBe(surface);
  });
});

describe("old revisions are collected once parts can be deleted", () => {
  it("drops records past the kept window and the buffers only they named", async () => {
    const store = new Map<string, Uint8Array>();
    const parts = {
      write: async (p: string, b: Uint8Array) => void store.set(p, b),
      read: async (p: string) => store.get(p) ?? null,
      list: async (prefix = "") => [...store.keys()].filter((k) => k.startsWith(prefix)),
    };
    const del = async (p: string) => store.delete(p);
    const rec = (manifest: string, buffers: string[]) =>
      new TextEncoder().encode(JSON.stringify({ v: 1, width: 1, height: 1, manifest, buffers, baked: "x" }));
    for (const sha of ["m1", "m2", "m3", "shared", "old"]) store.set(`px/${sha}.bin`, new Uint8Array([1]));
    store.set(recordPath("u1", 1), rec("m1", ["old", "shared"]));
    store.set(recordPath("u1", 2), rec("m2", ["shared"]));
    store.set(recordPath("u1", 3), rec("m3", ["shared"]));

    const out = await collectGarbage(parts as never, del, "u1", 3, 2);

    expect(out).toEqual({ records: 1, buffers: 2 });
    expect(store.has(recordPath("u1", 1))).toBe(false);
    expect(store.has(recordPath("u1", 2))).toBe(true);
    expect([...store.keys()].filter((k) => k.startsWith("px/")).sort()).toEqual([
      "px/m2.bin",
      "px/m3.bin",
      "px/shared.bin",
    ]);
  });
});
