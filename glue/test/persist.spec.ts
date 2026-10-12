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

// Commit an image's layers into the document and reopen them — over the
// real engine wasm and a fake editor that keeps parts, metadata and the
// frame's placed bytes. The stacks here fold without a GPU (an empty
// layer contributes nothing), so the whole round trip runs in Node.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import { createImageSession } from "../src/session";
import { sha256Hex } from "../src/session-store";
import { layeredPsdBytes, makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";

function open(fake: ReturnType<typeof makeFakeEditor>) {
  const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
  return { handle, session: createImageSession(handle.host) };
}

function setup() {
  const fake = makeFakeEditor();
  fake.placed.set("u1", psdBytes());
  fake.geometry.set("u1", {
    id: { kind: "rectangle", id: "u1" },
    pageId: "pg1",
    bounds: [0, 0, 10, 20],
  } as never);
  fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
  return fake;
}

describe("commit to the document, and reopen", () => {
  it("stores the layers, bakes the composite into the frame, and reopens them", async () => {
    const fake = setup();
    const a = open(fake);
    expect(await a.session.ingestSelection()).toBe(true);
    expect(await a.session.addLayer("Top")).toBe(true);
    a.session.setLayerName(1, "Highlights");
    expect(a.session.state().uncommitted).toBe(true);

    expect(await a.session.commitToDocument()).toBe(true);
    expect(a.session.state()).toMatchObject({ committedRev: 1, uncommitted: false });

    // ONE batch: the frame's image and its marker change together.
    const last = fake.mutations.at(-1) as { op: string; args: { ops: Array<{ op: string }> } };
    expect(last.op).toBe("batch");
    expect(last.args.ops.map((o) => o.op)).toEqual(["replaceImageBytes", "setPluginMetadata"]);

    // The frame now holds a PNG of the composite, and the marker points
    // at a stored revision whose baked hash is exactly those bytes.
    const placed = fake.placed.get("u1")!;
    expect(Array.from(placed.slice(0, 4))).toEqual([0x89, 0x50, 0x4e, 0x47]);
    const marker = JSON.parse(fake.metadata.get("u1")!.get("x-paged:media.paged.image")!);
    expect(marker).toMatchObject({ v: 2, data: { owns: "pixels", rev: 1 } });
    expect(marker.data.baked).toBe(await sha256Hex(placed));
    expect([...fake.parts.keys()].some((k) => k.endsWith("/r1.json"))).toBe(true);
    a.session.dispose();
    a.handle.dispose();

    // A later session on the same frame gets the layers back.
    const b = open(fake);
    expect(await b.session.ingestSelection()).toBe(true);
    expect(b.session.state().layers.layers.map((l) => l.name)).toEqual(["Background", "Highlights"]);
    expect(b.session.state().layersNote).toMatch(/Reopened 2 stored layers \(revision 1\)/);
    expect(b.session.state().committedRev).toBe(1);
    b.session.dispose();
    b.handle.dispose();
  });

  it("a second commit stores only what changed", async () => {
    const fake = setup();
    const a = open(fake);
    await a.session.ingestSelection();
    await a.session.addLayer("Top");
    await a.session.commitToDocument();
    const before = fake.parts.size;
    a.session.setLayerName(1, "Renamed");
    await a.session.setLayerVisible(1, false);
    expect(await a.session.commitToDocument()).toBe(true);
    expect(a.session.state().committedRev).toBe(2);
    // A new manifest and a new revision record; the pixel buffers are
    // unchanged and shared by hash. (The baked PNG is in the frame, not a part.)
    expect(fake.parts.size - before).toBe(2);
    expect(a.session.state().status).toMatch(/1 new piece/);
    a.session.dispose();
    a.handle.dispose();
  });

  it("a frame changed outside the plugin opens flat, and says why", async () => {
    const fake = setup();
    const a = open(fake);
    await a.session.ingestSelection();
    await a.session.addLayer("Top");
    await a.session.commitToDocument();
    a.session.dispose();
    a.handle.dispose();

    fake.placed.set("u1", psdBytes()); // someone relinked the original
    const b = open(fake);
    expect(await b.session.ingestSelection()).toBe(true);
    expect(b.session.state().layers.layers).toHaveLength(1);
    expect(b.session.state().layersNote).toMatch(/changed outside paged\.image/);
    b.session.dispose();
    b.handle.dispose();
  });

  // IMG-2 (filmed 2026-10-12). A file opened with one EMPTY frame
  // selected is that frame's image: the composite is baked into it and
  // the layers stored, exactly as a commit does, so the page shows it at
  // once and keeps it — before, the frame stayed empty and the import
  // could never be committed. This is also the step a host's Place hands
  // a PSD to, since no browser or renderer decodes one.
  it("an import into a selected empty frame is placed in it, layers and all", async () => {
    const fake = makeFakeEditor();
    fake.geometry.set("u2", {
      id: { kind: "rectangle", id: "u2" },
      pageId: "pg1",
      bounds: [0, 0, 10, 20],
    } as never);
    fake.emitSelection([{ kind: "rectangle", id: "u2" }]);
    const layered = await layeredPsdBytes(["Plate", "Signal"]);

    const a = open(fake);
    expect(await a.session.importBytes("annual-layers.psd", layered)).toBe(true);
    expect(a.session.state()).toMatchObject({ committedRev: 1, uncommitted: false });
    expect(a.session.state().status).toMatch(/2×1 decoded/);
    expect(a.session.state().status).toMatch(/placed in the selected frame/);
    const last = fake.mutations.at(-1) as { op: string; args: { ops: Array<{ op: string }> } };
    expect(last.op).toBe("batch");
    expect(last.args.ops.map((o) => o.op)).toEqual(["replaceImageBytes", "setPluginMetadata"]);
    const placed = fake.placed.get("u2")!;
    expect(Array.from(placed.slice(0, 4))).toEqual([0x89, 0x50, 0x4e, 0x47]);
    const marker = JSON.parse(fake.metadata.get("u2")!.get("x-paged:media.paged.image")!);
    expect(marker).toMatchObject({ v: 2, data: { owns: "pixels", rev: 1 } });
    expect(marker.data.baked).toBe(await sha256Hex(placed));
    // The PSD's own structure stays editable for the PSD exporter.
    expect(a.session.state().psd?.layers.map((l) => l.name)).toEqual(["Background", "Plate", "Signal"]);
    const stack = a.session.state().layers.layers.map((l) => l.name);
    a.session.dispose();
    a.handle.dispose();

    // Double-clicking the frame later gets the same layers back.
    const b = open(fake);
    expect(await b.session.ingestSelection()).toBe(true);
    expect(b.session.state().layers.layers.map((l) => l.name)).toEqual(stack);
    expect(b.session.state().committedRev).toBe(1);
    b.session.dispose();
    b.handle.dispose();
  });

  it("an import into a frame that holds an image leaves that image alone", async () => {
    const fake = setup(); // u1 holds the fixture PSD, and is selected
    const before = fake.placed.get("u1")!;
    const a = open(fake);
    expect(await a.session.importBytes("other.psd", psdBytes())).toBe(true);
    expect(fake.mutations).toHaveLength(0);
    expect(fake.placed.get("u1")).toBe(before);
    expect(a.session.state().committedRev).toBeNull();
    expect(a.session.state().status).toMatch(/decoded into the selected frame/);
    a.session.dispose();
    a.handle.dispose();
  });

  it("an import into an emptied frame continues its revisions instead of overwriting them", async () => {
    const fake = setup();
    const a = open(fake);
    await a.session.ingestSelection();
    await a.session.addLayer("Top");
    expect(await a.session.commitToDocument()).toBe(true);
    const r1 = [...fake.parts].find(([k]) => k.endsWith("/r1.json"))!;
    a.session.dispose();
    a.handle.dispose();

    // The frame's image is cleared; its marker (revision 1) remains.
    fake.placed.delete("u1");
    const b = open(fake);
    expect(await b.session.importBytes("next.psd", psdBytes())).toBe(true);
    expect(b.session.state().committedRev).toBe(2);
    expect(fake.parts.get(r1[0])).toEqual(r1[1]);
    const marker = JSON.parse(fake.metadata.get("u1")!.get("x-paged:media.paged.image")!);
    expect(marker.data.rev).toBe(2);
    b.session.dispose();
    b.handle.dispose();
  });

  it("an import (no frame) cannot be committed, and says so", async () => {
    const fake = makeFakeEditor();
    const a = open(fake);
    await a.session.importBytes("x.psd", psdBytes());
    expect(await a.session.commitToDocument()).toBe(false);
    expect(a.session.state().status).toMatch(/ingested from a frame/);
    a.session.dispose();
    a.handle.dispose();
  });
});
