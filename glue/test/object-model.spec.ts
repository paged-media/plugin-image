// @feat: image.object-model
//
// paged.image's object model (ADR 323) on the REAL plugin and the REAL
// engines: the bundle booted in the SDK's headless host (published
// canvas-wasm in Node) with its own committed image wasm for the CPU
// codec. Every property is set through `host.objects` and read back the
// same way; each write is one engine undo step; a 10-op batch is held to
// a count budget. No GPU is involved — the object model does not need
// one (a stack whose composite is not the identity is stored `pending`).
//
// REQUIRE_REAL_WASM=1 turns a missing artifact into a failure (the
// wasm-exec convention).

import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { createHeadlessHost, defineBundle, type HeadlessHost } from "@paged-media/plugin-sdk";
import type { BundleHost, PluginManifest } from "@paged-media/plugin-api";

import { imageBundle } from "../src/index";
import { bootEngine, freshIdentityParams, type ImageEngine } from "../src/engine";
import manifestJson from "../manifest.json";
import { contributeImageObjectModel, engineCodec } from "../src/object-model/contribute";
import { readPgil, writePgil } from "../src/object-model/pgil";
import { readAdjust } from "../src/object-model/adjust";
import { sha256Hex } from "../src/session-store";
import { minimalIdml } from "./fixtures/minimal-idml";

const HERE = dirname(fileURLToPath(import.meta.url));
const built = existsSync(join(HERE, "..", "wasm", "image_js_bg.wasm"));
if (process.env.REQUIRE_REAL_WASM === "1" && !built) {
  throw new Error("REQUIRE_REAL_WASM=1 but glue/wasm/image_js_bg.wasm is missing");
}
const lane = built ? describe : describe.skip;

const P = "plugin:media.paged.image";
const FRAME = "urect";
const silent = { debug() {}, info() {}, warn() {}, error() {} };
const value = (v: unknown) => ({ kind: "value", value: v });

let engine: ImageEngine;
beforeAll(async () => {
  if (built) engine = await bootEngine();
});

let live: HeadlessHost | null = null;
afterEach(() => {
  live?.dispose();
  live = null;
});

/** A W×H straight-RGBA8 image of one colour. */
const solid = (w: number, h: number, rgba: [number, number, number, number]) => {
  const px = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) px.set(rgba, i * 4);
  return px;
};
const png = (w: number, h: number, rgba: [number, number, number, number]) => engine.encode(solid(w, h, rgba), w, h, "png");
const dataUrl = (bytes: Uint8Array) => `data:image/png;base64,${Buffer.from(bytes).toString("base64")}`;
const decode = (bytes: Uint8Array) => {
  const d = engine.decode(bytes);
  try {
    return { w: d.width, h: d.height, px: engine.tile(d.handle, 0, 0, d.width, d.height) };
  } finally {
    engine.freeImage(d.handle);
  }
};

/** A document with one 4×4 grey image placed in the rectangle `urect`. */
async function imageDoc(bundle = imageBundle, opts: { capabilityMode?: "warn" } = {}) {
  const h = (live = await createHeadlessHost({ console: silent, ...opts }));
  await h.load(minimalIdml());
  h.loadBundle(bundle);
  const out = await h.host.document.mutate({
    op: "replaceImageBytes",
    args: { elementId: FRAME, bytes: Array.from(png(4, 4, [128, 128, 128, 255])) },
  } as never);
  expect(out.applied).toBe(true);
  return h;
}

const placedSha = async (h: HeadlessHost) => sha256Hex((await h.host.assets.getPlacedImage(FRAME))!.bytes);

lane("the PGIL codec is the engine's", () => {
  it("re-encodes a real engine export byte for byte, every layer kind included", async () => {
    const img = engine.ingestRgba8(8, 8, solid(8, 8, [10, 20, 30, 255]));
    try {
      engine.layersOpen(img.handle);
      engine.layerAdd("Top");
      engine.layerSetOpacity(1, 0.5);
      engine.layerSetBlend(1, "multiply");
      engine.layerSetName(1, "Überschrift"); // non-ASCII survives
      engine.layerAddMask(1, false);
      const params = freshIdentityParams();
      params.exposureEv = 0.5;
      params.posterizeLevels = 4;
      engine.layersAddAdjustment("Exposure", params);
      engine.layerMakeSmart(0);
      const [manifest, ...buffers] = engine.layersExport();
      const stack = readPgil(manifest!, buffers);
      expect(stack.layers.map((l) => [l.name, l.kind])).toEqual([
        ["Background", "smart"],
        ["Überschrift", "pixels"],
        ["Exposure", "adjustment"],
      ]);
      expect(stack.layers[1]!.blend).toBe("compose.multiply");
      expect(readAdjust(stack.layers[2]!.adjust!, "exposure.ev")).toBe(0.5);
      expect(readAdjust(stack.layers[2]!.adjust!, "posterize.enabled")).toBe(true);
      expect(readAdjust(stack.layers[2]!.adjust!, "posterize.levels")).toBe(4);
      const again = writePgil(stack);
      expect(again.length).toBe(buffers.length + 1);
      expect(Buffer.from(again[0]!).equals(Buffer.from(manifest!))).toBe(true);
      again.slice(1).forEach((b, i) => expect(Buffer.from(b).equals(Buffer.from(buffers[i]!))).toBe(true));
    } finally {
      engine.layersClose();
      engine.freeImage(img.handle);
    }
  });
});

lane("the object model on the real plugin, headless", () => {
  it("registers five kinds with schemas and a typed twin of every command", async () => {
    const h = await imageDoc();
    const kinds = (await h.objects.kinds()).filter((k) => k.owner === "media.paged.image").map((k) => k.kind);
    expect(kinds).toEqual([`${P}/image`, `${P}/layer`, `${P}/mask`, `${P}/adjustment`, `${P}/smartObject`]);
    const adj = await h.objects.schema(`${P}/adjustment`);
    expect(adj.length).toBeGreaterThan(80);
    expect(adj.find((r) => r.path === "exposure.ev")).toMatchObject({ type: { kind: "number" }, default: 0 });
    const layer = await h.objects.schema(`${P}/layer`);
    expect(layer.find((r) => r.path === "blendMode")!.type).toMatchObject({ kind: "enum" });
    expect((await h.objects.commands()).filter((c) => c.owner === "media.paged.image")).toHaveLength(40);
  });

  it("an unedited frame reads as its one Background layer", async () => {
    const h = await imageDoc();
    expect(await h.objects.query(`${P}/image`)).toEqual([`${P}/image/${FRAME}`]);
    expect(await h.objects.query(`${P}/layer`)).toEqual([`${P}/layer/${FRAME}/1`]);
    const L = `${P}/layer/${FRAME}/1`;
    expect(await h.objects.get(L, "name")).toEqual(value("Background"));
    expect(await h.objects.get(L, "opacity")).toEqual(value(100));
    expect(await h.objects.get(`${P}/image/${FRAME}`, "width")).toEqual(value(4));
    expect(await h.objects.get(`${P}/image/${FRAME}`, "revision")).toEqual(value(0));
    // Scoped by its frame, and selected by name.
    expect(await h.objects.query(`rectangle:${FRAME} > ${P}/layer[name="Background"]`)).toEqual([L]);
  });

  it("set → get round-trips each layer property, one undo step each, and undo restores the frame", async () => {
    const h = await imageDoc();
    const L = `${P}/layer/${FRAME}/1`;
    const before = await placedSha(h);
    for (const [path, v] of [
      ["name", "Logo"],
      ["locked", true],
      ["blendMode", "screen"],
      ["opacity", 40],
      ["visible", false],
      ["clipped", true],
    ] as const) {
      const r = await h.objects.set(L, path, v);
      expect(r, path).toMatchObject({ applied: true, undoSteps: 1 });
      expect(await h.objects.get(L, path), path).toEqual(value(v));
    }
    expect(await h.objects.get(`${P}/image/${FRAME}`, "revision")).toEqual(value(6));
    expect(await h.objects.query(`${P}/layer[name="Logo"]`)).toEqual([L]);
    // Six undos — one per write — land back on the untouched document.
    for (let i = 0; i < 6; i++) await h.host.document.undo();
    expect(await h.objects.get(L, "name")).toEqual(value("Background"));
    expect(await h.objects.get(L, "opacity")).toEqual(value(100));
    expect(await h.objects.get(`${P}/image/${FRAME}`, "revision")).toEqual(value(0));
    expect(await placedSha(h)).toBe(before);
    // Redo walks forward again (the parts of every revision are kept).
    await h.host.document.redo();
    expect(await h.objects.get(L, "name")).toEqual(value("Logo"));
  });

  it("a write whose composite is the identity bakes it into the frame; one that blends is stored pending", async () => {
    const h = await imageDoc();
    const L = `${P}/layer/${FRAME}/1`;
    const before = await placedSha(h);
    // Hidden: nothing contributes — a transparent canvas, baked here.
    await h.objects.set(L, "visible", false);
    const hidden = decode((await h.host.assets.getPlacedImage(FRAME))!.bytes);
    expect(Array.from(hidden.px.subarray(0, 4))).toEqual([0, 0, 0, 0]);
    expect(await h.objects.get(`${P}/image/${FRAME}`, "pending")).toEqual(value(false));
    await h.host.document.undo();
    expect(await placedSha(h)).toBe(before);
    // 40 % opacity is a blend (a GPU fold): the frame keeps its image.
    await h.objects.set(L, "opacity", 40);
    expect(await placedSha(h)).toBe(before);
    expect(await h.objects.get(`${P}/image/${FRAME}`, "pending")).toEqual(value(true));
    expect(await h.objects.get(L, "opacity")).toEqual(value(40));
  });

  it("refuses values outside the schema, and read-only paths, without writing", async () => {
    const h = await imageDoc();
    const L = `${P}/layer/${FRAME}/1`;
    expect(await h.objects.set(L, "opacity", 140)).toMatchObject({ applied: false, code: "invalidValue" });
    expect(await h.objects.set(L, "blendMode", "plasma")).toMatchObject({ applied: false, code: "invalidValue" });
    expect(await h.objects.set(L, "kind", "smart")).toMatchObject({ applied: false, code: "readOnly" });
    expect(await h.objects.set(`${P}/layer/${FRAME}/9`, "name", "x")).toMatchObject({ applied: false });
    expect(await h.objects.set(L, "pixelSource", "https://example.com/a.png")).toMatchObject({ applied: false });
    expect(await h.objects.get(`${P}/image/${FRAME}`, "revision")).toEqual(value(0));
  });

  it("pixel replacement: bytes, a data: URL or a placed image, placed as is and centred", async () => {
    const h = await imageDoc();
    const L = `${P}/layer/${FRAME}/1`;
    const red = dataUrl(png(2, 2, [255, 0, 0, 255]));
    expect(await h.objects.invoke("media.paged.image.command.replaceLayerPixels", { layer: L, source: red })).toMatchObject({
      applied: true,
      undoSteps: 1,
    });
    expect(await h.objects.get(L, "bounds")).toEqual(value([1, 1, 3, 3]));
    expect(await h.objects.get(L, "pixelSource")).toEqual(value(red));
    // One plain layer: the frame shows exactly the replacement.
    const shown = decode((await h.host.assets.getPlacedImage(FRAME))!.bytes);
    expect(Array.from(shown.px.subarray((1 * 4 + 1) * 4, (1 * 4 + 1) * 4 + 4))).toEqual([255, 0, 0, 255]);
    expect(Array.from(shown.px.subarray(0, 4))).toEqual([0, 0, 0, 0]);
    // `content` takes bytes; reading it answers the layer as PNG.
    expect(await h.objects.set(L, "content", png(4, 4, [0, 0, 255, 255]))).toMatchObject({ applied: true });
    const content = await h.objects.get(L, "content");
    expect(decode((content as { value: Uint8Array }).value).px.subarray(0, 4)).toEqual(new Uint8Array([0, 0, 255, 255]));
    // A locked layer's pixels are locked.
    await h.objects.set(L, "locked", true);
    expect(await h.objects.set(L, "pixelSource", red)).toMatchObject({ applied: false });
  });

  it("layers, adjustment layers, masks and smart objects through batches and typed commands", async () => {
    const h = await imageDoc();
    const image = `${P}/image/${FRAME}`;
    expect(await h.objects.invoke("media.paged.image.command.addLayer", { image, name: "Text" })).toMatchObject({ undoSteps: 1 });
    expect(await h.objects.query(`${P}/layer`)).toEqual([`${P}/layer/${FRAME}/2`, `${P}/layer/${FRAME}/1`]);
    // An adjustment layer, created with its parameters in one batch.
    expect(
      await h.objects.batch([
        { op: "create", kind: `${P}/adjustment`, within: image, props: { "exposure.ev": 0.75, "posterize.enabled": true } },
      ]),
    ).toMatchObject({ applied: true, undoSteps: 1 });
    const A = `${P}/adjustment/${FRAME}/3`;
    expect(await h.objects.query(`${P}/adjustment`)).toEqual([A]);
    expect(await h.objects.get(A, "exposure.ev")).toEqual(value(0.75));
    expect(await h.objects.set(A, "hueSaturation.ranges[2][1]", -0.5)).toMatchObject({ applied: true });
    expect(await h.objects.get(A, "hueSaturation.ranges[2][1]")).toEqual(value(-0.5));
    expect(await h.objects.set(A, "curves.composite", Array.from({ length: 256 }, (_, i) => 255 - i))).toMatchObject({ applied: true });
    expect(((await h.objects.get(A, "curves.composite")) as { value: number[] }).value[0]).toBe(255);
    // A hide-all mask on the background.
    const L1 = `${P}/layer/${FRAME}/1`;
    await h.objects.invoke("media.paged.image.command.addLayerMaskHideAll", { layer: L1 });
    const M = `${P}/mask/${FRAME}/1`;
    expect(await h.objects.get(L1, "hasMask")).toEqual(value(true));
    const mask = decode(((await h.objects.get(M, "content")) as { value: Uint8Array }).value);
    expect(Array.from(mask.px.subarray(0, 4))).toEqual([0, 0, 0, 255]);
    expect(await h.objects.set(M, "enabled", false)).toMatchObject({ applied: true });
    expect(await h.objects.get(M, "enabled")).toEqual(value(false));
    // Smart object: convert, then replace its contents.
    await h.objects.invoke("media.paged.image.command.convertLayerToSmart", { layer: L1 });
    const S = `${P}/smartObject/${FRAME}/1`;
    expect(await h.objects.get(L1, "kind")).toEqual(value("smart"));
    expect(await h.objects.get(S, "renderPending")).toEqual(value(false));
    expect(await h.objects.set(S, "source", dataUrl(png(6, 3, [0, 255, 0, 255])))).toMatchObject({ applied: true });
    expect(await h.objects.get(S, "sourceWidth")).toEqual(value(6));
    expect(await h.objects.get(S, "renderPending")).toEqual(value(true));
    // Delete a layer; the last one is refused.
    expect(await h.objects.batch([{ op: "delete", address: `${P}/layer/${FRAME}/2` }])).toMatchObject({ applied: true });
    expect(await h.objects.get(image, "layerCount")).toEqual(value(2));
  });

  it("a Photoshop-style data set is one host.objects batch per record (visibility + pixel replacement)", async () => {
    const h = await imageDoc();
    const image = `${P}/image/${FRAME}`;
    await h.objects.invoke("media.paged.image.command.addLayer", { image, name: "Badge" });
    const records = [
      { badge: true, photo: dataUrl(png(4, 4, [255, 0, 0, 255])) },
      { badge: false, photo: dataUrl(png(4, 4, [0, 0, 255, 255])) },
    ];
    // What a Binding::Property lowers to (ADR 558, Wave 5): targets are
    // SELECTORS over names, resolved per apply — never stored addresses.
    const [badge] = await h.objects.query(`${P}/layer[name="Badge"]`);
    const [photo] = await h.objects.query(`${P}/layer[name="Background"]`);
    for (const r of records) {
      const res = await h.objects.batch([
        { op: "set", address: badge!, path: "visible", value: r.badge },
        { op: "set", address: photo!, path: "pixelSource", value: r.photo },
      ]);
      expect(res).toMatchObject({ applied: true, undoSteps: 1 });
      expect(await h.objects.get(badge!, "visible")).toEqual(value(r.badge));
      expect(await h.objects.get(photo!, "pixelSource")).toEqual(value(r.photo));
    }
    // The second record hides the (empty) badge: one plain layer is left,
    // so the frame shows the record's photo.
    const shown = decode((await h.host.assets.getPlacedImage(FRAME))!.bytes);
    expect(Array.from(shown.px.subarray(0, 4))).toEqual([0, 0, 255, 255]);
  });

  it("the stored stack reopens from the document: a second session reads what the first wrote", async () => {
    // `warn`: the test drives the container export the bundle itself
    // never needs (and so does not declare).
    const h = await imageDoc(imageBundle, { capabilityMode: "warn" });
    const L = `${P}/layer/${FRAME}/1`;
    await h.objects.set(L, "name", "Kept");
    const bytes = await h.host.documents.exportPaged();
    const h2 = await createHeadlessHost({ console: silent });
    try {
      await h2.load(bytes);
      h2.loadBundle(imageBundle);
      expect(await h2.objects.get(L, "name")).toEqual(value("Kept"));
    } finally {
      h2.dispose();
    }
  });
});

lane("count budgets", () => {
  it("a 10-op batch is one engine mutation, one undo step, and a bounded number of reads", async () => {
    let om: ReturnType<typeof contributeImageObjectModel> = null;
    let partWrites = 0;
    // The object model alone, under the same manifest, so its counters
    // are reachable.
    const bundle = defineBundle({
      manifest: manifestJson as PluginManifest,
      activate(host: BundleHost) {
        const write = host.parts.write.bind(host.parts);
        (host.parts as { write: typeof write }).write = (p, b) => (partWrites++, write(p, b));
        om = contributeImageObjectModel(host, [], engineCodec());
        return { dispose: () => (om?.objects.dispose(), om?.handle.dispose()) } as never;
      },
    });
    const h = await imageDoc(bundle);
    const image = `${P}/image/${FRAME}`;
    await h.objects.invoke("media.paged.image.command.addLayer", { image, name: "Two" });
    const A = `${P}/layer/${FRAME}/1`;
    const B = `${P}/layer/${FRAME}/2`;
    let changes = 0;
    const sub = h.host.document.onDidChange(() => changes++);
    const c = om!.objects.counters;
    const start = { ...c };
    partWrites = 0;
    const ops = [
      ["name", "A"], ["opacity", 90], ["blendMode", "multiply"], ["locked", true], ["clipped", false],
    ].flatMap(([path, v]) => [
      { op: "set" as const, address: A, path: path as string, value: v },
      { op: "set" as const, address: B, path: path as string, value: path === "name" ? "B" : v },
    ]);
    expect(ops).toHaveLength(10);
    expect(await h.objects.batch(ops)).toMatchObject({ applied: true, undoSteps: 1 });
    sub.dispose();
    const spent = Object.fromEntries(Object.entries(c).map(([k, v]) => [k, v - start[k as keyof typeof start]]));
    // One commit, observed as one document change.
    expect(changes).toBe(1);
    // Reads: one tree; one placed-image read to re-validate the stack the
    // previous write left (the frame could have changed under it); no
    // stored-session read and no decode (that stack is cached under its
    // marker); no encode (two blended layers: the bake is pending).
    expect(spent).toEqual({ treeReads: 1, placedReads: 1, sessionReads: 0, decodes: 0, encodes: 0 });
    // Parts: the changed manifest + the record (unchanged pixels are
    // stored once, by content).
    expect(partWrites).toBe(2);
    expect(await h.objects.get(B, "name")).toEqual(value("B"));
    await h.host.document.undo();
    expect(await h.objects.get(A, "name")).toEqual(value("Background"));
    expect(await h.objects.get(B, "name")).toEqual(value("Two"));
  });
});
