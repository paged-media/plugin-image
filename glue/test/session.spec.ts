// The M4 ingest loop, end to end over the REAL engine wasm (Node = no
// navigator.gpu, so the GPU degrades honestly and only the IDENTITY
// composite runs — exactly the no-WebGPU contract; the GPU adjustments
// chain is covered natively in image-js/tests/ingest.rs and the
// conformance async-parity suite). What this pins: C-5 read → wasm
// decode → the v41 image scene-item submit sized to the frame, the
// Stage-A commit semantics, clear-on-deselect, and reset.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { createImageSession } from "../src/session";
import { isIdentity } from "../src/engine";
import {
  makeFakeEditor,
  mapBacking,
  psdBytes,
  shellStub,
  silentConsole,
  PSD_RGBA,
} from "./helpers";

function makeHost(fake: ReturnType<typeof makeFakeEditor>) {
  return createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
}

describe("the M4 ingest session (real engine wasm)", () => {
  it("ingests C-5 bytes, decodes, and composites the identity preview in-frame", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u42", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u42" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    expect(await session.ingestSelection()).toBe(true);
    const s = session.state();
    expect(s.engine).toBe("ready");
    expect(s.gpu).toBe(false); // Node has no navigator.gpu — honest
    expect(s.source).toMatchObject({
      width: 2,
      height: 1,
      origin: "selection",
      elementId: "u42",
    });

    // Identity Apply (the only lane without WebGPU): submits the v41
    // image scene item carrying the DECODED RGBA, full-frame dest.
    expect(await session.apply()).toBe(true);
    expect(fake.sceneLayers.submit).toHaveBeenCalledTimes(1);
    const [elementId, layer] = fake.sceneLayers.submit.mock.calls[0] as unknown as [
      string,
      { items: Array<Record<string, unknown>> },
    ];
    expect(elementId).toBe("u42");
    expect(layer.items).toHaveLength(1);
    expect(layer.items[0]).toMatchObject({
      kind: "image",
      width: 2,
      height: 1,
      rgba: PSD_RGBA,
    });

    // A NON-identity apply without WebGPU refuses honestly (no CPU
    // kernel fallback) — no second submit.
    session.setParams({ exposureEv: 1 });
    expect(await session.apply()).toBe(false);
    expect(fake.sceneLayers.submit).toHaveBeenCalledTimes(1);

    session.dispose();
    handle.dispose();
  });

  it("clears the in-frame layer when the composited frame is deselected", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u7", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u7" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    await session.ingestSelection();
    await session.apply();
    expect(session.state().compositedFrame).toBe("u7");

    fake.emitSelection([]);
    await new Promise((r) => setTimeout(r, 0)); // let the async clear settle
    expect(fake.sceneLayers.clear).toHaveBeenCalledWith("u7");
    expect(session.state().compositedFrame).toBeNull();

    session.dispose();
    handle.dispose();
  });

  it("imports bytes (the K-2 lane) and composites into the selected frame at Apply", async () => {
    const fake = makeFakeEditor();
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    expect(await session.importBytes("dropped.psd", psdBytes())).toBe(true);
    expect(session.state().source).toMatchObject({
      origin: "import",
      elementId: null,
    });

    // No frame selected — Apply states the gap instead of guessing.
    expect(await session.apply()).toBe(false);
    expect(fake.sceneLayers.submit).not.toHaveBeenCalled();

    fake.emitSelection([{ kind: "rectangle", id: "u9" }]);
    expect(await session.apply()).toBe(true);
    expect(fake.sceneLayers.submit).toHaveBeenCalledWith(
      "u9",
      expect.objectContaining({ items: expect.any(Array) }),
    );

    session.dispose();
    handle.dispose();
  });

  it("reset clears the layer and returns to identity params", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u1", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    await session.ingestSelection();
    session.setParams({ saturation: 2 });
    await session.apply(); // refused (no GPU) but params stay
    await session.reset();
    expect(session.state().params).toMatchObject({
      exposureEv: 0,
      brightness: 0,
      contrast: 1,
      saturation: 1,
    });
    expect(session.state().compositedFrame).toBeNull();

    session.dispose();
    handle.dispose();
  });

  it("auto-enhance derives levels + white balance from the histogram (real kernel)", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u5", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u5" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    // Before ingest: honest no-source state, params untouched.
    session.autoEnhance();
    expect(session.state().status).toMatch(/Nothing ingested/);
    expect(isIdentity(session.state().params)).toBe(true);

    await session.ingestSelection();
    session.autoEnhance();
    const s = session.state();
    // The fixture is a dark, narrow-range, blue-cast image → auto-levels
    // pulls the white point in (well under 1) and the result is non-identity;
    // exposure/brightness/contrast/saturation are left untouched.
    expect(s.params.levels.inWhite).toBeLessThan(1);
    expect(s.params.levels.inWhite).toBeGreaterThan(s.params.levels.inBlack);
    expect(isIdentity(s.params)).toBe(false);
    expect(s.params.exposureEv).toBe(0);
    expect(s.status).toMatch(/Auto-enhance/);

    // Deterministic (pure CPU readout): a second call yields the same points.
    const inWhite = s.params.levels.inWhite;
    session.autoEnhance();
    expect(session.state().params.levels.inWhite).toBe(inWhite);

    session.dispose();
    handle.dispose();
  });

  it("answers the honest no-bytes state for a frame without a placed image", async () => {
    const fake = makeFakeEditor();
    fake.emitSelection([{ kind: "rectangle", id: "empty" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    expect(await session.ingestSelection()).toBe(false);
    expect(session.state().source).toBeNull();
    expect(session.state().status).toMatch(/No placed image/);

    session.dispose();
    handle.dispose();
  });
});
