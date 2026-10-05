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

// LIVE PREVIEW orchestration. Node has no WebGPU, so the three GPU-bound
// engine calls (initGpu, resize, adjust) are stubbed around the REAL
// engine — everything else (decode, ingest, the session) is real. What
// this pins is the scheduling: a proxy built once per size, a burst of
// slider moves costing at most two previews, the full-resolution pass
// after the sliders rest, and the layout always at full-resolution size.
// The kernels themselves are covered natively (image-js/tests).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { ElementGeometryItem, PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import { createImageSession } from "../src/session";
import { makeFakeEditor, mapBacking, shellStub, silentConsole } from "./helpers";

const calls = { resize: [] as Array<[number, number]>, adjust: [] as number[] };
let realEncode: ((rgba: Uint8Array, w: number, h: number, f: "png") => Uint8Array) | null = null;
// The fake `adjust` answers pixels of the image's real size: the host's
// scene-image door rejects a buffer that is not `width*height*4` bytes.
const dims = new Map<number, [number, number]>();
let sourceDims: [number, number] = [0, 0];

vi.mock("../src/engine", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/engine")>();
  return {
    ...actual,
    bootEngine: async () => {
      const real = await actual.bootEngine();
      realEncode = real.encode.bind(real) as never;
      let next = 10_000;
      const fake = new Set<number>();
      return new Proxy(real, {
        get(t, prop, r) {
          if (prop === "initGpu") return async () => true;
          if (prop === "resize")
            return async (_h: number, w: number, h: number) => {
              calls.resize.push([w, h]);
              const handle = next++;
              fake.add(handle);
              dims.set(handle, [w, h]);
              return { handle, width: w, height: h, display: "assumedSrgb", depthReduced: false };
            };
          if (prop === "adjust")
            return async (h: number) => {
              calls.adjust.push(h);
              const [w, ht] = dims.get(h) ?? sourceDims;
              return new Uint8Array(w * ht * 4);
            };
          if (prop === "freeImage")
            return (h: number) => (fake.has(h) ? fake.delete(h) : t.freeImage(h));
          const v = Reflect.get(t, prop, r) as unknown;
          return typeof v === "function" ? (v as (...a: unknown[]) => unknown).bind(t) : v;
        },
      });
    },
  };
});

const geom = (bounds: [number, number, number, number]): ElementGeometryItem =>
  ({ id: { kind: "rectangle", id: "u1" }, pageId: "pg1", bounds }) as never;

async function open(sourceW: number, sourceH: number, frame: [number, number, number, number]) {
  sourceDims = [sourceW, sourceH];
  const fake = makeFakeEditor();
  const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
  const session = createImageSession(handle.host);
  // Boot once to reach the real encoder for the fixture.
  if (!realEncode) {
    await session.importBytes("boot.png", new Uint8Array(0));
  }
  const png = realEncode!(new Uint8Array(sourceW * sourceH * 4).fill(128), sourceW, sourceH, "png");
  fake.placed.set("u1", png);
  fake.geometry.set("u1", geom(frame));
  fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
  expect(await session.ingestSelection()).toBe(true);
  return { fake, handle, session };
}

describe("live preview", () => {
  beforeEach(() => {
    calls.resize.length = 0;
    calls.adjust.length = 0;
  });
  afterEach(() => vi.useRealTimers());

  it("a slider burst previews on ONE proxy, at most twice, then once at full resolution", async () => {
    // 400×200 source in a 50×25 pt frame → proxy 100×50 (2 px per pt).
    const { fake, handle, session } = await open(400, 200, [0, 0, 25, 50]);
    const src = session.state().source!;
    fake.sceneLayers.submit.mockClear();
    calls.adjust.length = 0;
    vi.useFakeTimers();

    for (let i = 1; i <= 20; i++) session.setParams({ exposureEv: i / 20 });
    await vi.waitFor(() => expect(calls.adjust.length).toBe(2));
    expect(calls.resize).toEqual([[100, 50]]);
    expect(calls.adjust.every((h) => h !== src.handle)).toBe(true);

    // The sliders rest → one full-resolution pass, on the source itself.
    await vi.advanceTimersByTimeAsync(400);
    await vi.waitFor(() => expect(calls.adjust).toContain(src.handle));
    expect(calls.adjust.filter((h) => h === src.handle)).toHaveLength(1);

    // Every submit is laid out at the SOURCE's size, proxy or not.
    const items = (fake.sceneLayers.submit.mock.calls as unknown as Array<[string, { items: Array<{ w: number; h: number; width: number }> }]>).map(
      (c) => c[1].items[0],
    );
    expect(items.length).toBe(3);
    for (const it of items) expect([it.w, it.h]).toEqual([50, 25]);
    expect(items.map((it) => it.width)).toEqual([100, 100, 400]);
    expect(session.state().ptPerPx).toBeCloseTo(50 / 400);

    session.dispose();
    handle.dispose();
  });

  it("a source no larger than the frame previews directly (no proxy)", async () => {
    const { handle, session } = await open(40, 20, [0, 0, 25, 50]);
    session.setParams({ exposureEv: 0.5 });
    await vi.waitFor(() => expect(calls.adjust.length).toBe(1));
    expect(calls.resize).toEqual([]);
    expect(calls.adjust[0]).toBe(session.state().source!.handle);
    session.dispose();
    handle.dispose();
  });

  it("switched off, a slider changes nothing on the page", async () => {
    const { fake, handle, session } = await open(400, 200, [0, 0, 25, 50]);
    session.setLivePreview(false);
    fake.sceneLayers.submit.mockClear();
    calls.adjust.length = 0;
    session.setParams({ exposureEv: 0.5 });
    await new Promise((r) => setTimeout(r, 20));
    expect(calls.adjust).toEqual([]);
    expect(fake.sceneLayers.submit).not.toHaveBeenCalled();
    session.dispose();
    handle.dispose();
  });
});
