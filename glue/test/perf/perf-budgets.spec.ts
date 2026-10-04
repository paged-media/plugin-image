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

// WORK BUDGETS for the bundle, in COUNTS: engine calls the session makes
// and what it hands the host. Never wall-clock. A budget equals the
// measurement (2026-10-04); it is lowered in the commit that earns it.
// Equality, not `<=`: an improvement must lower the number too.
//
// Node has no WebGPU, so these cover the session's orchestration — how
// often it recomposites, re-reads histograms, resubmits the page image —
// not kernel work (image-js/tests/perf_budgets.rs counts that natively).

import { describe, expect, it, vi } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import { createImageSession } from "../../src/session";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "../helpers";
import { countingEngine, type EngineLog } from "./counting-engine";

let engineLog: EngineLog | null = null;

vi.mock("../../src/engine", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/engine")>();
  return {
    ...actual,
    bootEngine: async () => {
      const counted = countingEngine(await actual.bootEngine());
      engineLog = counted.log;
      return counted.engine;
    },
  };
});

function setup() {
  const fake = makeFakeEditor();
  fake.placed.set("u1", psdBytes());
  fake.geometry.set("u1", {
    id: { kind: "rectangle", id: "u1" },
    bounds: [0, 0, 100, 200],
  } as never);
  fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
  const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
  const session = createImageSession(handle.host);
  return { fake, handle, session };
}

/** `rgba` elements (one JS number per byte) handed to the host's scene
 *  layer across every submit — the `number[]` wire type. */
function submittedElements(fake: ReturnType<typeof makeFakeEditor>): number {
  let n = 0;
  const visit = (v: unknown): void => {
    if (Array.isArray(v)) v.forEach(visit);
    else if (v && typeof v === "object") {
      for (const [k, x] of Object.entries(v)) {
        if (k === "rgba" && Array.isArray(x)) n += x.length;
        else visit(x);
      }
    }
  };
  for (const call of fake.sceneLayers.submit.mock.calls as unknown[][]) visit(call);
  return n;
}

describe("work budgets — layer opacity drag (20 steps)", () => {
  // A burst of 20 slider steps. Measured 2026-10-04 at 20 of each (every
  // step recomposited, re-read the histogram and channel stats and
  // resubmitted the whole page image, in no guaranteed order). Latest-wins
  // coalescing (src/coalesce.ts) made it 2: the first step's fold and one
  // trailing fold of the final state.
  const DRAG = {
    layersComposite: 2,
    histogram: 2,
    channelStats: 2,
    submits: 2,
    // 2 × the 2×1 fixture's 8 bytes. At 4000×3000 that is still 96 M
    // numbers per drag, until the binary scene-layer door.
    rgbaElements: 16,
  };

  it("counts what twenty opacity steps cost", async () => {
    const { fake, handle, session } = setup();
    expect(await session.ingestSelection()).toBe(true);
    expect(await session.addLayer("top")).toBe(true);
    engineLog!.reset();
    fake.sceneLayers.submit.mockClear();

    // The panel slider fires each step WITHOUT awaiting the last one
    // (React onChange), so the drag is a burst, not a sequence.
    const steps: Promise<boolean>[] = [];
    for (let i = 0; i < 20; i++) steps.push(session.setLayerOpacity(1, 1 - i / 40));
    await Promise.all(steps);
    // The final state is the last step's, whatever was coalesced.
    expect(session.state().layers.layers[1]?.opacity).toBeCloseTo(1 - 19 / 40);

    const got = {
      layersComposite: engineLog!.count("layersComposite"),
      histogram: engineLog!.count("histogram"),
      channelStats: engineLog!.count("channelStats"),
      submits: fake.sceneLayers.submit.mock.calls.length,
      rgbaElements: submittedElements(fake),
    };
    expect(got).toEqual(DRAG);

    session.dispose();
    handle.dispose();
  });
});
