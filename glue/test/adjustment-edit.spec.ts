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

// Editable adjustment layers: selecting one this session created binds
// the panel's sliders to it; slider moves then retune the layer instead
// of previewing; selecting any other layer gives the sliders back.
// Real engine wasm, counted (Node has no GPU, so the recomposite after
// each edit declines — the edit itself is CPU and lands).

import { describe, expect, it, vi } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import { createImageSession } from "../src/session";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";
import { countingEngine, type EngineLog } from "./perf/counting-engine";

let log: EngineLog | null = null;
const setCalls: Array<[number, number]> = [];

vi.mock("../src/engine", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/engine")>();
  return {
    ...actual,
    bootEngine: async () => {
      const real = await actual.bootEngine();
      const orig = real.layersSetAdjustment.bind(real);
      real.layersSetAdjustment = (i, p) => {
        setCalls.push([i, p.exposureEv]);
        orig(i, p);
      };
      const c = countingEngine(real);
      log = c.log;
      return c.engine;
    },
  };
});

describe("editable adjustment layers", () => {
  it("bind on select, edit through the sliders, unbind on another layer", async () => {
    const fake = makeFakeEditor();
    const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
      console: silentConsole,
      storage: mapBacking(),
      shell: shellStub(),
    });
    const session = createImageSession(handle.host);
    expect(await session.importBytes("a.psd", psdBytes())).toBe(true);

    session.setParams({ exposureEv: 0.5 });
    await session.addAdjustmentLayer();
    const rows = session.state().layers.layers;
    expect(rows.map((r) => r.kind)).toEqual(["pixels", "adjustment"]);

    // Back to the pixel layer: the sliders are the panel's own again.
    session.setParams({ exposureEv: 0 });
    session.setActiveLayer(0);
    expect(session.state().editingAdjustment).toBeNull();

    // Select the adjustment layer: its chain comes back into the sliders.
    session.setActiveLayer(1);
    expect(session.state().editingAdjustment?.index).toBe(1);
    expect(session.state().params.exposureEv).toBe(0.5);

    // A slider move retunes THE LAYER (no preview path).
    session.setParams({ exposureEv: -0.25 });
    expect(setCalls).toEqual([[1, -0.25]]);
    expect(log!.count("adjust")).toBe(0);

    // Select the pixel layer: unbound, and the panel's previous chain is back.
    session.setActiveLayer(0);
    expect(session.state().editingAdjustment).toBeNull();
    expect(session.state().params.exposureEv).toBe(0);

    // Re-binding shows the EDITED chain.
    session.setActiveLayer(1);
    expect(session.state().params.exposureEv).toBe(-0.25);

    session.dispose();
    handle.dispose();
  });
});
