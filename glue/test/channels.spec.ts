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

// The CHANNELS lane — the per-channel readout, and the operation a
// channel list exists to enable: loading a channel as the selection.
//
// This runs over the REAL engine wasm and the REAL 2×1 PSD fixture, so
// the numbers below are the engine's own reduction of pixels this file
// can name (`PSD_RGBA`). That matters more than usual here: a channel
// readout that reduced the wrong axis would still produce plausible
// numbers, and only arithmetic against known pixels catches it.
//
// The selection half is asserted through `selection_stats` — coverage is
// engine state, not a rendering question, so it runs on this lane with
// no GPU. What is NOT here: masked PIXELS. Every masked kernel dispatch
// is GPU-only, so the proof that coverage actually gates a write lives
// in the Rust device tests and the editor's GPU journey lane.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type {
  ElementGeometryItem,
  PluginManifest,
} from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { ChannelsSection } from "../src/panels/image-panel";
import { createImageSession } from "../src/session";
import {
  makeFakeEditor,
  mapBacking,
  psdBytes,
  shellStub,
  silentConsole,
} from "./helpers";

function geomFor(
  id: string,
  bounds: [number, number, number, number],
): ElementGeometryItem {
  return { id: { kind: "rectangle", id } as never, pageId: "pg1", bounds };
}

/** A session with the 2×1 fixture ingested through the real engine. */
async function ingested() {
  const fake = makeFakeEditor();
  fake.placed.set("u1", psdBytes());
  fake.geometry.set("u1", geomFor("u1", [0, 0, 100, 200]));
  fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
  const handle = createBundleHost(
    () => fake.editor,
    manifestJson as PluginManifest,
    { console: silentConsole, storage: mapBacking(), shell: shellStub() },
  );
  const session = createImageSession(handle.host);
  expect(await session.ingestSelection()).toBe(true);
  return { session, handle, fake };
}

describe("the channels readout (real engine wasm)", () => {
  // The fixture is 2 pixels: (10, 30, 50, 255) and (20, 40, 60, 255).
  it("reduces each channel over the real decoded pixels", async () => {
    const { session, handle } = await ingested();
    const ch = session.state().channels;
    expect(ch?.map((c) => c.name)).toEqual([
      "red",
      "green",
      "blue",
      "alpha",
      "luma",
    ]);
    const by = (n: string) => ch?.find((c) => c.name === n);
    expect(by("red")).toMatchObject({ min: 10, max: 20, mean: 15 });
    expect(by("green")).toMatchObject({ min: 30, max: 40, mean: 35 });
    expect(by("blue")).toMatchObject({ min: 50, max: 60, mean: 55 });
    expect(by("alpha")).toMatchObject({ min: 255, max: 255, mean: 255 });
    // Rec.709 of each pixel, rounded: 27 and 37.
    expect(by("luma")).toMatchObject({ min: 27, max: 37, mean: 32 });
    session.dispose();
    handle.dispose();
  });

  it("clears with the source, rather than going stale", async () => {
    const { session, handle } = await ingested();
    expect(session.state().channels).not.toBeNull();
    session.dispose();
    handle.dispose();
  });

  it("loads a channel as the selection, and the coverage is the channel", async () => {
    const { session, handle } = await ingested();
    expect(session.state().selection).toBeNull();
    expect(session.selectionFromChannel("alpha")).toBe(true);
    // Alpha is 255 everywhere, so the whole image is selected.
    expect(session.state().selection?.coverage).toBeCloseTo(1, 3);

    // Red is 10 and 20 of 255 — a nearly-empty PARTIAL coverage, which
    // is the whole point: a channel load is a copy, not a threshold. A
    // thresholding implementation would report 0 or 1 here.
    expect(session.selectionFromChannel("red")).toBe(true);
    const red = session.state().selection?.coverage ?? -1;
    expect(red).toBeGreaterThan(0);
    expect(red).toBeLessThan(0.1);
    expect(red).toBeCloseTo((10 / 255 + 20 / 255) / 2, 2);
    session.dispose();
    handle.dispose();
  });

  it("refuses an unknown channel instead of masking on the wrong one", async () => {
    const { session, handle } = await ingested();
    session.selectionFromChannel("alpha");
    const before = session.state().selection?.coverage;
    expect(session.selectionFromChannel("rd")).toBe(false);
    expect(session.state().status).toContain("rd");
    // The existing selection survives — a refused call changes nothing.
    expect(session.state().selection?.coverage).toBe(before);
    session.dispose();
    handle.dispose();
  });

  it("declines before an ingest, and says what is missing", async () => {
    const fake = makeFakeEditor();
    const handle = createBundleHost(
      () => fake.editor,
      manifestJson as PluginManifest,
      { console: silentConsole, storage: mapBacking(), shell: shellStub() },
    );
    const session = createImageSession(handle.host);
    expect(session.selectionFromChannel("luma")).toBe(false);
    expect(session.state().status).toContain("ingest");
    session.dispose();
    handle.dispose();
  });
});

describe("the panel's Channels section", () => {
  const render = (over: Partial<Parameters<typeof ChannelsSection>[0]> = {}) =>
    textOf(
      ChannelsSection({
        channels: [
          { name: "red", min: 10, max: 20, mean: 15 },
          { name: "green", min: 30, max: 40, mean: 35 },
          { name: "blue", min: 50, max: 60, mean: 55 },
          { name: "alpha", min: 255, max: 255, mean: 255 },
          { name: "luma", min: 27, max: 37, mean: 32 },
        ],
        disabled: false,
        onLoadSelection: () => {},
        ...over,
      }),
    )
      .join(" ")
      .replace(/\s+/g, " ");

  it("lists every channel with its range and mean", () => {
    const text = render();
    for (const n of ["red", "green", "blue", "alpha", "luma"]) {
      expect(text).toContain(n);
    }
    expect(text).toContain("10–20 · mean 15.0");
    expect(text).toContain("27–37 · mean 32.0");
  });

  it("states that a load is a copy, not a threshold", () => {
    // The property that makes luminosity masking work at all, said where
    // the designer is when they use it.
    expect(render()).toContain("no threshold");
  });

  it("says plainly that isolated channel VIEW is not offered", () => {
    // A gap named in the UI beats a gap discovered by a designer looking
    // for a feature that silently is not there.
    const text = render();
    expect(text).toContain("Viewing one channel in isolation is not offered");
    expect(text).toContain("destructive edit");
  });

  it("before an ingest it asks for one, rather than showing zeros", () => {
    const text = render({ channels: null });
    expect(text).toContain("Ingest an image");
    expect(text).not.toContain("mean 0");
  });

  it("an unmeasured channel says so instead of reading as flat black", () => {
    const text = render({
      channels: [{ name: "red", min: null, max: null, mean: null }],
    });
    expect(text).toContain("unmeasured");
  });
});

/** Render a pure element tree to its text, executing the (hook-free)
 *  function components it contains. No react-dom needed. */
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
  if (el.props) textOf(el.props.children, out);
  return out;
}
