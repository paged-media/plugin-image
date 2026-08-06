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

// The RASTER ↔ VECTOR bridge — the half of the catalog's "Paths / shapes"
// row that was missing. The vector side (Pen, shapes, Pathfinder) is
// host-owned and stays there; what these specs cover is the conversion
// in both directions, over the real engine wasm.
//
// Both directions cross a COORDINATE boundary — image pixels ↔ document
// points through the frame fit — which is exactly the seam that produced
// a plausible-looking 0-pixel diff once before in the paint lane. So the
// assertions here are arithmetic against a fit this file can compute by
// hand, not "something happened".

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type {
  ElementGeometryItem,
  PluginManifest,
} from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { PathsSection } from "../src/panels/image-panel";
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

// The fixture is 2×1 px. Fitted into a 200×100 pt frame at [0,0,100,200]
// (top, left, bottom, right) the aspect fit gives scale = min(200/2,
// 100/1) = 100, so the image occupies 200×100 pt with origin (0, 0).
const SCALE = 100;

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

describe("selection → path (real engine wasm)", () => {
  it("inserts one polygon per contour, in DOCUMENT coordinates", async () => {
    const { session, handle, fake } = await ingested();
    session.selectAll();
    expect(await session.selectionToPath()).toBe(true);

    expect(fake.mutations).toHaveLength(1);
    const m = fake.mutations[0] as {
      kind: string;
      parent: { kind: string; id: string };
      node: {
        kind: string;
        bounds: number[];
        anchors: Array<{
          anchor: [number, number];
          left: [number, number];
          right: [number, number];
        }>;
        subpath_open: boolean[];
        fill_color?: string | null;
      };
    };
    expect(m.kind).toBe("InsertNode");
    expect(m.parent).toEqual({ kind: "Page", id: "pg1" });
    expect(m.node.kind).toBe("polygon");

    // Select-all traces the image border: (0,0)-(2,1) in image px, which
    // at scale 100 with origin (0,0) is (0,0)-(200,100) in pt. Bounds are
    // [top, left, bottom, right].
    expect(m.node.bounds).toEqual([0, 0, 1 * SCALE, 2 * SCALE]);
    const xs = m.node.anchors.map((a) => a.anchor[0]).sort((a, b) => a - b);
    expect(xs[0]).toBe(0);
    expect(xs[xs.length - 1]).toBe(2 * SCALE);

    // A closed contour, and a PATH rather than a filled shape — filling
    // it would paint over the image it was traced from.
    expect(m.node.subpath_open).toEqual([false]);
    expect(m.node.fill_color ?? null).toBeNull();

    session.dispose();
    handle.dispose();
  });

  it("carries no bezier handles, because the mask has no curves", async () => {
    // Handles sitting ON the anchor is how a path expresses a corner.
    // Fitting curves to a traced staircase would invent smoothness.
    const { session, handle, fake } = await ingested();
    session.selectAll();
    await session.selectionToPath();
    const node = (fake.mutations[0] as { node: { anchors: Array<{
      anchor: [number, number];
      left: [number, number];
      right: [number, number];
    }> } }).node;
    for (const a of node.anchors) {
      expect(a.left).toEqual(a.anchor);
      expect(a.right).toEqual(a.anchor);
    }
    session.dispose();
    handle.dispose();
  });

  it("declines without a selection, and mutates nothing", async () => {
    const { session, handle, fake } = await ingested();
    expect(await session.selectionToPath()).toBe(false);
    expect(session.state().status).toContain("Nothing selected");
    expect(fake.mutations).toHaveLength(0);
    session.dispose();
    handle.dispose();
  });

  it("names the threshold when the cut traces nothing", async () => {
    // A selection that exists but is empty AT THIS CUT is a real state,
    // and the fix is a different threshold — so the message says which
    // one produced nothing rather than "no paths found".
    const { session, handle, fake } = await ingested();
    session.selectAll();
    session.invertSelection();
    expect(await session.selectionToPath(128)).toBe(false);
    expect(session.state().status).toContain("128");
    expect(fake.mutations).toHaveLength(0);
    session.dispose();
    handle.dispose();
  });
});

describe("path → selection (real engine wasm)", () => {
  /** A square path in DOCUMENT pt covering the LEFT half of the image
   *  (image px 0..1 of 2), with straight segments. */
  const leftHalfSquare = {
    id: { kind: "polygon", id: "p1" },
    pageId: "pg1",
    anchors: [
      { anchor: [0, 0], left: [0, 0], right: [0, 0] },
      { anchor: [SCALE, 0], left: [SCALE, 0], right: [SCALE, 0] },
      { anchor: [SCALE, SCALE], left: [SCALE, SCALE], right: [SCALE, SCALE] },
      { anchor: [0, SCALE], left: [0, SCALE], right: [0, SCALE] },
    ],
    subpathStarts: [],
  };

  it("turns a host path into the selection, in image pixels", async () => {
    const { session, handle, fake } = await ingested();
    fake.pathAnchors.set("p1", leftHalfSquare);
    fake.emitSelection([{ kind: "polygon", id: "p1" }]);

    expect(await session.selectionFromPath()).toBe(true);
    // The square covers one of the image's two pixels — 50% coverage.
    // A coordinate-space mistake would land at 0% or 100%, which is the
    // failure this arithmetic exists to catch.
    expect(session.state().selection?.coverage).toBeCloseTo(0.5, 2);
    expect(session.state().status).toContain("4-anchor path");
    session.dispose();
    handle.dispose();
  });

  it("needs exactly one non-image element selected, and says so", async () => {
    const { session, handle, fake } = await ingested();
    // Only the image frame is selected — there is no path to read.
    expect(await session.selectionFromPath()).toBe(false);
    expect(session.state().status).toContain("exactly ONE");

    fake.pathAnchors.set("p1", leftHalfSquare);
    fake.emitSelection([
      { kind: "polygon", id: "p1" },
      { kind: "polygon", id: "p2" },
    ]);
    expect(await session.selectionFromPath()).toBe(false);
    session.dispose();
    handle.dispose();
  });

  it("reports an element with no path geometry rather than erroring", async () => {
    // The contract documents a rectangle with no `<PathGeometry>` as
    // "nothing to draw"; a designer can act on that sentence, not on a
    // stack trace.
    const { session, handle, fake } = await ingested();
    fake.emitSelection([{ kind: "rectangle", id: "r9" }]);
    expect(await session.selectionFromPath()).toBe(false);
    expect(session.state().status).toContain("no path");
    session.dispose();
    handle.dispose();
  });

  it("flattens curves rather than dropping their bulge", async () => {
    // A segment whose handles are NOT on its anchors is a curve, and the
    // flattener must add intermediate points for it. A path that only
    // kept anchors would select a quadrilateral instead of a disc.
    const { session, handle, fake } = await ingested();
    const k = SCALE * 0.55; // circle-ish handle length
    fake.pathAnchors.set("c1", {
      id: { kind: "polygon", id: "c1" },
      pageId: "pg1",
      anchors: [
        { anchor: [SCALE, 0], left: [SCALE - k, 0], right: [SCALE + k, 0] },
        {
          anchor: [2 * SCALE, SCALE],
          left: [2 * SCALE, SCALE - k],
          right: [2 * SCALE, SCALE + k],
        },
        { anchor: [SCALE, 2 * SCALE], left: [SCALE + k, 2 * SCALE], right: [SCALE - k, 2 * SCALE] },
        { anchor: [0, SCALE], left: [0, SCALE + k], right: [0, SCALE - k] },
      ],
      subpathStarts: [],
    });
    fake.emitSelection([{ kind: "polygon", id: "c1" }]);
    expect(await session.selectionFromPath()).toBe(true);
    // 4 anchors + 3 interior points per curved segment × 4 segments.
    const reported = session.state().status;
    expect(reported).toContain("4-anchor path");
    const points = Number(/\((\d+) points/.exec(reported)?.[1] ?? 0);
    expect(points).toBeGreaterThan(4);
    session.dispose();
    handle.dispose();
  });
});

describe("the panel's Paths section", () => {
  const render = (over: Partial<Parameters<typeof PathsSection>[0]> = {}) =>
    textOf(
      PathsSection({
        hasSelection: true,
        threshold: 128,
        disabled: false,
        onThreshold: () => {},
        onToPath: () => {},
        onFromPath: () => {},
        ...over,
      }),
    )
      .join(" ")
      .replace(/\s+/g, " ");

  it("offers both directions and exposes the threshold", () => {
    const text = render();
    expect(text).toContain("Selection → path");
    expect(text).toContain("Path → selection");
    expect(text).toContain("Trace at coverage ≥");
  });

  it("says a traced path becomes a real host element", () => {
    // The property that makes this worth having rather than a private
    // path list: once traced, the whole vector toolchain applies.
    const text = render();
    expect(text).toContain("real vector polygon");
    expect(text).toContain("Pathfinder");
  });

  it("states that holes are separate paths", () => {
    expect(render()).toContain("a ring is two paths");
  });

  it("explains why the anchors have no handles", () => {
    expect(render()).toContain("invent a smoothness");
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
