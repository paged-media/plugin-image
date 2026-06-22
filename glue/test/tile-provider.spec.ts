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

// C-6 (I-06) — the renderer RESOURCE-PROVIDER consumer, end to end over
// the REAL engine wasm (Node decode lane) + the SDK adapter's
// needed→source→submit plumbing routed through the fake editor's images
// channel. What this pins: claim on an ingested image, the level-0 tile
// cut served back through host.images, the named gap (level>0 → null),
// and release on dispose / re-ingest.
//
// HONEST SUBSET: the provider serves only level 0 (windowed from the
// decoded RGBA8 buffer). The mip pyramid + Engine B (level,x,y) window
// eval are NOT wired across the wasm boundary yet — asserted here as the
// level>0 → no-submit behavior, never faked.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { createImageSession } from "../src/session";
import { TILE_SIZE } from "../src/tile-provider";
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

describe("the C-6 tile provider (real engine wasm + SDK routing)", () => {
  it("probes the door and claims an ingested image's tile resource", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u42", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u42" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    // The host wires the images channel → the feature is live.
    expect(handle.host.supports("rendering.resourceProvider@1")).toBe(true);

    expect(await session.ingestSelection()).toBe(true);
    expect(session.claimTiles()).toBe(true);
    expect(session.tilesClaimed()).toBe(true);

    // The claim crossed the wire with the provider-owned pyramid shape.
    expect(fake.imageClaims).toHaveLength(1);
    expect(fake.imageClaims[0]).toMatchObject({
      imageId: "u42",
      levels: 1,
      tileSize: TILE_SIZE,
      baseWidth: 2,
      baseHeight: 1,
      revision: 1,
    });

    session.dispose();
    handle.dispose();
  });

  it("serves the level-0 tile back through host.images on a needed event", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u42", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u42" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    await session.ingestSelection();
    session.claimTiles();

    // The renderer reports needing the level-0 tile at the grid origin
    // (0,0), generation 5. The 2×1 image fits in one tile.
    await fake.emitTilesNeeded({
      imageId: "u42",
      level: 0,
      tiles: [[0, 0]],
      generation: 5,
    });

    expect(fake.imageSubmits).toHaveLength(1);
    const s = fake.imageSubmits[0];
    expect(s.imageId).toBe("u42");
    expect(s.level).toBe(0);
    expect(s.generation).toBe(5); // echoed
    expect(s.tiles).toHaveLength(1);
    // Edge tile clamped to the 2×1 image extent.
    expect(s.tiles[0]).toMatchObject({ x: 0, y: 0, width: 2, height: 1 });
    // The decoded RGBA8 served verbatim (the PSD's two pixels).
    expect(s.tiles[0].rgba).toEqual(PSD_RGBA);

    session.dispose();
    handle.dispose();
  });

  it("serves nothing above level 0 (the named Engine-B-window-eval gap)", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u42", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u42" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    await session.ingestSelection();
    session.claimTiles();

    // A level-1 request: the honest subset returns null for every tile →
    // no submit (the renderer keeps its whole-image fallback). Not faked.
    await fake.emitTilesNeeded({
      imageId: "u42",
      level: 1,
      tiles: [[0, 0]],
      generation: 9,
    });
    expect(fake.imageSubmits).toHaveLength(0);

    session.dispose();
    handle.dispose();
  });

  it("releases the claim on dispose (and stops routing)", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u42", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u42" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    await session.ingestSelection();
    session.claimTiles();
    expect(session.tilesClaimed()).toBe(true);

    session.dispose();
    expect(fake.imageReleases).toContain("u42");

    // A post-dispose needed event drives no submit.
    await fake.emitTilesNeeded({
      imageId: "u42",
      level: 0,
      tiles: [[0, 0]],
      generation: 1,
    });
    expect(fake.imageSubmits).toHaveLength(0);

    handle.dispose();
  });

  it("re-ingesting the same frame releases the prior claim (no stale handle)", async () => {
    const fake = makeFakeEditor();
    fake.placed.set("u42", psdBytes());
    fake.emitSelection([{ kind: "rectangle", id: "u42" }]);
    const handle = makeHost(fake);
    const session = createImageSession(handle.host);

    await session.ingestSelection();
    session.claimTiles();
    expect(fake.imageClaims).toHaveLength(1);

    // Re-ingest (a fresh decode → new handle) frees the source, which
    // releases the prior claim. The session is no longer claimed.
    await session.ingestSelection();
    expect(fake.imageReleases).toContain("u42");
    expect(session.tilesClaimed()).toBe(false);

    session.dispose();
    handle.dispose();
  });
});
