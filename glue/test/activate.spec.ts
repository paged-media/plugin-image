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

// Registration wiring against the real in-process host adapter over the
// fake editor: the M0 honesty smoke test ("dispose leaves the shell
// exactly as found") extended with the M4 contributions — the second
// command and the K-2 raster importer. The ingest loop itself is
// session.spec.ts (real engine wasm).

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it, vi } from "vitest";

import { loadBundle } from "@paged-media/plugin-sdk";

import { imageBundle, PANEL_ID } from "../src";
import {
  makeFakeEditor,
  mapBacking,
  shellStub,
  silentConsole,
} from "./helpers";

function load(
  fake: ReturnType<typeof makeFakeEditor>,
  shell: ReturnType<typeof shellStub> = shellStub(),
) {
  return loadBundle(() => fake.editor, imageBundle, {
    console: silentConsole,
    storage: mapBacking(),
    shell,
  });
}

describe("imageBundle.activate", () => {
  it("registers the panel, the commands, the tools, and the raster importer", () => {
    const fake = makeFakeEditor();
    load(fake);
    expect(fake.panels.ids()).toEqual([PANEL_ID]);
    expect(fake.commands.ids()).toEqual([
      "media.paged.image.command.openImage",
      "media.paged.image.command.adjustSelected",
      "media.paged.image.command.autoEnhance",
      "media.paged.image.command.claimTiles",
      "media.paged.image.command.commitCrop",
      "media.paged.image.command.fillSelection",
      "media.paged.image.command.fillNoise",
      // The LAYER GRAPH's command-palette reach (the panel carries the
      // full palette).
      "media.paged.image.command.addLayer",
      "media.paged.image.command.bakeAdjustToLayer",
      "media.paged.image.command.undo",
      "media.paged.image.command.redo",
      "media.paged.image.command.applyToFile",
      "media.paged.image.command.saveToFile",
      "media.paged.image.command.loadBrushLibrary",
      // The raster↔vector bridge + the luminosity mask.
      "media.paged.image.command.selectionToPath",
      "media.paged.image.command.pathToSelection",
      "media.paged.image.command.channelToSelection",
      "media.paged.image.command.setType",
      "media.paged.image.command.contentAwareFill",
      "media.paged.image.command.selectAll",
      "media.paged.image.command.deselect",
      "media.paged.image.command.invertSelection",
      "media.paged.image.command.featherSelection",
    ]);
    expect(fake.tools.ids()).toEqual([
      "media.paged.image.tool.crop",
      "media.paged.image.tool.marqueeRect",
      "media.paged.image.tool.marqueeEllipse",
      "media.paged.image.tool.lasso",
      // The POLYGONAL lasso shares the lasso SLOT (same group): a
      // designer picks one or the other for a given edge, and neither
      // wants both on the rail at once. Click-driven where the freehand
      // one is drag-driven.
      "media.paged.image.tool.polygonal-lasso",
      "media.paged.image.tool.magicWand",
      "media.paged.image.tool.brush",
      "media.paged.image.tool.pencil",
      "media.paged.image.tool.eraser",
      // The retouching pair — the brush with a sampled paint layer.
      "media.paged.image.tool.clone",
      "media.paged.image.tool.heal",
      "media.paged.image.tool.type",
    ]);
    expect(fake.importers.ids()).toEqual(["media.paged.image.importer.raster"]);
  });

  it("the open command routes through host.shell.openPanel", () => {
    const fake = makeFakeEditor();
    const openPanel = vi.fn();
    load(fake, shellStub(openPanel));
    const cmd = fake.commands.get(
      "media.paged.image.command.openImage",
    ) as unknown as { handler: () => void };
    cmd.handler();
    expect(openPanel).toHaveBeenCalledWith(PANEL_ID);
  });

  it("the adjust command raises the panel before ingesting", () => {
    const fake = makeFakeEditor();
    const openPanel = vi.fn();
    load(fake, shellStub(openPanel));
    const cmd = fake.commands.get(
      "media.paged.image.command.adjustSelected",
    ) as unknown as { handler: () => void };
    cmd.handler();
    expect(openPanel).toHaveBeenCalledWith(PANEL_ID);
  });

  it("dispose leaves the shell exactly as found (honesty smoke test)", () => {
    const fake = makeFakeEditor();
    const loaded = load(fake);
    loaded.dispose();
    expect(fake.panels.ids()).toHaveLength(0);
    expect(fake.commands.ids()).toHaveLength(0);
    expect(fake.tools.ids()).toHaveLength(0);
    expect(fake.importers.ids()).toHaveLength(0);
  });
});

// ── the two manifests must be ONE manifest ───────────────────────────
//
// `manifest/manifest.json` is what `pnpm validate:manifest` checks and
// what the packaged bundle declares; `glue/manifest.json` is the COPY
// the bundle actually imports at build time. They are two tracked files
// with no sync step between them, so a capability declared in one and
// not the other produces a `PluginCapabilityError` at activate — which
// is exactly what happened when the `.abr` command was added. This is
// the guard that turns that into a one-line failure instead of a
// mystery.
describe("the manifest copy", () => {
  it("is byte-identical to the validated manifest", () => {
    const here = fileURLToPath(new URL(".", import.meta.url));
    const declared = readFileSync(`${here}../../manifest/manifest.json`, "utf8");
    const embedded = readFileSync(`${here}../manifest.json`, "utf8");
    expect(embedded).toBe(declared);
  });
});
