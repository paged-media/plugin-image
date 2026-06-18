// Registration wiring against the real in-process host adapter over the
// fake editor: the M0 honesty smoke test ("dispose leaves the shell
// exactly as found") extended with the M4 contributions — the second
// command and the K-2 raster importer. The ingest loop itself is
// session.spec.ts (real engine wasm).

import { describe, expect, it, vi } from "vitest";

import { loadBundle } from "@paged-media/plugin-sdk";

import { imageBundle, PANEL_ID } from "../src";
import { makeFakeEditor, mapBacking, shellStub, silentConsole } from "./helpers";

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
  it("registers the panel, the commands, the crop tool, and the raster importer", () => {
    const fake = makeFakeEditor();
    load(fake);
    expect(fake.panels.ids()).toEqual([PANEL_ID]);
    expect(fake.commands.ids()).toEqual([
      "media.paged.image.command.openImage",
      "media.paged.image.command.adjustSelected",
      "media.paged.image.command.autoEnhance",
      "media.paged.image.command.claimTiles",
      "media.paged.image.command.commitCrop",
    ]);
    expect(fake.tools.ids()).toEqual(["media.paged.image.tool.crop"]);
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
