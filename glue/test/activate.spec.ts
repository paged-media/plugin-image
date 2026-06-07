// Registration wiring against the real in-process host adapter over a
// minimal fake editor (the plugin-draw/plugin-web pattern). Engine
// behavior is not faked; this proves the contract wiring + the honesty
// smoke test ("dispose leaves the shell exactly as found") — the M0
// exit's "plugin bundle loads via SDK with zero core changes".

import { describe, expect, it, vi } from "vitest";

import type { PagedEditor } from "@paged-media/plugin-api";
import { loadBundle } from "@paged-media/plugin-sdk";

import { imageBundle, PANEL_ID } from "../src";

function fakeRegistry() {
  const byId = new Map<string, { id: string }>();
  return {
    ids: () => Array.from(byId.keys()),
    get: (id: string) => byId.get(id),
    register(c: { id: string }) {
      if (byId.has(c.id)) throw new Error(`duplicate id ${c.id}`);
      byId.set(c.id, c);
      return {
        dispose() {
          byId.delete(c.id);
        },
      };
    },
  };
}

function makeFakeEditor() {
  const panels = fakeRegistry();
  const commands = fakeRegistry();
  const editor = {
    registries: { panels, commands },
    selection: {
      elementSelection: [] as unknown[],
      setElementSelection: () => {},
      setElementGeometry: () => {},
    },
    camera: { camera: { scale: 1, tx: 0, ty: 0 } },
    client: {
      mutate: async () => ({ kind: "mutationApplied", payload: {} }),
      documentMeta: async () => ({ pageCount: 1, activePage: "pg1" }),
      collection: async () => [],
      setElementSelection: async (ids: unknown[]) => ids,
      elementGeometry: async () => [],
      subscribe: () => () => {},
    },
  };
  return { editor: editor as unknown as PagedEditor, panels, commands };
}

const silent = { debug() {}, info() {}, warn() {}, error() {} };
const mapBacking = () => {
  const m = new Map<string, string>();
  return {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    removeItem: (k: string) => void m.delete(k),
    keys: () => Array.from(m.keys()),
  };
};

describe("imageBundle.activate", () => {
  it("registers the adjustments panel + the open command", () => {
    const fake = makeFakeEditor();
    loadBundle(() => fake.editor, imageBundle, {
      console: silent,
      storage: mapBacking(),
      shell: { openPanel() {}, closePanel() {} },
    });
    expect(fake.panels.ids()).toEqual([PANEL_ID]);
    expect(fake.commands.ids()).toEqual(["media.paged.image.command.openImage"]);
  });

  it("the open command routes through host.shell.openPanel", () => {
    const fake = makeFakeEditor();
    const openPanel = vi.fn();
    loadBundle(() => fake.editor, imageBundle, {
      console: silent,
      storage: mapBacking(),
      shell: { openPanel, closePanel() {} },
    });
    const cmd = fake.commands.get(
      "media.paged.image.command.openImage",
    ) as unknown as { handler: () => void };
    cmd.handler();
    expect(openPanel).toHaveBeenCalledWith(PANEL_ID);
  });

  it("dispose leaves the shell exactly as found (honesty smoke test)", () => {
    const fake = makeFakeEditor();
    const loaded = loadBundle(() => fake.editor, imageBundle, {
      console: silent,
      storage: mapBacking(),
      shell: { openPanel() {}, closePanel() {} },
    });
    loaded.dispose();
    expect(fake.panels.ids()).toHaveLength(0);
    expect(fake.commands.ids()).toHaveLength(0);
  });
});
