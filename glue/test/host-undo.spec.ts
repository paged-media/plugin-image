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

// HOST UNDO in the rasterImage context (ADR 012, in-context tier): while
// the context is active, Cmd+Z steps the image's own journal. When the
// image has nothing to undo, the hooks must DECLINE so the keystroke
// falls through to the document — an image frame must never swallow it.

import { describe, expect, it } from "vitest";

import { loadBundle } from "@paged-media/plugin-sdk";

import { imageBundle } from "../src/index";
import { makeFakeEditor, mapBacking, shellStub, silentConsole } from "./helpers";

interface Hooks {
  type: string;
  toolIds?: string[];
  onUndo?(): boolean;
  onRedo?(): boolean;
  onCanUndo?(): boolean;
  onCanRedo?(): boolean;
}

function load() {
  const fake = makeFakeEditor();
  const contexts: Hooks[] = [];
  const loaded = loadBundle(() => fake.editor, imageBundle, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
    onEditContextRegistered: (c: unknown) => {
      contexts.push(c as Hooks);
      return { dispose() {} };
    },
  } as never);
  return { loaded, contexts };
}

describe("host Cmd+Z inside a raster image", () => {
  it("the rasterImage context declares all four undo hooks", () => {
    const { loaded, contexts } = load();
    const ctx = contexts.find((c) => c.type === "rasterImage");
    expect(ctx).toBeDefined();
    for (const hook of ["onUndo", "onRedo", "onCanUndo", "onCanRedo"] as const) {
      expect(typeof ctx![hook], hook).toBe("function");
    }
    loaded.dispose();
  });

  it("with nothing to undo it declines, so the document gets the keystroke", () => {
    const { loaded, contexts } = load();
    const ctx = contexts.find((c) => c.type === "rasterImage")!;
    expect(ctx.onCanUndo!()).toBe(false);
    expect(ctx.onCanRedo!()).toBe(false);
    expect(ctx.onUndo!()).toBe(false);
    expect(ctx.onRedo!()).toBe(false);
    loaded.dispose();
  });

  it("the Move tool is part of the context's tool set", () => {
    const { loaded, contexts } = load();
    const ctx = contexts.find((c) => c.type === "rasterImage")!;
    expect(ctx.toolIds).toContain("media.paged.image.tool.move");
    loaded.dispose();
  });
});
