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

// Features that were built engine-side with no way in: Define pattern →
// Fill with pattern, Shape blur (the pattern is its shape), smart
// objects. Real engine wasm; Node has no WebGPU, so the GPU fills are
// pinned to their honest decline, and the CPU half (the pattern
// capture) is checked for real.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import { MoveSection } from "../src/panels/sections/move-section";
import { PatternSection } from "../src/panels/sections/pattern-section";
import { SmartObjectSection } from "../src/panels/sections/smart-object-section";
import { createImageSession, type ImageSession } from "../src/session";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";

function open() {
  const fake = makeFakeEditor();
  const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
    console: silentConsole,
    storage: mapBacking(),
    shell: shellStub(),
  });
  return { handle, session: createImageSession(handle.host) };
}

describe("Define pattern / Fill with pattern / Shape blur", () => {
  it("fill and shape blur say a pattern is needed before one is defined", async () => {
    const { handle, session } = open();
    expect(await session.importBytes("p.psd", psdBytes())).toBe(true);
    expect(await session.fillWithPattern()).toBe(false);
    expect(session.state().status).toMatch(/No pattern defined/);
    expect(await session.shapeBlurWithPattern()).toBe(false);
    expect(session.state().status).toMatch(/define one first/);
    session.dispose();
    handle.dispose();
  });

  it("Define pattern captures the whole image when nothing is selected", async () => {
    const { handle, session } = open();
    expect(await session.importBytes("p.psd", psdBytes())).toBe(true);
    expect(session.definePattern()).toBe(true);
    expect(session.state().pattern).toEqual({ width: 2, height: 1 });
    session.dispose();
    handle.dispose();
  });

  it("with a pattern, the fill reaches the engine and declines only for want of a GPU", async () => {
    const { handle, session } = open();
    expect(await session.importBytes("p.psd", psdBytes())).toBe(true);
    session.definePattern();
    expect(session.state().gpu).toBe(false);
    expect(await session.fillWithPattern()).toBe(false);
    expect(session.state().status).toMatch(/GPU|WebGPU|init_gpu/);
    expect(await session.shapeBlurWithPattern()).toBe(false);
    expect(session.state().status).toMatch(/GPU|WebGPU|init_gpu/);
    session.dispose();
    handle.dispose();
  });

  it("Define pattern with nothing ingested says so", () => {
    const { handle, session } = open();
    expect(session.definePattern()).toBe(false);
    expect(session.state().status).toMatch(/Nothing ingested/);
    session.dispose();
    handle.dispose();
  });
});

describe("smart objects", () => {
  it("converting the active layer reaches the engine (a one-layer stack, no GPU)", async () => {
    const { handle, session } = open();
    expect(await session.importBytes("p.psd", psdBytes())).toBe(true);
    const active = session.state().layers.active;
    expect(active).toBe(0);
    expect(await session.makeLayerSmart(active)).toBe(true);
    expect(session.state().layers.layers[0]?.kind).toBe("smart");
    session.dispose();
    handle.dispose();
  });
});

// ── the panel sections that make them reachable ──────────────────────


type Props = Record<string, unknown>;

/** Every host element's props in a pure component tree (no DOM, no
 *  hooks run), so a test can find a control by its data attribute. */
function elements(node: unknown, out: Props[] = []): Props[] {
  if (node === null || node === undefined || typeof node !== "object") return out;
  if (Array.isArray(node)) {
    for (const n of node) elements(n, out);
    return out;
  }
  const el = node as { type?: unknown; props?: Props };
  if (typeof el.type === "function") {
    return elements((el.type as (p: unknown) => unknown)(el.props), out);
  }
  if (el.props) {
    out.push(el.props);
    elements(el.props.children, out);
  }
  return out;
}

const byData = (tree: unknown, attr: string): Props => {
  const hit = elements(tree).find((p) => attr in p);
  if (!hit) throw new Error(`no element with ${attr}`);
  return hit;
};

/** A session stand-in that records which door a control called. */
function recorder() {
  const calls: string[] = [];
  const session = new Proxy(
    {},
    {
      get: (_t, prop) => (...args: unknown[]) => {
        calls.push(`${String(prop)}(${args.join(",")})`);
        return Promise.resolve(true);
      },
    },
  ) as unknown as ImageSession;
  return { session, calls };
}

describe("panel sections", () => {
  it("Pattern: Define is live after ingest; Fill and Shape blur wait for a pattern and a GPU", () => {
    const { session, calls } = recorder();
    const props = {
      session,
      pattern: null,
      gpu: true,
      disabled: false,
      scale: 2,
      onScale: () => {},
      radius: 12,
      onRadius: () => {},
    };
    let tree = PatternSection(props);
    expect(byData(tree, "data-image-define-pattern").disabled).toBe(false);
    expect(byData(tree, "data-image-fill-pattern").disabled).toBe(true);
    expect(byData(tree, "data-image-shape-blur").disabled).toBe(true);

    tree = PatternSection({ ...props, pattern: { width: 4, height: 4 } });
    (byData(tree, "data-image-define-pattern").onClick as () => void)();
    (byData(tree, "data-image-fill-pattern").onClick as () => void)();
    (byData(tree, "data-image-shape-blur").onClick as () => void)();
    expect(calls).toEqual(["definePattern()", "fillWithPattern(2)", "shapeBlurWithPattern(12)"]);

    tree = PatternSection({ ...props, pattern: { width: 4, height: 4 }, gpu: false });
    expect(byData(tree, "data-image-fill-pattern").disabled).toBe(true);
  });

  it("Move: nudges the selection when there is one, the layer otherwise; Shift is 10 px", () => {
    const { session, calls } = recorder();
    const props = {
      session,
      hasSelection: true,
      size: { width: 100, height: 40 },
      gpu: true,
      disabled: false,
    };
    const right = (tree: unknown) =>
      elements(tree).find((p) => p["data-image-nudge"] === "→")!.onClick as (e: {
        shiftKey: boolean;
      }) => void;
    right(MoveSection(props))({ shiftKey: false });
    right(MoveSection({ ...props, hasSelection: false }))({ shiftKey: true });
    (byData(MoveSection(props), "data-image-offset-wrap").onClick as () => void)();
    expect(calls).toEqual(["moveSelection(1,0,0)", "offsetLayer(10,0,0)", "offsetFilter(50,20)"]);
  });

  it("Smart object: convert only a pixel layer; render only a smart one", () => {
    const { session, calls } = recorder();
    const props = {
      session,
      active: 0,
      isSmart: false,
      gpu: true,
      disabled: false,
      scale: 0.5,
      onScale: () => {},
    };
    let tree = SmartObjectSection(props);
    expect(byData(tree, "data-image-render-smart").disabled).toBe(true);
    (byData(tree, "data-image-make-smart").onClick as () => void)();
    tree = SmartObjectSection({ ...props, isSmart: true });
    expect(byData(tree, "data-image-make-smart").disabled).toBe(true);
    (byData(tree, "data-image-render-smart").onClick as () => void)();
    expect(calls).toEqual(["makeLayerSmart(0)", "renderLayerSmart(0,0.5)"]);
  });
});
