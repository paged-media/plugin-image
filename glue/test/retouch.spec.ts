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

// The RETOUCHING pair — clone stamp and healing brush.
//
// The pixels are Rust's half (device tests in `image-js/src/stroke.rs`,
// including the one that measures a heal landing nearer its destination
// than a clone does). What lives here is the part that is glue: the
// anchor's lifecycle, the refusal to paint without one, the alt-click
// that must NOT start a stroke, and what the panel says about the
// healing brush's limit.
//
// Node has no WebGPU, so every stroke here declines at the GPU gate, and
// the ORDER of the two refusals turns out to be a real decision. It is
// GPU FIRST, deliberately: on a machine with no device the tool cannot
// paint whatever the user does, so "alt-click to set a source" would
// invite an action that still fails. The first draft of this spec
// asserted the opposite order and was wrong about the user, not about
// the code.
//
// That leaves the no-anchor refusal unreachable in Node, so it is proven
// where a device exists: `image_editor_clone_without_an_anchor_paints_
// nothing` in `image-js/src/stroke.rs` runs on a real adapter and
// asserts the pixels are untouched.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type {
  CanvasPointerEvent,
  ElementGeometryItem,
  PluginManifest,
} from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { makeBrushGesture } from "../src/brush-tool";
import { SAMPLING_TOOLS, STROKE_TOOLS } from "../src/engine";
import { RetouchSection } from "../src/panels/image-panel";
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

function ev(
  pagePoint: [number, number] | null,
  over: Partial<CanvasPointerEvent> = {},
): CanvasPointerEvent {
  return {
    pageId: "pg1",
    pagePoint,
    pressure: 1,
    pointerType: "mouse",
    modifiers: { shift: false, alt: false, ctrl: false, meta: false },
    ...over,
  } as CanvasPointerEvent;
}

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

describe("the retouching tools", () => {
  it("are registered as painting tools and marked as sampling ones", () => {
    expect(STROKE_TOOLS).toContain("clone");
    expect(STROKE_TOOLS).toContain("heal");
    expect(SAMPLING_TOOLS).toEqual(["clone", "heal"]);
    // The brush is NOT a sampling tool — the flag is what routes the
    // alt-click and the no-anchor refusal, so a wrong answer here would
    // make the ordinary brush demand a clone source.
    expect(SAMPLING_TOOLS).not.toContain("brush");
  });

  it("keeps the anchor on the SESSION, so it survives a stroke", async () => {
    const { session, handle } = await ingested();
    expect(session.state().cloneSource).toBeNull();
    session.setCloneSource([12, 34]);
    expect(session.state().cloneSource).toEqual({
      x: 12,
      y: 34,
      aligned: true,
    });
    expect(session.state().status).toContain("12, 34");
    // Aligned toggles without losing the anchor.
    session.setCloneAligned(false);
    expect(session.state().cloneSource).toEqual({
      x: 12,
      y: 34,
      aligned: false,
    });
    session.dispose();
    handle.dispose();
  });

  it("reports the missing DEVICE before the missing anchor", async () => {
    // Both are missing in Node. The device wins because it is the one
    // the user cannot work around: telling them to alt-click would send
    // them down a path that still cannot paint.
    const { session, handle } = await ingested();
    expect(await session.brushBegin("clone")).toBe(false);
    expect(session.state().status).toContain("GPU-only");
    expect(session.state().status).not.toContain("Alt-click");
    session.dispose();
    handle.dispose();
  });

  it("asks for an anchor once a device exists", async () => {
    // The check itself, exercised by pretending the device is there —
    // the pixel proof that a source-less clone paints NOTHING lives in
    // the Rust device test, which is the only place it can.
    const { session, handle } = await ingested();
    (session.state() as { gpu: boolean }).gpu = true;
    expect(await session.brushBegin("clone")).toBe(false);
    expect(session.state().status).toContain("Alt-click");
    expect(await session.brushBegin("heal")).toBe(false);
    expect(session.state().status).toContain("Alt-click");
    session.dispose();
    handle.dispose();
  });

  it("an ordinary brush never asks for an anchor", async () => {
    const { session, handle } = await ingested();
    (session.state() as { gpu: boolean }).gpu = true;
    // No anchor, but the brush paints from a colour — so it proceeds to
    // whatever the engine says, never to the clone instruction.
    await session.brushBegin("brush");
    expect(session.state().status).not.toContain("Alt-click");
    session.dispose();
    handle.dispose();
  });
});

describe("the alt-click anchor gesture", () => {
  /** A live clone gesture over the real host, with the frame fit
   *  resolved (page (100,100) is image (1,1) at this geometry). */
  async function gestureOn(tool: "clone" | "heal" | "brush") {
    const made = await ingested();
    const gesture = makeBrushGesture(made.handle.host, made.session, tool);
    gesture.onActivate?.(undefined as never);
    await new Promise((r) => setTimeout(r, 0));
    return { ...made, gesture };
  }

  it("sets the source and does NOT start a stroke", async () => {
    // Painting the anchor click is the one thing a retoucher never
    // wants, so the alt-click returns before `machine.down`.
    const { session, handle, gesture } = await gestureOn("clone");
    gesture.onPointerDown?.(
      ev([100, 100], {
        modifiers: { shift: false, alt: true } as never,
      }),
    );
    await new Promise((r) => setTimeout(r, 0));
    expect(session.state().cloneSource).toMatchObject({ x: 1, y: 1 });
    expect(session.state().strokeActive, "no stroke was opened").toBe(false);
    session.dispose();
    handle.dispose();
  });

  it("a plain click on a clone tool is a stroke, not an anchor", async () => {
    const { session, handle, gesture } = await gestureOn("clone");
    gesture.onPointerDown?.(ev([100, 100]));
    await new Promise((r) => setTimeout(r, 0));
    // No anchor was set — the click went down the painting path (which
    // then declines here, GPU-less, and that is the point: it did not
    // silently become an anchor).
    expect(session.state().cloneSource).toBeNull();
    session.dispose();
    handle.dispose();
  });

  it("an alt-click on the ordinary BRUSH sets no anchor", async () => {
    // Otherwise alt-dragging the brush would stop painting.
    const { session, handle, gesture } = await gestureOn("brush");
    gesture.onPointerDown?.(
      ev([100, 100], {
        modifiers: { shift: false, alt: true } as never,
      }),
    );
    await new Promise((r) => setTimeout(r, 0));
    expect(session.state().cloneSource).toBeNull();
    session.dispose();
    handle.dispose();
  });
});

describe("the panel's Retouch section", () => {
  const render = (over: Partial<Parameters<typeof RetouchSection>[0]> = {}) =>
    textOf(
      RetouchSection({
        source: { x: 12, y: 34, aligned: true },
        hasSelection: true,
        onAligned: () => {},
        onContentAwareFill: () => {},
        ...over,
      }),
    )
      .join(" ")
      .replace(/\s+/g, " ");

  it("shows the anchor, or says there is none", () => {
    expect(render()).toContain("12, 34");
    expect(render({ source: null })).toContain("not set");
  });

  it("describes what the healing brush actually does, and where it stops", () => {
    // This text used to warn that a mean match still seams across a
    // ramp. The solve removed that limit, so the warning had to go —
    // leaving it would be a different kind of lie. What remains is the
    // real edge case: no boundary, no correction.
    const text = render();
    expect(text).toContain("gradient domain");
    expect(text).toContain("follows a ramp");
    expect(text).toContain("falls back to a plain clone");
    expect(text).not.toContain("still shows a seam");
  });

  it("offers content-aware fill, and states what it does and does not do", () => {
    // This section used to say the feature was absent. It ships now, so
    // the note describes the real thing — including the two limits that
    // are genuinely there rather than the absence that no longer is.
    const text = render();
    expect(text).toContain("Content-aware fill");
    expect(text).toContain("copied from real image data");
    expect(text).toContain("coarse to fine");
    expect(text).toContain("still WINDOWED");
    expect(text).not.toContain("is not offered");
  });

  it("asks for a selection before it will fill", () => {
    // The fill synthesises the SELECTION; with none there is no hole,
    // and a button that did nothing would read as broken.
    expect(render({ hasSelection: false })).toContain("select an area first");
  });

  it("explains alt-click and aligned", () => {
    const text = render();
    expect(text).toContain("Alt-click");
    expect(text).toContain("fixed offset");
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
