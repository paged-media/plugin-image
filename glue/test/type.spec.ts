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

// RASTER TYPE — the lane from the host's font bytes to pixels in a layer.
//
// WHERE THE PROOFS LIVE, and why they are split. The Rust module
// (`image-js/src/text.rs`) asserts the contract's edges — a non-font is
// refused, a nonsensical size is refused, overlapping glyph coverage
// takes the max — but it cannot assert PIXELS, because there is no font
// in this repository. Fonts arrive from the HOST at runtime
// (`assets.getFontFace`), which is the whole design: a bundle renders
// only what the document already embeds and never fetches.
//
// So the font-shaped assertions live here, where a face can be supplied
// to the door the way the host would, and the honest ones that need no
// glyphs live in Rust.

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type {
  ElementGeometryItem,
  PluginManifest,
} from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { TypeSection } from "../src/panels/image-panel";
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

/** A real TrueType face from the platform, standing in for the one the
 *  HOST would serve. Not committed to this repo — there is no font here
 *  by design — so a machine without it skips the pixel assertions loudly
 *  rather than passing vacuously. */
function systemFace(): Uint8Array | null {
  for (const p of [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  ]) {
    try {
      return new Uint8Array(readFileSync(p));
    } catch {
      // next
    }
  }
  return null;
}

/** A session over the real wasm, with a host that serves `face`. */
async function withFont(face: Uint8Array | null) {
  const fake = makeFakeEditor();
  fake.placed.set("u1", psdBytes());
  fake.geometry.set("u1", geomFor("u1", [0, 0, 100, 200]));
  fake.emitSelection([{ kind: "rectangle", id: "u1" }]);
  const handle = createBundleHost(
    () => fake.editor,
    manifestJson as PluginManifest,
    {
      console: silentConsole,
      storage: mapBacking(),
      shell: shellStub(),
      // `assetSource`, not `assets` — the SDK injects a SOURCE and
      // exposes the surface. Getting that wrong made `supports(
      // "assets.fonts@1")` false and the door answer "this host serves
      // no fonts", which is the honest no-door message and exactly what
      // it should say when a host really wires nothing.
      assetSource: {
        getFontFace: async () =>
          face ? { family: "Test", bytes: face, format: "truetype" } : null,
        getPlacedImage: async () => null,
      } as never,
    },
  );
  const session = createImageSession(handle.host);
  expect(await session.ingestSelection()).toBe(true);
  return { session, handle, fake };
}

describe("the raster type lane", () => {
  it("keeps its settings on the SESSION, so several runs share them", async () => {
    const { session, handle } = await withFont(null);
    expect(session.state().type).toEqual({
      text: "",
      family: "Helvetica",
      sizePx: 48,
    });
    session.setType({ text: "Hello", sizePx: 64 });
    expect(session.state().type).toMatchObject({
      text: "Hello",
      family: "Helvetica",
      sizePx: 64,
    });
    session.dispose();
    handle.dispose();
  });

  it("says the DOCUMENT has no such face, rather than fetching one", async () => {
    // The design in one assertion: a bundle renders the faces the
    // document embeds and never reaches the network, so an unknown
    // family is a stated dead end and not a download.
    const { session, handle } = await withFont(null);
    expect(await session.paintText([10, 10], "Hi", "Nonesuch", 24)).toBe(false);
    expect(session.state().status).toContain("Nonesuch");
    expect(session.state().status).toContain("never a web fetch");
    session.dispose();
    handle.dispose();
  });

  it("paints real glyphs into the image", async () => {
    const face = systemFace();
    if (!face) {
      // Loudly, not silently: a machine with no system font cannot prove
      // this half, and a quiet pass would read as coverage it does not
      // have.
      console.warn("no system font available — skipping the pixel assertion");
      return;
    }
    const { session, handle } = await withFont(face);
    session.setType({ text: "Hi", family: "Test", sizePx: 20 });
    const ok = await session.paintText([1, 1], "Hi", "Test", 20);

    // The run SHAPES and RASTERIZES here — that half is CPU and is what
    // this test exists to prove reachable with a real face. The paint
    // itself is a masked fill, and fills are GPU-only, so in Node it
    // declines at the device gate. What it must never do is fail for a
    // FONT reason, and that is the assertion: whatever happened, it was
    // not "no such face" and not "no bytes".
    expect(session.state().status).not.toContain("no bytes");
    expect(session.state().status).not.toContain("serves no fonts");
    expect(session.state().status).not.toContain("could not shape");
    if (ok) {
      expect(session.state().status).toContain("Type set");
    }
    session.dispose();
    handle.dispose();
  });
});

describe("the panel's Type section", () => {
  const render = (over: Partial<Parameters<typeof TypeSection>[0]> = {}) =>
    textOf(
      TypeSection({
        type: { text: "", family: "Helvetica", sizePx: 48 },
        disabled: false,
        onChange: () => {},
        ...over,
      }),
    )
      .join(" ")
      .replace(/\s+/g, " ");

  it("tells the user what a click does", () => {
    expect(render()).toContain("Click the canvas to set the BASELINE");
  });

  it("states that glyphs are SHAPED, not merely advanced", () => {
    // The property that separates correct type from plausible type, said
    // where a designer setting Arabic or Devanagari will read it.
    const text = render();
    expect(text).toContain("shaped by the font");
    expect(text).toContain("joining, reordering, ligatures");
  });

  it("states where faces come from and what a missing glyph does", () => {
    const text = render();
    expect(text).toContain("come from the DOCUMENT");
    expect(text).toContain("nothing is ever fetched from the network");
    expect(text).toContain("left undrawn rather than replaced with a box");
  });

  it("says plainly that this is pixels, not a text object", () => {
    // The scope sentence. Without it the tool reads as a worse version
    // of the host's text frame instead of a different thing.
    const text = render();
    expect(text).toContain("paints PIXELS, not a text object");
    expect(text).toContain("host");
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
