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

// Foreground / background colour and the eyedropper. Real engine wasm:
// the sample reads the composite through the CPU tile window.

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import { averageRgba8, fromHex, swapped, toHex, DEFAULT_COLORS } from "../src/color-state";
import { createImageSession } from "../src/session";
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

describe("colour-state", () => {
  it("averages premultiplied, so a transparent neighbour does not darken the sample", () => {
    const px = Uint8Array.from([255, 0, 0, 255, 0, 0, 0, 0]);
    expect(averageRgba8(px)).toEqual([1, 0, 0, 0.5]);
    expect(averageRgba8(new Uint8Array(0))).toBeNull();
  });

  it("hex round-trips, and bad input is refused", () => {
    expect(toHex([1, 0.5, 0, 1])).toBe("#ff8000");
    expect(fromHex("#ff8000")).toEqual([1, 128 / 255, 0, 1]);
    expect(fromHex("#f80")).toEqual([1, 136 / 255, 0, 1]);
    expect(fromHex("nope")).toBeNull();
  });

  it("swap exchanges, and the default is black on white", () => {
    expect(swapped(DEFAULT_COLORS)).toEqual({ fg: [1, 1, 1, 1], bg: [0, 0, 0, 1] });
  });
});

describe("the session's colours", () => {
  it("the foreground IS the brush colour, both ways", () => {
    const { handle, session } = open();
    session.setForeground([1, 0, 0, 1]);
    expect(session.state().brush.color).toEqual([1, 0, 0, 1]);
    session.setBrushParams({ color: [0, 0, 1, 1] });
    expect(session.state().colors.fg).toEqual([0, 0, 1, 1]);
    session.swapColors();
    expect(session.state().colors).toEqual({ fg: [1, 1, 1, 1], bg: [0, 0, 1, 1] });
    expect(session.state().brush.color).toEqual([1, 1, 1, 1]);
    session.resetColors();
    expect(session.state().colors).toEqual(DEFAULT_COLORS);
    session.dispose();
    handle.dispose();
  });

  it("the eyedropper takes the composite's pixel as the foreground", async () => {
    const { handle, session } = open();
    // psdBytes(): 2×1, pixels (10,30,50) and (20,40,60).
    expect(await session.importBytes("c.psd", psdBytes())).toBe(true);
    const c = session.sampleColor([1, 0], 1);
    expect(c).toEqual([20 / 255, 40 / 255, 60 / 255, 1]);
    expect(session.state().colors.fg).toEqual(c);
    expect(session.state().brush.color).toEqual(c);
    // 3×3 around the left pixel clips to the image: the two pixels' mean.
    const avg = session.sampleColor([0, 0], 3, "bg")!;
    expect(avg.map((v) => Math.round(v * 255))).toEqual([15, 35, 55, 255]);
    expect(session.state().colors.bg).toEqual(avg);
    // Outside the image: nothing.
    expect(session.sampleColor([5, 0])).toBeNull();
    session.dispose();
    handle.dispose();
  });
});
