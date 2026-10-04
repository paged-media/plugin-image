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

// The gradient and red-eye tools: the geometry they hand the engine, and
// their honest decline in Node (both are GPU kernels).

import { describe, expect, it } from "vitest";

import { createBundleHost } from "@paged-media/plugin-sdk";
import type { PluginManifest } from "@paged-media/plugin-api";
import manifestJson from "@paged-media/image-manifest/manifest.json";

import { constrain45 } from "../src/gradient-tool";
import { redEyeBox } from "../src/red-eye-tool";
import { createImageSession } from "../src/session";
import { makeFakeEditor, mapBacking, psdBytes, shellStub, silentConsole } from "./helpers";

describe("gradient tool", () => {
  it("Shift snaps the line to 45° and keeps its length", () => {
    const to = constrain45([0, 0], [10, 1]);
    expect(to[0]).toBeCloseTo(Math.hypot(10, 1));
    expect(to[1]).toBeCloseTo(0);
    const diag = constrain45([0, 0], [10, 9]);
    expect(diag[0]).toBeCloseTo(diag[1]);
  });
});

describe("red eye tool", () => {
  it("a drag is its box, either direction; a click is a 24 px box around the point", () => {
    expect(redEyeBox([10, 10], [4, 20])).toEqual({ x: 4, y: 10, w: 6, h: 10 });
    expect(redEyeBox([50, 50], [50.5, 50])).toEqual({ x: 38, y: 38, w: 24, h: 24 });
  });
});

describe("in Node (no GPU)", () => {
  it("both decline with the GPU reason", async () => {
    const fake = makeFakeEditor();
    const handle = createBundleHost(() => fake.editor, manifestJson as PluginManifest, {
      console: silentConsole,
      storage: mapBacking(),
      shell: shellStub(),
    });
    const session = createImageSession(handle.host);
    expect(await session.importBytes("t.psd", psdBytes())).toBe(true);
    expect(await session.fillGradientLine([0, 0], [2, 0])).toBe(false);
    expect(session.state().status).toMatch(/GPU-only/);
    expect(await session.applyRedEye({ x: 0, y: 0, w: 2, h: 1 })).toBe(false);
    expect(session.state().status).toMatch(/GPU-only/);
    // A degenerate box does nothing at all.
    expect(await session.applyRedEye({ x: 0, y: 0, w: 0, h: 0 })).toBe(false);
    session.setGradientKind("radial");
    expect(session.state().gradientKind).toBe("radial");
    session.dispose();
    handle.dispose();
  });
});
