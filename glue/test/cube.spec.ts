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

// .cube parsing and the resample to the lookup kernel's 9×9×9 cube.

import { describe, expect, it } from "vitest";

import { KERNEL_EDGE, parseCube, toKernelCube } from "../src/cube";

/** An N³ cube text whose entry is f(r, g, b) on the 0–1 lattice. */
function cubeText(n: number, f: (r: number, g: number, b: number) => number[], extra = ""): string {
  const lines = [`TITLE "test"`, `LUT_3D_SIZE ${n}`, extra];
  for (let b = 0; b < n; b++)
    for (let g = 0; g < n; g++)
      for (let r = 0; r < n; r++) lines.push(f(r / (n - 1), g / (n - 1), b / (n - 1)).join(" "));
  return lines.join("\n");
}

describe(".cube", () => {
  it("an identity table of any size resamples to the identity cube", () => {
    for (const n of [2, 17, 33]) {
      const k = toKernelCube(parseCube(cubeText(n, (r, g, b) => [r, g, b])));
      expect(k.length).toBe(KERNEL_EDGE ** 3 * 3);
      const e = KERNEL_EDGE - 1;
      // lattice point (r=3, g=5, b=8) → (3/8, 5/8, 1)
      const i = ((8 * KERNEL_EDGE + 5) * KERNEL_EDGE + 3) * 3;
      expect(k[i]).toBeCloseTo(3 / e, 6);
      expect(k[i + 1]).toBeCloseTo(5 / e, 6);
      expect(k[i + 2]).toBeCloseTo(1, 6);
    }
  });

  it("keeps the title and reads red-fastest order", () => {
    const c = parseCube(cubeText(2, (r, g, b) => [r, g * 0.5, b * 0.25]));
    expect(c.title).toBe("test");
    expect(Array.from(c.data.slice(3, 6))).toEqual([1, 0, 0]); // r=1, g=0, b=0
  });

  it("a DOMAIN maps the input range onto the lattice", () => {
    // Domain 0–2: an input of 1 sits halfway along the file's lattice.
    const c = parseCube(cubeText(2, (r) => [r, 0, 0], "DOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2"));
    const k = toKernelCube(c);
    expect(k[(KERNEL_EDGE - 1) * 3]).toBeCloseTo(0.5, 6); // input r = 1 → half
  });

  it("refuses malformed files with the line", () => {
    expect(() => parseCube("LUT_3D_SIZE 2\n0 0 0\n1 x 0")).toThrow(/line 3/);
    expect(() => parseCube("LUT_3D_SIZE 2\n0 0 0")).toThrow(/needs 8/);
    expect(() => parseCube("LUT_1D_SIZE 4")).toThrow(/1D/);
    expect(() => parseCube("0 0 0")).toThrow(/LUT_3D_SIZE/);
  });
});
