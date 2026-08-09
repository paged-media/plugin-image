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

// The QUICK-SELECTION growth rule, against CONSTRUCTED images (no wasm):
// the point of the module being pure is that every claim in its header —
// stop at an edge, accumulate across dabs, gate on the painted evidence
// rather than a fixed threshold, cap, and stay O(painted region) — is a
// pixel assertion here rather than a comment. The engine-side commit (the
// coverage plane reaching the selection through the channel door, under
// every combine mode) is proven over the REAL wasm in selection.spec.ts.

import { describe, expect, it } from "vitest";

import {
  coverageToRgba8,
  createQuickSelectStroke,
  QUICK_SELECT_MAX_PIXELS,
  type QuickSelectPixels,
} from "../src/quick-select";

/** A synthetic image whose `readWindow` reproduces the engine C-6 tile
 *  contract exactly (clipped to the extent, tightly packed, empty when
 *  fully outside) and RECORDS every window it was asked for — the reads
 *  are what the O(painted region) claim is measured on. */
function testImage(
  width: number,
  height: number,
  at: (x: number, y: number) => [number, number, number, number],
) {
  const windows: Array<[number, number, number, number]> = [];
  const pixels: QuickSelectPixels = {
    width,
    height,
    readWindow(x, y, w, h) {
      windows.push([x, y, w, h]);
      const x0 = Math.min(x, width);
      const y0 = Math.min(y, height);
      const x1 = Math.min(x + w, width);
      const y1 = Math.min(y + h, height);
      if (x1 <= x0 || y1 <= y0) return new Uint8Array(0);
      const tw = x1 - x0;
      const th = y1 - y0;
      const out = new Uint8Array(tw * th * 4);
      for (let row = 0; row < th; row++) {
        for (let col = 0; col < tw; col++) {
          const p = at(x0 + col, y0 + row);
          const o = (row * tw + col) * 4;
          out[o] = p[0];
          out[o + 1] = p[1];
          out[o + 2] = p[2];
          out[o + 3] = p[3];
        }
      }
      return out;
    },
  };
  return { pixels, windows };
}

const DARK: [number, number, number, number] = [60, 60, 60, 255];
const LIGHT: [number, number, number, number] = [180, 180, 180, 255];

/** 32×16, split down the middle: DARK left of x = 16, LIGHT from it. */
const splitImage = () =>
  testImage(32, 16, (x) => (x < 16 ? DARK : LIGHT));

const selectedCount = (cov: Uint8Array) => {
  let n = 0;
  for (const v of cov) if (v !== 0) n++;
  return n;
};

describe("quick selection — growth from a painted dab", () => {
  it("fills the flat region it was painted in and STOPS at the hard edge", () => {
    const { pixels } = splitImage();
    const stroke = createQuickSelectStroke(pixels, { radius: 4 });

    // One dab well inside the dark half (the disc never touches the
    // boundary, so nothing light is ever painted evidence).
    const added = stroke.dab(6, 8);

    expect(added).toBe(16 * 16);
    expect(stroke.count(), "the whole dark half, and only it").toBe(16 * 16);
    expect(stroke.bounds()).toEqual({ x: 0, y: 0, w: 16, h: 16 });

    const cov = stroke.coverage();
    // Every dark pixel in, every light pixel out — asserted per pixel on
    // both sides of the boundary column, which is where a missing edge
    // stop shows up first.
    for (let y = 0; y < 16; y++) {
      expect(cov[y * 32 + 15], `dark (15,${y}) selected`).toBe(255);
      expect(cov[y * 32 + 16], `light (16,${y}) NOT selected`).toBe(0);
      expect(cov[y * 32 + 31], `light (31,${y}) NOT selected`).toBe(0);
    }
    expect(selectedCount(cov)).toBe(16 * 16);

    // σ ≈ 0 on a flat region, so the gate stayed at the documented floor
    // — the reason the edge held.
    const st = stroke.statistics();
    expect(st.deviation[0]).toBeCloseTo(0, 6);
    expect(st.tolerance[0]).toBeCloseTo(16, 6);
    expect(st.mean[0]).toBeCloseTo(60, 6);
  });

  it("a SECOND dab in a differently-coloured region extends the selection (it accumulates)", () => {
    const { pixels } = splitImage();
    const stroke = createQuickSelectStroke(pixels, { radius: 4 });

    stroke.dab(6, 8);
    expect(stroke.count()).toBe(16 * 16);

    // Paint the light half too. The painted evidence now spans both
    // colours, so μ moves between them and σ opens the gate wide enough
    // to cross the boundary the FIRST dab correctly refused.
    const added = stroke.dab(26, 8);

    expect(added, "the second dab adds the light half").toBe(16 * 16);
    expect(stroke.count(), "both halves — the first is NOT replaced").toBe(
      32 * 16,
    );
    expect(stroke.bounds()).toEqual({ x: 0, y: 0, w: 32, h: 16 });

    const cov = stroke.coverage();
    // The load-bearing assertion for accumulation: pixels that only the
    // FIRST dab could have selected are still selected.
    for (let y = 0; y < 16; y++) {
      expect(cov[y * 32 + 0], `dark (0,${y}) still selected`).toBe(255);
      expect(cov[y * 32 + 15], `dark (15,${y}) still selected`).toBe(255);
      expect(cov[y * 32 + 31], `light (31,${y}) now selected`).toBe(255);
    }
    expect(selectedCount(cov)).toBe(32 * 16);

    const st = stroke.statistics();
    expect(st.mean[0], "μ sits between the two painted colours").toBeCloseTo(
      120,
      6,
    );
    expect(st.deviation[0]).toBeCloseTo(60, 6);
  });

  it("the running-statistics gate takes a textured region a FIXED threshold cannot", () => {
    // The constructed contrast. A checkerboard of 100/140 (a "uniform but
    // noisy" region) beside a flat 220 outlier region. Both strokes get
    // the same base tolerance and the same dab; only the variance weight
    // differs — k = 0 IS the fixed threshold.
    const { pixels } = testImage(32, 16, (x, y) => {
      if (x >= 16) return [220, 220, 220, 255];
      const v = (x + y) % 2 === 0 ? 100 : 140;
      return [v, v, v, 255];
    });

    const adaptive = createQuickSelectStroke(pixels, {
      radius: 4,
      baseTolerance: 16,
      varianceWeight: 1.5,
    });
    adaptive.dab(6, 8);

    const fixed = createQuickSelectStroke(pixels, {
      radius: 4,
      baseTolerance: 16,
      varianceWeight: 0,
    });
    fixed.dab(6, 8);

    // Adaptive: σ ≈ 20 widens the gate to ≈ 46, which spans the texture
    // and still refuses the 220 region.
    expect(adaptive.statistics().deviation[0]).toBeGreaterThan(15);
    expect(adaptive.statistics().tolerance[0]).toBeGreaterThan(40);
    expect(adaptive.count(), "the whole textured region").toBe(16 * 16);
    const covA = adaptive.coverage();
    for (let y = 0; y < 16; y++) {
      expect(covA[y * 32 + 16], `outlier (16,${y}) refused`).toBe(0);
    }

    // Fixed: the gate stays at 16, which is NARROWER than the texture's
    // own spread, so growth dies at the painted disc.
    expect(fixed.statistics().tolerance[0]).toBeCloseTo(16, 6);
    expect(fixed.count()).toBe(fixed.statistics().painted);
    expect(fixed.count()).toBeLessThan(80);
    expect(
      adaptive.count(),
      "the two rules give materially different answers",
    ).toBeGreaterThan(fixed.count() * 3);

    // And the documented reason for preferring it: the adaptive answer
    // does not depend on WHICH pixel the drag happened to start on. The
    // neighbouring start sits on the opposite checkerboard parity — a
    // seed-pixel threshold would flip its whole notion of "similar".
    const shifted = createQuickSelectStroke(pixels, {
      radius: 4,
      baseTolerance: 16,
      varianceWeight: 1.5,
    });
    shifted.dab(7, 8);
    expect(shifted.count()).toBe(adaptive.count());
  });

  it("the pixel CAP caps — growth stops and latches, and the plane agrees", () => {
    const { pixels } = testImage(200, 200, () => DARK);

    const uncapped = createQuickSelectStroke(pixels, { radius: 4 });
    uncapped.dab(100, 100);
    expect(uncapped.count(), "a flat image floods whole without a cap").toBe(
      200 * 200,
    );
    expect(uncapped.capped()).toBe(false);

    const capped = createQuickSelectStroke(pixels, {
      radius: 4,
      maxPixels: 500,
    });
    capped.dab(100, 100);
    expect(capped.count()).toBe(500);
    expect(capped.capped()).toBe(true);
    // The counter and the plane must not be able to disagree — a cap
    // that only stops the counter would still have written the pixels.
    expect(selectedCount(capped.coverage())).toBe(500);

    // A further dab cannot smuggle more past the cap.
    capped.dab(150, 150);
    expect(capped.count()).toBe(500);
    expect(selectedCount(capped.coverage())).toBe(500);

    expect(QUICK_SELECT_MAX_PIXELS).toBe(4_000_000);
  });

  it("reads only where it grew — O(painted region), not O(image)", () => {
    // 200×200 at the 64 px cache grid is 16 tiles. A capped stroke in the
    // middle must touch ONE of them: the guarantee that a pointer move
    // does not re-flood (or even re-read) the whole image.
    const { pixels, windows } = testImage(200, 200, () => DARK);
    const stroke = createQuickSelectStroke(pixels, {
      radius: 4,
      maxPixels: 500,
    });
    stroke.dab(100, 100);
    expect(stroke.count()).toBe(500);
    expect(windows.length, `touched ${windows.length} of 16 tiles`).toBeLessThan(
      4,
    );

    // And a second dab inside the already-grown region re-reads nothing.
    const before = windows.length;
    stroke.dab(100, 100);
    expect(windows.length).toBe(before);
  });

  it("a dab outside the image is a no-op, and an empty stroke has no bounds", () => {
    const { pixels } = splitImage();
    const stroke = createQuickSelectStroke(pixels, { radius: 2 });
    expect(stroke.dab(-50, -50)).toBe(0);
    expect(stroke.count()).toBe(0);
    expect(stroke.bounds()).toBeNull();
  });
});

describe("quick selection — the coverage plane the commit hands over", () => {
  it("expands to straight RGBA8 with the coverage in every colour channel", () => {
    const cov = Uint8Array.from([0, 255, 0, 255]);
    expect(Array.from(coverageToRgba8(cov))).toEqual([
      0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255,
    ]);
  });

  it("is BINARY 0/255 — the magic wand's own coverage vocabulary", () => {
    const { pixels } = splitImage();
    const stroke = createQuickSelectStroke(pixels, { radius: 4 });
    stroke.dab(6, 8);
    const cov = stroke.coverage();
    expect(cov).toBeInstanceOf(Uint8Array);
    expect(cov.length).toBe(32 * 16);
    for (const v of cov) expect(v === 0 || v === 255).toBe(true);
  });
});
