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

// QUICK SELECTION — paint-to-grow region growing, host-agnostic and
// engine-agnostic (it reads pixels through an injected window reader and
// hands back a u8 coverage plane; the selection machine owns the engine
// doors). This module is pure so the growth rule can be tested against
// constructed images without booting wasm.
//
// WHAT IT IS, AND WHY IT IS NOT A WAND
// ────────────────────────────────────
// The magic wand is a ONE-SHOT flood: it reads the seed pixel, freezes a
// Chebyshev threshold around THAT pixel's colour, and floods. Its whole
// character — and the complaint that it "feels unpredictable" — follows
// from that: the answer depends on exactly which pixel you happened to
// click. Click one pixel into a JPEG ringing artefact and you get a
// different selection than the pixel next door.
//
// Quick selection replaces the seed-pixel threshold with the STATISTICS
// OF WHAT THE USER HAS PAINTED. Every brush dab folds its disc of pixels
// into a running mean μ and standard deviation σ (per RGBA channel), and
// growth accepts a candidate when it sits inside
//
//     |p_c − μ_c| ≤ min(maxTolerance, baseTolerance + k·σ_c)   for all c
//
// So the tolerance is a property of the painted evidence, not of one
// arbitrary pixel: painting a swath across a noisy-but-uniform region
// raises σ and the region is taken whole, while a dab inside a genuinely
// flat area keeps σ ≈ 0 and the growth stays tight. Starting the same
// drag one pixel to the left changes almost nothing, which is precisely
// what the wand cannot promise.
//
// WHY THE STATISTICS COME FROM THE PAINTED PIXELS, NOT THE GROWN ONES
// ───────────────────────────────────────────────────────────────────
// Feeding accepted-by-growth pixels back into μ/σ is the classic
// region-growing drift: a pixel accepted at the edge of tolerance widens
// σ, which admits the next one, which widens σ again — and a drag over a
// smooth gradient walks off across the entire image. Only pixels the
// brush actually covered are evidence of intent, so only they move the
// distribution. Grown pixels are judged by it and never move it. That is
// the bound that makes "paint a bit more to take a bit more" behave.
//
// COST
// ────
// Per dab the work is O(dab area + selection perimeter + newly accepted)
// — never O(image). Three things buy that: coverage/flag planes carried
// ACROSS dabs (interior pixels are never revisited), a persistent
// REJECT SHELL (the pixels tested and refused last time — re-queued on
// the next dab because μ/σ moved, so widening the statistics resumes
// growth from the boundary instead of re-flooding), and a lazily filled
// tile cache so pixels are only ever read where the growth actually
// reached.

/** A windowed reader over the source pixels — the shape of the engine's
 *  C-6 tile door (`ImageEngine.tile`): straight RGBA8, tightly packed,
 *  the window CLIPPED to the image extent, empty when fully outside. */
export interface QuickSelectPixels {
  width: number;
  height: number;
  readWindow(x: number, y: number, w: number, h: number): Uint8Array;
}

/** Brush radius in IMAGE px for one dab (the painted disc). */
export const QUICK_SELECT_RADIUS_DEFAULT = 8;

/** The floor of the growth tolerance, per channel of 255. It exists so a
 *  dab on a perfectly flat region (σ = 0) still tolerates codec noise
 *  instead of selecting literally-equal pixels only. */
export const QUICK_SELECT_BASE_TOLERANCE = 16;

/** How many standard deviations of the painted evidence widen the
 *  tolerance (the `k` above). 1.5 takes the bulk of a normal-ish
 *  distribution while leaving genuine outliers outside. */
export const QUICK_SELECT_VARIANCE_WEIGHT = 1.5;

/** The tolerance CEILING, per channel of 255. Even a deliberately wild
 *  drag across a rainbow cannot open the gate past this; without it a
 *  large enough σ degenerates into "select everything". */
export const QUICK_SELECT_MAX_TOLERANCE = 72;

/** PERFORMANCE GUARD — the hard cap on how many pixels one quick-select
 *  stroke may hold (4 MP ≈ a 2000×2000 region). It bounds three costs at
 *  once: the BFS frontier work, the tile cache the growth pulls in, and
 *  the single full-plane coverage upload the commit performs. Reaching
 *  it stops growth (and further seeding) and latches `capped()`, so the
 *  tool reports a truthful partial rather than freezing the canvas. A
 *  stroke that has swallowed 4 MP has stopped being a *quick* selection
 *  — that is a colour-range job. */
export const QUICK_SELECT_MAX_PIXELS = 4_000_000;

/** Tile edge (px) for the lazy pixel cache. 64² × 4 = 16 KiB per tile:
 *  small enough that growth along a boundary does not drag whole
 *  megabytes in, large enough that the door is not called per pixel. */
export const QUICK_SELECT_TILE = 64;

export interface QuickSelectOptions {
  radius?: number;
  baseTolerance?: number;
  varianceWeight?: number;
  maxTolerance?: number;
  maxPixels?: number;
  tileSize?: number;
}

/** The painted evidence and the gate it currently opens — exposed so a
 *  panel (and the tests) can state the rule rather than guess it. */
export interface QuickSelectStatistics {
  /** How many pixels the brush has actually covered. */
  painted: number;
  /** Per-channel mean of the painted pixels (RGBA). */
  mean: [number, number, number, number];
  /** Per-channel standard deviation of the painted pixels (RGBA). */
  deviation: [number, number, number, number];
  /** The per-channel acceptance half-width actually in force (RGBA). */
  tolerance: [number, number, number, number];
}

export interface QuickSelectBounds {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface QuickSelectStroke {
  /** Paint one dab centred at `(x, y)` (image px) and grow from it.
   *  Returns how many pixels this dab ADDED (0 is a legitimate answer:
   *  a dab inside the already-selected region adds nothing). */
  dab(x: number, y: number, radius?: number): number;
  /** The accumulated coverage plane: `width·height` bytes, 0 or 255 —
   *  byte-for-byte the representation the magic wand produces, so the
   *  two are indistinguishable to everything downstream. */
  coverage(): Uint8Array;
  /** How many pixels are selected. */
  count(): number;
  /** The tight bounding box of the coverage, or null when empty. */
  bounds(): QuickSelectBounds | null;
  /** True once the pixel cap stopped the growth. */
  capped(): boolean;
  /** The running statistics (see [`QuickSelectStatistics`]). */
  statistics(): QuickSelectStatistics;
}

/** Flag plane bits (one byte per pixel, alongside the coverage plane). */
const PAINTED = 1;
const REJECTED = 2;

/**
 * Open a quick-selection stroke over `px`. Nothing is read until the
 * first `dab`.
 */
export function createQuickSelectStroke(
  px: QuickSelectPixels,
  options: QuickSelectOptions = {},
): QuickSelectStroke {
  const width = Math.max(0, Math.floor(px.width));
  const height = Math.max(0, Math.floor(px.height));
  const npx = width * height;

  const defaultRadius = options.radius ?? QUICK_SELECT_RADIUS_DEFAULT;
  const baseTolerance = options.baseTolerance ?? QUICK_SELECT_BASE_TOLERANCE;
  const varianceWeight =
    options.varianceWeight ?? QUICK_SELECT_VARIANCE_WEIGHT;
  const maxTolerance = options.maxTolerance ?? QUICK_SELECT_MAX_TOLERANCE;
  const maxPixels = Math.max(0, options.maxPixels ?? QUICK_SELECT_MAX_PIXELS);
  const tileSize = Math.max(1, options.tileSize ?? QUICK_SELECT_TILE);

  const cov = new Uint8Array(npx);
  const flags = new Uint8Array(npx);

  let count = 0;
  let capped = false;
  let minX = 0;
  let minY = 0;
  let maxX = -1;
  let maxY = -1;

  // Painted-pixel accumulators (RGBA): n, Σv, Σv².
  let n = 0;
  const sum = [0, 0, 0, 0];
  const sumSq = [0, 0, 0, 0];

  /** Tested-and-refused pixels — the shell the next dab re-queues. */
  let rejects: number[] = [];

  // ── lazy tile cache ────────────────────────────────────────────────
  const tilesX = Math.ceil(width / tileSize) || 1;
  const cache = new Map<number, Uint8Array>();
  const sample = new Uint8Array(4);

  const readPixel = (x: number, y: number): boolean => {
    const tx = Math.floor(x / tileSize);
    const ty = Math.floor(y / tileSize);
    const key = ty * tilesX + tx;
    let tile = cache.get(key);
    if (tile === undefined) {
      tile = px.readWindow(tx * tileSize, ty * tileSize, tileSize, tileSize);
      cache.set(key, tile);
    }
    const tw = Math.min(tileSize, width - tx * tileSize);
    const off = ((y - ty * tileSize) * tw + (x - tx * tileSize)) * 4;
    if (off < 0 || off + 4 > tile.length) return false;
    sample[0] = tile[off];
    sample[1] = tile[off + 1];
    sample[2] = tile[off + 2];
    sample[3] = tile[off + 3];
    return true;
  };

  // ── running statistics ─────────────────────────────────────────────
  const mean: [number, number, number, number] = [0, 0, 0, 0];
  const deviation: [number, number, number, number] = [0, 0, 0, 0];
  const tolerance: [number, number, number, number] = [
    baseTolerance,
    baseTolerance,
    baseTolerance,
    baseTolerance,
  ];

  const recomputeGate = () => {
    if (n === 0) return;
    for (let c = 0; c < 4; c++) {
      const m = sum[c] / n;
      // Var = E[v²] − E[v]²; clamped because float cancellation can push
      // a genuinely-zero variance a hair below zero.
      const variance = Math.max(0, sumSq[c] / n - m * m);
      const sd = Math.sqrt(variance);
      mean[c] = m;
      deviation[c] = sd;
      tolerance[c] = Math.min(maxTolerance, baseTolerance + varianceWeight * sd);
    }
  };

  const accepts = (): boolean => {
    for (let c = 0; c < 4; c++) {
      if (Math.abs(sample[c] - mean[c]) > tolerance[c]) return false;
    }
    return true;
  };

  // ── coverage bookkeeping ───────────────────────────────────────────
  const select = (idx: number, x: number, y: number) => {
    cov[idx] = 255;
    flags[idx] &= ~REJECTED;
    count++;
    if (maxX < minX) {
      minX = maxX = x;
      minY = maxY = y;
    } else {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  };

  return {
    dab(cx, cy, radius = defaultRadius) {
      if (npx === 0) return 0;
      const before = count;
      const r = Math.max(0, radius);
      const x0 = Math.max(0, Math.floor(cx - r));
      const x1 = Math.min(width - 1, Math.ceil(cx + r));
      const y0 = Math.max(0, Math.floor(cy - r));
      const y1 = Math.min(height - 1, Math.ceil(cy + r));
      const r2 = r * r;

      // 1. The painted disc. Every covered pixel is evidence — it enters
      //    the statistics once (PAINTED latches) — and is selected
      //    unconditionally: the user said so with the brush.
      const frontier: number[] = [];
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dx = x - cx;
          const dy = y - cy;
          if (dx * dx + dy * dy > r2) continue;
          const idx = y * width + x;
          if ((flags[idx] & PAINTED) === 0) {
            if (!readPixel(x, y)) continue;
            flags[idx] |= PAINTED;
            n++;
            for (let c = 0; c < 4; c++) {
              sum[c] += sample[c];
              sumSq[c] += sample[c] * sample[c];
            }
          }
          if (cov[idx] === 0) {
            if (count >= maxPixels) {
              capped = true;
              continue;
            }
            select(idx, x, y);
          }
          frontier.push(idx);
        }
      }
      if (n === 0) return 0;
      recomputeGate();

      // 2. The reject shell goes back in the queue: μ/σ have moved, so
      //    pixels refused under the old gate deserve a fresh test. This
      //    (not a re-flood) is what makes the stroke ACCUMULATE — and it
      //    is O(perimeter), not O(image).
      const queue: number[] = frontier;
      for (const idx of rejects) {
        flags[idx] &= ~REJECTED;
        queue.push(idx);
      }
      rejects = [];

      // 3. Grow. 4-connected, Chebyshev-per-channel over RGBA — the same
      //    connectivity and metric the magic-wand door uses, so "similar"
      //    means one thing across both tools.
      for (let head = 0; head < queue.length; head++) {
        if (capped) break;
        const idx = queue[head];
        const x = idx % width;
        const y = (idx - x) / width;
        for (let d = 0; d < 4; d++) {
          const nx = d === 0 ? x - 1 : d === 1 ? x + 1 : x;
          const ny = d === 2 ? y - 1 : d === 3 ? y + 1 : y;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const ni = ny * width + nx;
          if (cov[ni] !== 0 || (flags[ni] & REJECTED) !== 0) continue;
          if (count >= maxPixels) {
            capped = true;
            break;
          }
          if (!readPixel(nx, ny)) continue;
          if (accepts()) {
            select(ni, nx, ny);
            queue.push(ni);
          } else {
            flags[ni] |= REJECTED;
            rejects.push(ni);
          }
        }
      }
      return count - before;
    },

    coverage: () => cov,
    count: () => count,
    capped: () => capped,
    bounds: () =>
      maxX < minX
        ? null
        : { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 },
    statistics: () => ({
      painted: n,
      mean: [...mean] as [number, number, number, number],
      deviation: [...deviation] as [number, number, number, number],
      tolerance: [...tolerance] as [number, number, number, number],
    }),
  };
}

/**
 * Expand a `width·height` coverage plane into straight RGBA8 with the
 * coverage in every colour channel and an opaque alpha.
 *
 * This is the shape the commit needs: the engine takes an arbitrary
 * coverage plane only through `selection_from_channel`, which reads a
 * registered image's channel bytes AS coverage (a copy, not a
 * threshold). Going through that door — rather than emitting a pile of
 * per-run rectangles — is what keeps quick selection indistinguishable
 * downstream: the coverage lands via the same `apply_shape` fold as the
 * wand's, under any of the four combine modes including `intersect`,
 * which a sequence of per-run shapes could not express.
 */
export function coverageToRgba8(coverage: Uint8Array): Uint8Array {
  const rgba = new Uint8Array(coverage.length * 4);
  for (let i = 0; i < coverage.length; i++) {
    const v = coverage[i];
    const o = i * 4;
    rgba[o] = v;
    rgba[o + 1] = v;
    rgba[o + 2] = v;
    rgba[o + 3] = 255;
  }
  return rgba;
}
