// Foreground / background colour — Photoshop's two-colour model.
//
// The foreground is what paint lays down (brush, pencil, fills that ask
// for one colour); the background is the second gradient stop. Before
// this the panel offered eight preset wells and nothing else: a user
// could not paint in the colour of the image they were retouching.
//
// Pure functions over straight RGBA in [0,1]; the session owns the
// state and wires it to the brush.

import type { Rgba01 } from "./engine";

export interface ColorPair {
  fg: Rgba01;
  bg: Rgba01;
}

/** Photoshop's D: black foreground, white background. */
export const DEFAULT_COLORS: ColorPair = {
  fg: [0, 0, 0, 1],
  bg: [1, 1, 1, 1],
};

export function freshColors(): ColorPair {
  return { fg: [...DEFAULT_COLORS.fg], bg: [...DEFAULT_COLORS.bg] };
}

/** Photoshop's X. */
export function swapped(c: ColorPair): ColorPair {
  return { fg: [...c.bg], bg: [...c.fg] };
}

/** Sampling sizes the eyedropper offers: point, 3×3, 5×5 average. */
export type SampleSize = 1 | 3 | 5;

/**
 * The average of a straight-RGBA8 window (as `engine.tile` returns it),
 * as straight RGBA in [0,1]. Colour is averaged PREMULTIPLIED, so a
 * transparent neighbour does not drag the sample toward black; alpha is
 * the plain mean. An empty window yields null.
 */
export function averageRgba8(px: Uint8Array): Rgba01 | null {
  const n = Math.floor(px.length / 4);
  if (n === 0) return null;
  let r = 0;
  let g = 0;
  let b = 0;
  let a = 0;
  for (let i = 0; i < n * 4; i += 4) {
    const al = px[i + 3] / 255;
    r += (px[i] / 255) * al;
    g += (px[i + 1] / 255) * al;
    b += (px[i + 2] / 255) * al;
    a += al;
  }
  if (a === 0) return [0, 0, 0, 0];
  return [r / a, g / a, b / a, a / n];
}

/** `#rrggbb` for a colour (alpha dropped — the hex field is RGB). */
export function toHex(c: Rgba01): string {
  const h = (v: number) =>
    Math.round(Math.min(1, Math.max(0, v)) * 255)
      .toString(16)
      .padStart(2, "0");
  return `#${h(c[0])}${h(c[1])}${h(c[2])}`;
}

/** Parse `#rgb` / `#rrggbb` (opaque), or null. */
export function fromHex(s: string): Rgba01 | null {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(s.trim());
  if (!m) return null;
  const hex = m[1].length === 3 ? m[1].replace(/./g, (ch) => ch + ch) : m[1];
  const v = (i: number) => parseInt(hex.slice(i, i + 2), 16) / 255;
  return [v(0), v(2), v(4), 1];
}
