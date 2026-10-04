// Color Lookup tables in the .cube format (the format colour-grading
// tools exchange): a header with LUT_3D_SIZE N (and optionally
// DOMAIN_MIN / DOMAIN_MAX), then N³ lines of "r g b" with RED changing
// fastest.
//
// The engine's lookup kernel carries a 9×9×9 cube (its parameter block
// cannot hold more without a change to the frozen kernel ABI), so a file
// at 17, 33 or 65 is resampled to 9 trilinearly, which is exact at the 9
// lattice points the two grids share when N − 1 is a multiple of 8, and
// a smooth approximation otherwise.

export const KERNEL_EDGE = 9;

export interface Cube {
  size: number;
  /** size³ rgb triples, red fastest, domain-normalised to 0–1 input. */
  data: Float32Array;
  title: string | null;
  /** The INPUT range the lattice spans (DOMAIN_MIN / DOMAIN_MAX). */
  domain: { min: [number, number, number]; max: [number, number, number] };
}

/** Parse a .cube file. Throws with the line number on malformed input. */
export function parseCube(text: string): Cube {
  let size = 0;
  let title: string | null = null;
  let min: [number, number, number] = [0, 0, 0];
  let max: [number, number, number] = [1, 1, 1];
  const values: number[] = [];
  const lines = text.split(/\r?\n/);
  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (!line || line.startsWith("#")) return;
    const [key, ...rest] = line.split(/\s+/);
    switch (key) {
      case "TITLE":
        title = line.slice(5).trim().replace(/^"|"$/g, "");
        return;
      case "LUT_3D_SIZE":
        size = Number(rest[0]);
        return;
      case "LUT_1D_SIZE":
        throw new Error(`.cube line ${i + 1}: 1D lookup tables are not supported (a 3D table is)`);
      case "DOMAIN_MIN":
        min = [Number(rest[0]), Number(rest[1]), Number(rest[2])];
        return;
      case "DOMAIN_MAX":
        max = [Number(rest[0]), Number(rest[1]), Number(rest[2])];
        return;
      default: {
        const v = line.split(/\s+/).map(Number);
        if (v.length !== 3 || v.some((x) => !Number.isFinite(x))) {
          throw new Error(`.cube line ${i + 1}: expected "r g b", got "${line}"`);
        }
        values.push(v[0], v[1], v[2]);
      }
    }
  });
  if (!Number.isInteger(size) || size < 2 || size > 256) {
    throw new Error(".cube: LUT_3D_SIZE missing or out of range (2–256)");
  }
  if (values.length !== size * size * size * 3) {
    throw new Error(`.cube: ${values.length / 3} entries, LUT_3D_SIZE ${size} needs ${size ** 3}`);
  }
  if ([0, 1, 2].some((k) => !(max[k] > min[k]))) {
    throw new Error(".cube: DOMAIN_MAX must exceed DOMAIN_MIN on every channel");
  }
  return { size, data: Float32Array.from(values), title, domain: { min, max } };
}

/** Trilinear sample of a cube at normalised (r, g, b) ∈ [0, 1]. */
function sample(c: Cube, r: number, g: number, b: number): [number, number, number] {
  const n = c.size - 1;
  const at = (v: number) => {
    const x = Math.min(Math.max(v, 0), 1) * n;
    const i0 = Math.min(Math.floor(x), n - 1);
    return [i0, x - i0] as const;
  };
  const [r0, fr] = at(r);
  const [g0, fg] = at(g);
  const [b0, fb] = at(b);
  const idx = (ri: number, gi: number, bi: number) => ((bi * c.size + gi) * c.size + ri) * 3;
  const out: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    let acc = 0;
    for (let db = 0; db < 2; db++)
      for (let dg = 0; dg < 2; dg++)
        for (let dr = 0; dr < 2; dr++) {
          const w = (dr ? fr : 1 - fr) * (dg ? fg : 1 - fg) * (db ? fb : 1 - fb);
          if (w) acc += w * c.data[idx(r0 + dr, g0 + dg, b0 + db) + k];
        }
    out[k] = acc;
  }
  return out;
}

/** The kernel's 9³ cube (729 rgb triples, red fastest) for a parsed file. */
export function toKernelCube(c: Cube): Float32Array {
  const e = KERNEL_EDGE;
  const out = new Float32Array(e * e * e * 3);
  // An input value v sits at (v − min) / (max − min) along the file's
  // lattice; outside the domain it clamps to the edge entry.
  const { min, max } = c.domain;
  const toDomain = (v: number, k: number) => (v - min[k]) / (max[k] - min[k]);
  let o = 0;
  for (let b = 0; b < e; b++)
    for (let g = 0; g < e; g++)
      for (let r = 0; r < e; r++) {
        const v = sample(c, toDomain(r / (e - 1), 0), toDomain(g / (e - 1), 1), toDomain(b / (e - 1), 2));
        out[o++] = v[0];
        out[o++] = v[1];
        out[o++] = v[2];
      }
  return out;
}
