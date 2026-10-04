#!/usr/bin/env node
// The Photoshop lane's STIMULI: deterministic 128x128 RGB images written
// as 16-bit PNGs with no colour profile (untagged, so Photoshop opens
// them in the working space without converting a single number).
//
//   node scripts/photoshop/lib/stimuli.mjs <out-dir>
//
// The same formulas are NOT re-implemented in Rust: the replay reads
// these committed PNGs back, so the stimulus is whatever bytes were
// handed to Photoshop.
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

export const SIZE = 128;

/** Encode a 16-bit RGB PNG (colour type 2, depth 16), no ancillary chunks. */
export function png16(width, height, sample) {
  const raw = Buffer.alloc(height * (1 + width * 6));
  let o = 0;
  for (let y = 0; y < height; y++) {
    raw[o++] = 0; // filter: none
    for (let x = 0; x < width; x++) {
      const px = sample(x, y);
      for (let c = 0; c < 3; c++) {
        const v = Math.max(0, Math.min(65535, Math.round(px[c] * 65535)));
        raw.writeUInt16BE(v, o);
        o += 2;
      }
    }
  }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, "latin1"), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(zlib.crc32(td) >>> 0);
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 16; // bit depth
  ihdr[9] = 2; // RGB
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlib.deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function hsv(h, s, v) {
  const i = Math.floor(h / 60) % 6;
  const f = h / 60 - Math.floor(h / 60);
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  return [
    [v, t, p],
    [q, v, p],
    [p, v, t],
    [p, q, v],
    [t, p, v],
    [v, p, q],
  ][i];
}

const N = SIZE - 1;

/** Every stimulus: name -> sample(x, y) -> [r, g, b] in [0, 1]. */
export const STIMULI = {
  // Adjustments and filters: hue across x; the top half walks value at
  // full saturation, the bottom half walks saturation at full value, and
  // the last 8 rows are a neutral grey ramp.
  "colour-sweep": (x, y) => {
    if (y >= SIZE - 8) return [x / N, x / N, x / N];
    const h = (x / SIZE) * 360;
    return y < 60 ? hsv(h, 1, (y + 4) / 64) : hsv(h, (SIZE - 8 - y) / 60, 1);
  },
  // Blend modes: two layers whose channels sweep independently, so every
  // (backdrop, source) pair along the 0..1 square is visited.
  "blend-bottom": (x, y) => [x / N, y / N, ((x + y) % SIZE) / N],
  "blend-top": (x, y) => [(N - y) / N, x / N, Math.abs(x - y) / N],
};

export function writeStimuli(dir) {
  fs.mkdirSync(dir, { recursive: true });
  const out = {};
  for (const [name, f] of Object.entries(STIMULI)) {
    const p = path.join(dir, `${name}.png`);
    fs.writeFileSync(p, png16(SIZE, SIZE, f));
    out[name] = p;
  }
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const dir = process.argv[2];
  if (!dir) {
    console.error("usage: stimuli.mjs <out-dir>");
    process.exit(2);
  }
  for (const [n, p] of Object.entries(writeStimuli(dir))) console.log(`${n} -> ${p}`);
}
