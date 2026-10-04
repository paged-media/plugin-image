#!/usr/bin/env node
// Record libvips's answer for every kernel the registry names libvips
// as an oracle for (`oracle: vips | both` in registry/kernels.yaml).
//
//   node scripts/vips/record.mjs            # every row
//   node scripts/vips/record.mjs math.add   # just these ids
//
// Maintainer-only (needs the `vips` CLI). CI never runs this: it replays
// the committed fixtures through image-conformance/tests/oracle_vips.rs.
//
// For each row this writes the deterministic float stimuli (formula
// below; the replay regenerates them bit-identically), runs the row's
// vips command chain, casts the result to float, and commits
//   image-conformance/fixtures/vips/<id>.v          vips's output
//   image-conformance/fixtures/vips/<id>.vips.json  provenance
// and regenerates registry/vips-oracle.yaml from ROWS (keeping the
// measured tolerances the replay wrote there).
//
// A row is either a command chain or `diverges: <reason>` — libvips
// implementing a different formula or convention is a finding, not a
// gap, and the reason is the record.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
const OUT = path.join(ROOT, "image-conformance/fixtures/vips");
const YAML = path.join(ROOT, "registry/vips-oracle.yaml");
const N = 64;

// ── stimuli ────────────────────────────────────────────────────────────
// h = (7x + 13y + 29c + 31·seed + 17·((x·y) mod 11)) mod 257; every value
// below is k/256, k/128 or k/4 — exact in f16, so the GPU, the scalar
// reference and libvips all start from the same numbers.
export function hash(seed, x, y, c) {
  return (7 * x + 13 * y + 29 * c + 31 * seed + 17 * ((x * y) % 11)) % 257;
}
export const KINDS = {
  unit: (h) => h / 256,
  signed: (h) => (h - 128) / 128,
  pos: (h) => (64 + (h % 193)) / 256,
  quarter: (h) => (h % 5) / 4,
  binary: (h) => h % 2,
};
function stimulus(kind, seed) {
  const buf = Buffer.alloc(N * N * 4 * 4);
  let o = 0;
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++)
      for (let c = 0; c < 4; c++) {
        const h = hash(seed, x, y, c);
        const v = kind === "opaque" ? (c === 3 ? 1 : KINDS.unit(h)) : KINDS[kind](h);
        buf.writeFloatLE(v, o);
        o += 4;
      }
  return buf;
}

const v = (...xs) => xs.join(" ");
const hueMatrix = (deg) => {
  const t = (deg * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [
    [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
    [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.14, 0.072 - c * 0.072 - s * 0.283],
    [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
  ];
};
const satMatrix = (sat) => {
  const L = [0.3, 0.59, 0.11];
  return [0, 1, 2].map((r) => [0, 1, 2].map((c) => (1 - sat) * L[c] + (r === c ? sat : 0)));
};
/** A 4x4 recomb matrix file: the 3x3 colour block, alpha passed through. */
const mat4 = (m3) =>
  `4 4\n${[...m3.map((r) => [...r, 0]), [0, 0, 0, 1]].map((r) => r.join(" ")).join("\n")}\n`;

// ── the rows ───────────────────────────────────────────────────────────
// in: stimulus kinds for in0[, in1] (seed 1, 2). out: [w, h] of OUR
// output. region: [x, y, w, h] of our output compared (default all).
// steps: vips argv lists; A, B = inputs, T1.. temporaries, O = output.
// files: extra files a step names. scale: final multiplier into our units.
// params: our kernel's parameters, as the replay builds them.
const unaryLinear = (id, a, b, kind = "unit") => ({
  id,
  in: [kind],
  params: {},
  steps: [["linear", "A", "O", a, b]],
});
export const ROWS = [
  { id: "math.linear", in: ["unit"], params: { gain: 1.5, bias: -0.125 }, steps: [["linear", "A", "O", "1.5", "-0.125"]] },
  { ...unaryLinear("math.invert", "-1", "1"), note: "1 - x via linear (vips invert is format-relative)" },
  { id: "math.add", in: ["unit", "unit"], params: {}, steps: [["add", "A", "B", "O"]] },
  { id: "math.sub", in: ["unit", "unit"], params: {}, steps: [["subtract", "A", "B", "O"]] },
  { id: "math.mul", in: ["unit", "unit"], params: {}, steps: [["multiply", "A", "B", "O"]] },
  { id: "math.div", in: ["unit", "pos"], params: {}, steps: [["divide", "A", "B", "O"]] },
  { id: "math.add_const", in: ["unit"], params: { v: 0.375 }, steps: [["linear", "A", "O", "1", "0.375"]] },
  { id: "math.mul_const", in: ["unit"], params: { v: 1.75 }, steps: [["linear", "A", "O", "1.75", "0"]] },
  { id: "math.abs", in: ["signed"], params: {}, steps: [["abs", "A", "O"]] },
  { id: "math.sign", in: ["signed"], params: {}, steps: [["sign", "A", "O"]] },
  { ...unaryLinear("math.neg", "-1", "0", "signed") },
  ...[
    ["rel.eq", "equal"],
    ["rel.ne", "noteq"],
    ["rel.lt", "less"],
    ["rel.le", "lesseq"],
    ["rel.gt", "more"],
    ["rel.ge", "moreeq"],
  ].map(([id, op]) => ({
    id,
    in: ["quarter", "quarter"],
    params: {},
    steps: [["relational", "A", "B", "T1", op]],
    scale: 1 / 255,
    note: "vips answers 255/0 in uchar; scaled to 1/0",
  })),
  ...[
    ["bool.and", "and"],
    ["bool.or", "or"],
    ["bool.xor", "eor"],
  ].map(([id, op]) => ({
    id,
    in: ["binary", "binary"],
    params: {},
    steps: [["boolean", "A", "B", "T1", op]],
    note: "vips casts to int and works bitwise; on 0/1 stimuli that is the engine's truthiness logic",
  })),
  {
    id: "bool.not",
    in: ["binary"],
    params: {},
    steps: [["relational_const", "A", "T1", "equal", "0"]],
    scale: 1 / 255,
    note: "not x == (x == 0) on 0/1 stimuli",
  },
  {
    id: "band.extract",
    in: ["unit"],
    params: { channel: 1 },
    steps: [
      ["extract_band", "A", "T1", "1"],
      ["bandjoin", "T1 T1 T1", "T2"],
      ["bandjoin_const", "T2", "O", "1"],
    ],
  },
  {
    id: "band.set_alpha",
    in: ["unit"],
    params: { alpha: 0.5 },
    steps: [
      ["extract_band", "A", "T1", "0", "--n", "3"],
      ["bandjoin_const", "T1", "O", "0.5"],
    ],
  },
  { id: "math.min", in: ["unit", "unit"], params: {}, steps: [["minpair", "A", "B", "O"]] },
  { id: "math.max", in: ["unit", "unit"], params: {}, steps: [["maxpair", "A", "B", "O"]] },
  { id: "math.clamp", in: ["unit"], params: { lo: 0.25, hi: 0.75 }, steps: [["clamp", "A", "O", "--min", "0.25", "--max", "0.75"]] },
  { id: "math.min_const", in: ["unit"], params: { v: 0.5 }, steps: [["clamp", "A", "O", "--min", "-1000", "--max", "0.5"]] },
  { id: "math.max_const", in: ["unit"], params: { v: 0.5 }, steps: [["clamp", "A", "O", "--min", "0.5", "--max", "1000"]] },
  // ── windowed / module kernels (GPU only in the replay) ──
  {
    id: "conv.box",
    in: ["unit"],
    params: {},
    out: [62, 62],
    files: { "box.mat": "3 3 9 0\n1 1 1\n1 1 1\n1 1 1\n" },
    steps: [
      ["conv", "A", "T1", "box.mat", "--precision", "float"],
      ["extract_area", "T1", "O", "1", "1", "62", "62"],
    ],
  },
  {
    id: "conv.gaussian_h",
    in: ["unit"],
    params: { sigma: 1.5, radius: 5 },
    out: [16, 64],
    steps: [
      ["gaussmat", "G", "1.5", "0.001", "--separable", "--precision", "float"],
      ["conv", "A", "T1", "G", "--precision", "float"],
      ["extract_area", "T1", "O", "24", "0", "16", "64"],
    ],
    note: "vips gaussmat (min_ampl 0.001 gives the same 11 taps, normalised). The kernel's window halo is its MAX radius (24), so a 64-wide window yields 16 columns centred at 24..40",
  },
  {
    id: "conv.gaussian_v",
    in: ["unit"],
    params: { sigma: 1.5, radius: 5 },
    out: [64, 16],
    steps: [
      ["gaussmat", "G", "1.5", "0.001", "--separable", "--precision", "float"],
      ["rot", "G", "GV", "d90"],
      ["conv", "A", "T1", "GV", "--precision", "float"],
      ["extract_area", "T1", "O", "0", "24", "64", "16"],
    ],
    note: "as conv.gaussian_h, vertically",
  },
  {
    id: "conv.unsharp",
    in: ["unit", "unit"],
    params: { amount: 0.8, threshold: 0 },
    steps: [
      ["subtract", "A", "B", "T1"],
      ["linear", "T1", "T2", "0.8", "0"],
      ["add", "A", "T2", "O"],
    ],
    note: "in1 is the blurred image; a + amount * (a - blur)",
  },
  {
    id: "resample.nearest",
    in: ["unit"],
    params: { inv_scale: 0.5 },
    out: [128, 128],
    steps: [["resize", "A", "O", "2", "--kernel", "nearest"]],
  },
  {
    id: "resample.mitchell",
    in: ["unit"],
    params: { inv_scale: 2 },
    out: [32, 32],
    steps: [["reduce", "A", "O", "2", "2", "--kernel", "mitchell"]],
  },
  {
    id: "resample.lanczos3",
    in: ["unit"],
    params: { inv_scale: 2 },
    out: [32, 32],
    steps: [["reduce", "A", "O", "2", "2", "--kernel", "lanczos3"]],
  },
  { id: "geom.flip_h", in: ["unit"], params: {}, steps: [["flip", "A", "O", "horizontal"]] },
  { id: "geom.flip_v", in: ["unit"], params: {}, steps: [["flip", "A", "O", "vertical"]] },
  { id: "geom.rotate90_cw", in: ["unit"], params: {}, steps: [["rot", "A", "O", "d90"]] },
  { id: "geom.rotate90_ccw", in: ["unit"], params: {}, steps: [["rot", "A", "O", "d270"]] },
  {
    id: "geom.crop",
    in: ["unit"],
    params: { off_x: 8, off_y: 4 },
    out: [40, 32],
    steps: [["extract_area", "A", "O", "8", "4", "40", "32"]],
  },
  {
    id: "geom.rotate_bilinear",
    in: ["unit"],
    params: { degrees: 30, centre: 32 },
    region: [12, 12, 40, 40],
    steps: [
      [
        "affine",
        "A",
        "O",
        v(Math.cos(Math.PI / 6), -Math.sin(Math.PI / 6), Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)),
        "--interpolate",
        "bilinear",
        "--oarea",
        "0 0 64 64",
        "--idx",
        "-31.5",
        "--idy",
        "-31.5",
        "--odx",
        "31.5",
        "--ody",
        "31.5",
      ],
    ],
    note: "rotation about the centre; only the interior 40x40 is compared (edge rules differ)",
  },
  ...[
    ["morph.dilate", "8"],
    ["morph.erode", "0"],
    ["rank.median3", "4"],
  ].map(([id, idx]) => ({
    id,
    in: ["unit"],
    params: {},
    out: [62, 62],
    steps: [
      ["rank", "A", "T1", "3", "3", idx],
      ["extract_area", "T1", "O", "1", "1", "62", "62"],
    ],
  })),
  // ── adjust (opaque stimulus: alpha 1, so premultiplied = straight) ──
  {
    id: "adjust.exposure",
    in: ["opaque"],
    params: { ev: 0.5 },
    steps: [["linear", "A", "O", v(...Array(3).fill(2 ** 0.5), 1), "0 0 0 0"]],
  },
  {
    id: "adjust.brightness_contrast",
    in: ["opaque"],
    params: { brightness: 0.1, contrast: 1.3 },
    steps: [["linear", "A", "O", "1.3 1.3 1.3 1", v(...Array(3).fill(0.5 - 0.65 + 0.1), 0)]],
  },
  {
    id: "adjust.levels",
    in: ["opaque"],
    params: { in_black: 0.1, in_white: 0.9, gamma: 1.3, out_black: 0.05, out_white: 0.95 },
    steps: [
      ["linear", "A", "T1", "1.25 1.25 1.25 1", "-0.125 -0.125 -0.125 0"],
      ["clamp", "T1", "T2", "--min", "0", "--max", "1"],
      ["math2_const", "T2", "T3", "pow", v(...Array(3).fill(1 / 1.3), 1)],
      ["linear", "T3", "O", "0.9 0.9 0.9 1", "0.05 0.05 0.05 0"],
    ],
  },
  {
    id: "adjust.levels_rgb",
    in: ["opaque"],
    params: { r: [0.05, 0.95, 1.2], g: [0.1, 0.9, 0.8], b: [0, 1, 1.5] },
    steps: [
      ["linear", "A", "T1", v(1 / 0.9, 1 / 0.8, 1, 1), v(-0.05 / 0.9, -0.1 / 0.8, 0, 0)],
      ["clamp", "T1", "T2", "--min", "0", "--max", "1"],
      ["math2_const", "T2", "O", "pow", v(1 / 1.2, 1 / 0.8, 1 / 1.5, 1)],
    ],
  },
  {
    id: "adjust.saturation",
    in: ["opaque"],
    params: { sat: 0.6 },
    files: { "sat.mat": mat4(satMatrix(0.6)) },
    steps: [["recomb", "A", "O", "sat.mat"]],
  },
  {
    id: "adjust.hue_rotate",
    in: ["opaque"],
    params: { degrees: 30 },
    files: { "hue.mat": mat4(hueMatrix(30)) },
    steps: [["recomb", "A", "O", "hue.mat"]],
    note: "the W3C feColorMatrix hueRotate matrix the kernel documents",
  },
  {
    id: "adjust.invert_rgb",
    in: ["opaque"],
    params: {},
    steps: [["linear", "A", "O", "-1 -1 -1 1", "1 1 1 0"]],
  },
  {
    id: "adjust.white_balance",
    in: ["opaque"],
    params: { temp: 0.3, tint: -0.2 },
    steps: [["linear", "A", "O", "1.3 0.8 0.7 1", "0 0 0 0"]],
  },
];

// ── running vips ───────────────────────────────────────────────────────
/** vips options that take no value. */
const FLAGS = new Set(["--separable"]);
function vips(args, cwd) {
  return execFileSync("vips", args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
}

function record(row, version) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "paged-vips-"));
  const names = ["A", "B"];
  row.in.forEach((kind, i) => {
    const raw = path.join(dir, `${names[i]}.raw`);
    fs.writeFileSync(raw, stimulus(kind, i + 1));
    vips(["rawload", "--", raw, `${names[i]}8.v`, String(N), String(N), "16"], dir);
    vips(["copy", `${names[i]}8.v`, `${names[i]}.v`, "--format", "float", "--bands", "4"], dir);
  });
  for (const [f, text] of Object.entries(row.files || {})) fs.writeFileSync(path.join(dir, f), text);
  const sub = (a) =>
    a
      .split(" ")
      .map((t) => (/^(A|B|O|G|GV|T\d)$/.test(t) ? `${t}.v` : t))
      .join(" ");
  const commands = [];
  for (const step of row.steps) {
    const [op, ...rest] = step;
    // Options as --name=value BEFORE a "--", positionals after it: a
    // negative constant ("-0.125", "-1 -1 -1 1") would otherwise be read
    // as an option by the vips CLI.
    const opts = [];
    const pos = [];
    for (let i = 0; i < rest.length; i++) {
      const t = rest[i];
      if (t.startsWith("--")) {
        if (FLAGS.has(t)) opts.push(t);
        else opts.push(`${t}=${rest[++i]}`);
      } else pos.push(sub(t));
    }
    const argv = [op, ...opts, "--", ...pos];
    commands.push(`vips ${argv.map((a) => (a.includes(" ") ? `"${a}"` : a)).join(" ")}`);
    vips(argv, dir);
  }
  // The last step's output, whatever it was called, cast to float and
  // scaled into the engine's units.
  const tail = row.steps[row.steps.length - 1];
  const last = sub(tail[lastOutIndex(tail)]);
  vips(["cast", last, "F.v", "float"], dir);
  const final = row.scale ? (vips(["linear", "F.v", "S.v", String(row.scale), "0"], dir), "S.v") : "F.v";
  if (row.scale) commands.push(`vips linear <out> <out> ${row.scale} 0`);
  fs.mkdirSync(OUT, { recursive: true });
  fs.copyFileSync(path.join(dir, final), path.join(OUT, `${row.id}.v`));
  const prov = {
    id: row.id,
    oracle: "libvips",
    vips_version: version,
    recorded_at: new Date().toISOString(),
    runner: "scripts/vips/record.mjs",
    stimulus: {
      size: [N, N],
      bands: 4,
      formula: "h = (7x + 13y + 29c + 31*seed + 17*((x*y) mod 11)) mod 257; inputs use seed 1, 2",
      kinds: row.in,
    },
    params: row.params,
    out: row.out || [N, N],
    region: row.region || null,
    commands,
    note: row.note || null,
  };
  fs.writeFileSync(path.join(OUT, `${row.id}.vips.json`), JSON.stringify(prov, null, 2) + "\n");
  fs.rmSync(dir, { recursive: true, force: true });
}

/** Index of the output operand in a step: the token after the inputs. */
function lastOutIndex(step) {
  const outs = step.map((t, i) => [t, i]).filter(([t, i]) => i > 0 && /^(O|T\d)$/.test(t));
  return outs[outs.length - 1][1];
}

// ── the registry file ──────────────────────────────────────────────────
function measured() {
  // Keep the replay's measured tolerances across regenerations.
  const keep = {};
  if (!fs.existsSync(YAML)) return keep;
  let id = null;
  for (const line of fs.readFileSync(YAML, "utf8").split("\n")) {
    const m = line.match(/^- id: (\S+)/);
    if (m) id = m[1];
    const t = line.match(/^ {2}(oracle_tolerance|gpu_tolerance_measured): (.+)$/);
    if (id && t) (keep[id] ||= {})[t[1]] = t[2];
  }
  return keep;
}

export const DIVERGES = {
};

function writeYaml() {
  const keep = measured();
  let s = `# libvips oracle table — one row per registry/kernels.yaml kernel whose
# \`oracle\` is \`vips\` or \`both\`. Each row is a vips command chain
# (recorded by scripts/vips/record.mjs into image-conformance/fixtures/vips/)
# or \`diverges: <reason>\` where libvips implements a different formula.
# Replayed by image-conformance/tests/oracle_vips.rs; the tolerances are
# MEASURED there (max |ours - vips| over the compared region, rounded up)
# and written back with PAGED_VIPS_TOLERANCE=write. GENERATED — edit
# scripts/vips/record.mjs, not this file.
`;
  for (const row of ROWS) {
    s += `\n- id: ${row.id}\n`;
    if (DIVERGES[row.id]) {
      s += `  diverges: ${JSON.stringify(DIVERGES[row.id])}\n`;
      continue;
    }
    s += `  fixture: image-conformance/fixtures/vips/${row.id}.v\n`;
    s += `  vips: ${JSON.stringify(row.steps.map((st) => `vips ${st.join(" ")}`).join(" && "))}\n`;
    const k = keep[row.id] || {};
    s += `  oracle_tolerance: ${k.oracle_tolerance ?? "unmeasured"}\n`;
    s += `  gpu_tolerance_measured: ${k.gpu_tolerance_measured ?? "unmeasured"}\n`;
  }
  fs.writeFileSync(YAML, s);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const want = process.argv.slice(2).filter((a) => !a.startsWith("--"));
  const yamlOnly = process.argv.includes("--yaml-only");
  if (!yamlOnly) {
    const version = execFileSync("vips", ["--version"], { encoding: "utf8" }).trim();
    for (const row of ROWS) {
      if (want.length && !want.includes(row.id)) continue;
      if (DIVERGES[row.id]) continue;
      record(row, version);
      console.log(`recorded ${row.id}`);
    }
  }
  writeYaml();
  console.log(`wrote ${path.relative(ROOT, YAML)}`);
}
