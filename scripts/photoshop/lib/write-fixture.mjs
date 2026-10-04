#!/usr/bin/env node
// Judge Photoshop's raw reply and, only if it is a recording, copy the
// stimuli and outputs into the fixtures dir and write the provenance
// JSON. Split out of run-probe.sh so a kept staging dir can be judged
// again without driving the app:
//
//   node scripts/photoshop/lib/write-fixture.mjs \
//     <raw.json> <stage-dir> <fixtures-dir> <probe> <script-path> <script-sha256> <macos>
//
// "Judge by the artifact": osascript exiting 0 says only that Photoshop
// returned a string. This says whether the string, and the files it
// names, are an answer.
import fs from "node:fs";
import path from "node:path";
import { SIZE } from "./stimuli.mjs";

const [raw, stage, outDir, name, script, sha, os] = process.argv.slice(2);
const fail = (why) => {
  console.error(`run-probe: NOT A RECORDING -- ${why}`);
  process.exit(1);
};

/** Width, height, bit depth and colour type of a PNG, from its IHDR. */
export function pngHeader(buf) {
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (!sig.every((b, i) => buf[i] === b)) return null;
  return {
    width: buf.readUInt32BE(16),
    height: buf.readUInt32BE(20),
    depth: buf[24],
    colorType: buf[25],
  };
}

let reply;
try {
  reply = JSON.parse(fs.readFileSync(raw, "utf8"));
} catch (e) {
  fail(`Photoshop's reply is not JSON (${e.message})`);
}
if (reply.error) fail(`the probe failed before its cases: ${reply.error}`);
if (reply.close_error) fail(`a probe document did not close: ${reply.close_error}`);
if (reply.probe !== name) fail(`the reply is for "${reply.probe}", not "${name}"`);
if (!Array.isArray(reply.cases) || reply.cases.length === 0) fail("no cases");
if (reply.documents_after !== reply.documents_before) {
  fail(`${reply.documents_before} document(s) before, ${reply.documents_after} after`);
}
const ids = reply.cases.map((c) => c.id);
if (new Set(ids).size !== ids.length) fail("duplicate case ids");
const broken = reply.cases.filter((c) => c.error || c.close_error);
if (broken.length > 0) {
  fail(
    `${broken.length} of ${reply.cases.length} cases raised:\n` +
      broken.map((c) => `    ${c.id}: ${c.error || c.close_error}`).join("\n"),
  );
}
for (const c of reply.cases) {
  if (!c.output) continue; // a case may record values instead of an image
  const p = path.join(stage, c.output);
  if (!fs.existsSync(p)) fail(`${c.id}: Photoshop reported ${c.output} but wrote nothing`);
  const h = pngHeader(fs.readFileSync(p));
  if (!h) fail(`${c.id}: ${c.output} is not a PNG`);
  if (h.width !== SIZE || h.height !== SIZE) fail(`${c.id}: ${h.width}x${h.height}, not ${SIZE}x${SIZE}`);
  if (h.depth !== 16) fail(`${c.id}: ${h.depth}-bit output; the lane records 16-bit`);
}

fs.mkdirSync(path.join(outDir, name), { recursive: true });
fs.mkdirSync(path.join(outDir, "stimuli"), { recursive: true });
for (const f of fs.readdirSync(path.join(stage, "stimuli"))) {
  fs.copyFileSync(path.join(stage, "stimuli", f), path.join(outDir, "stimuli", f));
}
for (const c of reply.cases) {
  if (c.output) fs.copyFileSync(path.join(stage, c.output), path.join(outDir, c.output));
  for (const extra of c.files || []) {
    fs.copyFileSync(path.join(stage, extra), path.join(outDir, extra));
  }
}
const fixture = {
  fixture: name,
  produced_by: {
    app: reply.app.name,
    version: reply.app.version,
    build: reply.app.build,
    locale: reply.app.locale,
    color_settings: reply.color_settings,
    script,
    script_sha256: sha,
    runner: "scripts/photoshop/run-probe.sh",
    recorded_at: new Date().toISOString(),
    host_os: `macOS ${os}`,
  },
  stimulus_size: SIZE,
  cases: reply.cases.map(({ id, params, output, profile, bits, files, values }) => ({
    id,
    params,
    output,
    profile,
    bits,
    ...(files ? { files } : {}),
    ...(values ? { values } : {}),
  })),
};
fs.writeFileSync(path.join(outDir, `${name}.photoshop.json`), JSON.stringify(fixture, null, 2) + "\n");
console.log(`run-probe: recorded ${reply.cases.length} case(s)`);
