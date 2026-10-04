// Storing an image session IN THE DOCUMENT, so the layers survive saving
// and reopening.
//
// Layout under the plugin's own part namespace (`paged/media.paged.image/`):
//
//   px/<sha256>.bin           one buffer: a manifest, a layer's pixels, a
//                             mask or a smart object's source. Named by
//                             content, so a layer that did not change
//                             between two commits is stored once.
//   f/<frameId>/r<rev>.json   one revision: the manifest and buffer hashes,
//                             the size, and the hash of the BAKED image
//                             that revision put in the frame.
//
// Parts are written BEFORE the frame's image and marker change, and are
// never overwritten, so the document's own undo can only ever land on a
// marker whose revision exists. The frame's placed image is the baked
// composite — the page renders, prints and exports without this plugin.

import type { PartsSurface } from "@paged-media/plugin-api";

/** One stored revision (`f/<frameId>/r<rev>.json`). */
export interface SessionRecord {
  v: 1;
  width: number;
  height: number;
  /** Hash of the manifest buffer. */
  manifest: string;
  /** Hashes of the buffers, in slot order. */
  buffers: string[];
  /** Hash of the baked image bytes placed in the frame. */
  baked: string;
}

/** The frame marker's `data` once a session has been committed. */
export interface CommittedMarker {
  owns: "pixels";
  rev: number;
  record: string;
  baked: string;
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const d = await globalThis.crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

export const blobPath = (sha: string) => `px/${sha}.bin`;
export const recordPath = (frameId: string, rev: number) =>
  `f/${frameId.replace(/[^A-Za-z0-9_-]/g, "_")}/r${rev}.json`;

/**
 * Write a session: every buffer not already stored, then the record.
 * `exported` is the engine's `layersExport()` — manifest first.
 * Returns the record's path and how many buffers were new.
 */
export async function writeSession(
  parts: PartsSurface,
  frameId: string,
  rev: number,
  exported: Uint8Array[],
  size: { width: number; height: number },
  bakedSha: string,
): Promise<{ path: string; written: number }> {
  const have = new Set(await parts.list("px/"));
  const hashes = await Promise.all(exported.map(sha256Hex));
  let written = 0;
  for (let i = 0; i < exported.length; i++) {
    const p = blobPath(hashes[i]);
    if (have.has(p)) continue;
    await parts.write(p, exported[i]);
    have.add(p);
    written++;
  }
  const record: SessionRecord = {
    v: 1,
    width: size.width,
    height: size.height,
    manifest: hashes[0],
    buffers: hashes.slice(1),
    baked: bakedSha,
  };
  const path = recordPath(frameId, rev);
  await parts.write(path, new TextEncoder().encode(JSON.stringify(record)));
  return { path, written };
}

/** Read a session back: the record, the manifest and the buffers, each
 *  verified against its hash. Null (with the reason) when anything is
 *  missing or does not match. */
export async function readSession(
  parts: PartsSurface,
  path: string,
): Promise<
  | { ok: true; record: SessionRecord; manifest: Uint8Array; buffers: Uint8Array[] }
  | { ok: false; reason: string }
> {
  const raw = await parts.read(path);
  if (!raw) return { ok: false, reason: `the stored session ${path} is missing` };
  let record: SessionRecord;
  try {
    record = JSON.parse(new TextDecoder().decode(raw)) as SessionRecord;
  } catch {
    return { ok: false, reason: `the stored session ${path} is not readable` };
  }
  if (record.v !== 1) return { ok: false, reason: `stored session version ${record.v}` };
  const fetch = async (sha: string) => {
    const b = await parts.read(blobPath(sha));
    if (!b) throw new Error(`stored layer data ${sha.slice(0, 12)}… is missing`);
    if ((await sha256Hex(b)) !== sha) throw new Error(`stored layer data ${sha.slice(0, 12)}… is damaged`);
    return b;
  };
  try {
    const manifest = await fetch(record.manifest);
    const buffers = await Promise.all(record.buffers.map(fetch));
    return { ok: true, record, manifest, buffers };
  } catch (err) {
    return { ok: false, reason: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * Delete what no reachable revision needs: this frame's records older
 * than its newest `keep` (revision `rev` is the newest), then every
 * stored buffer that no remaining record — of any frame — names. Parts
 * are written before the document points at them and never rewritten,
 * so deleting only the unreachable ones keeps every revision the
 * document's undo can land on. Returns how many parts went.
 */
export async function collectGarbage(
  parts: PartsSurface,
  del: (path: string) => Promise<boolean>,
  frameId: string,
  rev: number,
  keep: number,
): Promise<{ records: number; buffers: number }> {
  const dir = recordPath(frameId, 0).replace(/r0\.json$/, "");
  let records = 0;
  for (const path of await parts.list(dir)) {
    const m = /\/r(\d+)\.json$/.exec(path);
    if (m && Number(m[1]) <= rev - keep) {
      if (await del(path)) records++;
    }
  }
  const live = new Set<string>();
  for (const path of await parts.list("f/")) {
    if (!path.endsWith(".json")) continue;
    const raw = await parts.read(path);
    if (!raw) continue;
    try {
      const r = JSON.parse(new TextDecoder().decode(raw)) as SessionRecord;
      live.add(blobPath(r.manifest));
      for (const b of r.buffers) live.add(blobPath(b));
    } catch {
      // An unreadable record keeps nothing alive and is left for a human.
    }
  }
  let buffers = 0;
  for (const path of await parts.list("px/")) {
    if (!live.has(path) && (await del(path))) buffers++;
  }
  return { records, buffers };
}
