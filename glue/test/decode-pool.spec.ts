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

// K-3 / S-07 / I-02 — the paged.image DECODE POOL (the live consumer of
// host.workers) over a MOCK WorkerBackend whose workers run the REAL
// image-js decode in-process (Node, CPU lanes — the same decode the worker
// module performs, here without a Worker realm so it is unit-testable).
// Pins: the pool spawns the GRANTED cap of workers, a decode round-trips
// the codec output, a batch fans out across the pool, dispose terminates
// every worker, and the honest null when the host wires no workers.

import { describe, expect, it } from "vitest";

import { createBundleHost, type WorkerBackend } from "@paged-media/plugin-sdk";
import type { PagedEditor, PluginManifest } from "@paged-media/plugin-api";

import manifestJson from "@paged-media/image-manifest/manifest.json";

import { createDecodePool, DECODE_WORKER_MODULE } from "../src/decode-pool";
import type { DecodeReply, DecodeRequest } from "../src/decode-worker";
import { bootEngine } from "../src/engine";
import { makeFakeEditor, mapBacking, psdBytes, silentConsole, PSD_RGBA } from "./helpers";

// A mock WorkerBackend whose every "worker" decodes via the REAL engine
// wasm in-process (the worker module's logic, run on the main thread for
// the test). Records spawns + terminations so the cap + teardown can be
// asserted. The injected module path is checked against DECODE_WORKER_MODULE
// (the declared-only contract).
function makeDecodingBackend() {
  const spawned: Array<{ module: string; alive: boolean }> = [];
  const enginePromise = bootEngine();
  const backend: WorkerBackend = {
    async spawn(_pluginId, module) {
      expect(module).toBe(DECODE_WORKER_MODULE);
      const record = { module, alive: true };
      spawned.push(record);
      const handlers = new Set<(m: unknown) => void>();
      return {
        post(message) {
          const req = message as DecodeRequest;
          void (async () => {
            const engine = await enginePromise;
            let reply: DecodeReply;
            try {
              // The worker decodes to an engine handle, then windows the
              // whole image out as RGBA8 (the decode-worker's contract).
              const info = engine.decode(req.bytes);
              const rgba = engine.tile(info.handle, 0, 0, info.width, info.height);
              engine.freeImage(info.handle);
              const out = new Uint8Array(rgba.length);
              out.set(rgba);
              reply = {
                id: req.id,
                ok: true,
                width: info.width,
                height: info.height,
                rgba: out.buffer,
              };
            } catch (err) {
              reply = {
                id: req.id,
                ok: false,
                error: err instanceof Error ? err.message : String(err),
              };
            }
            for (const h of handlers) h(reply);
          })();
        },
        onMessage(handler) {
          handlers.add(handler);
          return { dispose: () => handlers.delete(handler) };
        },
        terminate() {
          record.alive = false;
          handlers.clear();
        },
      };
    },
  };
  return { backend, spawned };
}

function makeHost(withWorkers: boolean) {
  const fake = makeFakeEditor();
  const mock = makeDecodingBackend();
  const handle = createBundleHost(
    () => fake.editor as unknown as PagedEditor,
    manifestJson as PluginManifest,
    {
      console: silentConsole,
      storage: mapBacking(),
      ...(withWorkers ? { workers: mock.backend } : {}),
    },
  );
  return { ...handle, mock };
}

describe("the paged.image decode pool (K-3 consumer)", () => {
  it("spawns the granted cap of workers and decodes off-thread", async () => {
    const { host, mock } = makeHost(true);
    const pool = await createDecodePool(host);
    expect(pool).not.toBeNull();
    // The manifest declares max 4; the grant is min(4, hwConcurrency, 8) ≥ 1.
    const grant = host.workers.concurrency();
    expect(pool!.size()).toBe(grant);
    expect(mock.spawned.length).toBe(grant);
    expect(mock.spawned.every((s) => s.module === DECODE_WORKER_MODULE)).toBe(true);

    // A real decode round-trips the codec output.
    const decoded = await pool!.decode(psdBytes());
    expect(decoded.width).toBe(2);
    expect(decoded.height).toBe(1);
    expect(Array.from(decoded.rgba)).toEqual(PSD_RGBA);

    pool!.dispose();
  });

  it("decodes a batch in parallel across the pool", async () => {
    const { host } = makeHost(true);
    const pool = await createDecodePool(host);
    expect(pool).not.toBeNull();
    const batch = [psdBytes(), psdBytes(), psdBytes()];
    const results = await pool!.decodeBatch(batch);
    expect(results).toHaveLength(3);
    for (const r of results) {
      expect(r.width).toBe(2);
      expect(Array.from(r.rgba)).toEqual(PSD_RGBA);
    }
    pool!.dispose();
  });

  it("dispose terminates every worker", async () => {
    const { host, mock } = makeHost(true);
    const pool = await createDecodePool(host);
    expect(mock.spawned.some((s) => s.alive)).toBe(true);
    pool!.dispose();
    expect(mock.spawned.every((s) => !s.alive)).toBe(true);
  });

  it("bundle dispose ALSO terminates the pool's workers (host teardown)", async () => {
    const { host, dispose, mock } = makeHost(true);
    await createDecodePool(host);
    expect(mock.spawned.some((s) => s.alive)).toBe(true);
    // The host facade tracks every spawned worker; tearing the bundle down
    // terminates them even without an explicit pool.dispose().
    dispose();
    expect(mock.spawned.every((s) => !s.alive)).toBe(true);
  });

  it("returns null when the host wires no workers (honest main-thread fallback)", async () => {
    const { host } = makeHost(false);
    expect(host.supports("workers@1")).toBe(false);
    const pool = await createDecodePool(host);
    expect(pool).toBeNull();
  });
});
