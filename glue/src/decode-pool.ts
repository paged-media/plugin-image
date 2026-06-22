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

// K-3 / S-07 / I-02 — the paged.image DECODE POOL: the live consumer of
// host.workers. It spawns N decode workers (N from the GRANTED cap —
// host.workers.concurrency(), itself min(declared.max, hardwareConcurrency,
// 8)) and parallelises the M4 ingest decode lane across them. A batch of
// placed images (or a multi-page ingest) decodes embarrassingly in
// parallel; a single decode runs off the main thread, keeping the editor
// responsive.
//
// DEGRADES HONESTLY (the no-speculative-surface / brand-honesty rule):
// when the host wires no WorkerBackend (supports("workers@1") false) or
// the granted cap is 0, `create` returns null and the session falls back
// to the main-thread engine.decode — same pixels, just not off-thread. No
// fake parallelism, no silent main-thread stall pretending to be a pool.
//
// The DECLARED module path ("workers/decode.js") is what the manifest
// lists under capabilities.workers' shipped module and what the editor's
// WorkerBackend resolver maps to the served worker URL. A bundle can only
// spawn a module it ships (declared-only, like the wasm artifacts).

import type { BundleHost, BundleWorker } from "@paged-media/plugin-api";

import type { DecodeReply, DecodeRequest } from "./decode-worker";

/** The DECLARED, bundle-relative decode-worker module path. The manifest
 *  lists it; the editor's WorkerBackend resolver maps it to a served URL. */
export const DECODE_WORKER_MODULE = "workers/decode.js";

/** One decoded image — straight RGBA8 + natural pixel extent. */
export interface DecodedRGBA {
  width: number;
  height: number;
  rgba: Uint8Array;
}

/** The decode pool surface the session drives. */
export interface DecodePool {
  /** Worker count the pool actually spawned (the granted cap). */
  size(): number;
  /** Decode one image off-thread; rejects with the engine's honest
   *  message on an unsupported input. Dispatched to the least-busy worker. */
  decode(bytes: Uint8Array): Promise<DecodedRGBA>;
  /** Decode a batch in parallel across the pool (the M4 ingest lane).
   *  Resolves in input order; a per-item failure surfaces as a rejected
   *  entry via `Promise.allSettled` semantics (callers map results). */
  decodeBatch(items: readonly Uint8Array[]): Promise<DecodedRGBA[]>;
  /** Terminate every worker (also auto-run by the host on bundle dispose). */
  dispose(): void;
}

interface PendingDecode {
  resolve(v: DecodedRGBA): void;
  reject(e: Error): void;
}

interface PoolWorker {
  worker: BundleWorker;
  /** In-flight request count (for least-busy dispatch). */
  inflight: number;
  /** Pending decodes keyed by request id. */
  pending: Map<number, PendingDecode>;
}

/**
 * Spawn a decode pool over `host.workers`. Returns null when the host
 * wires no worker backend or grants no workers (the honest fallback
 * signal — the session decodes on the main thread instead). Spawning is
 * async (each `spawn` resolves a Worker); `create` awaits them all so the
 * caller gets a ready pool or null.
 */
export async function createDecodePool(
  host: BundleHost,
): Promise<DecodePool | null> {
  if (!host.supports("workers@1")) {
    host.log.debug(
      "decode pool: host wires no workers (workers@1 false) — main-thread decode",
    );
    return null;
  }
  const cap = host.workers.concurrency();
  if (cap <= 0) return null;

  const workers: PoolWorker[] = [];
  let nextId = 1;

  const wireWorker = (bw: BundleWorker): PoolWorker => {
    const pw: PoolWorker = { worker: bw, inflight: 0, pending: new Map() };
    bw.onMessage((msg) => {
      const reply = msg as DecodeReply;
      const p = pw.pending.get(reply.id);
      if (!p) return;
      pw.pending.delete(reply.id);
      pw.inflight = Math.max(0, pw.inflight - 1);
      if (reply.ok) {
        p.resolve({
          width: reply.width,
          height: reply.height,
          rgba: new Uint8Array(reply.rgba),
        });
      } else {
        p.reject(new Error(reply.error));
      }
    });
    return pw;
  };

  // Spawn up to the granted cap. A spawn that rejects (a backend hiccup)
  // is tolerated — the pool runs with whatever workers came up; an EMPTY
  // pool degrades to null (main-thread fallback).
  for (let i = 0; i < cap; i++) {
    try {
      const bw = await host.workers.spawn({
        module: DECODE_WORKER_MODULE,
        name: `paged.image decode #${i}`,
      });
      workers.push(wireWorker(bw));
    } catch (err) {
      host.log.warn(`decode pool: worker ${i} failed to spawn`, err);
    }
  }
  if (workers.length === 0) return null;
  host.log.info(`decode pool: ${workers.length} worker(s) ready`);

  const leastBusy = (): PoolWorker =>
    workers.reduce((a, b) => (b.inflight < a.inflight ? b : a));

  const decodeOn = (pw: PoolWorker, bytes: Uint8Array): Promise<DecodedRGBA> =>
    new Promise<DecodedRGBA>((resolve, reject) => {
      const id = nextId++;
      pw.pending.set(id, { resolve, reject });
      pw.inflight++;
      // Transfer the input bytes (the main realm is done with them).
      const req: DecodeRequest = { id, bytes };
      pw.worker.post(req, [bytes.buffer]);
    });

  let disposed = false;
  return {
    size: () => workers.length,
    decode(bytes) {
      if (disposed) return Promise.reject(new Error("decode pool disposed"));
      return decodeOn(leastBusy(), bytes);
    },
    async decodeBatch(items) {
      if (disposed) throw new Error("decode pool disposed");
      // Round-robin across workers so the batch fans out evenly.
      return Promise.all(
        items.map((bytes, i) => decodeOn(workers[i % workers.length], bytes)),
      );
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const pw of workers) {
        for (const p of pw.pending.values()) {
          p.reject(new Error("decode pool disposed"));
        }
        pw.pending.clear();
        pw.worker.terminate();
      }
    },
  };
}
