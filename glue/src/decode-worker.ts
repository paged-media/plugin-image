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

// K-3 / S-07 / I-02 — the paged.image DECODE WORKER (the real consumer
// of the host.workers door). An off-main-thread ES-module worker that
// boots its OWN image-js instance and decodes PSD/PNG/JPEG bytes to
// straight RGBA8, posting the result back. The M4 ingest lane's codec/PSD
// decode is CPU work; moving it here keeps the editor's main flow
// responsive while a multi-image batch decodes in parallel across the
// pool (decode-pool.ts dispatches across N of these).
//
// TRUST LINE: this worker has NO ambient authority — no engine/DOM/
// network handle, no canvas client. The editor's WorkerBackend constructs
// it as a plain module worker; it talks ONLY to the bundle's own glue
// over postMessage. It boots a SECOND, worker-local image-js (the same
// `--target web` artifact the main realm boots) purely for the CPU decode
// lanes — it never requests a GPU device (kernels are GPU-only and stay
// on the main realm; decode is pure CPU).
//
// The wasm is loaded the browser way (engine.ts's path): the glue JS via
// a bare import, the `_bg.wasm` via a `?url` asset import. In a worker the
// bundler serves both as worker chunks.

/// <reference lib="webworker" />

// The decode subset of the image-js wasm-bindgen surface (a structural
// subset of manifest/wasm/image_js.d.ts — only what the worker drives).
interface DecodeWasm {
  default(input?: unknown): Promise<unknown>;
  decode_image(bytes: Uint8Array): {
    handle: number;
    width: number;
    height: number;
    free(): void;
  };
  image_tile_rgba8(
    handle: number,
    x: number,
    y: number,
    w: number,
    h: number,
  ): Uint8Array;
  free_image(handle: number): void;
}

/** A decode request from the pool. `id` correlates the reply. */
export interface DecodeRequest {
  id: number;
  bytes: Uint8Array;
}

/** The worker's reply for one request. On success carries the straight
 *  RGBA8 buffer (transferred) + the natural pixel extent; on failure the
 *  engine's honest message (16-bit / CMYK / unsupported …). */
export type DecodeReply =
  | {
      id: number;
      ok: true;
      width: number;
      height: number;
      rgba: ArrayBuffer;
    }
  | { id: number; ok: false; error: string };

let wasmPromise: Promise<DecodeWasm> | null = null;

/** Boot the worker-local image-js (browser path, like engine.ts). */
async function ensureWasm(): Promise<DecodeWasm> {
  if (!wasmPromise) {
    wasmPromise = (async () => {
      // The glue + the `_bg.wasm` URL — resolved as worker chunks by the
      // bundler (the editor's `?worker&url` build includes them).
      const mod = (await import(
        // @ts-ignore — artifact built by build-wasm.sh; absent from the source tree.
        "../wasm/image_js.js"
      )) as unknown as DecodeWasm;
      const wasmUrl = (await import(
        // @ts-ignore — `?url` is a bundler affordance, untyped.
        "../wasm/image_js_bg.wasm?url"
      )) as { default: string };
      await mod.default({ module_or_path: wasmUrl.default });
      return mod;
    })();
  }
  return wasmPromise;
}

/** Decode bytes to a full RGBA8 buffer via a level-0 window cut over the
 *  whole image (image_tile_rgba8 with the natural extent — the same pure
 *  windowing the tile provider uses, here for the whole frame). Frees the
 *  engine handle before returning (the worker holds nothing between
 *  requests). */
async function decode(bytes: Uint8Array): Promise<{
  width: number;
  height: number;
  rgba: Uint8Array;
}> {
  const wasm = await ensureWasm();
  const decoded = wasm.decode_image(bytes);
  const { handle, width, height } = decoded;
  decoded.free();
  try {
    const rgba = wasm.image_tile_rgba8(handle, 0, 0, width, height);
    // Copy out before freeing the handle (the view aliases wasm memory).
    const out = new Uint8Array(rgba.length);
    out.set(rgba);
    return { width, height, rgba: out };
  } finally {
    wasm.free_image(handle);
  }
}

// The worker message loop. Guarded by `self instanceof WorkerGlobalScope`
// so importing the module for its TYPES (the pool / a test) never wires a
// listener on the main thread.
const scope = self as unknown as DedicatedWorkerGlobalScope;
if (
  typeof DedicatedWorkerGlobalScope !== "undefined" &&
  scope instanceof DedicatedWorkerGlobalScope
) {
  scope.addEventListener("message", (ev: MessageEvent<DecodeRequest>) => {
    const { id, bytes } = ev.data;
    void (async () => {
      try {
        const { width, height, rgba } = await decode(bytes);
        // `out` was allocated as a fresh `new Uint8Array(len)`, so its
        // buffer is a plain (non-shared) ArrayBuffer — assert it for the
        // transfer typing.
        const buffer = rgba.buffer as ArrayBuffer;
        const reply: DecodeReply = {
          id,
          ok: true,
          width,
          height,
          rgba: buffer,
        };
        // Transfer the RGBA buffer (zero-copy hand-back).
        scope.postMessage(reply, [buffer]);
      } catch (err) {
        const reply: DecodeReply = {
          id,
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        };
        scope.postMessage(reply);
      }
    })();
  });
}
