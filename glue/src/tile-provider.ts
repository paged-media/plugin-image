// C-6 (I-06) — the renderer RESOURCE-PROVIDER consumer. Claim a placed
// image's pyramid through host.images.claimImageResource and serve tiles
// from the engine's decoded buffer; the renderer pulls the tiles its
// current scale needs (the v44 wire), the SDK adapter owns the
// needed→source→submit plumbing — this module supplies only the `source`
// callback (+ `revision`).
//
// ENGINE-B WINDOW-EVAL GAP (the honest subset — do NOT fake):
// Engine B (image-graph::BufferGraph::request) DOES support full
// (node, region, level) mip-aware window evaluation in Rust — but its
// output is rgba16float and, crucially, it is NOT exposed across the
// image-js wasm boundary yet (image-js publishes only decode/adjust/
// free + the new image_tile_rgba8 window cut). So this provider serves
// the LEVEL-0 lane only: it windows the already-decoded RGBA8 buffer
// (image_tile_rgba8 — pure slicing, no kernel) into tile_size tiles.
// Levels above 0 return null (the renderer holds the whole-image
// fallback / the best level it has). Wiring Engine B's tiled mip eval
// (and an rgba16float→rgba8 downconvert) to wasm is the M2/Stage-B
// follow-on; until then this is the truthful partial, named here.

import type { BundleHost, Disposable, TileBytes } from "@paged-media/plugin-api";

import type { ImageEngine } from "./engine";

/** What the provider needs about the ingested image to claim it. */
export interface TileSource {
  /** The frame to claim (the renderer's image_id namespace). */
  elementId: string;
  /** The engine handle holding the decoded RGBA8 pixels. */
  handle: number;
  /** Natural pixel extent (level 0). */
  width: number;
  height: number;
}

/** The grid step for the level-0 lane. 256 keeps a tile under the
 *  per-message budget (256² × 4 = 256 KiB) while covering a 50 MP image
 *  in a few hundred tiles. */
export const TILE_SIZE = 256;

/**
 * Claim `src.elementId`'s image resource and serve LEVEL-0 tiles from the
 * engine's decoded buffer. Returns the claim's Disposable (dispose →
 * release; the renderer drops to the whole-image fallback lane). When the
 * host wires no resource channel
 * (`supports("rendering.resourceProvider@1")` false) the claim is the
 * SDK's inert no-op door — still safe to dispose.
 *
 * `getHandle` is read on every tile pull so a re-ingest (new handle for
 * the same frame) is picked up without re-claiming; it returns null once
 * the source is freed (the provider then serves transparent misses).
 */
export function claimImageTiles(
  host: BundleHost,
  src: TileSource,
  engine: ImageEngine,
  getHandle: () => number | null,
): Disposable {
  // The provider is honest about what it has: one level. `levels: 1`
  // tells the renderer not to ask above level 0 — but the SDK adapter
  // still routes whatever the worker reports, so `source` also guards
  // `level > 0` (defense in depth + the named gap).
  let revision = 1;
  return host.images.claimImageResource(src.elementId, {
    levels: 1,
    tileSize: TILE_SIZE,
    baseWidth: src.width,
    baseHeight: src.height,
    revision: () => revision,
    source: async (level, x, y): Promise<TileBytes | null> => {
      // GAP: no mip pyramid wired across wasm yet — only level 0.
      if (level !== 0) return null;
      const handle = getHandle();
      if (handle === null) return null;
      // `x`/`y` arrive as LEVEL-space tile origins (the SDK echoes the
      // worker's grid). Cut the window from the decoded buffer.
      let rgba: Uint8Array;
      try {
        rgba = engine.tile(handle, x, y, TILE_SIZE, TILE_SIZE);
      } catch (err) {
        host.log.debug(`tile(${x}, ${y}) cut failed`, err);
        return null;
      }
      if (rgba.length === 0) return null; // fully outside the image
      // The cut is clamped to the image extent — recover the real tile
      // dims (edge tiles are narrower/shorter than TILE_SIZE).
      const tw = Math.min(TILE_SIZE, src.width - x);
      const th = Math.min(TILE_SIZE, src.height - y);
      if (tw <= 0 || th <= 0) return null;
      return { x, y, width: tw, height: th, rgba };
    },
  });
}

/** Bump on a content change (a re-decode of the same frame) so the
 *  renderer re-pulls. Exposed for the session to call on re-ingest. */
export function nextRevision(prev: number): number {
  return prev + 1;
}
