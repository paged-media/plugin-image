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

// Shared test fixtures: the minimal fake editor (the plugin-draw/
// plugin-web pattern, widened with the M4 doors — sceneLayers channel,
// importers registry, the placedAssetBytes wire reply) and a tiny
// hand-assembled PSD. Engine behavior is NOT faked: the session specs
// boot the REAL image-js wasm in Node (decode lanes are CPU; GPU init
// degrades honestly — no navigator.gpu in Node).

import { vi } from "vitest";

import type { ElementGeometryItem, PagedEditor } from "@paged-media/plugin-api";

export function fakeRegistry() {
  const byId = new Map<string, { id: string }>();
  return {
    ids: () => Array.from(byId.keys()),
    get: (id: string) => byId.get(id),
    register(c: { id: string }) {
      if (byId.has(c.id)) throw new Error(`duplicate id ${c.id}`);
      byId.set(c.id, c);
      return {
        dispose() {
          byId.delete(c.id);
        },
      };
    },
  };
}

/** A 2×1 8-bit RGB PSD (RAW composite) — pixels (10,30,50), (20,40,60). */
export function psdBytes(): Uint8Array {
  const b: number[] = [];
  const pushU16 = (v: number) => b.push((v >> 8) & 0xff, v & 0xff);
  const pushU32 = (v: number) =>
    b.push((v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff);
  b.push(0x38, 0x42, 0x50, 0x53); // "8BPS"
  pushU16(1); // version (PSD)
  b.push(0, 0, 0, 0, 0, 0); // reserved
  pushU16(3); // channels
  pushU32(1); // height
  pushU32(2); // width
  pushU16(8); // depth
  pushU16(3); // RGB
  pushU32(0); // color mode data (empty)
  pushU32(0); // image resources (empty)
  pushU32(0); // layer & mask info (empty)
  pushU16(0); // RAW compression
  b.push(10, 20, 30, 40, 50, 60); // R, G, B planes
  return Uint8Array.from(b);
}

/** The 2×1 RGBA8 pixels `psdBytes()` decodes to. */
export const PSD_RGBA = [10, 30, 50, 255, 20, 40, 60, 255];

export function makeFakeEditor() {
  const panels = fakeRegistry();
  const commands = fakeRegistry();
  const importers = fakeRegistry();
  const tools = fakeRegistry();
  // The overlay signal sink (host.overlay.setToolPreview forwards here);
  // records the last published shape so a test can assert the crop frame.
  const overlayShapes: unknown[] = [];
  const overlaySignals = {
    setToolPreview: (shape: unknown) => {
      overlayShapes.push(shape);
    },
  };
  const subscribers = new Set<(msg: unknown) => void>();
  const sceneLayers = {
    submit: vi.fn(async () => {}),
    clear: vi.fn(async () => {}),
  };
  // C-6 — a fake renderer resource channel (the editor's PagedEditor.images
  // member). Records claims/releases/submits and lets a test EMIT a
  // resourceTilesNeeded notification, driving the SDK adapter's
  // pull→submit plumbing against the bundle's provider.
  const imageNeededListeners = new Set<
    (need: {
      imageId: string;
      level: number;
      tiles: [number, number][];
      generation: number;
    }) => void
  >();
  const imageClaims: unknown[] = [];
  const imageReleases: string[] = [];
  const imageSubmits: Array<{
    imageId: string;
    level: number;
    tiles: Array<{ x: number; y: number; width: number; height: number; rgba: number[] }>;
    generation: number;
  }> = [];
  const images = {
    claim: vi.fn(async (claim: unknown) => {
      imageClaims.push(claim);
    }),
    release: vi.fn(async (imageId: string) => {
      imageReleases.push(imageId);
    }),
    submitTiles: vi.fn(
      async (
        imageId: string,
        level: number,
        tiles: Array<{
          x: number;
          y: number;
          width: number;
          height: number;
          rgba: number[];
        }>,
        generation: number,
      ) => {
        imageSubmits.push({ imageId, level, tiles, generation });
      },
    ),
    onResourceTilesNeeded: (
      listener: (need: {
        imageId: string;
        level: number;
        tiles: [number, number][];
        generation: number;
      }) => void,
    ) => {
      imageNeededListeners.add(listener);
      return () => imageNeededListeners.delete(listener);
    },
  };
  /** elementId → placed ORIGINAL bytes (the C-5 store). */
  const placed = new Map<string, Uint8Array>();
  /** elementId → element geometry (the crop tool's frame-box read). */
  const geometry = new Map<string, ElementGeometryItem>();
  const editor = {
    registries: { panels, commands, importers, tools },
    overlaySignals,
    selection: {
      elementSelection: [] as unknown[],
      setElementSelection: () => {},
      setElementGeometry: () => {},
    },
    camera: { camera: { scale: 1, tx: 0, ty: 0 } },
    sceneLayers,
    images,
    client: {
      mutate: async () => ({ kind: "mutationApplied", payload: {} }),
      documentMeta: async () => ({ pageCount: 1, activePage: "pg1" }),
      collection: async () => [],
      setElementSelection: async (ids: unknown[]) => ids,
      elementGeometry: async (ids: Array<{ id?: string }>) =>
        ids
          .map((i) => (i.id ? geometry.get(i.id) : undefined))
          .filter((g): g is ElementGeometryItem => g !== undefined),
      subscribe: (fn: (msg: unknown) => void) => {
        subscribers.add(fn);
        return () => subscribers.delete(fn);
      },
      // C-5 — the requestPlacedAssetBytes wire pair, answered from the
      // fake placed store (found:false otherwise, the honest miss).
      send: async (msg: { kind: string; payload?: { elementId?: string } }) => {
        if (msg.kind === "requestPlacedAssetBytes") {
          const id = msg.payload?.elementId ?? "";
          const bytes = placed.get(id);
          return {
            kind: "placedAssetBytes",
            payload: bytes
              ? {
                  elementId: id,
                  found: true,
                  uri: "links/sample.psd",
                  width: 2,
                  height: 1,
                  encoded: Array.from(bytes),
                }
              : {
                  elementId: id,
                  found: false,
                  uri: "",
                  width: 0,
                  height: 0,
                  encoded: [],
                },
          };
        }
        throw new Error(`fake editor: unhandled ${msg.kind}`);
      },
    },
  };
  const emitSelection = (ids: unknown[]) => {
    editor.selection.elementSelection = ids;
    for (const fn of [...subscribers]) {
      fn({ kind: "elementSelectionApplied", payload: { ids } });
    }
  };
  /** Drive the worker→main resourceTilesNeeded event; resolves after the
   *  SDK adapter's async source→submit microtasks settle. */
  const emitTilesNeeded = async (need: {
    imageId: string;
    level: number;
    tiles: [number, number][];
    generation: number;
  }) => {
    for (const l of [...imageNeededListeners]) l(need);
    await new Promise((r) => setTimeout(r, 0));
  };
  return {
    editor: editor as unknown as PagedEditor,
    panels,
    commands,
    importers,
    tools,
    sceneLayers,
    images,
    overlayShapes,
    geometry,
    imageClaims,
    imageReleases,
    imageSubmits,
    emitTilesNeeded,
    placed,
    emitSelection,
  };
}

export const silentConsole = { debug() {}, info() {}, warn() {}, error() {} };

export const mapBacking = () => {
  const m = new Map<string, string>();
  return {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    removeItem: (k: string) => void m.delete(k),
    keys: () => Array.from(m.keys()),
  };
};

export const shellStub = (openPanel: (id: string) => void = () => {}) => ({
  openPanel,
  closePanel() {},
  pickFile: async () => [],
});
