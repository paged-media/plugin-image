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

// ADR-023 phase D — paged.image answers the HOST's Layers panel while
// the `rasterImage` context is active.
//
// THE RULE THIS IMPLEMENTS: the document is a paged document with
// layers, and a plugin content type has layers of its OWN. When you are
// inside a plugin's content, the Layers panel shows THAT content's
// layers, not the document's. One panel, one place to look, and what it
// shows follows where you are — which is the same principle ADR-023
// applied to values, applied here to a collection.
//
// The breadcrumb, not this list, answers "where am I". That division is
// deliberate: mixing the host's layer rows and the image's into one
// list would mean this provider synthesizing a row for a core-owned
// layer it does not own and cannot speak for, and the panel would show
// two authorities' rows with no way to tell them apart.
//
// WHAT IT SERVES, and what it deliberately does not:
//
//   ROWS — the raster layer stack, in core's `LayerSummary` shape. The
//   image's layer GROUPS become real tree rows, because `LayerSummary`
//   carries `parentId` and the host list already declares
//   `tree: { parentField: "parentId" }`. Groups render as folders for
//   free; nothing here draws a tree.
//
//   OPS — `layerMove` / `layerSetVisible` / `layerSetLocked` /
//   `layerSetName` / `layerRemove`, the structural edits the host
//   panel's own actions send. Taking first refusal is what makes the
//   panel's buttons act on the RASTER stack instead of quietly editing
//   the document's layers behind a panel that is showing ours.
//
//   NOT opacity, NOT blend mode. `LayerSummary` models neither, and the
//   host Layers panel has no control for either — so those stay in the
//   image panel, where they are image vocabulary with an image
//   audience. Retiring them into a panel that cannot show them would be
//   a silent feature loss dressed up as consolidation.

import type {
  BindingCollection,
  BindingProvider,
  BindingProviderScope,
  BindingWrite,
  MutationInput,
} from "@paged-media/plugin-api";

import type { ImageSession } from "../session";
import type { LayerGroupInfo, LayerInfo } from "../engine";

/** Core ops this provider takes first refusal on — exactly the ones the
 *  host Layers panel's actions and drag-reorder emit. An op NOT listed
 *  here reaches core untouched, which is correct: the panel is showing
 *  our rows, but a verb we do not implement is not ours to swallow. */
const SERVED_OPS: BindingProviderScope["ops"] = [
  "layerMove",
  "layerSetVisible",
  "layerSetLocked",
  "layerSetName",
  "layerRemove",
];

/** A group's row id. Groups and layers share one id space in the host
 *  list, and the raster model numbers them independently, so the two
 *  are prefixed apart rather than hoping they never collide. */
const groupRowId = (id: number): string => `g${id}`;
/** A layer's row id. Stable across reorders — it is the engine's own
 *  stable layer id, not the index. */
const layerRowId = (id: number): string => `l${id}`;

/** Parse a row id back to what it addresses. Returns null for anything
 *  this provider did not mint. */
export function parseRowId(
  selfId: string,
): { kind: "group" | "layer"; id: number } | null {
  const m = /^([gl])(\d+)$/.exec(selfId);
  if (!m) return null;
  return { kind: m[1] === "g" ? "group" : "layer", id: Number(m[2]) };
}

/**
 * One raster layer as core's `LayerSummary`.
 *
 * `printable` is the interesting field: the raster model has no such
 * concept, and the contract's rule for a field the provider's model
 * cannot answer is the provider's HONEST DEFAULT, not a decline and not
 * a lie. A raster layer that is visible prints, so `true` is the honest
 * answer rather than a convenient one — and the alternative, mirroring
 * `visible`, would invent a second control that silently moved the
 * first.
 */
function layerRow(l: LayerInfo, z: number): Record<string, unknown> {
  return {
    selfId: layerRowId(l.id),
    name: l.name,
    visible: l.visible,
    locked: l.locked,
    printable: true,
    z,
    parentId: l.group === null ? null : groupRowId(l.group),
  };
}

/** A layer GROUP as a `LayerSummary` row — the folder its members hang
 *  under. Groups nest one level in the raster model's own terms (a
 *  group's parent is always the stack root), so `parentId` is null. */
function groupRow(g: LayerGroupInfo, z: number): Record<string, unknown> {
  return {
    selfId: groupRowId(g.id),
    name: g.name,
    visible: g.visible,
    // The raster model has no group-level pixel lock — locking is
    // per-layer. Honest default rather than a control that would do
    // nothing.
    locked: false,
    printable: true,
    z,
    parentId: null,
  };
}

export interface LayersBindingProvider {
  provider: BindingProvider;
}

export function makeLayersBindingProvider(
  session: ImageSession,
): LayersBindingProvider {
  /** The index `layerMove` and friends address, from a row id. The host
   *  panel speaks row ids; the raster engine speaks INDEXES, and the
   *  translation is here rather than in the session so the session
   *  never learns that a host panel exists. */
  const indexOfRow = (selfId: string): number | null => {
    const parsed = parseRowId(selfId);
    if (!parsed || parsed.kind !== "layer") return null;
    const i = session.state().layers.layers.findIndex((l) => l.id === parsed.id);
    return i < 0 ? null : i;
  };

  const provider: BindingProvider = {
    provides: {
      collections: ["layers"],
      ops: SERVED_OPS,
    },

    readCollection(request): BindingCollection {
      if (request.collection !== "layers") {
        return { kind: "decline", reason: "only the layers collection" };
      }
      const stack = session.state().layers;
      if (stack.layers.length === 0) {
        // A DECLINE, not an empty row set. Empty rows would CLAIM the
        // panel and assert "this raster frame has no layers", which is
        // false — there is no open stack here at all, and the document's
        // own layers are then the truthful thing to show.
        return { kind: "decline", reason: "no raster layer stack is open" };
      }
      // FRONT-FIRST is the panel's job, not ours: the host list declares
      // `displayOrder: "frontFirst"` and re-orders what we hand it. We
      // emit the engine's own order and let one place own that rule.
      //
      // Groups first so a folder exists before the rows that point at
      // it; the host builds the tree from `parentId` regardless, but a
      // parent-after-child list is the kind of thing that works until a
      // renderer stops sorting.
      const rows: Record<string, unknown>[] = [];
      stack.groups.forEach((g, i) => rows.push(groupRow(g, i)));
      stack.layers.forEach((l, i) => rows.push(layerRow(l, stack.groups.length + i)));
      return { kind: "rows", rows };
    },

    async applyMutation(mutation: MutationInput): Promise<BindingWrite> {
      switch (mutation.op) {
        case "layerSetVisible": {
          const a = mutation.args as { id?: string; visible?: boolean };
          const i = a.id ? indexOfRow(a.id) : null;
          if (i === null) return notOurs(a.id);
          const ok = await session.setLayerVisible(i, a.visible !== false);
          return applied(ok);
        }
        case "layerSetLocked": {
          const a = mutation.args as { id?: string; locked?: boolean };
          const i = a.id ? indexOfRow(a.id) : null;
          if (i === null) return notOurs(a.id);
          return applied(session.setLayerLocked(i, a.locked !== false));
        }
        case "layerSetName": {
          const a = mutation.args as { id?: string; name?: string };
          const i = a.id ? indexOfRow(a.id) : null;
          if (i === null) return notOurs(a.id);
          return applied(session.setLayerName(i, a.name ?? ""));
        }
        case "layerRemove": {
          const a = mutation.args as { id?: string };
          const i = a.id ? indexOfRow(a.id) : null;
          if (i === null) return notOurs(a.id);
          return applied(await session.removeLayer(i));
        }
        case "layerMove": {
          const a = mutation.args as { id?: string; to?: number };
          const from = a.id ? indexOfRow(a.id) : null;
          if (from === null || typeof a.to !== "number") return notOurs(a.id);
          return applied(await session.reorderLayer(from, a.to));
        }
        default:
          // Declared ops only ever route here, so this is unreachable by
          // the registry's own gate — kept as the honest floor rather
          // than a cast that assumes it.
          return {
            kind: "decline",
            reason: `paged.image does not serve "${mutation.op}"`,
          };
      }
    },
  };

  return { provider };
}

/** A row this provider did not mint — a GROUP row, or a stale id. The
 *  host is showing our rows, so this is a real miss worth naming rather
 *  than a silent false. */
function notOurs(id: string | undefined): BindingWrite {
  return {
    kind: "decline",
    reason: `"${id ?? "(no id)"}" is not a raster layer row`,
  };
}

/**
 * The engine's boolean, as the contract's outcome. A refusal is a
 * RESOLVED VALUE here, never a throw — the raster engine declines for
 * ordinary reasons (a locked layer, no open stack) and the panel needs
 * to hear that rather than catch an exception.
 *
 * `createdId: null` and `pageIds: []` are both literally true rather
 * than filler: a layer edit creates no element, and it touches no
 * DOCUMENT page — the raster stack lives in the plugin's own wasm and
 * the host has no page to invalidate for it. Claiming a page here would
 * make the host re-render something that did not change.
 */
function applied(ok: boolean): BindingWrite {
  return ok
    ? { kind: "applied", outcome: { applied: true, createdId: null, pageIds: [] } }
    : { kind: "decline", reason: "the raster engine refused the edit" };
}
