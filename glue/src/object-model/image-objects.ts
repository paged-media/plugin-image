// paged.image's OBJECT MODEL (ADR 323): every image frame's layer stack
// as addressable, typed objects that a data binding, a script, the Node
// CLI or a panel field reads and writes through `host.objects`.
//
// Kinds (addresses are `plugin:media.paged.image/<kind>/<id>`):
//
//   image        `<frame>`             the frame's stack: size, layer count
//   layer        `<frame>/<layerId>`   name, visible, opacity, blendMode,
//                                      order, locked, clipped, kind, the
//                                      pixels (`content`, `pixelSource`)
//   mask         `<frame>/<layerId>`   a layer's mask: enabled, coverage
//   adjustment   `<frame>/<layerId>`   an adjustment layer's parameters —
//                                      GENERATED from the kernels (adjust.ts)
//   smartObject  `<frame>/<layerId>`   a smart object's preserved source
//
// `<frame>` is the frame's Self id and `<layerId>` the engine's stable
// layer id: SESSION handles. InDesign renumbers Self ids and drops the
// container parts, so nothing persisted names one — a stored binding
// selects by layer `name` inside a frame it finds by name or label.
//
// HEADLESS by construction: the stack is read and written as the stored
// manifest (pgil.ts), never through the engine's one layer slot (which
// the interactive session owns), and the engine is used only for its CPU
// codec (decode the placed image / a replacement, encode the bake).
//
// WRITES use the image session store's own label-hash pattern
// (session-store.ts): content-addressed parts, then ONE engine batch that
// moves the frame's `x-paged:media.paged.image` marker to the new
// revision — and, when the composite can be formed without a GPU, also
// places the baked image. Undo reverts the marker (and the image) in one
// step; the previous revision's parts are still there.
//
// The BAKE: the composite is a GPU kernel fold (constitution: no CPU
// blend path ships). A stack whose composite is the identity (one plain
// visible layer) bakes here; any other stack is stored with
// `pending: true` and the frame keeps its previous image until a GPU
// realm opens the frame (the session re-composites and commits on exit).

import type {
  Address,
  BundleHost,
  ElementId,
  ObjectKindContribution,
  ObjectOp,
  ObjectValue,
  ObjectWrite,
  PartChange,
  PropertySchema,
  SceneTreeNode,
} from "@paged-media/plugin-api";

import { escapeLabelValue, fromElementId, parseAddress } from "@paged-media/plugin-sdk";

import { blobPath, readSession, sha256Hex, type CommittedMarker, type SessionRecord } from "../session-store";
import { adjustSchemaRows, IMAGE_SCHEMA, readAdjust, widenAdjust, writeAdjust, defaultAdjust } from "./adjust";
import { clonePgil, readPgil, stackFromImage, writePgil, type PgilLayer, type PgilStack } from "./pgil";

export const PLUGIN_ID = "media.paged.image";
export const LABEL_KEY = `x-paged:${PLUGIN_ID}`;
export const KIND_PREFIX = `plugin:${PLUGIN_ID}/`;

/** The engine's CPU codec — all the object model needs of the wasm. */
export interface ImageCodec {
  decode(bytes: Uint8Array): { width: number; height: number; rgba: Uint8Array };
  encodePng(rgba: Uint8Array, width: number, height: number): Uint8Array;
}

/** What one stored revision adds to the session store's record. */
interface ObjectRecord extends SessionRecord {
  /** The last pixel source applied per layer id (a binding reads it). */
  sources?: Record<string, string>;
  /** Layer ids whose smart-object render is older than their source. */
  pendingRender?: number[];
  /** The frame still shows the previous bake (the composite needs a GPU). */
  pending?: boolean;
}

export interface ImageMarker extends CommittedMarker {
  /** The frame's image is NOT this revision's composite yet. */
  pending?: boolean;
}

interface Frame {
  frameId: string;
  elementId: ElementId;
}

interface FrameState {
  frame: Frame;
  stack: PgilStack;
  marker: ImageMarker | null;
  /** SHA-256 of the frame's placed image as it is now. */
  placedSha: string;
  sources: Record<string, string>;
  pendingRender: Set<number>;
}

class Refused extends Error {}

const value = (v: unknown): ObjectValue => ({ kind: "value", value: v });
const absent: ObjectValue = { kind: "absent" };
const refusedValue = (code: "unknownAddress" | "unknownPath" | "failed", reason: string): ObjectValue => ({
  kind: "refused",
  code,
  reason,
});

// ------------------------------------------------------------- schemas

const BLEND_MODES = IMAGE_SCHEMA.blendModes;

const IMAGE_ROWS: PropertySchema[] = [
  { path: "width", type: { kind: "number", integer: true }, access: "readOnly", summary: "Canvas width in image pixels." },
  { path: "height", type: { kind: "number", integer: true }, access: "readOnly", summary: "Canvas height in image pixels." },
  { path: "layerCount", type: { kind: "number", integer: true }, access: "readOnly" },
  { path: "revision", type: { kind: "number", integer: true }, access: "readOnly", summary: "The stored revision the frame's marker points at; 0 = never committed." },
  { path: "pending", type: { kind: "bool" }, access: "readOnly", summary: "The frame still shows an older composite: this stack needs a GPU realm to bake." },
];

const LAYER_ROWS: PropertySchema[] = [
  { path: "name", type: { kind: "text" }, summary: "The layer's name (what a stored binding selects by)." },
  { path: "visible", type: { kind: "bool" }, default: true },
  { path: "opacity", type: { kind: "number", unit: "percent" }, range: { min: 0, max: 100 }, default: 100 },
  { path: "blendMode", type: { kind: "enum", members: BLEND_MODES }, default: "normal", summary: "A registered compose.* kernel (generated from the kernel registry)." },
  { path: "order", type: { kind: "number", integer: true }, range: { min: 0 }, summary: "Stack position, 0 = bottom." },
  { path: "locked", type: { kind: "bool" }, default: false, summary: "Locks the layer's PIXELS; its properties stay editable." },
  { path: "clipped", type: { kind: "bool" }, default: false, summary: "Clipped to the layer beneath." },
  { path: "kind", type: { kind: "enum", members: ["pixels", "adjustment", "smart"] }, access: "readOnly" },
  { path: "id", type: { kind: "number", integer: true }, access: "readOnly", summary: "The engine's stable layer id (a session handle)." },
  { path: "hasMask", type: { kind: "bool" }, access: "readOnly" },
  { path: "bounds", type: { kind: "bounds" }, access: "readOnly", summary: "[top, left, bottom, right] in image pixels." },
  {
    path: "pixelSource",
    type: { kind: "asset" },
    // Up to 2 KiB of data: URL per layer: `getAll` skips it unless asked.
    heavy: true,
    summary:
      "Pixel replacement (Photoshop's pixel-replacement variable): a data: URL, or the address of a page item " +
      "whose placed image supplies the pixels. Placed as is, centred. Reads the last source applied.",
  },
];

const MASK_ROWS: PropertySchema[] = [
  { path: "enabled", type: { kind: "bool" }, default: true, summary: "Whether the mask applies (disabling keeps the coverage)." },
  { path: "bounds", type: { kind: "bounds" }, access: "readOnly" },
];

const SMART_ROWS: PropertySchema[] = [
  { path: "source", type: { kind: "asset" }, heavy: true, summary: "Replace contents: a data: URL or a page item whose placed image becomes the preserved source." },
  { path: "sourceWidth", type: { kind: "number", integer: true }, access: "readOnly" },
  { path: "sourceHeight", type: { kind: "number", integer: true }, access: "readOnly" },
  { path: "scale", type: { kind: "number" }, access: "readOnly", summary: "Re-rendering at a new scale is a GPU resample: the editor's panel." },
  { path: "renderPending", type: { kind: "bool" }, access: "readOnly", summary: "The cached render predates the source (re-rendered in a GPU realm)." },
];

export const KIND_ROWS = {
  image: IMAGE_ROWS,
  layer: LAYER_ROWS,
  mask: MASK_ROWS,
  adjustment: adjustSchemaRows(),
  smartObject: SMART_ROWS,
} as const;

export type ImageKind = keyof typeof KIND_ROWS;

export const KIND_TITLES: Record<ImageKind, string> = {
  image: "Image (layer stack)",
  layer: "Image layer",
  mask: "Layer mask",
  adjustment: "Adjustment layer",
  smartObject: "Smart object",
};

export const KIND_CONTENT: Partial<Record<ImageKind, { kind: "pixels"; mime: string[] }>> = {
  layer: { kind: "pixels", mime: ["image/png", "image/jpeg", "image/vnd.adobe.photoshop"] },
  mask: { kind: "pixels", mime: ["image/png", "image/jpeg"] },
};

// ------------------------------------------------------------ addresses

export function imageAddress(frameId: string): Address {
  return `${KIND_PREFIX}image/${encodeURIComponent(frameId)}`;
}
export function layerAddress(kind: Exclude<ImageKind, "image">, frameId: string, layerId: number): Address {
  return `${KIND_PREFIX}${kind}/${encodeURIComponent(frameId)}/${layerId}`;
}

/** `{ kind, frameId, layerId }` of one of this plugin's addresses. */
export function parseImageAddress(
  address: Address,
): { kind: ImageKind; frameId: string; layerId: number | null } | null {
  if (!address.startsWith(KIND_PREFIX)) return null;
  const [kind, frame, layer, ...extra] = address.slice(KIND_PREFIX.length).split("/");
  if (!kind || !(kind in KIND_ROWS) || !frame || extra.length > 0) return null;
  let frameId: string;
  try {
    frameId = decodeURIComponent(frame);
  } catch {
    return null;
  }
  if (kind === "image") return layer === undefined ? { kind, frameId, layerId: null } : null;
  if (layer === undefined || !/^\d+$/.test(layer)) return null;
  return { kind: kind as ImageKind, frameId, layerId: Number(layer) };
}

// -------------------------------------------------------------- the store

/** Counts the store's own work, so the perf spec can hold it to budgets. */
export interface ImageObjectCounters {
  treeReads: number;
  placedReads: number;
  sessionReads: number;
  decodes: number;
  encodes: number;
  /** Planner calls (one per `host.objects` batch, whatever its kinds). */
  plans: number;
}

export interface ImageObjects {
  kinds: ObjectKindContribution[];
  /** The PLUGIN-level planner (`ObjectModelContribution.batch`): every op
   *  of a batch on any of the five kinds, in one plan per frame — layers,
   *  masks, adjustments and smart objects share one stack. */
  batch(ops: readonly ObjectOp[]): Promise<ObjectWrite>;
  counters: ImageObjectCounters;
  /** Forget every cached stack (the document changed). */
  invalidate(): void;
  dispose(): void;
}

const ITEM_KINDS = new Set(["rectangle", "oval", "polygon"]);
const KEEP_SOURCE_CHARS = 2048;

/**
 * Build the five kinds over `host`. `codec` boots lazily (the wasm's CPU
 * codec — no GPU is ever requested here).
 */
export function createImageObjects(host: BundleHost, codec: () => Promise<ImageCodec>): ImageObjects {
  const counters: ImageObjectCounters = { treeReads: 0, placedReads: 0, sessionReads: 0, decodes: 0, encodes: 0, plans: 0 };
  /** Stacks by `frameId|marker`. A marker names a content-addressed
   *  record, so an entry never goes stale for its key — but the frame's
   *  image can change under an unchanged marker, so each entry is
   *  re-VALIDATED (one placed-image read) after every document change.
   *  Bounded; oldest first out. */
  const cache = new Map<string, FrameState>();
  const validated = new Set<string>();
  const CACHE_MAX = 32;
  /** Frames found to carry no placed image, until the document changes. */
  const notImages = new Set<string>();
  let treeMemo: Promise<SceneTreeNode[]> | null = null;

  const remember = (key: string, state: FrameState) => {
    cache.delete(key);
    cache.set(key, state);
    while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value!);
  };
  const invalidate = () => {
    validated.clear();
    notImages.clear();
    treeMemo = null;
  };
  const sub = host.document.onDidChange?.(() => invalidate());

  const tree = () => {
    if (!treeMemo) {
      counters.treeReads++;
      treeMemo = host.document.tree();
      // A tree is good for one turn of work; the next read sees edits.
      void treeMemo.finally(() => setTimeout(() => (treeMemo = null), 0));
    }
    return treeMemo;
  };

  /** Every page item that could hold an image, with its marker value. */
  async function frames(): Promise<Array<{ node: SceneTreeNode; marker: string | null; within: string[] }>> {
    const out: Array<{ node: SceneTreeNode; marker: string | null; within: string[] }> = [];
    const walk = (nodes: readonly SceneTreeNode[], within: string[]) => {
      for (const n of nodes) {
        const id = n.id as ElementId | null | undefined;
        const ownId = id && typeof id.id === "string" ? id.id : null;
        if (id && ownId && ITEM_KINDS.has(id.kind)) {
          const m = n.pluginMetadata?.find((e) => e.key === LABEL_KEY)?.value ?? null;
          out.push({ node: n, marker: m, within });
        }
        // Groups scope their members. Page / spread scopes never reach
        // `list`: the registry resolves them through `hostOf`.
        if (n.children) walk(n.children, ownId ? [...within, ownId] : within);
      }
    };
    walk(await tree(), []);
    return out;
  }

  async function findFrame(frameId: string) {
    return (await frames()).find((f) => (f.node.id as ElementId).id === frameId) ?? null;
  }

  function parseMarker(raw: string | null): ImageMarker | null {
    if (!raw) return null;
    try {
      const env = JSON.parse(raw) as { v?: number; data?: Partial<ImageMarker> };
      if (env.v !== 2 || !env.data?.record || !env.data.baked) return null;
      return env.data as ImageMarker;
    } catch {
      return null;
    }
  }

  async function load(frameId: string): Promise<FrameState> {
    const found = await findFrame(frameId);
    if (!found) throw new Refused(`no page item ${frameId} that can hold an image`);
    const key = frameKey(frameId, found.marker);
    const hit = cache.get(key);
    if (hit && validated.has(key)) return hit;
    if (notImages.has(frameId)) throw new Refused(`page item ${frameId} has no placed image`);
    counters.placedReads++;
    const placed = await host.assets.getPlacedImage(frameId);
    if (!placed) {
      notImages.add(frameId);
      throw new Refused(`page item ${frameId} has no placed image`);
    }
    const placedSha = await sha256Hex(placed.bytes);
    if (hit && hit.placedSha === placedSha) {
      validated.add(key);
      return hit;
    }
    const frame: Frame = { frameId, elementId: found.node.id as ElementId };
    const marker = parseMarker(found.marker);
    let state: FrameState | null = null;
    // The session's rule: restore the stored stack only when the frame
    // shows exactly what that revision baked (else it was changed outside
    // paged.image and opens flat).
    if (marker && marker.baked === placedSha) {
      counters.sessionReads++;
      const stored = await readSession(host.parts, marker.record);
      if (stored.ok) {
        const record = stored.record as ObjectRecord;
        state = {
          frame,
          stack: readPgil(stored.manifest, stored.buffers),
          marker,
          placedSha,
          sources: { ...(record.sources ?? {}) },
          pendingRender: new Set(record.pendingRender ?? []),
        };
      }
    }
    if (!state) {
      counters.decodes++;
      const d = (await codec()).decode(placed.bytes);
      state = {
        frame,
        stack: stackFromImage(d.width, d.height, d.rgba),
        marker: null,
        placedSha,
        sources: {},
        pendingRender: new Set(),
      };
    }
    remember(key, state);
    validated.add(key);
    return state;
  }

  const layerOf = (s: FrameState, layerId: number): PgilLayer => {
    const l = s.stack.layers.find((x) => x.id === layerId);
    if (!l) throw new Refused(`frame ${s.frame.frameId} has no layer ${layerId}`);
    return l;
  };

  // ------------------------------------------------------------ reads

  function readLayer(s: FrameState, l: PgilLayer, path: string): ObjectValue {
    const bounds = l.rect
      ? [l.rect.y, l.rect.x, l.rect.y + l.rect.h, l.rect.x + l.rect.w]
      : [0, 0, s.stack.height, s.stack.width];
    switch (path) {
      case "name": return value(l.name);
      case "visible": return value(l.visible);
      case "opacity": return value(Math.round(l.opacity * 100 * 1000) / 1000);
      case "blendMode": return value(l.blend.replace(/^compose\./, ""));
      case "order": return value(s.stack.layers.indexOf(l));
      case "locked": return value(l.locked);
      case "clipped": return value(l.clipped);
      case "kind": return value(l.kind);
      case "id": return value(l.id);
      case "hasMask": return value(l.mask !== null);
      case "bounds": return value(bounds);
      case "pixelSource": {
        const src = s.sources[String(l.id)];
        return src === undefined ? absent : value(src);
      }
      default: return refusedValue("unknownPath", `layer has no path "${path}"`);
    }
  }

  async function readContent(s: FrameState, l: PgilLayer, mask: boolean): Promise<ObjectValue> {
    const { width: W, height: H } = s.stack;
    let rgba: Uint8Array;
    if (mask) {
      if (!l.mask) return absent;
      const cov = l.maskRect ? expandMask(l, W, H) : l.mask;
      rgba = new Uint8Array(W * H * 4);
      for (let i = 0; i < W * H; i++) rgba.set([cov[i]!, cov[i]!, cov[i]!, 255], i * 4);
    } else {
      if (l.kind === "adjustment") return absent;
      if (l.depth !== 8) return refusedValue("failed", "reading 16-bit layer pixels is not supported here");
      rgba = l.rect ? expandBounded(l, W, H) : l.pixels;
    }
    counters.encodes++;
    return value((await codec()).encodePng(rgba, W, H));
  }

  async function get(address: Address, path: string): Promise<ObjectValue> {
    const a = parseImageAddress(address);
    if (!a) return refusedValue("unknownAddress", `${address} is not a paged.image address`);
    let s: FrameState;
    try {
      s = await load(a.frameId);
    } catch (err) {
      return refusedValue("unknownAddress", err instanceof Error ? err.message : String(err));
    }
    if (a.kind === "image") {
      switch (path) {
        case "width": return value(s.stack.width);
        case "height": return value(s.stack.height);
        case "layerCount": return value(s.stack.layers.length);
        case "revision": return value(s.marker?.rev ?? 0);
        case "pending": return value(s.marker?.pending === true);
        default: return refusedValue("unknownPath", `image has no path "${path}"`);
      }
    }
    const l = s.stack.layers.find((x) => x.id === a.layerId);
    if (!l) return refusedValue("unknownAddress", `frame ${a.frameId} has no layer ${a.layerId}`);
    switch (a.kind) {
      case "layer":
        return path === "content" ? readContent(s, l, false) : readLayer(s, l, path);
      case "mask":
        if (!l.mask) return refusedValue("unknownAddress", `layer ${l.id} has no mask`);
        if (path === "content") return readContent(s, l, true);
        if (path === "enabled") return value(l.maskEnabled);
        if (path === "bounds") {
          const r = l.maskRect;
          return value(r ? [r.y, r.x, r.y + r.h, r.x + r.w] : [0, 0, s.stack.height, s.stack.width]);
        }
        return refusedValue("unknownPath", `mask has no path "${path}"`);
      case "adjustment": {
        if (l.kind !== "adjustment" || !l.adjust) return refusedValue("unknownAddress", `layer ${l.id} is not an adjustment layer`);
        const v = readAdjust(l.adjust, path);
        return v === undefined ? refusedValue("unknownPath", `adjustment has no path "${path}"`) : value(v);
      }
      case "smartObject": {
        if (l.kind !== "smart" || !l.smart) return refusedValue("unknownAddress", `layer ${l.id} is not a smart object`);
        switch (path) {
          case "source": {
            const src = s.sources[`smart:${l.id}`];
            return src === undefined ? absent : value(src);
          }
          case "sourceWidth": return value(l.smart.width);
          case "sourceHeight": return value(l.smart.height);
          case "scale": return value(l.smart.scale);
          case "renderPending": return value(s.pendingRender.has(l.id));
          default: return refusedValue("unknownPath", `smartObject has no path "${path}"`);
        }
      }
    }
    return refusedValue("unknownPath", path);
  }

  // ------------------------------------------------------------ lists

  async function list(kind: ImageKind, within: string | undefined): Promise<Address[]> {
    let all = await frames();
    if (within !== undefined && within !== "doc") {
      const w = parseAddress(within);
      const id =
        w && (w.kind === "item" || w.kind === "frame")
          ? w.id
          : w && w.kind === "plugin"
            ? parseImageAddress(within)?.frameId
            : undefined;
      if (id === undefined) return [];
      all = all.filter((f) => (f.node.id as ElementId).id === id || f.within.includes(id));
    }
    const out: Address[] = [];
    for (const f of all) {
      const frameId = (f.node.id as ElementId).id as string;
      let s: FrameState;
      try {
        s = await load(frameId);
      } catch {
        continue; // not an image frame
      }
      if (kind === "image") {
        out.push(imageAddress(frameId));
        continue;
      }
      for (const l of [...s.stack.layers].reverse()) {
        const keep =
          kind === "layer" ||
          (kind === "mask" && l.mask !== null) ||
          (kind === "adjustment" && l.kind === "adjustment") ||
          (kind === "smartObject" && l.kind === "smart");
        if (keep) out.push(layerAddress(kind, frameId, l.id));
      }
    }
    return out;
  }

  // ------------------------------------------------------------ writes

  /** Pixels from an `asset` string or bytes, decoded by the CPU codec. */
  async function pixelsFrom(src: unknown): Promise<{ width: number; height: number; rgba: Uint8Array }> {
    let bytes: Uint8Array;
    if (src instanceof Uint8Array) bytes = src;
    else if (typeof src === "string" && src.startsWith("data:")) {
      const comma = src.indexOf(",");
      if (comma < 0 || !/;base64$/i.test(src.slice(0, comma))) throw new Refused("a data: URL source must be base64");
      const bin = atob(src.slice(comma + 1));
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    } else if (typeof src === "string" && /^[a-z]+:\/\//i.test(src)) {
      throw new Refused(
        "a network URL needs a fetch door this bundle does not have (manifest network: false); " +
          "pass the bytes, a data: URL or a placed image's page item",
      );
    } else if (typeof src === "string") {
      const a = parseAddress(src);
      const id = a && (a.kind === "item" || a.kind === "frame") ? a.id : null;
      if (!id) throw new Refused(`"${src}" is neither a data: URL nor a page item address`);
      counters.placedReads++;
      const placed = await host.assets.getPlacedImage(id);
      if (!placed) throw new Refused(`${src} has no placed image`);
      bytes = placed.bytes;
    } else throw new Refused("a pixel source is bytes, a data: URL or a page item address");
    counters.decodes++;
    try {
      return (await codec()).decode(bytes);
    } catch (err) {
      throw new Refused(`the source does not decode: ${err instanceof Error ? err.message : err}`);
    }
  }

  /** Replace a pixel layer's pixels "as is", centred (no resample: the
   *  scaling placements are GPU kernels). */
  function placeAsIs(s: FrameState, l: PgilLayer, img: { width: number; height: number; rgba: Uint8Array }) {
    const { width: W, height: H } = s.stack;
    const x0 = Math.floor((W - img.width) / 2);
    const y0 = Math.floor((H - img.height) / 2);
    const left = Math.max(0, x0), top = Math.max(0, y0);
    const right = Math.min(W, x0 + img.width), bottom = Math.min(H, y0 + img.height);
    const w = right - left, h = bottom - top;
    if (w <= 0 || h <= 0) throw new Refused("the replacement covers no pixel of the canvas");
    const px = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) {
      const sy = top - y0 + y;
      const sx = left - x0;
      px.set(img.rgba.subarray((sy * img.width + sx) * 4, (sy * img.width + sx + w) * 4), y * w * 4);
    }
    l.depth = 8;
    if (w === W && h === H) {
      l.pixels = px;
      l.rect = null;
    } else {
      l.pixels = px;
      l.rect = { x: left, y: top, w, h };
    }
  }

  function applyLayerSet(s: FrameState, l: PgilLayer, path: string, v: unknown) {
    switch (path) {
      case "name": l.name = v as string; return;
      case "visible": l.visible = v as boolean; return;
      case "opacity": l.opacity = Math.fround((v as number) / 100); return;
      case "blendMode": l.blend = `compose.${v as string}`; return;
      case "locked": l.locked = v as boolean; return;
      case "clipped": l.clipped = v as boolean; return;
      case "order": {
        if (s.stack.groups.length > 0) throw new Refused("reordering inside a grouped stack is the editor's (group runs must stay contiguous)");
        const to = v as number;
        if (to >= s.stack.layers.length) throw new Refused(`order ${to} is past the top (${s.stack.layers.length - 1})`);
        const activeId = s.stack.layers[s.stack.active]?.id;
        const from = s.stack.layers.indexOf(l);
        s.stack.layers.splice(from, 1);
        s.stack.layers.splice(to, 0, l);
        s.stack.active = Math.max(0, s.stack.layers.findIndex((x) => x.id === activeId));
        return;
      }
    }
    throw new Refused(`layer path "${path}" is not writable`);
  }

  async function replacePixels(s: FrameState, l: PgilLayer, src: unknown, label: string | null) {
    if (l.kind === "adjustment") throw new Refused("an adjustment layer has no pixels");
    if (l.locked) throw new Refused(`layer "${l.name}" is locked (its pixels are)`);
    const img = await pixelsFrom(src);
    if (l.kind === "smart") {
      replaceSmartSource(s, l, img, label);
      return;
    }
    placeAsIs(s, l, img);
    rememberSource(s, String(l.id), label);
  }

  function replaceSmartSource(s: FrameState, l: PgilLayer, img: { width: number; height: number; rgba: Uint8Array }, label: string | null) {
    l.smart = { width: img.width, height: img.height, scale: l.smart?.scale ?? 1, source: img.rgba };
    // The cached render is now older than the source; re-rendering is a
    // GPU resample, so it is recorded and done where a device exists.
    s.pendingRender.add(l.id);
    rememberSource(s, `smart:${l.id}`, label);
  }

  function rememberSource(s: FrameState, key: string, label: string | null) {
    if (label !== null && label.length <= KEEP_SOURCE_CHARS) s.sources[key] = label;
    else delete s.sources[key];
  }

  async function applyOp(kind: ImageKind, s: FrameState, op: ObjectOp) {
    if (op.op === "create") {
      const props = op.props ?? {};
      if (kind === "layer" || kind === "adjustment") {
        const l = newLayer(s, kind === "layer" ? "pixels" : "adjustment");
        for (const [p, v] of Object.entries(props)) {
          if (kind === "adjustment") {
            const why = writeAdjust(l.adjust!, p, v);
            if (why) throw new Refused(why);
          } else if (p === "content" || p === "pixelSource") {
            await replacePixels(s, l, v, p === "pixelSource" ? (v as string) : null);
          } else applyLayerSet(s, l, p, v);
        }
        return;
      }
      // A mask or a smart object is created ON a layer: `within` is it.
      const on = op.within ? parseImageAddress(op.within) : null;
      if (!on || on.layerId === null) throw new Refused(`create ${kind} needs within: the layer it goes on`);
      const l = layerOf(s, on.layerId);
      if (kind === "mask") {
        if (l.mask) throw new Refused(`layer "${l.name}" already has a mask`);
        // Reveal all (white); `content` in the same batch paints it.
        l.mask = new Uint8Array(s.stack.width * s.stack.height).fill(255);
        l.maskRect = null;
        l.maskEnabled = props.enabled === undefined ? true : (props.enabled as boolean);
        return;
      }
      if (kind === "smartObject") {
        if (l.kind !== "pixels") throw new Refused(`layer "${l.name}" is ${l.kind === "smart" ? "already a smart object" : "an adjustment layer"}`);
        if (l.depth !== 8) throw new Refused("a 16-bit layer becomes a smart object in the editor");
        // `make_smart`: the layer's own canvas pixels become the source.
        const canvas = l.rect ? expandBounded(l, s.stack.width, s.stack.height) : l.pixels;
        l.kind = "smart";
        l.pixels = canvas;
        l.rect = null;
        l.smart = { width: s.stack.width, height: s.stack.height, scale: 1, source: canvas };
        return;
      }
      throw new Refused(`create ${kind} is not supported`);
    }
    if (op.op === "invoke") throw new Refused("invoke is not a write");
    const a = parseImageAddress(op.address)!;
    if (a.kind === "image") throw new Refused("the image object has no writable paths");
    const l = layerOf(s, a.layerId!);
    if (op.op === "delete") {
      if (kind === "mask") {
        l.mask = null;
        l.maskRect = null;
        return;
      }
      if (kind !== "layer") throw new Refused(`delete a ${kind} by deleting its layer`);
      if (s.stack.layers.length === 1) throw new Refused("a document keeps at least one layer");
      const activeId = s.stack.layers[s.stack.active]?.id;
      s.stack.layers.splice(s.stack.layers.indexOf(l), 1);
      const at = s.stack.layers.findIndex((x) => x.id === activeId);
      s.stack.active = at >= 0 ? at : s.stack.layers.length - 1;
      delete s.sources[String(l.id)];
      return;
    }
    const { path, value: v } = op;
    switch (kind) {
      case "layer":
        if (path === "content") return replacePixels(s, l, v, null);
        if (path === "pixelSource") return replacePixels(s, l, v, v as string);
        return applyLayerSet(s, l, path, v);
      case "mask": {
        if (!l.mask) throw new Refused(`layer ${l.id} has no mask (addLayerMask first)`);
        if (path === "enabled") {
          l.maskEnabled = v as boolean;
          return;
        }
        if (path === "content") {
          const img = await pixelsFrom(v);
          const { width: W, height: H } = s.stack;
          if (img.width !== W || img.height !== H) throw new Refused(`a mask image must be ${W}×${H}`);
          const cov = new Uint8Array(W * H);
          for (let i = 0; i < W * H; i++) {
            const [r, g, b] = [img.rgba[i * 4]!, img.rgba[i * 4 + 1]!, img.rgba[i * 4 + 2]!];
            cov[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
          }
          l.mask = cov;
          l.maskRect = null;
          return;
        }
        throw new Refused(`mask path "${path}" is not writable`);
      }
      case "adjustment": {
        if (l.kind !== "adjustment" || !l.adjust) throw new Refused(`layer ${l.id} is not an adjustment layer`);
        l.adjust = widenAdjust(l.adjust);
        const why = writeAdjust(l.adjust, path, v);
        if (why) throw new Refused(why);
        return;
      }
      case "smartObject": {
        if (l.kind !== "smart") throw new Refused(`layer ${l.id} is not a smart object`);
        if (path !== "source") throw new Refused(`smartObject path "${path}" is not writable`);
        replaceSmartSource(s, l, await pixelsFrom(v), v as string);
        return;
      }
    }
  }

  function newLayer(s: FrameState, kind: "pixels" | "adjustment"): PgilLayer {
    const { width: W, height: H } = s.stack;
    const l: PgilLayer = {
      id: s.stack.nextId++,
      name: kind === "adjustment" ? "Adjustment" : `Layer ${s.stack.layers.length + 1}`,
      visible: true, locked: false, opacity: 1, blend: "compose.normal", clipped: false, group: null,
      maskEnabled: true, kind, depth: 8,
      // An adjustment layer's pixels are a transparent placeholder.
      pixels: new Uint8Array(W * H * 4), rect: null, blendGamma: 0, mask: null, maskRect: null,
      adjust: kind === "adjustment" ? defaultAdjust() : null, smart: null,
    };
    s.stack.layers.push(l);
    s.stack.active = s.stack.layers.length - 1;
    return l;
  }

  /** The kind an op writes: the created kind, or its address's. */
  function kindOfOp(op: ObjectOp): ImageKind | null {
    if (op.op === "create") {
      const k = op.kind.startsWith(KIND_PREFIX) ? op.kind.slice(KIND_PREFIX.length) : "";
      return k in KIND_ROWS ? (k as ImageKind) : null;
    }
    return op.op === "set" || op.op === "delete" ? (parseImageAddress(op.address)?.kind ?? null) : null;
  }

  /** Plan every op (on ANY of this plugin's kinds) into ONE write: one
   *  stack per frame, every op applied to it in batch order. */
  async function plan(ops: readonly ObjectOp[]): Promise<ObjectWrite> {
    counters.plans++;
    const byFrame = new Map<string, Array<{ kind: ImageKind; op: ObjectOp }>>();
    for (const op of ops) {
      let frameId: string | undefined;
      const kind = kindOfOp(op);
      if (!kind) return { kind: "rejected", reason: `${op.op} names no paged.image kind` };
      if (op.op === "create") {
        const w = op.within ? parseImageAddress(op.within) : null;
        // A mask / smart object create names the LAYER it goes on.
        const core = op.within && !w ? parseAddress(op.within) : null;
        frameId = w?.frameId ?? (core && (core.kind === "item" || core.kind === "frame") ? core.id : undefined);
        if (!frameId) return { kind: "rejected", reason: `create ${kind} needs within: an image or its frame` };
      } else if (op.op === "set" || op.op === "delete") {
        frameId = parseImageAddress(op.address)?.frameId;
        if (!frameId) return { kind: "rejected", reason: `${op.address} is not a paged.image address` };
      } else return { kind: "rejected", reason: `${op.op} is not a ${kind} write` };
      const list = byFrame.get(frameId) ?? [];
      list.push({ kind, op });
      byFrame.set(frameId, list);
    }
    const planned: Array<{ base: FrameState; next: FrameState }> = [];
    try {
      for (const [frameId, frameOps] of byFrame) {
        const base = await load(frameId);
        const next: FrameState = {
          ...base,
          stack: clonePgil(base.stack),
          sources: { ...base.sources },
          pendingRender: new Set(base.pendingRender),
        };
        for (const { kind, op } of frameOps) await applyOp(kind, next, op);
        planned.push({ base, next });
      }
    } catch (err) {
      return { kind: "rejected", reason: err instanceof Error ? err.message : String(err) };
    }
    return commitPlan(planned);
  }

  async function commitPlan(planned: Array<{ base: FrameState; next: FrameState }>): Promise<ObjectWrite> {
    const parts: PartChange[] = [];
    const mutations: Array<{ op: string; args: Record<string, unknown> }> = [];
    const labels: Array<{ frame: Frame; value: string }> = [];
    const have = new Set(await host.parts.list("px/"));
    const c = await codec();
    for (const { base, next } of planned) {
      const exported = writePgil(next.stack);
      const hashes = await Promise.all(exported.map(sha256Hex));
      exported.forEach((bytes, i) => {
        const p = blobPath(hashes[i]!);
        if (!have.has(p)) {
          have.add(p);
          parts.push({ path: p, bytes });
        }
      });
      const rgba = next.pendingRender.size === 0 ? trivialComposite(next.stack) : null;
      let baked = base.placedSha;
      if (rgba) {
        counters.encodes++;
        const png = c.encodePng(rgba, next.stack.width, next.stack.height);
        baked = await sha256Hex(png);
        mutations.push({ op: "replaceImageBytes", args: { elementId: next.frame.frameId, bytes: Array.from(png) } });
      }
      const record: ObjectRecord = {
        v: 1,
        width: next.stack.width,
        height: next.stack.height,
        manifest: hashes[0]!,
        buffers: hashes.slice(1),
        baked,
        ...(Object.keys(next.sources).length > 0 ? { sources: next.sources } : {}),
        ...(next.pendingRender.size > 0 ? { pendingRender: [...next.pendingRender].sort((a, b) => a - b) } : {}),
        ...(rgba ? {} : { pending: true }),
      };
      const recordBytes = new TextEncoder().encode(JSON.stringify(record));
      const rev = (base.marker?.rev ?? 0) + 1;
      // Content-addressed, so a revision a redo can still reach is never
      // overwritten by a different write that reuses its number.
      const path = objectRecordPath(next.frame.frameId, rev, (await sha256Hex(recordBytes)).slice(0, 16));
      parts.push({ path, bytes: recordBytes });
      const marker: ImageMarker = { owns: "pixels", rev, record: path, baked, ...(rgba ? {} : { pending: true }) };
      const labelValue = JSON.stringify({ v: 2, data: marker });
      labels.push({ frame: next.frame, value: labelValue });
      // The state this write leaves, under the marker it will carry —
      // unused if the commit never lands (its marker never appears).
      remember(frameKey(next.frame.frameId, escapeLabelValue(labelValue)), {
        ...next,
        marker,
        placedSha: baked,
      });
    }
    const [first, ...rest] = labels;
    if (!first) return { kind: "rejected", reason: "nothing to write" };
    // The registry sets the FIRST frame's marker; any further frame's
    // goes in the same batch (one undo step for all of them).
    for (const l of rest) {
      mutations.push({
        op: "setPluginMetadata",
        args: { elementId: l.frame.elementId, key: LABEL_KEY, value: escapeLabelValue(l.value), caller: PLUGIN_ID },
      });
    }
    // The canonical item address (`rectangle:<id>`), never the `frame:`
    // alias: the frame's element id carries its kind.
    const hostAddress = fromElementId(first.frame.elementId);
    return mutations.length > 0
      ? { kind: "both", mutations: mutations as never, parts, host: hostAddress, labelKey: LABEL_KEY, labelValue: first.value }
      : { kind: "state", parts, host: hostAddress, labelKey: LABEL_KEY, labelValue: first.value };
  }

  const kinds: ObjectKindContribution[] = (Object.keys(KIND_ROWS) as ImageKind[]).map((kind) => ({
    kind,
    title: KIND_TITLES[kind],
    schema: KIND_ROWS[kind],
    ...(KIND_CONTENT[kind] ? { content: KIND_CONTENT[kind] } : {}),
    // Async: the canonical address needs the frame's item kind, which
    // only the scene tree knows. Also how the registry scopes
    // `page:#n > plugin:media.paged.image/layer`.
    async hostOf(address: Address) {
      const a = parseImageAddress(address);
      const found = a ? await findFrame(a.frameId) : null;
      return found ? fromElementId(found.node.id as ElementId) : null;
    },
    list: (q) => list(kind, q.within),
    get,
    // The fallback for a registry without the plugin-level `batch`
    // (pre-0.2.43): one kind per plan.
    set: (address, path, v) => plan([{ op: "set", address, path, value: v }]),
    batch: (ops) => plan(ops),
  }));

  return {
    kinds,
    batch: plan,
    counters,
    invalidate,
    dispose() {
      sub?.dispose();
      invalidate();
    },
  };
}

const frameKey = (frameId: string, marker: string | null) => `${frameId}|${marker ?? ""}`;

/** `f/<frame>/r<rev>-<hash>.json` — the session store's layout, made
 *  content-addressed (its garbage collector reads both spellings). */
export const objectRecordPath = (frameId: string, rev: number, hash: string) =>
  `f/${frameId.replace(/[^A-Za-z0-9_-]/g, "_")}/r${rev}-${hash}.json`;

function expandBounded(l: PgilLayer, W: number, H: number): Uint8Array {
  const out = new Uint8Array(W * H * 4);
  const r = l.rect!;
  for (let y = 0; y < r.h; y++) {
    out.set(l.pixels.subarray(y * r.w * 4, (y + 1) * r.w * 4), ((r.y + y) * W + r.x) * 4);
  }
  return out;
}

function expandMask(l: PgilLayer, W: number, H: number): Uint8Array {
  const r = l.maskRect!;
  const out = new Uint8Array(W * H).fill(r.outside);
  for (let y = 0; y < r.h; y++) out.set(l.mask!.subarray(y * r.w, (y + 1) * r.w), (r.y + y) * W + r.x);
  return out;
}

const fullyTransparent = (px: Uint8Array) => {
  for (let i = 3; i < px.length; i += 4) if (px[i] !== 0) return false;
  return true;
};

/**
 * The composite when it is the IDENTITY (the engine's
 * `composite_is_trivial`): nothing contributes (a transparent canvas), or
 * exactly one plain, visible, 8-bit pixel layer — its pixels ARE the
 * composite. Anything that blends is a GPU kernel fold: null.
 */
export function trivialComposite(stack: PgilStack): Uint8Array | null {
  const { width: W, height: H } = stack;
  if (stack.groups.length > 0) return null;
  const live = stack.layers.filter(
    (l) => l.visible && l.opacity > 0 && !(l.kind !== "adjustment" && fullyTransparent(l.pixels)),
  );
  if (live.length === 0) return new Uint8Array(W * H * 4);
  if (live.length > 1) return null;
  const l = live[0]!;
  const plain =
    l.kind !== "adjustment" &&
    l.opacity >= 1 &&
    l.blend === "compose.normal" &&
    !(l.mask && l.maskEnabled) &&
    !l.clipped &&
    l.depth === 8;
  if (!plain) return null;
  return l.rect ? expandBounded(l, W, H) : l.pixels;
}
