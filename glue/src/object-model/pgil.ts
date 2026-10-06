// The layer manifest (`PGIL`) read and written in TypeScript, so the
// object model can read and change a stored layer stack WITHOUT the
// engine: a property write is a manifest edit, and needs neither the
// wasm's one-stack-per-realm slot (which the interactive session owns)
// nor a GPU.
//
// The format is the engine's (image-js/src/layers_persist.rs): a compact
// manifest plus the buffers it names by slot. Versions 1–3 are read;
// version 3 is written. `writePgil(readPgil(m, b))` reproduces the
// engine's own export byte for byte, including the slot order — the
// object-model spec proves that against the real wasm, so a drift in
// either codec fails there rather than in a document.

const MAGIC = [0x50, 0x47, 0x49, 0x4c]; // "PGIL"
export const PGIL_VERSION = 3;

export interface PgilRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PgilGroup {
  id: number;
  name: string;
  visible: boolean;
  /** 0–1, as the engine holds it (an f32). */
  opacity: number;
  /** The `compose.*` kernel id. */
  blend: string;
  passThrough: boolean;
  parent: number | null;
  maskEnabled: boolean;
  /** Canvas-sized coverage, one byte per pixel. */
  mask: Uint8Array | null;
}

export type PgilLayerKind = "pixels" | "adjustment" | "smart";

export interface PgilSmart {
  width: number;
  height: number;
  scale: number;
  /** Straight RGBA8, `width·height·4`. */
  source: Uint8Array;
}

export interface PgilLayer {
  /** Stable within the stack: survives reorder, persisted. */
  id: number;
  name: string;
  visible: boolean;
  locked: boolean;
  opacity: number;
  blend: string;
  clipped: boolean;
  group: number | null;
  maskEnabled: boolean;
  kind: PgilLayerKind;
  depth: 8 | 16;
  /** The layer's pixels: canvas-sized, or `rect`-sized when bounded. */
  pixels: Uint8Array;
  rect: PgilRect | null;
  /** Raw f32; 0 = none. */
  blendGamma: number;
  mask: Uint8Array | null;
  /** A bounded mask's rectangle and the coverage outside it. */
  maskRect: (PgilRect & { outside: number }) | null;
  /** An adjustment layer's `AdjustParams::to_wire` block. */
  adjust: Float32Array | null;
  smart: PgilSmart | null;
}

export interface PgilStack {
  version: number;
  width: number;
  height: number;
  active: number;
  nextId: number;
  groups: PgilGroup[];
  /** Bottom-first. */
  layers: PgilLayer[];
}

class Reader {
  private at = 0;
  private readonly view: DataView;
  constructor(private readonly b: Uint8Array) {
    this.view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  }
  private need(n: number): number {
    const at = this.at;
    if (at + n > this.b.length) throw new Error("layer manifest is truncated");
    this.at += n;
    return at;
  }
  u8 = () => this.b[this.need(1)]!;
  u32 = () => this.view.getUint32(this.need(4), true);
  i32 = () => this.view.getInt32(this.need(4), true);
  i64 = () => Number(this.view.getBigInt64(this.need(8), true));
  f32 = () => this.view.getFloat32(this.need(4), true);
  str = () => {
    const n = this.u32();
    const at = this.need(n);
    return new TextDecoder("utf-8", { fatal: true }).decode(this.b.subarray(at, at + n));
  };
  bytes = (n: number) => {
    const at = this.need(n);
    return this.b.subarray(at, at + n);
  };
}

class Writer {
  private chunks: number[] = [];
  private scratch = new DataView(new ArrayBuffer(8));
  private push(n: number) {
    for (let i = 0; i < n; i++) this.chunks.push(this.scratch.getUint8(i));
  }
  raw(b: ArrayLike<number>) {
    for (let i = 0; i < b.length; i++) this.chunks.push(b[i]!);
  }
  u8 = (v: number) => this.chunks.push(v & 0xff);
  u32 = (v: number) => (this.scratch.setUint32(0, v, true), this.push(4));
  i32 = (v: number) => (this.scratch.setInt32(0, v, true), this.push(4));
  i64 = (v: number) => (this.scratch.setBigInt64(0, BigInt(v), true), this.push(8));
  f32 = (v: number) => (this.scratch.setFloat32(0, v, true), this.push(4));
  str = (s: string) => {
    const b = new TextEncoder().encode(s);
    this.u32(b.length);
    this.raw(b);
  };
  done = () => Uint8Array.from(this.chunks);
}

/** Decode a manifest and resolve its slots against `buffers`. Throws a
 *  reason (never a half-read stack). */
export function readPgil(manifest: Uint8Array, buffers: readonly Uint8Array[]): PgilStack {
  const r = new Reader(manifest);
  for (const m of MAGIC) if (r.u8() !== m) throw new Error("not a layer manifest");
  const version = r.u32();
  if (version < 1 || version > PGIL_VERSION) {
    throw new Error(`layer manifest version ${version} (this reader reads 1–${PGIL_VERSION})`);
  }
  const width = r.u32();
  const height = r.u32();
  const active = r.u32();
  const nextId = r.u32();
  const buf = (slot: number) => {
    const b = buffers[slot];
    if (!b) throw new Error(`layer manifest: no buffer ${slot}`);
    return b;
  };
  const opt = (v: number) => (v < 0 ? null : v);
  const groups: PgilGroup[] = [];
  for (let n = r.u32(), i = 0; i < n; i++) {
    const id = r.u32();
    const name = r.str();
    const visible = r.u8() !== 0;
    const opacity = r.f32();
    const blend = r.str();
    const passThrough = r.u8() !== 0;
    const parent = opt(r.i64());
    let maskEnabled = true;
    let mask: Uint8Array | null = null;
    if (version >= 2) {
      maskEnabled = r.u8() !== 0;
      const slot = opt(r.i64());
      mask = slot === null ? null : buf(slot);
    }
    groups.push({ id, name, visible, opacity, blend, passThrough, parent, maskEnabled, mask });
  }
  const layers: PgilLayer[] = [];
  for (let n = r.u32(), i = 0; i < n; i++) {
    const id = r.u32();
    const name = r.str();
    const visible = r.u8() !== 0;
    const locked = r.u8() !== 0;
    const opacity = r.f32();
    const blend = r.str();
    const clipped = r.u8() !== 0;
    const group = opt(r.i64());
    const maskEnabled = r.u8() !== 0;
    const kindCode = r.u8();
    const depth = r.u8();
    if (depth !== 8 && depth !== 16) throw new Error(`layer manifest: depth ${depth}`);
    const pixels = buf(r.u32());
    let rect: PgilRect | null = null;
    let blendGamma = 0;
    if (version >= 3) {
      const x = r.i32(), y = r.i32(), w = r.u32(), h = r.u32();
      rect = x >= 0 ? { x, y, w, h } : null;
      blendGamma = r.f32();
    }
    const maskSlot = opt(r.i64());
    let maskRect: PgilLayer["maskRect"] = null;
    if (version >= 3) {
      const x = r.i32(), y = r.i32(), w = r.u32(), h = r.u32(), outside = r.u8();
      maskRect = x >= 0 ? { x, y, w, h, outside } : null;
    }
    const mask = maskSlot === null ? null : buf(maskSlot);
    let kind: PgilLayerKind;
    let adjust: Float32Array | null = null;
    let smart: PgilSmart | null = null;
    if (kindCode === 0) kind = "pixels";
    else if (kindCode === 1) {
      kind = "adjustment";
      const count = r.u32();
      adjust = new Float32Array(count);
      for (let k = 0; k < count; k++) adjust[k] = r.f32();
    } else if (kindCode === 2) {
      kind = "smart";
      const sw = r.u32(), sh = r.u32(), scale = r.f32();
      smart = { width: sw, height: sh, scale, source: buf(r.u32()) };
    } else throw new Error(`layer manifest: layer kind ${kindCode}`);
    layers.push({
      id, name, visible, locked, opacity, blend, clipped, group, maskEnabled, kind,
      depth, pixels, rect, blendGamma, mask, maskRect, adjust, smart,
    });
  }
  return { version, width, height, active, nextId, groups, layers };
}

/** Encode a stack as the engine's version-3 export: the manifest first,
 *  then the buffers, slots assigned in the engine's order (group masks,
 *  then per layer: pixels, mask, smart source). */
export function writePgil(stack: PgilStack): Uint8Array[] {
  const w = new Writer();
  const buffers: Uint8Array[] = [];
  const slot = (b: Uint8Array) => (buffers.push(b), buffers.length - 1);
  w.raw(MAGIC);
  w.u32(PGIL_VERSION);
  w.u32(stack.width);
  w.u32(stack.height);
  w.u32(stack.active);
  w.u32(stack.nextId);
  w.u32(stack.groups.length);
  for (const g of stack.groups) {
    w.u32(g.id);
    w.str(g.name);
    w.u8(g.visible ? 1 : 0);
    w.f32(g.opacity);
    w.str(g.blend);
    w.u8(g.passThrough ? 1 : 0);
    w.i64(g.parent ?? -1);
    w.u8(g.maskEnabled ? 1 : 0);
    w.i64(g.mask ? slot(g.mask) : -1);
  }
  w.u32(stack.layers.length);
  for (const l of stack.layers) {
    w.u32(l.id);
    w.str(l.name);
    w.u8(l.visible ? 1 : 0);
    w.u8(l.locked ? 1 : 0);
    w.f32(l.opacity);
    w.str(l.blend);
    w.u8(l.clipped ? 1 : 0);
    w.i64(l.group ?? -1);
    w.u8(l.maskEnabled ? 1 : 0);
    w.u8(l.kind === "pixels" ? 0 : l.kind === "adjustment" ? 1 : 2);
    w.u8(l.depth);
    w.u32(slot(l.pixels));
    if (l.rect) {
      w.i32(l.rect.x);
      w.i32(l.rect.y);
      w.u32(l.rect.w);
      w.u32(l.rect.h);
    } else {
      w.i32(-1);
      w.i32(-1);
      w.u32(0);
      w.u32(0);
    }
    w.f32(l.blendGamma);
    w.i64(l.mask ? slot(l.mask) : -1);
    if (l.mask && l.maskRect) {
      w.i32(l.maskRect.x);
      w.i32(l.maskRect.y);
      w.u32(l.maskRect.w);
      w.u32(l.maskRect.h);
      w.u8(l.maskRect.outside);
    } else {
      w.i32(-1);
      w.i32(-1);
      w.u32(0);
      w.u32(0);
      w.u8(0);
    }
    if (l.kind === "adjustment") {
      const a = l.adjust ?? new Float32Array(0);
      w.u32(a.length);
      for (const v of a) w.f32(v);
    } else if (l.kind === "smart") {
      const s = l.smart!;
      w.u32(s.width);
      w.u32(s.height);
      w.f32(s.scale);
      w.u32(slot(s.source));
    }
  }
  return [w.done(), ...buffers];
}

/** A deep-enough copy to edit without touching the original (buffers
 *  are shared: an edit REPLACES a buffer, never writes into one). */
export function clonePgil(s: PgilStack): PgilStack {
  return {
    ...s,
    groups: s.groups.map((g) => ({ ...g })),
    layers: s.layers.map((l) => ({
      ...l,
      rect: l.rect ? { ...l.rect } : null,
      maskRect: l.maskRect ? { ...l.maskRect } : null,
      adjust: l.adjust ? l.adjust.slice() : null,
      smart: l.smart ? { ...l.smart } : null,
    })),
  };
}

/** The engine's one-layer stack over an image (`LayerStack::from_image`):
 *  layer id 1, "Background", next id 2. */
export function stackFromImage(width: number, height: number, rgba: Uint8Array): PgilStack {
  return {
    version: PGIL_VERSION,
    width,
    height,
    active: 0,
    nextId: 2,
    groups: [],
    layers: [
      {
        id: 1, name: "Background", visible: true, locked: false, opacity: 1,
        blend: "compose.normal", clipped: false, group: null, maskEnabled: true,
        kind: "pixels", depth: 8, pixels: rgba, rect: null, blendGamma: 0,
        mask: null, maskRect: null, adjust: null, smart: null,
      },
    ],
  };
}
