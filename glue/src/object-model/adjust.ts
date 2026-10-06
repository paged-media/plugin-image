// An adjustment layer's parameters as object-model paths, read from and
// written to the block the engine stores (`AdjustParams::to_wire`).
//
// Nothing here is a hand list: the rows come from
// `generated/image-schema.json`, which image-js/tests/object_schema.rs
// emits from the kernels' `ParamsLayout` and the wire layout, and checks
// against both (every kernel field bound or internal, every wire slot
// covered once and live). Regenerate it there.

import type { PropertySchema, ValueType } from "@paged-media/plugin-api";

import generated from "./generated/image-schema.json";

export interface AdjustRow {
  path: string;
  kernel: string | null;
  field: string | null;
  component?: number;
  offset: number;
  type: "number" | "bool" | "lut";
  default: number | boolean | null;
  /** `lut`: how many 256-entry tables, and the slot that gates them. */
  tables?: number;
  gate?: number;
  /** A value the engine keeps only while this slot is 1. */
  gatedBy?: number;
}

export const IMAGE_SCHEMA = generated as {
  generatedBy: string;
  wireLength: number;
  kernels: string[];
  blendModes: string[];
  adjust: AdjustRow[];
};

export const ADJUST_ROWS: readonly AdjustRow[] = IMAGE_SCHEMA.adjust;
const BY_PATH = new Map(ADJUST_ROWS.map((r) => [r.path, r]));

const LUT_TABLE: ValueType = { kind: "list", of: { kind: "number", integer: true }, maxItems: 256 };

/** The schema rows (ADR 132 shape) for the `adjustment` kind. */
export function adjustSchemaRows(): PropertySchema[] {
  return ADJUST_ROWS.map((r): PropertySchema => {
    const gate = r.gatedBy !== undefined ? ADJUST_ROWS.find((g) => g.offset === r.gatedBy)?.path : undefined;
    const source = r.kernel
      ? `kernel ${r.kernel}${r.field ? `.${r.field}${r.component !== undefined ? `[${r.component}]` : ""}` : " (stage gate)"}`
      : "the CPU curves pass";
    if (r.type === "lut") {
      return {
        path: r.path,
        type: r.tables === 1 ? LUT_TABLE : { kind: "list", of: LUT_TABLE, maxItems: r.tables },
        nullable: true,
        default: null,
        summary: `${r.tables === 1 ? "A 256-entry tone table" : `${r.tables} tone tables (red, green, blue)`}; null = off. From ${source}.`,
      };
    }
    return {
      path: r.path,
      type: r.type === "bool" ? { kind: "bool" } : { kind: "number" },
      default: r.default,
      summary: `From ${source}${gate ? `; takes effect while ${gate} is true` : ""}.`,
    };
  });
}

/** Read one path from a stored block (a short v1/v2 block reads the
 *  missing slots as their defaults, as the engine does). */
export function readAdjust(block: Float32Array, path: string): unknown {
  const r = BY_PATH.get(path);
  if (!r) return undefined;
  const at = (i: number, fallback: number) => (i < block.length ? block[i]! : fallback);
  if (r.type === "lut") {
    if (at(r.gate!, 0) === 0) return null;
    const tables: number[][] = [];
    for (let t = 0; t < r.tables!; t++) {
      tables.push(Array.from(block.subarray(r.offset + t * 256, r.offset + (t + 1) * 256)));
    }
    return r.tables === 1 ? tables[0] : tables;
  }
  const dflt = r.type === "bool" ? (r.default ? 1 : 0) : Number(r.default);
  const v = at(r.offset, dflt);
  return r.type === "bool" ? v !== 0 : v;
}

/** The block widened to the current wire length (a short stored block
 *  keeps its values; new slots take their defaults). */
export function widenAdjust(block: Float32Array): Float32Array {
  if (block.length === IMAGE_SCHEMA.wireLength) return block.slice();
  const out = defaultAdjust();
  out.set(block.subarray(0, Math.min(block.length, out.length)));
  return out;
}

export function defaultAdjust(): Float32Array {
  const out = new Float32Array(IMAGE_SCHEMA.wireLength);
  for (const r of ADJUST_ROWS) {
    if (r.type === "number") out[r.offset] = Number(r.default);
    else if (r.type === "bool") out[r.offset] = r.default ? 1 : 0;
  }
  return out;
}

/** Write one path into a (full-length) block; a reason when the value
 *  does not fit the row. */
export function writeAdjust(block: Float32Array, path: string, value: unknown): string | null {
  const r = BY_PATH.get(path);
  if (!r) return `no adjustment parameter "${path}"`;
  if (r.type === "lut") {
    if (value === null) {
      block[r.gate!] = 0;
      block.fill(0, r.offset, r.offset + 256 * r.tables!);
      return null;
    }
    const tables = (r.tables === 1 ? [value] : value) as unknown[];
    if (!Array.isArray(tables) || tables.length !== r.tables) return `${path} takes ${r.tables} table(s)`;
    for (const [t, table] of tables.entries()) {
      if (!Array.isArray(table) || table.length !== 256) return `${path}: each table has 256 entries`;
      for (const [i, v] of table.entries()) {
        if (!Number.isInteger(v) || (v as number) < 0 || (v as number) > 255) return `${path}[${t}][${i}] must be 0–255`;
        block[r.offset + t * 256 + i] = v as number;
      }
    }
    block[r.gate!] = 1;
    return null;
  }
  if (r.type === "bool") {
    if (typeof value !== "boolean") return `${path} must be a boolean`;
    block[r.offset] = value ? 1 : 0;
    return null;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) return `${path} must be a number`;
  block[r.offset] = value;
  return null;
}
