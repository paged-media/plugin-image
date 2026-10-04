/**
 * TOOL OPTIONS in the host's tool-options bar: brush, pencil, eraser,
 * magic wand, paint bucket and gradient declare their fields, and the
 * values the host holds flow into the session (`tools.settings@1`).
 *
 * Declared only where the host can report the values back: a host
 * without the door would show controls that change nothing. The panel
 * keeps its own controls either way (the host offers no write-back, so
 * a change made in the panel is not reflected in the bar).
 */
import type { BundleHost, Disposable } from "@paged-media/plugin-api";
import { GRADIENT_KINDS, type GradientKind } from "./engine";
import { toolSettings, type ToolSettingValue } from "./host66";
import type { ImageSession } from "./session";

type ToolOptionField =
  | { kind: "number"; key: string; label: string; min?: number; max?: number; step?: number; unit?: string }
  | { kind: "toggle"; key: string; label: string }
  | { kind: "select"; key: string; label: string; options: Array<{ value: string; label: string }> };

const pct = (key: string, label: string): ToolOptionField => ({
  kind: "number",
  key,
  label,
  min: 0,
  max: 100,
  step: 1,
  unit: "%",
});

const SIZE: ToolOptionField = { kind: "number", key: "size", label: "Size", min: 1, max: 2000, step: 1, unit: "px" };
const WAND: ToolOptionField[] = [
  { kind: "number", key: "tolerance", label: "Tolerance", min: 0, max: 255, step: 1 },
  { kind: "toggle", key: "contiguous", label: "Contiguous" },
];

/** The fields each tool declares, by tool id. */
export function toolOptionFields(ids: {
  brush: string;
  pencil: string;
  eraser: string;
  wand: string;
  bucket: string;
  gradient: string;
}): Record<string, ToolOptionField[]> {
  return {
    [ids.brush]: [SIZE, pct("hardness", "Hardness"), pct("opacity", "Opacity"), pct("flow", "Flow")],
    [ids.pencil]: [SIZE, pct("opacity", "Opacity")],
    [ids.eraser]: [SIZE, pct("hardness", "Hardness"), pct("opacity", "Opacity")],
    [ids.wand]: WAND,
    [ids.bucket]: WAND,
    [ids.gradient]: [
      {
        kind: "select",
        key: "kind",
        label: "Gradient",
        options: GRADIENT_KINDS.map((k) => ({ value: k, label: k })),
      },
    ],
  };
}

/** Apply one tool's settings to the session (keys it does not know are
 *  ignored; a value of the wrong type is ignored, not coerced). */
export function applyToolSettings(
  session: ImageSession,
  kind: "brush" | "wand" | "gradient",
  v: Readonly<Record<string, ToolSettingValue>>,
): void {
  const num = (k: string) => (typeof v[k] === "number" ? (v[k] as number) : undefined);
  if (kind === "brush") {
    const p: Parameters<ImageSession["setBrushParams"]>[0] = {};
    if (num("size") !== undefined) p.size = Math.max(1, num("size")!);
    if (num("hardness") !== undefined) p.hardness = num("hardness")! / 100;
    if (num("opacity") !== undefined) p.opacity = num("opacity")! / 100;
    if (num("flow") !== undefined) p.flow = num("flow")! / 100;
    if (Object.keys(p).length) session.setBrushParams(p);
  } else if (kind === "wand") {
    const o: Parameters<ImageSession["setWandOptions"]>[0] = {};
    if (num("tolerance") !== undefined) o.tolerance = num("tolerance")!;
    if (typeof v.contiguous === "boolean") o.contiguous = v.contiguous;
    if (Object.keys(o).length) session.setWandOptions(o);
  } else if (typeof v.kind === "string" && (GRADIENT_KINDS as string[]).includes(v.kind)) {
    session.setGradientKind(v.kind as GradientKind);
  }
}

/** Read each tool's current settings into the session and follow its
 *  changes. Null when the host cannot report them. */
export function bindToolSettings(
  host: BundleHost,
  session: ImageSession,
  tools: Record<string, "brush" | "wand" | "gradient">,
): Disposable | null {
  const surface = toolSettings(host);
  if (!surface) return null;
  const subs = Object.entries(tools).map(([id, kind]) => {
    applyToolSettings(session, kind, surface.settings(id));
    return surface.onDidChangeSettings(id, (v) => applyToolSettings(session, kind, v));
  });
  return { dispose: () => subs.forEach((s) => s.dispose()) };
}
