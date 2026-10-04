/**
 * The protocol-66 host doors, typed here so the bundle compiles against
 * the published contract it pins (0.2.37, protocol 64/65) and uses each
 * door where the running host offers it. Every accessor answers `null`
 * when the door is absent, and every caller keeps its older path for
 * that case — nothing here is assumed.
 *
 * The shapes are plugin-api 0.2.38's (`SceneLayerSurface.submitImage`,
 * `PartsSurface.delete`, `DocumentSurface.onWillSave`,
 * `ShellSurface.enterEditContext`, `ToolsSurface`,
 * `WidgetSurface.ColorPicker`, `DocumentSurface.mutateWithBytes`). When
 * the pin moves to 0.2.38 these become the contract's own types and this
 * module shrinks to the capability probes.
 */
import type { ComponentType } from "react";
import type {
  BundleHost,
  Disposable,
  ElementId,
  MutationOutcome,
  SceneLayerSurface,
} from "@paged-media/plugin-api";

type Mutation = Parameters<BundleHost["document"]["mutate"]>[0];

export interface SceneImage {
  rgba: Uint8Array;
  width: number;
  height: number;
  /** `[x, y, w, h]` in frame-content points. */
  dest: [number, number, number, number];
}

export interface SceneImageTile {
  x: number;
  y: number;
  width: number;
  height: number;
  rgba: Uint8Array;
}

interface ImageSceneSurface {
  submitImage(elementId: string, image: SceneImage, options?: { transfer?: boolean }): Promise<void>;
  submitImageTiles(
    elementId: string,
    tiles: readonly SceneImageTile[],
    options?: { transfer?: boolean },
  ): Promise<void>;
}

/** The scene surface's image doors, or null on an older host. */
export function imageScene(surface: SceneLayerSurface): ImageSceneSurface | null {
  const s = surface as SceneLayerSurface & Partial<ImageSceneSurface>;
  return typeof s.submitImage === "function" && typeof s.submitImageTiles === "function"
    ? (s as ImageSceneSurface)
    : null;
}

/** Whether the host takes scene images as bytes (no `number[]`). */
export const binaryScene = (host: BundleHost) => host.supports("rendering.sceneLayer.binary@1");

/** `host.parts.delete`, or null when the host cannot delete parts. */
export function partsDelete(host: BundleHost): ((path: string) => Promise<boolean>) | null {
  const parts = host.parts as BundleHost["parts"] & { delete?: (path: string) => Promise<boolean> };
  if (!host.supports("storage.parts@2") || typeof parts.delete !== "function") return null;
  return (path) => parts.delete!(path);
}

/** Subscribe to the host's save of the document, when it announces one. */
export function onWillSave(
  host: BundleHost,
  listener: (e: { format: "paged" }) => void | Promise<void>,
): Disposable | null {
  const doc = host.document as BundleHost["document"] & {
    onWillSave?: (l: (e: { format: "paged" }) => void | Promise<void>) => Disposable;
  };
  if (!host.supports("document.onWillSave@1") || typeof doc.onWillSave !== "function") return null;
  return doc.onWillSave(listener);
}

/** Enter one of this bundle's edit contexts on `elementId`. Resolves
 *  false where the host cannot (the caller then says how to enter). */
export async function enterEditContext(
  host: BundleHost,
  type: string,
  elementId: ElementId,
): Promise<boolean> {
  const shell = host.shell as BundleHost["shell"] & {
    enterEditContext?: (type: string, id: ElementId) => Promise<boolean>;
  };
  if (!host.supports("shell.enterEditContext@1") || typeof shell.enterEditContext !== "function") {
    return false;
  }
  try {
    return await shell.enterEditContext(type, elementId);
  } catch {
    return false;
  }
}

export type ToolSettingValue = number | boolean | string;

interface ToolsSurface {
  settings(toolId: string): Readonly<Record<string, ToolSettingValue>>;
  onDidChangeSettings(
    toolId: string,
    listener: (settings: Readonly<Record<string, ToolSettingValue>>) => void,
  ): Disposable;
}

/** The host's tool-options values for this bundle's tools, or null. */
export function toolSettings(host: BundleHost): ToolsSurface | null {
  const tools = (host as BundleHost & { tools?: Partial<ToolsSurface> }).tools;
  if (
    !host.supports("tools.settings@1") ||
    !tools ||
    typeof tools.settings !== "function" ||
    typeof tools.onDidChangeSettings !== "function"
  ) {
    return null;
  }
  return tools as ToolsSurface;
}

export interface ColorPickerProps {
  /** `#rrggbb`. */
  value: string;
  onChange(next: string): void;
  onCommit?(next: string): void;
  disabled?: boolean;
  ariaLabel?: string;
}

/** The host's colour picker widget, or null (the panel's own then). */
export function hostColorPicker(host: BundleHost): ComponentType<ColorPickerProps> | null {
  const widgets = host.widgets as BundleHost["widgets"] & {
    ColorPicker?: ComponentType<ColorPickerProps>;
  };
  return host.supports("widgets.colorPicker@1") && widgets.ColorPicker ? widgets.ColorPicker : null;
}

/** Whether a mutation's image bytes can cross as bytes (no JSON array),
 *  so the size of a baked image is not bounded by the JSON lane. */
export const binaryMutations = (host: BundleHost) => host.supports("document.mutateBinary@1");

/**
 * `mutation` with `bytes` as the image of its first `replaceImageBytes`
 * (whose own `bytes` is `[]`): over the binary lane where the host has
 * one, else the bytes spliced in as the `number[]` the JSON lane takes.
 */
export async function mutateWithBytes(
  host: BundleHost,
  mutation: Mutation,
  bytes: Uint8Array,
): Promise<MutationOutcome> {
  const doc = host.document as BundleHost["document"] & {
    mutateWithBytes?: (m: Mutation, b: Uint8Array) => Promise<MutationOutcome>;
  };
  if (binaryMutations(host) && typeof doc.mutateWithBytes === "function") {
    return doc.mutateWithBytes(mutation, bytes);
  }
  return host.document.mutate(spliceBytes(mutation, bytes));
}

/** The JSON-lane form: the first `replaceImageBytes` gets the bytes. */
export function spliceBytes(mutation: Mutation, bytes: Uint8Array): Mutation {
  let done = false;
  const visit = (m: Mutation): Mutation => {
    const any = m as { op: string; args: Record<string, unknown> };
    if (!done && any.op === "replaceImageBytes") {
      done = true;
      return { ...any, args: { ...any.args, bytes: Array.from(bytes) } } as Mutation;
    }
    if (any.op === "batch") {
      const ops = (any.args.ops as Mutation[]).map(visit);
      return { ...any, args: { ...any.args, ops } } as Mutation;
    }
    return m;
  };
  return visit(mutation);
}
