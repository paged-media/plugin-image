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

// ADR 323 (object-model design §4 "Panels") — the image frame's layer
// stack as a SCHEMA panel of property rows. The host renders each row with
// its schema-driven `PropertyField` over `host.objects` (type from the
// layer / mask / adjustment / smartObject schema row, one undo step per
// commit, "Bind to data…" on every field). The bundle declares fields,
// never widgets; the existing image panel keeps its tool and preview UI.
//
// A layer list (publishObjectList over `<selected frame> > layer`) picks
// ONE layer; its mask, adjustment and smart-object objects share the
// layer's id, so their addresses are derived from the picked layer and
// published as bindings, with booleans gating the sections that apply.

import type {
  BundleHost,
  Disposable,
  PanelSchema,
  PanelSchemaPropertyRow,
  PanelSchemaRow,
  PropertyFieldAddress,
} from "@paged-media/plugin-api";
import { propertyRows, publishObjectList } from "@paged-media/plugin-sdk";

import { KIND_PREFIX, PLUGIN_ID, parseImageAddress, layerAddress } from "../object-model/image-objects";

export const PROPERTIES_PANEL_ID = `${PLUGIN_ID}.panel.properties`;

const LAYER = `${KIND_PREFIX}layer`;
const MASK = `${KIND_PREFIX}mask`;
const ADJUSTMENT = `${KIND_PREFIX}adjustment`;
const SMART = `${KIND_PREFIX}smartObject`;

/** Published bindings the list, the property rows and the gates share. */
export const BIND = {
  layers: "image.properties.layers",
  layer: "image.properties.layer",
  mask: "image.properties.mask",
  adjustment: "image.properties.adjustment",
  smartObject: "image.properties.smartObject",
  hasMask: "image.properties.hasMask",
  isAdjustment: "image.properties.isAdjustment",
  isSmart: "image.properties.isSmart",
} as const;

/** One declared field: a schema row of `kind`, shown in section `group`. */
export interface ImagePanelField {
  kind: "layer" | "mask" | "adjustment" | "smartObject";
  path: string;
  group: string;
}

/**
 * The declared panel fields: EVERY writable row of the layer, mask, smart
 * object and adjustment kinds (a spec holds this to the schemas), grouped
 * into sections. Adjustment parameters stay flat scalar rows (the per-range
 * hue/saturation triples, the 3×4 channel mixer, per-channel levels and
 * colour balance are indexed scalar paths of the generated schema); the two
 * curves rows are list values: `curves.composite` one 256-entry table,
 * `curves.channels` three (red, green, blue).
 */
export const IMAGE_PANEL_FIELDS: ReadonlyArray<ImagePanelField> = [
  ...["name", "visible", "opacity", "blendMode", "locked", "clipped", "order", "pixelSource"].map((path) => ({ kind: "layer" as const, path, group: "Layers" })),
  { kind: "mask", path: "enabled", group: "Mask" },
  { kind: "smartObject", path: "source", group: "Smart object" },
  ...[
    "exposure.ev",
    "brightnessContrast.brightness",
    "brightnessContrast.contrast",
    "saturation.sat",
    "vibrance.vibrance",
    "whiteBalance.temp",
    "whiteBalance.tint",
    "levels.inBlack",
    "levels.inWhite",
    "levels.gamma",
    "levels.outBlack",
    "levels.outWhite",
  ].map((path) => ({ kind: "adjustment" as const, path, group: "Tone" })),
  ...[
    "hueSaturation.master[0]",
    "hueSaturation.master[1]",
    "hueSaturation.master[2]",
    "hueSaturation.colorize[0]",
    "hueSaturation.colorize[1]",
    "hueSaturation.colorize[2]",
    "hueSaturation.colorize[3]",
  ].map((path) => ({ kind: "adjustment" as const, path, group: "Hue/saturation" })),
  ...[
    "hueSaturation.ranges[0][0]",
    "hueSaturation.ranges[0][1]",
    "hueSaturation.ranges[0][2]",
    "hueSaturation.ranges[1][0]",
    "hueSaturation.ranges[1][1]",
    "hueSaturation.ranges[1][2]",
    "hueSaturation.ranges[2][0]",
    "hueSaturation.ranges[2][1]",
    "hueSaturation.ranges[2][2]",
    "hueSaturation.ranges[3][0]",
    "hueSaturation.ranges[3][1]",
    "hueSaturation.ranges[3][2]",
    "hueSaturation.ranges[4][0]",
    "hueSaturation.ranges[4][1]",
    "hueSaturation.ranges[4][2]",
    "hueSaturation.ranges[5][0]",
    "hueSaturation.ranges[5][1]",
    "hueSaturation.ranges[5][2]",
  ].map((path) => ({ kind: "adjustment" as const, path, group: "Hue/saturation ranges" })),
  ...[
    "colorBalance.shCr",
    "colorBalance.shMg",
    "colorBalance.shYb",
    "colorBalance.midCr",
    "colorBalance.midMg",
    "colorBalance.midYb",
    "colorBalance.hiCr",
    "colorBalance.hiMg",
    "colorBalance.hiYb",
  ].map((path) => ({ kind: "adjustment" as const, path, group: "Colour balance" })),
  ...[
    "blackWhite.enabled",
    "blackWhite.reds",
    "blackWhite.yellows",
    "blackWhite.greens",
    "blackWhite.cyans",
    "blackWhite.blues",
    "blackWhite.magentas",
  ].map((path) => ({ kind: "adjustment" as const, path, group: "Black & white" })),
  ...[
    "channelMixer.rr",
    "channelMixer.rg",
    "channelMixer.rb",
    "channelMixer.rc",
    "channelMixer.gr",
    "channelMixer.gg",
    "channelMixer.gb",
    "channelMixer.gc",
    "channelMixer.br",
    "channelMixer.bg",
    "channelMixer.bb",
    "channelMixer.bc",
  ].map((path) => ({ kind: "adjustment" as const, path, group: "Channel mixer" })),
  ...[
    "levelsRgb.rInBlack",
    "levelsRgb.rInWhite",
    "levelsRgb.rGamma",
    "levelsRgb.gInBlack",
    "levelsRgb.gInWhite",
    "levelsRgb.gGamma",
    "levelsRgb.bInBlack",
    "levelsRgb.bInWhite",
    "levelsRgb.bGamma",
  ].map((path) => ({ kind: "adjustment" as const, path, group: "Levels per channel" })),
  ...[
    "photoFilter.density",
    "photoFilter.fr",
    "photoFilter.fg",
    "photoFilter.fb",
    "photoFilter.preserve",
  ].map((path) => ({ kind: "adjustment" as const, path, group: "Photo filter" })),
  ...["curves.composite", "curves.channels"].map((path) => ({ kind: "adjustment" as const, path, group: "Curves" })),
  ...[
    "gaussianBlur.sigma",
    "unsharp.amount",
    "hueRotate.degrees",
    "posterize.enabled",
    "posterize.levels",
    "threshold.enabled",
    "threshold.threshold",
    "invert.enabled",
  ].map((path) => ({ kind: "adjustment" as const, path, group: "Filters" })),
];

const pathsOf = (group: string) => IMAGE_PANEL_FIELDS.filter((f) => f.group === group).map((f) => f.path);

const HSL = ["Hue", "Saturation", "Lightness"];
const RANGES = ["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas"];
const RGB = ["Red", "Green", "Blue"];
const MIX_SOURCE: Record<string, string> = { r: "red", g: "green", b: "blue", c: "constant" };
const TONES: Record<string, string> = { sh: "Shadows", mid: "Midtones", hi: "Highlights" };
const BALANCE: Record<string, string> = { Cr: "cyan–red", Mg: "magenta–green", Yb: "yellow–blue" };

const LABELS: Readonly<Record<string, string>> = {
  ...Object.fromEntries(HSL.map((l, i) => [`hueSaturation.master[${i}]`, l])),
  "hueSaturation.colorize[0]": "Colorize",
  ...Object.fromEntries(HSL.map((l, i) => [`hueSaturation.colorize[${i + 1}]`, `Colorize ${l.toLowerCase()}`])),
  ...Object.fromEntries(
    RANGES.flatMap((r, i) => HSL.map((l, j) => [`hueSaturation.ranges[${i}][${j}]`, `${r} ${l.toLowerCase()}`])),
  ),
  ...Object.fromEntries(
    Object.entries(TONES).flatMap(([k, t]) =>
      Object.entries(BALANCE).map(([b, l]) => [`colorBalance.${k}${b}`, `${t} ${l}`]),
    ),
  ),
  ...Object.fromEntries(
    RGB.flatMap((out) =>
      Object.entries(MIX_SOURCE).map(([s, l]) => [`channelMixer.${out[0]!.toLowerCase()}${s}`, `${out} from ${l}`]),
    ),
  ),
  ...Object.fromEntries(
    RGB.flatMap((c) =>
      [
        ["InBlack", "input black"],
        ["InWhite", "input white"],
        ["Gamma", "gamma"],
      ].map(([k, l]) => [`levelsRgb.${c[0]!.toLowerCase()}${k}`, `${c} ${l}`]),
    ),
  ),
  "photoFilter.fr": "Filter red",
  "photoFilter.fg": "Filter green",
  "photoFilter.fb": "Filter blue",
  "photoFilter.preserve": "Preserve luminosity",
  "curves.composite": "Composite curve (256 levels)",
  "curves.channels": "Channel curves (red, green, blue)",
  order: "Stack position",
  pixelSource: "Replace pixels",
};

/** Per-row presentation. The curves rows keep the typed list field (a
 *  number list; the channel tables as their JSON) rather than an items
 *  editor: both are nullable (null = curves off), and an items editor
 *  cannot start a list from null. */
const ROW_OPTIONS: Readonly<Record<string, Omit<PanelSchemaPropertyRow, "field" | "path" | "address">>> = {
  blendMode: { style: "select" },
};

const layerList: PanelSchemaRow = {
  widget: "paged.list",
  list: {
    items: { kind: "binding", bind: BIND.layers },
    labelField: "name",
    secondaryField: "secondary",
    selectionBinding: BIND.layer,
  },
};

/** The adjustment sections, in panel order. */
const ADJUSTMENT_GROUPS = [
  "Tone",
  "Hue/saturation",
  "Hue/saturation ranges",
  "Colour balance",
  "Black & white",
  "Channel mixer",
  "Levels per channel",
  "Photo filter",
  "Curves",
  "Filters",
] as const;

const rowsOf = (kind: string, group: string, address: PropertyFieldAddress) =>
  propertyRows(kind, pathsOf(group), address, {
    labels: LABELS,
    row: (p) => ROW_OPTIONS[p] ?? {},
  });

export const IMAGE_PROPERTIES_PANEL: PanelSchema = {
  id: PROPERTIES_PANEL_ID,
  title: "Image properties",
  icon: "panel-layers",
  defaultDock: "right",
  sections: [
    { title: "Layers", rows: [layerList, ...rowsOf(LAYER, "Layers", { bind: BIND.layer })] },
    { title: "Mask", visible: { bind: BIND.hasMask }, rows: rowsOf(MASK, "Mask", { bind: BIND.mask }) },
    { title: "Smart object", visible: { bind: BIND.isSmart }, rows: rowsOf(SMART, "Smart object", { bind: BIND.smartObject }) },
    ...ADJUSTMENT_GROUPS.map((group) => ({
      title: group === "Tone" ? "Adjustment" : group,
      collapsible: true,
      visible: { bind: BIND.isAdjustment },
      rows: rowsOf(ADJUSTMENT, group, { bind: BIND.adjustment }),
    })),
  ],
};

/** Keep the layer list and the picked layer's derived addresses live. */
export function publishPropertyLists(host: BundleHost): Disposable {
  const list = publishObjectList(host, {
    rows: BIND.layers,
    select: BIND.layer,
    selector: (sel) => (sel.length === 1 ? `${sel[0]} > ${LAYER}` : null),
    secondaryPath: "kind",
  });
  let generation = 0;
  const derive = async () => {
    const mine = ++generation;
    const chosen = host.bindings.get(BIND.layer);
    const a = typeof chosen === "string" ? parseImageAddress(chosen) : null;
    let kind: unknown = null;
    let hasMask: unknown = false;
    if (a && a.kind === "layer" && a.layerId !== null) {
      const [k, m] = await Promise.all([
        host.objects.get(chosen as string, "kind").catch(() => null),
        host.objects.get(chosen as string, "hasMask").catch(() => null),
      ]);
      kind = k?.kind === "value" ? k.value : null;
      hasMask = m?.kind === "value" ? m.value === true : false;
    }
    if (mine !== generation) return;
    const at = (k: "mask" | "adjustment" | "smartObject", on: boolean, bind: string) => {
      if (on && a && a.layerId !== null) host.bindings.publish(bind, layerAddress(k, a.frameId, a.layerId));
      else host.bindings.delete(bind);
    };
    at("mask", hasMask === true, BIND.mask);
    at("adjustment", kind === "adjustment", BIND.adjustment);
    at("smartObject", kind === "smart", BIND.smartObject);
    host.bindings.publish(BIND.hasMask, hasMask === true);
    host.bindings.publish(BIND.isAdjustment, kind === "adjustment");
    host.bindings.publish(BIND.isSmart, kind === "smart");
  };
  const subs = [
    host.bindings.onDidChange((name) => {
      if (name === BIND.layer) void derive();
    }),
    host.objects.onDidChange(() => void derive()),
  ];
  void derive();
  return {
    dispose() {
      generation++;
      list.dispose();
      for (const s of subs) s.dispose();
    },
  };
}
