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

import type { BundleHost, Disposable, PanelSchema, PanelSchemaRow } from "@paged-media/plugin-api";
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

/**
 * The declared panel fields (object-model.md "Panel fields that should
 * become a shared PropertyField"). Adjustments: the tone, hue/saturation
 * master and filter scalars — 22 of the 89 generated rows; the per-range
 * hue/saturation, channel mixer, per-channel levels, colour balance,
 * black & white and curves rows stay in the image panel (lists and
 * 3×4 matrices read better as their own widgets).
 */
export const IMAGE_PANEL_FIELDS: ReadonlyArray<{ kind: "layer" | "mask" | "adjustment" | "smartObject"; path: string }> = [
  ...["name", "visible", "opacity", "blendMode", "locked", "clipped"].map((path) => ({ kind: "layer" as const, path })),
  { kind: "mask", path: "enabled" },
  { kind: "smartObject", path: "source" },
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
    "hueSaturation.master[0]",
    "hueSaturation.master[1]",
    "hueSaturation.master[2]",
    "gaussianBlur.sigma",
    "unsharp.amount",
    "hueRotate.degrees",
    "posterize.levels",
    "threshold.threshold",
    "photoFilter.density",
    "invert.enabled",
  ].map((path) => ({ kind: "adjustment" as const, path })),
];

const pathsOf = (kind: string) => IMAGE_PANEL_FIELDS.filter((f) => f.kind === kind).map((f) => f.path);

const LABELS: Readonly<Record<string, string>> = {
  "hueSaturation.master[0]": "Hue",
  "hueSaturation.master[1]": "Saturation",
  "hueSaturation.master[2]": "Lightness",
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

export const IMAGE_PROPERTIES_PANEL: PanelSchema = {
  id: PROPERTIES_PANEL_ID,
  title: "Image properties",
  icon: "panel-layers",
  defaultDock: "right",
  sections: [
    {
      title: "Layers",
      rows: [
        layerList,
        ...propertyRows(LAYER, pathsOf("layer"), { bind: BIND.layer }, { row: (p) => (p === "blendMode" ? { style: "select" } : {}) }),
      ],
    },
    { title: "Mask", visible: { bind: BIND.hasMask }, rows: propertyRows(MASK, pathsOf("mask"), { bind: BIND.mask }) },
    { title: "Smart object", visible: { bind: BIND.isSmart }, rows: propertyRows(SMART, pathsOf("smartObject"), { bind: BIND.smartObject }) },
    {
      title: "Adjustment",
      collapsible: true,
      visible: { bind: BIND.isAdjustment },
      rows: propertyRows(ADJUSTMENT, pathsOf("adjustment"), { bind: BIND.adjustment }, { labels: LABELS }),
    },
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
