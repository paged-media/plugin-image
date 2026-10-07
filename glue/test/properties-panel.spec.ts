// @feat: image.object-model
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

// ADR 323 — the image properties panel declares property rows (fields,
// not widgets); every row is a row of its kind's schema and the address
// forms are well-formed (plugin-sdk validatePanelSchema).

import { describe, expect, it } from "vitest";

import type { PropertySchema } from "@paged-media/plugin-api";
import { isPropertyRow, propertyRowsOf, validatePanelSchema } from "@paged-media/plugin-sdk";

import manifest from "../manifest.json";
import { KIND_PREFIX, KIND_ROWS } from "../src/object-model/image-objects";
import { BIND, IMAGE_PANEL_FIELDS, IMAGE_PROPERTIES_PANEL, PROPERTIES_PANEL_ID } from "../src/panels/properties-panel";

const schemaOf = (k: string): readonly PropertySchema[] | undefined =>
  k.startsWith(KIND_PREFIX) ? (KIND_ROWS as Record<string, readonly PropertySchema[]>)[k.slice(KIND_PREFIX.length)] : undefined;

describe("image properties panel (property rows)", () => {
  it("validates against the object-model schemas", () => {
    expect(validatePanelSchema(IMAGE_PROPERTIES_PANEL, { pluginId: manifest.id, schemaOf })).toEqual([]);
  });
  it("covers every declared field, each through a property row on its kind", () => {
    const rows = propertyRowsOf(IMAGE_PROPERTIES_PANEL).map((r) => `${r.field.kind!.slice(KIND_PREFIX.length)}.${r.field.path}`);
    expect(rows.sort()).toEqual(IMAGE_PANEL_FIELDS.map((f) => `${f.kind}.${f.path}`).sort());
  });
  it("every field is writable on its kind (an editing panel, not a readout)", () => {
    for (const f of IMAGE_PANEL_FIELDS) {
      const row = schemaOf(`${KIND_PREFIX}${f.kind}`)!.find((r) => r.path === f.path)!;
      expect(row.access ?? "readWrite", `${f.kind}.${f.path}`).toBe("readWrite");
    }
  });
  it("layer rows edit the picked layer; mask/adjustment/smart rows their derived bindings", () => {
    const want: Record<string, string> = {
      [`${KIND_PREFIX}layer`]: BIND.layer,
      [`${KIND_PREFIX}mask`]: BIND.mask,
      [`${KIND_PREFIX}adjustment`]: BIND.adjustment,
      [`${KIND_PREFIX}smartObject`]: BIND.smartObject,
    };
    for (const s of IMAGE_PROPERTIES_PANEL.sections)
      for (const r of s.rows) if (isPropertyRow(r)) expect(r.address).toEqual({ bind: want[r.kind!] });
  });
  it("covers EVERY writable row of every kind (read-only and derived rows are readouts, not fields)", () => {
    const declared = new Set(propertyRowsOf(IMAGE_PROPERTIES_PANEL).map((r) => `${r.field.kind!.slice(KIND_PREFIX.length)}.${r.field.path}`));
    const writable = Object.entries(KIND_ROWS).flatMap(([kind, rows]) =>
      (rows as readonly PropertySchema[]).filter((r) => (r.access ?? "readWrite") === "readWrite").map((r) => `${kind}.${r.path}`),
    );
    expect(writable.length).toBeGreaterThan(90);
    expect(writable.filter((w) => !declared.has(w))).toEqual([]);
  });
  it("list-valued adjustment rows: the curves tables keep the typed list field (nullable: null = off)", () => {
    const rows = propertyRowsOf(IMAGE_PROPERTIES_PANEL).map((r) => r.field);
    const composite = rows.find((r) => r.path === "curves.composite")!;
    const channels = rows.find((r) => r.path === "curves.channels")!;
    const adj = schemaOf(`${KIND_PREFIX}adjustment`)!;
    expect(adj.find((r) => r.path === "curves.composite")!.type).toMatchObject({ kind: "list", of: { kind: "number" }, maxItems: 256 });
    expect(adj.find((r) => r.path === "curves.channels")!.type).toMatchObject({ kind: "list", maxItems: 3, of: { kind: "list" } });
    expect(composite.items).toBeUndefined();
    expect(channels.items).toBeUndefined();
  });
  it("labels the indexed scalars (per-range hue/saturation, channel mixer, colour balance)", () => {
    const label = (p: string) => propertyRowsOf(IMAGE_PROPERTIES_PANEL).find((r) => r.field.path === p)!.field.label;
    expect(label("hueSaturation.ranges[2][1]")).toBe("Greens saturation");
    expect(label("channelMixer.gc")).toBe("Green from constant");
    expect(label("colorBalance.hiYb")).toBe("Highlights yellow–blue");
    expect(label("levelsRgb.bGamma")).toBe("Blue gamma");
    expect(label("hueSaturation.colorize[0]")).toBe("Colorize");
  });
  it("is declared in the manifest", () => {
    expect(manifest.contributes.panels).toContain(PROPERTIES_PANEL_ID);
  });
});
