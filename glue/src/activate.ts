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

// The paged.image bundle entry — the M4 "editor enablement" slice. The
// platform doors this consumes (all probed, never assumed): C-5
// host.assets.getPlacedImage (a placed frame's ORIGINAL bytes), C-1
// Stage-A sceneLayer with the v41 image item (in-frame composite of the
// adjusted RGBA), K-2 contribute.importer (File/Open + drag-drop routes
// PSD/PNG/JPEG bytes here). The engine wasm (image-js: codec/PSD decode
// + the GPU-only Engine A adjustments) boots LAZILY in the bundle realm
// on first ingest — the GPU device is created THERE, not inside
// loadBundleWasm's no-authority sandbox (BREAKAGE I-07).

import type { BundleHandle, BundleHost } from "@paged-media/plugin-api";
import { contributePanel, contributeTool } from "@paged-media/plugin-sdk";

import manifest from "../manifest.json";

import { createImageSession } from "./session";
import { makeImagePanel } from "./panels/image-panel";
import { makeCropGesture } from "./crop-tool";
import { makeSelectionGesture } from "./selection-tool";
import { makeBrushGesture, PAINT_CURSOR } from "./brush-tool";

const PANEL_ID = "media.paged.image.panel.adjustments";
const CROP_TOOL_ID = "media.paged.image.tool.crop";
const MARQUEE_RECT_TOOL_ID = "media.paged.image.tool.marqueeRect";
const MARQUEE_ELLIPSE_TOOL_ID = "media.paged.image.tool.marqueeEllipse";
const LASSO_TOOL_ID = "media.paged.image.tool.lasso";
const MAGIC_WAND_TOOL_ID = "media.paged.image.tool.magicWand";
const BRUSH_TOOL_ID = "media.paged.image.tool.brush";
const PENCIL_TOOL_ID = "media.paged.image.tool.pencil";
const ERASER_TOOL_ID = "media.paged.image.tool.eraser";

export function activate(host: BundleHost): BundleHandle {
  const session = createImageSession(host);

  contributePanel(host, {
    id: PANEL_ID,
    title: "Image",
    icon: "panel-canvas",
    component: makeImagePanel(session),
    defaultDock: "right",
  });

  host.contribute.command({
    id: "media.paged.image.command.openImage",
    title: "Open image panel",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
    },
  });

  // The selection-driven flow: "Adjust image" on a placed image frame —
  // ingest the frame's original bytes (C-5) and raise the panel; the
  // panel's committed Apply runs the GPU chain + the in-frame composite.
  host.contribute.command({
    id: "media.paged.image.command.adjustSelected",
    title: "Adjust image",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      void session.ingestSelection();
    },
  });

  // Auto-enhance: derive auto-levels + a gray-world white balance from the
  // ingested image's histogram (pure CPU readout) and set them on the panel
  // params. Like every edit it's PREVIEW-only — the user commits with Apply.
  host.contribute.command({
    id: "media.paged.image.command.autoEnhance",
    title: "Auto-enhance image",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      session.autoEnhance();
    },
  });

  // C-6 (I-06) — claim the ingested image's tile resource so the renderer
  // pulls level-0 tiles for the placed frame at its current scale (the
  // honest subset; the mip pyramid + Engine B window eval are the named
  // gap in tile-provider.ts). Degrades honestly when the host wires no
  // resource channel (rendering.resourceProvider@1 is false).
  host.contribute.command({
    id: "media.paged.image.command.claimTiles",
    title: "Serve image tiles to the renderer",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      session.claimTiles();
    },
  });

  // Crop + straighten TOOL (the on-canvas crop affordance). Registers
  // into the transform rail; its gesture drives the session's crop machine
  // (image_core::crop geometry) and renders the crop frame through the
  // LIVE host.overlay door. The COMMIT rides the commitCrop command (and
  // the panel button) so it's a deliberate, single action.
  contributeTool(host, {
    id: CROP_TOOL_ID,
    title: "Crop",
    icon: "tool-crop",
    group: CROP_TOOL_ID,
    section: "transform",
    // shift+x — yields "c" to the built-in Scissors tool (InDesign-canonical);
    // INV-REG-1 (editor registry-invariants) keeps tool shortcuts unique.
    shortcut: "shift+x",
    gesture: () => makeCropGesture(host, session),
  });

  // The crop commit command (also surfaced as the panel's "Apply crop"
  // button): cut the machine's rect out of the engine source + recomposite.
  host.contribute.command({
    id: "media.paged.image.command.commitCrop",
    title: "Apply crop",
    category: "Image",
    handler: () => {
      void session.commitCrop();
    },
  });

  // ── SELECTION tools (spec §6.1 — the mask ABI's editor reach) ──
  //
  // Four tools over ONE gesture architecture (the crop pattern: gesture →
  // machine → Rust coverage → marching-ants overlay via
  // host.overlay.setToolPreview). The selection is ENGINE state bound to
  // the ingested source; the committed Apply masks every adjust/filter
  // dispatch (GPU `mix(a, result, mask)`) + the CPU curves pass by it.
  // Modifier convention on all four (read from the gesture events'
  // `modifiers`): shift = add, alt = subtract, shift+alt = intersect.
  // SHORTCUTS (INV-REG-1, globally unique tool shortcuts): "y" is the
  // last free single letter; shift+y / shift+l / shift+w are free in the
  // shift register (draw holds shift+a/b/c/i/j/k/m/n/r/u; this bundle
  // holds shift+x) — verified against both manifests + the editor
  // built-ins at pick time.
  contributeTool(host, {
    id: MARQUEE_RECT_TOOL_ID,
    title: "Marquee (rectangle)",
    icon: "tool-marquee-rect",
    group: MARQUEE_RECT_TOOL_ID,
    section: "selection",
    shortcut: "y",
    gesture: () => makeSelectionGesture(host, session, "rect"),
  });
  contributeTool(host, {
    id: MARQUEE_ELLIPSE_TOOL_ID,
    title: "Marquee (ellipse)",
    icon: "tool-marquee-ellipse",
    group: MARQUEE_ELLIPSE_TOOL_ID,
    section: "selection",
    shortcut: "shift+y",
    gesture: () => makeSelectionGesture(host, session, "ellipse"),
  });
  contributeTool(host, {
    id: LASSO_TOOL_ID,
    title: "Lasso",
    icon: "tool-lasso",
    group: LASSO_TOOL_ID,
    section: "selection",
    shortcut: "shift+l",
    gesture: () => makeSelectionGesture(host, session, "lasso"),
  });
  // Magic wand: click-select by color distance (v0 fixed tolerance
  // 32/255 per channel, contiguous flood — documented in
  // selection-machine.ts; a tool-options tolerance slider is a
  // follow-up).
  contributeTool(host, {
    id: MAGIC_WAND_TOOL_ID,
    title: "Magic wand",
    icon: "tool-magic-wand",
    group: MAGIC_WAND_TOOL_ID,
    section: "selection",
    shortcut: "shift+w",
    gesture: () => makeSelectionGesture(host, session, "wand"),
  });

  // ── PAINT tools (spec §6.3 — the brush engine's editor reach) ──
  //
  // Three tools over ONE gesture architecture (the selection pattern:
  // gesture → machine → engine doors → overlay affordance). They paint
  // RASTER pixels into the ingested image, which is what distinguishes
  // them from paged.draw's VECTOR Paintbrush / Blob Brush / Eraser on
  // the same rail — hence the "(raster)" in every title. They paint the
  // ACTIVE LAYER of the session's layer stack, and a committed stroke is
  // journaled (undoable, within the stated bound). The exact scope is
  // spelled out in the panel's Brush + Layers sections
  // (BRUSH_SCOPE_NOTE / LAYERS_SCOPE_NOTE), not only here.
  //
  // SHORTCUTS (INV-REG-1, globally unique tool shortcuts): "q" is the
  // last free single letter, and shift+f / shift+e are free in the shift
  // register — verified at pick time against the editor built-ins (v a
  // shift+p u b t shift+t \ p n f m l c e r s o g shift+g i k h z),
  // paged.draw (= - shift+c/u/n/a/m/b/r/j/k/i/d/s/q) and this bundle's
  // own five (shift+x, y, shift+y, shift+l, shift+w). "[" and "]" are
  // deliberately LEFT FREE for brush-size nudging.
  contributeTool(host, {
    id: BRUSH_TOOL_ID,
    title: "Brush (raster)",
    icon: "tool-paintbrush",
    group: BRUSH_TOOL_ID,
    section: "drawType",
    shortcut: "q",
    cursor: PAINT_CURSOR,
    gesture: () => makeBrushGesture(host, session, "brush"),
  });
  contributeTool(host, {
    id: PENCIL_TOOL_ID,
    title: "Pencil (raster)",
    icon: "tool-pencil",
    group: PENCIL_TOOL_ID,
    section: "drawType",
    shortcut: "shift+f",
    cursor: PAINT_CURSOR,
    gesture: () => makeBrushGesture(host, session, "pencil"),
  });
  contributeTool(host, {
    id: ERASER_TOOL_ID,
    title: "Eraser (raster)",
    icon: "tool-erase",
    group: ERASER_TOOL_ID,
    section: "drawType",
    shortcut: "shift+e",
    cursor: PAINT_CURSOR,
    gesture: () => makeBrushGesture(host, session, "eraser"),
  });

  // GENERATE — the `gen.*` family's editor reach. Fills the CURRENT
  // SELECTION (the whole image when there is none) with a fixed
  // two-stop gradient or with noise, composited through the coverage
  // mask on the GPU. DESTRUCTIVE by design (it swaps the engine-held
  // source like a crop commit) — see fill.rs for why a generator cannot
  // be a re-runnable stage of the adjust chain. The commands carry the
  // v0 defaults (black→white linear, noise amount 0.5); the panel's
  // Generate section exposes the pickers.
  host.contribute.command({
    id: "media.paged.image.command.fillSelection",
    title: "Fill selection with a gradient",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      void session.fillSelection({
        kind: "gradient",
        gradient: "linear",
        c0: [0, 0, 0, 1],
        c1: [1, 1, 1, 1],
      });
    },
  });
  host.contribute.command({
    id: "media.paged.image.command.fillNoise",
    title: "Fill selection with noise",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      void session.fillSelection({ kind: "noise", amount: 0.5 });
    },
  });

  // ── LAYER GRAPH commands (§6.2 — the layer stack's palette reach) ──
  //
  // The panel's Layers section carries the full palette (order,
  // visibility, opacity, blend, lock, the active-layer choice); these
  // are the command-palette reach for the four that deserve a shortcut
  // surface. Undo/redo act on the plugin's OWN pixel journal — paint,
  // fills and bakes on the ingested image — not on the host document's
  // history, which the editor owns and which this plugin never mutates.
  host.contribute.command({
    id: "media.paged.image.command.addLayer",
    title: "Add image layer",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      void session.addLayer();
    },
  });
  host.contribute.command({
    id: "media.paged.image.command.bakeAdjustToLayer",
    title: "Bake adjustments into the active layer",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      void session.bakeAdjustToLayer();
    },
  });
  host.contribute.command({
    id: "media.paged.image.command.undo",
    title: "Undo image edit",
    category: "Image",
    handler: () => {
      void session.undo();
    },
  });
  host.contribute.command({
    id: "media.paged.image.command.redo",
    title: "Redo image edit",
    category: "Image",
    handler: () => {
      void session.redo();
    },
  });

  // SAVE-BACK — bake the adjustments into the SOURCE FILE's bytes.
  //
  // DELIVERY SEAM (honest, and the reason this is a "stage + export"
  // flow rather than a one-click write): the host wires NO save-file
  // door. `host.shell.pickFile` is a READ picker — it returns
  // `PickedFile { name, bytes }`, i.e. bytes coming IN — and there is no
  // counterpart that takes bytes out (probed here at activate time so
  // the claim is checked, not assumed). So the panel button + this
  // command COMPUTE and STAGE the bytes (reporting size + the honest
  // layer-structure note), and the Export Center's exporters below are
  // the whole delivery lane. A `shell.saveFile@1` door is the RFI-worthy
  // gap; the day it exists this handler writes directly.
  host.contribute.command({
    id: "media.paged.image.command.applyToFile",
    title: "Apply adjustments to the file",
    category: "Image",
    handler: () => {
      host.shell.openPanel(PANEL_ID);
      void session.applyToFile();
    },
  });

  // Selection commands (the panel buttons + command-palette reach).
  host.contribute.command({
    id: "media.paged.image.command.selectAll",
    title: "Select all (image)",
    category: "Image",
    handler: () => {
      session.selectAll();
    },
  });
  host.contribute.command({
    id: "media.paged.image.command.deselect",
    title: "Deselect (image)",
    category: "Image",
    handler: () => {
      session.deselect();
    },
  });
  host.contribute.command({
    id: "media.paged.image.command.invertSelection",
    title: "Invert selection (image)",
    category: "Image",
    handler: () => {
      session.invertSelection();
    },
  });
  // Fixed v0 sigma (session FEATHER_SIGMA_DEFAULT) — a slider follows.
  host.contribute.command({
    id: "media.paged.image.command.featherSelection",
    title: "Feather selection (image)",
    category: "Image",
    handler: () => {
      session.featherSelection();
    },
  });

  // K-2 — the raster importer: opening/dropping a PSD/PNG/JPEG routes
  // its bytes HERE (decode into the session, raise the panel; it does
  // NOT replace the document). Degrades honestly on an older host.
  if (host.supports("contribute.importer@1")) {
    host.contribute.importer({
      id: "media.paged.image.importer.raster",
      title: "Image (PSD/PNG/JPEG)",
      extensions: [".psd", ".psb", ".png", ".jpg", ".jpeg"],
      mimeTypes: [
        "image/vnd.adobe.photoshop",
        "image/png",
        "image/jpeg",
      ],
      import: async ({ name, bytes }) => {
        host.shell.openPanel(PANEL_ID);
        await session.importBytes(name, bytes);
      },
    });
  }
  // SAVE-BACK exporters — the delivery lane for the adjusted pixels.
  //
  // PSD: re-emits the (possibly layer-edited) PSD with full
  // carry-through preservation. When the panel is at IDENTITY the bytes
  // are the untouched re-emit (zero-edit ⇒ BYTE-IDENTICAL — §10.4
  // survives); when it is not, the full-resolution adjusted composite is
  // written first (`replace_channel_pixels` on the single canvas-sized
  // content layer, else a flatten into a new single-layer PSD, which the
  // panel/status announces). Null when no PSD is loaded — the Export
  // Center shows the honest empty result.
  //
  // PNG / JPEG: the non-PSD lane, one exporter per format so the
  // registry's declared `extension` is never a lie. The panel's "Apply
  // to file" button picks the source's own format (JPEG only when the
  // ingested bytes WERE a JPEG — a re-encode never invents a lossy
  // format); these two let the user ask for either explicitly.
  if (host.supports("contribute.exporter@1")) {
    host.contribute.exporter({
      id: "media.paged.image.exporter.psd",
      title: "PSD (adjusted pixels + edited layers)",
      extension: ".psd",
      mimeType: "image/vnd.adobe.photoshop",
      export: () => session.psdExportBytes(),
    });
    host.contribute.exporter({
      id: "media.paged.image.exporter.png",
      title: "PNG (adjusted pixels)",
      extension: ".png",
      mimeType: "image/png",
      export: () => session.rasterExportBytes("png"),
    });
    host.contribute.exporter({
      id: "media.paged.image.exporter.jpeg",
      title: "JPEG (adjusted pixels)",
      extension: ".jpg",
      mimeType: "image/jpeg",
      export: () => session.rasterExportBytes("jpeg"),
    });
  }

  // The probe behind the save-back delivery seam (see the applyToFile
  // command): `shell.pickFile@1` is the only file door the contract has,
  // and it READS. Logged, not assumed — if a save door ever appears the
  // log line is the first place the gap stops being true.
  host.log.debug(
    `save-back delivery: exporter registry only (shell.pickFile@1=${host.supports(
      "shell.pickFile@1",
    )} is a READ picker; no save-file door in the contract)`,
  );

  host.log.info(`activated (apiVersion ${manifest.apiVersion})`);

  return {
    dispose() {
      session.dispose();
    },
  };
}

export { manifest, PANEL_ID };
