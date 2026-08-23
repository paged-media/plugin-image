/**
 * paged.image — the menu bar entries.
 *
 * 23 commands and, until `contribute.menu()` (plugin-api 0.2.33), no
 * route to a menu: every verb lived behind Cmd+K, where a raster editor's
 * whole vocabulary — adjust, fill, feather, bake — appeared as a flat
 * list of ids next to everything else in the app.
 *
 * A TOP-LEVEL `Image` MENU, which is what every raster editor has and
 * where a user will look first. Two verbs merge elsewhere instead,
 * because they are about the DOCUMENT rather than the image: the
 * selection-to-path pair belongs beside the host's own selection verbs
 * in `Edit`.
 *
 * THREE COMMANDS ARE DELIBERATELY ABSENT, and the reasons differ:
 *
 *   · `undo` / `redo` — the host owns `Edit ▸ Undo`. A second undo in a
 *     second menu, operating on a DIFFERENT stack (the image session's,
 *     not the document's), is the most dangerous kind of duplicate: it
 *     looks like the thing you know and destroys something else. It
 *     stays on its panel, where the stack it belongs to is visible.
 *   · `claimTiles` — "serve image tiles to the renderer" is plumbing the
 *     compositor calls, not a verb a person performs.
 */

import type { BundleHost, Disposable } from "@paged-media/plugin-api";

const C = "media.paged.image.command";

/** `[path, command suffix, group]`. */
const ENTRIES: [path: string, suffix: string, group: string][] = [
  ["Image/Open image panel", "openImage", "session"],

  ["Image/Adjust…", "adjustSelected", "adjust"],
  ["Image/Auto-enhance", "autoEnhance", "adjust"],
  ["Image/Bake adjustments into layer", "bakeAdjustToLayer", "adjust-bake"],

  ["Image/Layer/Add", "addLayer", "layer"],

  ["Image/Selection/Select all", "selectAll", "sel"],
  ["Image/Selection/Deselect", "deselect", "sel"],
  ["Image/Selection/Invert", "invertSelection", "sel"],
  ["Image/Selection/Feather…", "featherSelection", "sel-edit"],
  ["Image/Selection/From luminosity channel", "channelToSelection", "sel-conv"],

  ["Image/Fill/Gradient…", "fillSelection", "fill"],
  ["Image/Fill/Noise…", "fillNoise", "fill"],
  ["Image/Fill/Content-aware", "contentAwareFill", "fill"],

  ["Image/Apply crop", "commitCrop", "crop"],
  ["Image/Set type", "setType", "type"],
  ["Image/Load brush library (.abr)…", "loadBrushLibrary", "brush"],

  ["Image/File/Apply adjustments to the file", "applyToFile", "file"],
  ["Image/File/Save the adjusted file", "saveToFile", "file"],

  // These two convert between the DOCUMENT's paths and the image's
  // selection, so they belong beside the host's selection verbs rather
  // than buried three levels into a plugin menu.
  ["Edit/Selection to path", "selectionToPath", "img-sel-conv"],
  ["Edit/Path to selection", "pathToSelection", "img-sel-conv"],
];

/**
 * Register every entry; one Disposable drops them all.
 *
 * Degrades on a host older than plugin-api 0.2.33 by contributing
 * nothing and saying so — the shape every optional door in this bundle
 * already uses.
 */
export function contributeMenu(host: BundleHost): Disposable {
  const contribute = host.contribute as BundleHost["contribute"] & {
    menu?: (c: {
      path: string;
      command: string;
      order?: number;
      group?: string;
    }) => Disposable;
  };
  if (typeof contribute.menu !== "function") {
    host.log.info(
      "host predates contribute.menu (plugin-api 0.2.33) — " +
        `${ENTRIES.length} menu entries not contributed; every command ` +
        "remains reachable through the command palette",
    );
    return { dispose() {} };
  }

  const handles: Disposable[] = [];
  const perGroup = new Map<string, number>();
  for (const [path, suffix, group] of ENTRIES) {
    const n = (perGroup.get(group) ?? 0) + 1;
    perGroup.set(group, n);
    handles.push(
      contribute.menu({ path, command: `${C}.${suffix}`, group, order: n * 10 }),
    );
  }
  host.log.info(`contributed ${handles.length} menu entries`);
  return {
    dispose() {
      for (const h of handles) h.dispose();
      handles.length = 0;
    },
  };
}

/** Exported for the bundle's own test. */
export const MENU_ENTRIES = ENTRIES;
export const MENU_COMMAND_PREFIX = C;
