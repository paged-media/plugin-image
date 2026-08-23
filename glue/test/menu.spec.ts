/**
 * paged.image — the menu entries name real commands.
 *
 * `contribute.menu()` REFUSES an entry whose command the bundle has not
 * registered, and that refusal happens at activate time in a browser —
 * so a mistyped suffix would surface as one quietly missing menu item
 * and nowhere else. Comparing the table against the manifest catches it
 * in CI instead. It also catches the reverse drift: a command renamed in
 * the manifest while the menu table still points at the old id.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { describe, expect, it } from "vitest";

import { MENU_COMMAND_PREFIX, MENU_ENTRIES } from "../src/menu";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST = join(HERE, "..", "manifest.json");

const declared: string[] = JSON.parse(readFileSync(MANIFEST, "utf8"))
  .contributes.commands;

describe("image menu entries", () => {
  it("every entry points at a command the manifest declares", () => {
    const set = new Set(declared);
    const missing = MENU_ENTRIES.map(
      ([, s]) => `${MENU_COMMAND_PREFIX}.${s}`,
    ).filter((id) => !set.has(id));
    expect(missing, `unknown commands: ${missing.join(", ")}`).toEqual([]);
  });

  it("undo and redo are NOT in the menu", () => {
    // The host owns Edit ▸ Undo. A second undo in a second menu driving
    // a DIFFERENT stack looks like the thing you know and destroys
    // something else — the most dangerous kind of duplicate.
    const suffixes = MENU_ENTRIES.map(([, s]) => s);
    expect(suffixes).not.toContain("undo");
    expect(suffixes).not.toContain("redo");
    // …and `claimTiles` is plumbing the compositor calls, not a verb.
    expect(suffixes).not.toContain("claimTiles");
  });

  it("no command and no path appears twice", () => {
    const suffixes = MENU_ENTRIES.map(([, s]) => s);
    expect(suffixes.length).toBe(new Set(suffixes).size);
    const paths = MENU_ENTRIES.map(([p]) => p);
    expect(paths.length).toBe(new Set(paths).size);
  });

  it("only Image and host top-levels are used", () => {
    const stray = MENU_ENTRIES.map(([p]) => p.split("/")[0]).filter(
      (t) => t !== "Image" && !["Edit", "Object"].includes(t),
    );
    expect(stray, `unexpected top-level menus: ${stray.join(", ")}`).toEqual([]);
  });

  it("covers a real share of the bundle's commands", () => {
    // A floor: with an empty table every assertion above passes
    // vacuously.
    expect(MENU_ENTRIES.length).toBeGreaterThan(15);
    expect(MENU_ENTRIES.length).toBeLessThanOrEqual(declared.length);
  });
});
