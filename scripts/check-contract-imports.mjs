#!/usr/bin/env node
// The clean-checkout proof's enforcement mechanism (mirrors plugin-draw
// / plugin-web): every import in this repo's TS source must come through
// the sanctioned contract surface. The lint IS the "no private
// backdoors" guarantee for the isolation contract (spec §2.1): an
// editor-internal import cannot land silently. (The Rust half of the
// SDK-only rule is deny.toml's [sources] + the cargo-tree CI guards.)
//
// Allowed: the plugin contract (@paged-media/plugin-api / plugin-sdk),
// this repo's own packages, react (panels are React expert leaves, v0
// exception — same as plugin-web), and relative paths. Everything else
// fails the build with a pointer to BREAKAGE_LOG.md (the a/b/c
// disposition: promote to plugin-api, use an existing capability, or
// record).

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import process from "node:process";

const ROOT = new URL("..", import.meta.url).pathname;

const ALLOWED_PREFIXES = [
  "@paged-media/plugin-api",
  "@paged-media/plugin-sdk",
  "@paged-media/image-", // this repo's own packages
  "react", // panels are React expert leaves (v0 exception)
];

// The TS packages live at top level per the spec layout (§4).
const PACKAGE_DIRS = ["glue", "manifest"];

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name.startsWith(".")) continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path, out);
    else if (/\.(ts|tsx)$/.test(name) && !/\.(spec|test)\./.test(name)) {
      out.push(path);
    }
  }
  return out;
}

const IMPORT = /(?:^|\n)\s*(?:import|export)[^"'`;]*?from\s*["']([^"']+)["']/g;

const violations = [];
for (const pkg of PACKAGE_DIRS) {
  const dir = join(ROOT, pkg);
  if (!existsSync(dir)) continue;
  for (const file of walk(dir)) {
    if (!file.includes("/src/")) continue;
    const text = readFileSync(file, "utf8");
    IMPORT.lastIndex = 0;
    let m;
    while ((m = IMPORT.exec(text)) !== null) {
      const spec = m[1];
      if (spec.startsWith(".") || spec.startsWith("..")) continue;
      if (ALLOWED_PREFIXES.some((p) => spec.startsWith(p))) continue;
      violations.push(`${relative(ROOT, file)} → "${spec}"`);
    }
  }
}

if (violations.length > 0) {
  console.error(
    "contract-import lint: imports outside the plugin surface " +
      "(disposition each: promote to plugin-api / use an existing " +
      "capability / record in BREAKAGE_LOG.md):",
  );
  for (const v of violations) console.error(`  - ${v}`);
  process.exit(1);
}
console.log("contract-import lint: clean (plugin surface only)");
