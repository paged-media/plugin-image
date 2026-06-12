#!/usr/bin/env bash
# Build the paged.image engine wasm (image-js) and land the wasm-bindgen
# `--target web` output in manifest/wasm/ — the path the manifest
# declares under capabilities.wasm[] (governance + the plugin-cli size
# gate; the gate only measures a file that exists, so building HERE
# makes it live). The bundle loads it via the wbindgen glue in the
# BUNDLE REALM (the core/canvas-wasm pattern), NOT via loadBundleWasm —
# the sandbox has no navigator.gpu (BREAKAGE I-07).
#
# wasm-opt: CI pins binaryen (old apt binaryen breaks wasm-bindgen
# externref table grow — the "Table.grow failed" gotcha); locally it is
# applied when present, skipped with a warning when absent.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=manifest/wasm
BUDGET=$((8 * 1024 * 1024))

cargo build --release --target wasm32-unknown-unknown -p image-js

# Pin check: wasm-bindgen-cli must match the Cargo.lock wasm-bindgen.
LOCKED=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | grep version | head -1 | cut -d'"' -f2)
CLI=$(wasm-bindgen --version | awk '{print $2}')
if [ "$LOCKED" != "$CLI" ]; then
  echo "error: wasm-bindgen-cli $CLI != Cargo.lock wasm-bindgen $LOCKED" >&2
  echo "       cargo install wasm-bindgen-cli --version $LOCKED" >&2
  exit 1
fi

wasm-bindgen target/wasm32-unknown-unknown/release/image_js.wasm \
  --target web --out-dir "$OUT"

if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Oz "$OUT/image_js_bg.wasm" -o "$OUT/image_js_bg.wasm"
else
  echo "warning: wasm-opt not found — shipping unoptimized wasm (CI optimizes)" >&2
fi

SIZE=$(wc -c < "$OUT/image_js_bg.wasm" | tr -d ' ')
echo "image_js_bg.wasm: $SIZE bytes (budget $BUDGET)"
if [ "$SIZE" -gt "$BUDGET" ]; then
  echo "error: wasm artifact exceeds the 8 MiB plugin budget" >&2
  exit 1
fi
