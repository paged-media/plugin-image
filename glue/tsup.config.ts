import { defineConfig } from "tsup";

// Two entries (index + the decode worker). The wasm artifact in ../wasm is
// left for the consuming bundler (the editor's Vite) — `?url` asset imports and
// the wasm-bindgen glue must not be bundled by esbuild.
export default defineConfig({
  entry: ["src/index.ts", "src/decode-worker.ts"],
  format: ["esm"],
  dts: true,
  clean: true,
  external: [/\?url$/, /wasm\//],
});
