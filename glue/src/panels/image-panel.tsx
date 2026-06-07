// The M0 panel — an expert React leaf (the only panel form the v0.2
// SDK carries). Honest seams: it states what the engine can do TODAY
// and which SDK doors are pending, instead of faking controls. The
// real adjustments surface (gesture sliders over Engine B params) is
// the M2 panel sketched in manifest/panels/image-adjustments.panel.json.

import type { CSSProperties } from "react";

import type { BundleHost } from "@paged-media/plugin-api";

import manifest from "@paged-media/image-manifest/manifest.json";

const row: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  gap: "var(--space-2, 8px)",
  padding: "var(--space-1, 4px) 0",
  borderBottom: "1px solid var(--pg-border, rgba(127,127,127,0.25))",
};

const kicker: CSSProperties = {
  textTransform: "uppercase",
  letterSpacing: "var(--tracking-wide, 0.08em)",
  fontSize: "11px",
  opacity: 0.7,
};

export function makeImagePanel(host: BundleHost) {
  return function ImagePanel() {
    return (
      <div style={{ padding: "var(--space-2, 8px)", fontSize: "12px" }}>
        <div style={kicker}>paged.image — M0</div>
        <div style={row}>
          <span>Bundle</span>
          <span style={{ fontFamily: "var(--font-mono, monospace)" }}>
            {manifest.id}@{manifest.version}
          </span>
        </div>
        <div style={row}>
          <span>Engine</span>
          <span>T0 kernels at gpu↔ref parity; PSD structural round-trip</span>
        </div>
        <div style={row}>
          <span>Pending SDK doors</span>
          <span style={{ textAlign: "right" }}>
            asset bytes (I-04) · GPU surface (I-01/I-07) · PSD open handler
            (I-05)
          </span>
        </div>
        <button
          type="button"
          style={{ marginTop: "var(--space-2, 8px)" }}
          onClick={() => {
            host.diagnostics.set("status", [
              {
                severity: "info",
                message:
                  "paged.image M0 — engine wasm lane lands with M1 ingest",
                source: "media.paged.image.panel.adjustments",
              },
            ]);
            host.log.info("status pinged from the adjustments panel");
          }}
        >
          Log engine status
        </button>
      </div>
    );
  };
}
