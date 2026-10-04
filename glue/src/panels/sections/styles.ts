// Shared panel styles (moved out of image-panel.tsx so sections can
// live in their own files without copying them).

import type { CSSProperties } from "react";

export const row: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: "var(--space-2, 8px)",
  padding: "var(--space-1, 4px) 0",
  borderBottom: "1px solid var(--pg-border, rgba(127,127,127,0.25))",
};

export const kicker: CSSProperties = {
  textTransform: "uppercase",
  letterSpacing: "var(--tracking-wide, 0.08em)",
  fontSize: "11px",
  opacity: 0.7,
};

export const sectionTitle: CSSProperties = {
  ...kicker,
  marginTop: "var(--space-3, 12px)",
  marginBottom: "var(--space-1, 4px)",
};

export const mono: CSSProperties = {
  fontFamily: "var(--font-mono, monospace)",
};

export const note: CSSProperties = {
  fontSize: "11px",
  opacity: 0.65,
  marginTop: "var(--space-2, 8px)",
};
