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

/** The panel's own heading: a plain title, not a kicker. */
export const panelTitle: CSSProperties = {
  fontSize: "13px",
  fontWeight: 600,
  margin: "0 0 var(--space-1, 4px)",
};

/** The commit bar: Apply, Reset and the status line stay in view while
 *  the sections below scroll. Bleeds over the panel padding so the
 *  sections do not show through at its edges. */
export const actionBar: CSSProperties = {
  position: "sticky",
  top: 0,
  zIndex: 1,
  background: "var(--pg-bg)",
  margin: "0 calc(-1 * var(--space-2, 8px))",
  padding: "var(--space-2, 8px)",
  borderBottom: "1px solid var(--chrome-divider, var(--pg-border))",
};

/** A small colour chip beside a colour select (the select itself keeps
 *  the theme's surface, so its text stays readable). */
export const swatch = (css: string): CSSProperties => ({
  display: "inline-block",
  width: 14,
  height: 14,
  flex: "none",
  borderRadius: 3,
  border: "1px solid var(--pg-border)",
  background: css,
});
