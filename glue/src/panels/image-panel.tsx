// The M4 adjustments panel — an expert React leaf over the ingest
// session. Honest seams everywhere: the engine/GPU state is stated, the
// composite is named a PREVIEW (the document is never mutated — save-
// back of adjusted pixels is a later milestone), and parameter edits
// only hit the GPU + scene channel on the committed Apply (the Stage-A
// re-submit-on-commit contract; per-drag interactivity is Stage B/M2).

import { useEffect, useReducer } from "react";
import type { CSSProperties } from "react";

import manifest from "@paged-media/image-manifest/manifest.json";

import type { ImageSession } from "../session";
import type { AdjustParams } from "../engine";

const row: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
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

const mono: CSSProperties = {
  fontFamily: "var(--font-mono, monospace)",
};

const note: CSSProperties = {
  fontSize: "11px",
  opacity: 0.65,
  marginTop: "var(--space-2, 8px)",
};

interface SliderSpec {
  key: keyof AdjustParams;
  label: string;
  min: number;
  max: number;
  step: number;
}

const SLIDERS: SliderSpec[] = [
  { key: "exposureEv", label: "Exposure (EV)", min: -5, max: 5, step: 0.1 },
  { key: "brightness", label: "Brightness", min: -1, max: 1, step: 0.05 },
  { key: "contrast", label: "Contrast", min: 0, max: 4, step: 0.05 },
  { key: "saturation", label: "Saturation", min: 0, max: 4, step: 0.05 },
];

export function makeImagePanel(session: ImageSession) {
  return function ImagePanel() {
    const [, bump] = useReducer((n: number) => n + 1, 0);
    useEffect(() => {
      const sub = session.onDidChange(bump);
      return () => sub.dispose();
    }, []);

    const s = session.state();
    const engineLine =
      s.engine === "ready"
        ? s.gpu
          ? "ready (WebGPU)"
          : "ready (no WebGPU — adjustments disabled)"
        : s.engine;

    return (
      <div style={{ padding: "var(--space-2, 8px)", fontSize: "12px" }}>
        <div style={kicker}>paged.image — adjustments (M4 ingest slice)</div>

        <div style={row}>
          <span>Bundle</span>
          <span style={mono}>
            {manifest.id}@{manifest.version}
          </span>
        </div>
        <div style={row}>
          <span>Engine</span>
          <span style={{ textAlign: "right" }}>{engineLine}</span>
        </div>
        {s.engineDetail ? <div style={note}>{s.engineDetail}</div> : null}

        <div style={row}>
          <span>Source</span>
          <span style={{ ...mono, textAlign: "right" }}>
            {s.source
              ? `${s.source.name} ${s.source.width}×${s.source.height}`
              : "none"}
          </span>
        </div>

        <div
          style={{
            display: "flex",
            gap: "var(--space-2, 8px)",
            marginTop: "var(--space-2, 8px)",
          }}
        >
          <button
            type="button"
            disabled={s.busy}
            onClick={() => void session.ingestSelection()}
          >
            Use selected frame
          </button>
        </div>

        {SLIDERS.map(({ key, label, min, max, step }) => (
          <div style={row} key={key}>
            <label htmlFor={`pg-image-${key}`}>{label}</label>
            <span
              style={{
                display: "flex",
                gap: "var(--space-1, 4px)",
                alignItems: "center",
              }}
            >
              <input
                id={`pg-image-${key}`}
                type="range"
                min={min}
                max={max}
                step={step}
                value={s.params[key]}
                disabled={s.busy || !s.source}
                onChange={(e) =>
                  session.setParams({ [key]: Number(e.target.value) })
                }
              />
              <span style={{ ...mono, minWidth: "3.5em", textAlign: "right" }}>
                {s.params[key].toFixed(2)}
              </span>
            </span>
          </div>
        ))}

        <div
          style={{
            display: "flex",
            gap: "var(--space-2, 8px)",
            marginTop: "var(--space-2, 8px)",
          }}
        >
          <button
            type="button"
            disabled={s.busy || !s.source}
            onClick={() => void session.apply()}
          >
            Apply
          </button>
          <button type="button" disabled={s.busy} onClick={() => void session.reset()}>
            Reset
          </button>
        </div>

        <div style={note}>{s.status}</div>
        <div style={note}>
          Apply composites an in-frame PREVIEW layer (C-1 Stage A) — the
          document and the placed file are unchanged. Adjusted-pixel
          save-back and interactive (per-drag) preview are later milestones
          (Stage B / M2), not faked here.
        </div>
      </div>
    );
  };
}
