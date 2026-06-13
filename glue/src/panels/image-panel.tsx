// The adjustments panel — an expert React leaf over the ingest session.
// Phase 6 grew it from the four base sliders into the levels / curves /
// white-balance panel with histograms, plus the crop controls. Honest
// seams throughout: the engine/GPU state is stated; the composite is named
// a PREVIEW (the document is never mutated); parameter edits hit the GPU +
// scene channel only on the committed Apply (Stage-A re-submit-on-commit,
// per-drag interactivity is Stage B / M2). The histogram + curve LUT +
// crop geometry are the engine's deterministic Rust (image_core / the
// reduce histogram); this leaf only renders + forwards.

import { useEffect, useReducer, useState } from "react";
import type { CSSProperties } from "react";

import manifest from "@paged-media/image-manifest/manifest.json";

import type { ImageSession } from "../session";
import type { AdjustParams, ImageHistogram } from "../engine";
import type { AspectPreset } from "../crop-machine";

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

const sectionTitle: CSSProperties = {
  ...kicker,
  marginTop: "var(--space-3, 12px)",
  marginBottom: "var(--space-1, 4px)",
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
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  disabled: boolean;
}

function Slider({ label, min, max, step, value, onChange, disabled }: SliderSpec) {
  return (
    <div style={row}>
      <label>{label}</label>
      <span style={{ display: "flex", gap: "var(--space-1, 4px)", alignItems: "center" }}>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <span style={{ ...mono, minWidth: "3.5em", textAlign: "right" }}>
          {value.toFixed(2)}
        </span>
      </span>
    </div>
  );
}

// ── histogram ────────────────────────────────────────────────────────

const HIST_W = 256;
const HIST_H = 60;

/** Render one channel's 256 bins as a normalized SVG area path. */
function channelPath(bins: Uint32Array, max: number): string {
  if (max <= 0) return "";
  let d = `M0,${HIST_H}`;
  for (let i = 0; i < 256; i++) {
    const y = HIST_H - (bins[i] / max) * HIST_H;
    d += ` L${i},${y.toFixed(2)}`;
  }
  d += ` L${HIST_W - 1},${HIST_H} Z`;
  return d;
}

function HistogramView({ hist }: { hist: ImageHistogram }) {
  // Shared vertical scale across channels (the tallest bin overall), so
  // the channels are comparable — clipping the very tallest bin (often a
  // spike at 0/255) by capping at the 2nd-tallest keeps detail visible.
  const peak = (b: Uint32Array) => {
    const s = [...b].sort((x, y) => y - x);
    return s[1] ?? s[0] ?? 1;
  };
  const max = Math.max(peak(hist.r), peak(hist.g), peak(hist.b), peak(hist.luma), 1);
  return (
    <svg
      viewBox={`0 0 ${HIST_W} ${HIST_H}`}
      preserveAspectRatio="none"
      style={{
        width: "100%",
        height: HIST_H,
        background: "var(--pg-surface-2, rgba(127,127,127,0.12))",
        borderRadius: "3px",
      }}
      role="img"
      aria-label="RGB and luma histogram"
    >
      <path d={channelPath(hist.luma, max)} fill="rgba(160,160,160,0.5)" />
      <path d={channelPath(hist.r, max)} fill="rgba(220,60,60,0.55)" />
      <path d={channelPath(hist.g, max)} fill="rgba(60,200,90,0.55)" />
      <path d={channelPath(hist.b, max)} fill="rgba(70,120,235,0.55)" />
    </svg>
  );
}

// ── curves (control-point editor) ────────────────────────────────────

/** The default identity curve control points (input, output) in [0,1]. */
const IDENTITY_CURVE: Array<[number, number]> = [
  [0, 0],
  [0.5, 0.5],
  [1, 1],
];

const CURVE_SIZE = 140;

/** A draggable control-point curve editor. The points drive the LUT the
 *  curves kernel-stage consumes (built engine-side by `engine.curveLut`);
 *  this UI only edits the points + previews the polyline through them.
 *  v0 previews a straight polyline between points (the engine applies the
 *  monotone-cubic LUT) — the honest UI subset. */
function CurveEditor({
  points,
  onChange,
  disabled,
}: {
  points: Array<[number, number]>;
  onChange: (p: Array<[number, number]>) => void;
  disabled: boolean;
}) {
  const [drag, setDrag] = useState<number | null>(null);

  const toScreen = (p: [number, number]): [number, number] => [
    p[0] * CURVE_SIZE,
    (1 - p[1]) * CURVE_SIZE,
  ];

  const updatePoint = (i: number, clientX: number, clientY: number, svg: SVGSVGElement) => {
    const r = svg.getBoundingClientRect();
    let x = (clientX - r.left) / r.width;
    let y = 1 - (clientY - r.top) / r.height;
    x = Math.min(1, Math.max(0, x));
    y = Math.min(1, Math.max(0, y));
    const next = points.map((p, j) => (j === i ? ([x, y] as [number, number]) : p));
    // Endpoints keep their input fixed (0 and 1) — only their output moves.
    if (i === 0) next[0] = [0, y];
    if (i === points.length - 1) next[i] = [1, y];
    onChange(next);
  };

  const line =
    "M" +
    points
      .map((p) => {
        const s = toScreen(p);
        return `${s[0].toFixed(1)},${s[1].toFixed(1)}`;
      })
      .join(" L");

  return (
    <svg
      viewBox={`0 0 ${CURVE_SIZE} ${CURVE_SIZE}`}
      style={{
        width: CURVE_SIZE,
        height: CURVE_SIZE,
        background: "var(--pg-surface-2, rgba(127,127,127,0.12))",
        borderRadius: "3px",
        touchAction: "none",
        opacity: disabled ? 0.5 : 1,
      }}
      onPointerMove={(e) => {
        if (drag === null || disabled) return;
        updatePoint(drag, e.clientX, e.clientY, e.currentTarget);
      }}
      onPointerUp={() => setDrag(null)}
      onPointerLeave={() => setDrag(null)}
      role="img"
      aria-label="Tone curve editor"
    >
      <line x1="0" y1={CURVE_SIZE} x2={CURVE_SIZE} y2="0" stroke="rgba(127,127,127,0.3)" />
      <path d={line} fill="none" stroke="var(--pg-accent, #6ab0ff)" strokeWidth="1.5" />
      {points.map((p, i) => {
        const s = toScreen(p);
        return (
          <circle
            key={i}
            cx={s[0]}
            cy={s[1]}
            r={5}
            fill="var(--pg-accent, #6ab0ff)"
            style={{ cursor: disabled ? "default" : "grab" }}
            onPointerDown={(e) => {
              if (disabled) return;
              (e.target as Element).setPointerCapture?.(e.pointerId);
              setDrag(i);
            }}
          />
        );
      })}
    </svg>
  );
}

// ── crop ─────────────────────────────────────────────────────────────

const ASPECTS: AspectPreset[] = ["free", "original", "1:1", "3:2", "4:3", "16:9"];

export function makeImagePanel(session: ImageSession) {
  return function ImagePanel() {
    const [, bump] = useReducer((n: number) => n + 1, 0);
    const [curvePoints, setCurvePoints] =
      useState<Array<[number, number]>>(IDENTITY_CURVE);
    const [aspect, setAspect] = useState<AspectPreset>("free");
    const [angle, setAngle] = useState(0);

    useEffect(() => {
      const sub = session.onDidChange(bump);
      return () => sub.dispose();
    }, []);

    const s = session.state();
    const p = s.params;
    const disabled = s.busy || !s.source;
    const engineLine =
      s.engine === "ready"
        ? s.gpu
          ? "ready (WebGPU)"
          : "ready (no WebGPU — adjustments disabled)"
        : s.engine;

    const setBase = (k: keyof AdjustParams, v: number) => session.setParams({ [k]: v });

    const pushCurve = (next: Array<[number, number]>) => {
      setCurvePoints(next);
      session.setCurvePoints(next);
    };

    const machine = session.cropMachine();

    return (
      <div style={{ padding: "var(--space-2, 8px)", fontSize: "12px" }}>
        <div style={kicker}>paged.image — levels / curves / white balance</div>

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
            {s.source ? `${s.source.name} ${s.source.width}×${s.source.height}` : "none"}
          </span>
        </div>

        <div style={{ display: "flex", gap: "var(--space-2, 8px)", marginTop: "var(--space-2, 8px)" }}>
          <button type="button" disabled={s.busy} onClick={() => void session.ingestSelection()}>
            Use selected frame
          </button>
        </div>

        {/* Histogram */}
        <div style={sectionTitle}>Histogram (R / G / B / luma)</div>
        {s.histogram ? (
          <HistogramView hist={s.histogram} />
        ) : (
          <div style={note}>Ingest an image to see its histogram.</div>
        )}

        {/* Tone / color base */}
        <div style={sectionTitle}>Tone</div>
        <Slider label="Exposure (EV)" min={-5} max={5} step={0.1} value={p.exposureEv} disabled={disabled} onChange={(v) => setBase("exposureEv", v)} />
        <Slider label="Brightness" min={-1} max={1} step={0.05} value={p.brightness} disabled={disabled} onChange={(v) => setBase("brightness", v)} />
        <Slider label="Contrast" min={0} max={4} step={0.05} value={p.contrast} disabled={disabled} onChange={(v) => setBase("contrast", v)} />
        <Slider label="Saturation" min={0} max={4} step={0.05} value={p.saturation} disabled={disabled} onChange={(v) => setBase("saturation", v)} />

        {/* White balance */}
        <div style={sectionTitle}>White balance</div>
        <Slider label="Temp" min={-1} max={1} step={0.02} value={p.temp} disabled={disabled} onChange={(v) => setBase("temp", v)} />
        <Slider label="Tint" min={-1} max={1} step={0.02} value={p.tint} disabled={disabled} onChange={(v) => setBase("tint", v)} />

        {/* Levels */}
        <div style={sectionTitle}>Levels (composite)</div>
        <Slider label="In black" min={0} max={1} step={0.01} value={p.levels.inBlack} disabled={disabled} onChange={(v) => session.setLevels({ inBlack: v })} />
        <Slider label="Gamma" min={0.1} max={4} step={0.05} value={p.levels.gamma} disabled={disabled} onChange={(v) => session.setLevels({ gamma: v })} />
        <Slider label="In white" min={0} max={1} step={0.01} value={p.levels.inWhite} disabled={disabled} onChange={(v) => session.setLevels({ inWhite: v })} />
        <Slider label="Out black" min={0} max={1} step={0.01} value={p.levels.outBlack} disabled={disabled} onChange={(v) => session.setLevels({ outBlack: v })} />
        <Slider label="Out white" min={0} max={1} step={0.01} value={p.levels.outWhite} disabled={disabled} onChange={(v) => session.setLevels({ outWhite: v })} />

        {/* Curves */}
        <div style={sectionTitle}>Curves</div>
        <div style={{ display: "flex", gap: "var(--space-2, 8px)", alignItems: "flex-start" }}>
          <CurveEditor points={curvePoints} onChange={pushCurve} disabled={disabled} />
          <button
            type="button"
            disabled={disabled}
            onClick={() => pushCurve(IDENTITY_CURVE.map((q) => [...q] as [number, number]))}
          >
            Reset curve
          </button>
        </div>

        {/* Crop + straighten */}
        <div style={sectionTitle}>Crop + straighten</div>
        <div style={row}>
          <label htmlFor="pg-image-aspect">Aspect</label>
          <select
            id="pg-image-aspect"
            value={aspect}
            disabled={disabled || !machine}
            onChange={(e) => {
              const a = e.target.value as AspectPreset;
              setAspect(a);
              machine?.setPreset(a);
              bump();
            }}
          >
            {ASPECTS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
        <Slider
          label="Straighten°"
          min={-45}
          max={45}
          step={0.5}
          value={angle}
          disabled={disabled || !machine}
          onChange={(v) => {
            setAngle(v);
            machine?.setAngle(v);
            bump();
          }}
        />
        <div style={{ display: "flex", gap: "var(--space-2, 8px)", marginTop: "var(--space-1, 4px)" }}>
          <button type="button" disabled={disabled || !machine} onClick={() => void session.commitCrop()}>
            Apply crop
          </button>
          <button
            type="button"
            disabled={disabled || !machine}
            onClick={() => {
              if (s.source) machine?.reset(s.source.width, s.source.height);
              setAspect("free");
              setAngle(0);
              bump();
            }}
          >
            Reset crop
          </button>
        </div>

        {/* Commit */}
        <div style={{ display: "flex", gap: "var(--space-2, 8px)", marginTop: "var(--space-3, 12px)" }}>
          <button type="button" disabled={disabled} onClick={() => void session.apply()}>
            Apply
          </button>
          <button
            type="button"
            disabled={s.busy}
            onClick={() => {
              void session.reset();
              setCurvePoints(IDENTITY_CURVE);
              setAspect("free");
              setAngle(0);
            }}
          >
            Reset
          </button>
        </div>

        <div style={note}>{s.status}</div>
        <div style={note}>
          Apply composites an in-frame PREVIEW layer (C-1 Stage A) — the
          document and the placed file are unchanged. The crop commit cuts
          the engine source (axis-aligned; the straighten angle previews the
          frame but the rotation resample is a follow-on stage). Adjusted-
          pixel save-back and per-drag preview are later milestones.
        </div>
      </div>
    );
  };
}
