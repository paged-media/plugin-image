/*
 * This file is part of paged (https://paged.media).
 *
 * paged is free software: you may redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License, version 3, as published by
 * the Free Software Foundation, OR under the Paged Media Enterprise License
 * (PMEL), a commercial license available from And The Next GmbH. Full
 * copyright and license information is available in LICENSE.md, distributed
 * with this source code.
 *
 * paged is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the licenses for details.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

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
import type { CSSProperties, ReactNode } from "react";

import manifest from "../../manifest.json";

import type { ImageSession } from "../session";
import type {
  AdjustParams,
  BrushParams,
  BrushStats,
  GradientKind,
  ImageHistogram,
  LayerHistory,
  LayerInfo,
  LevelsChannel,
  PressureTarget,
  ResampleFilter,
  Rgba01,
} from "../engine";
import {
  DEFAULT_BW_WEIGHTS,
  displayTreatmentLabel,
  GRADIENT_KINDS,
  PRESSURE_TARGETS,
} from "../engine";
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

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
  disabled,
}: SliderSpec) {
  return (
    <div style={row}>
      <label>{label}</label>
      <span
        style={{
          display: "flex",
          gap: "var(--space-1, 4px)",
          alignItems: "center",
        }}
      >
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

/** A compact numeric cell — the Channel-Mixer / per-channel-Levels grid
 *  idiom (sliders would not fit three columns; the number IS the value). */
function Num({
  value,
  onChange,
  disabled,
  step = 0.05,
  title,
  width = 46,
  testAttr,
}: {
  value: number;
  onChange: (v: number) => void;
  disabled: boolean;
  step?: number;
  title?: string;
  width?: number;
  testAttr?: string;
}) {
  const attrs = testAttr ? { [testAttr]: "" } : {};
  return (
    <input
      type="number"
      step={step}
      value={value}
      title={title}
      disabled={disabled}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ width, font: "11px var(--font-mono, monospace)" }}
      {...attrs}
    />
  );
}

/** A labelled row of compact numbers (the grid idiom's line). */
function NumRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div style={{ ...row, gap: "var(--space-1, 4px)" }}>
      <label style={{ flex: 1, minWidth: 0 }}>{label}</label>
      <span style={{ display: "flex", gap: 3 }}>{children}</span>
    </div>
  );
}

/** A checkbox gate (the "this looks destructive, opt in" idiom). */
function Gate({
  label,
  checked,
  onChange,
  disabled,
  testAttr,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled: boolean;
  testAttr?: string;
}) {
  const attrs = testAttr ? { [testAttr]: "" } : {};
  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        font: "12px var(--font-sans, sans-serif)",
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        {...attrs}
      />
      {label}
    </label>
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
  const max = Math.max(
    peak(hist.r),
    peak(hist.g),
    peak(hist.b),
    peak(hist.luma),
    1,
  );
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

  const updatePoint = (
    i: number,
    clientX: number,
    clientY: number,
    svg: SVGSVGElement,
  ) => {
    const r = svg.getBoundingClientRect();
    let x = (clientX - r.left) / r.width;
    let y = 1 - (clientY - r.top) / r.height;
    x = Math.min(1, Math.max(0, x));
    y = Math.min(1, Math.max(0, y));
    const next = points.map((p, j) =>
      j === i ? ([x, y] as [number, number]) : p,
    );
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
      <line
        x1="0"
        y1={CURVE_SIZE}
        x2={CURVE_SIZE}
        y2="0"
        stroke="rgba(127,127,127,0.3)"
      />
      <path
        d={line}
        fill="none"
        stroke="var(--pg-accent, #6ab0ff)"
        strokeWidth="1.5"
      />
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

const ASPECTS: AspectPreset[] = [
  "free",
  "original",
  "1:1",
  "3:2",
  "4:3",
  "16:9",
];

// ── colour presets ───────────────────────────────────────────────────

/** A few conventional photo-filter gels (straight RGB in [0,1]). The
 *  kernel takes an arbitrary colour — this is the shortlist the panel
 *  offers instead of a colour picker the shell does not provide. */
const PHOTO_FILTERS: Array<{ name: string; rgb: [number, number, number] }> = [
  { name: "Warming (85)", rgb: [0.925, 0.639, 0.365] },
  { name: "Warming (81)", rgb: [0.92, 0.78, 0.55] },
  { name: "Cooling (80)", rgb: [0.0, 0.44, 0.88] },
  { name: "Cooling (82)", rgb: [0.44, 0.71, 0.94] },
  { name: "Sepia", rgb: [0.67, 0.51, 0.25] },
  { name: "Red", rgb: [0.91, 0.14, 0.16] },
  { name: "Green", rgb: [0.14, 0.71, 0.32] },
  { name: "Blue", rgb: [0.11, 0.28, 0.82] },
];

/** The gradient stop wells (no colour-picker door in the contract, so
 *  the panel offers a shortlist — the ENGINE takes any RGBA). */
const STOPS: Array<{ name: string; rgba: Rgba01 }> = [
  { name: "Black", rgba: [0, 0, 0, 1] },
  { name: "White", rgba: [1, 1, 1, 1] },
  { name: "Red", rgba: [1, 0, 0, 1] },
  { name: "Green", rgba: [0, 1, 0, 1] },
  { name: "Blue", rgba: [0, 0, 1, 1] },
  { name: "Transparent", rgba: [0, 0, 0, 0] },
];

const stopCss = (c: Rgba01) =>
  `rgba(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(
    c[2] * 255,
  )}, ${c[3]})`;

// ── brush ────────────────────────────────────────────────────────────

/** The brush colour wells. The engine takes ANY straight RGBA; this is
 *  the shortlist the panel offers because the host contract wires no
 *  colour-picker door (the same reason the gradient stops are a list). */
const BRUSH_COLORS: Array<{ name: string; rgba: Rgba01 }> = [
  { name: "Black", rgba: [0, 0, 0, 1] },
  { name: "White", rgba: [1, 1, 1, 1] },
  { name: "50% grey", rgba: [0.5, 0.5, 0.5, 1] },
  { name: "Red", rgba: [1, 0, 0, 1] },
  { name: "Green", rgba: [0, 1, 0, 1] },
  { name: "Blue", rgba: [0, 0, 1, 1] },
  { name: "Yellow", rgba: [1, 1, 0, 1] },
  { name: "Magenta", rgba: [1, 0, 1, 1] },
  { name: "Cyan", rgba: [0, 1, 1, 1] },
];

/**
 * THE SCOPE OF PAINTING, in the UI rather than in a code comment.
 *
 * Exported so a spec pins the WORDING: an honesty note that can be
 * edited away silently is not a guarantee (the paged.draw Appearance
 * panel's `APPEARANCE_BAKE_NOTE` convention).
 */
export const BRUSH_SCOPE_NOTE =
  "A stroke paints the ACTIVE LAYER, not the flattened image — add a layer " +
  "and what is underneath survives. A committed stroke is UNDOABLE: the " +
  "tiles it touched are journaled and Undo restores them byte for byte. " +
  "That history is BOUNDED — 32 steps or 256 MB, whichever runs out first; " +
  "past the bound the oldest edits are dropped and become permanent, and " +
  "the panel says how many. Painting is still destructive INTO its own " +
  "layer: paint on the background and what it covered is gone. Layer " +
  "STRUCTURE — add, remove, reorder, opacity, blend — is not journaled, so " +
  "a removed layer does not come back. The document and the source file " +
  "are never touched (the in-frame result stays a preview layer). A stroke " +
  "still in flight can be abandoned — the base pixels are held until you " +
  "release. While you drag, the frame shows the whole stack with the " +
  "stroke composited into it; the adjustment chain re-runs on release.";

/**
 * THE SCOPE OF THE LAYER GRAPH itself — what a layer here is and, more
 * to the point, what it is not. Exported and spec-pinned for the same
 * reason as [`BRUSH_SCOPE_NOTE`].
 */
export const LAYERS_SCOPE_NOTE =
  "Layers are canvas-extent PIXEL layers composited bottom-up through the " +
  "engine's own compose.* kernels — the same 26 blend modes the brush " +
  "paints through, so nothing here is a second implementation. There are " +
  "no groups, no clipping masks, no per-layer masks and no adjustment " +
  "layers: an adjustment is either the panel's re-runnable preview over " +
  "the whole composite, or a one-way bake into the active layer. Undo " +
  "covers PIXEL edits only — paint, fills and bakes; adding, removing, " +
  "reordering and re-blending are not journaled, and removing a layer " +
  "clears the history outright (its entries could never be replayed). A " +
  "crop, resize or straighten changes the canvas extent and therefore " +
  "FLATTENS the stack, and the undo history goes with it. Export and " +
  "save-back write the FLATTENED composite, not the layers — this stack " +
  "lives in the session, and a PSD saved back is still one layer or a " +
  "flatten, exactly as it was before. A PSD opens as its own layers only " +
  "when this model reproduces it exactly — flat, unclipped, unmasked, " +
  "8-bit RGB; anything else keeps Photoshop's merged composite as a " +
  "single layer and says why.";

/** The paint parameters section — a PURE component (props in, elements
 *  out, no hooks) so a spec can render it without a DOM and assert what
 *  it says. */
export function BrushSection({
  brush,
  blendModes,
  strokeActive,
  strokeStats,
  masked,
  gpu,
  disabled,
  onChange,
}: {
  brush: BrushParams;
  /** From the engine's `compose.*` registry — never a hardcoded list. */
  blendModes: readonly string[];
  strokeActive: boolean;
  strokeStats: BrushStats | null;
  /** A selection exists, so strokes are clipped to it. */
  masked: boolean;
  gpu: boolean;
  disabled: boolean;
  onChange: (patch: Partial<BrushParams>) => void;
}) {
  const colorIndex = BRUSH_COLORS.findIndex(
    (c) =>
      c.rgba[0] === brush.color[0] &&
      c.rgba[1] === brush.color[1] &&
      c.rgba[2] === brush.color[2] &&
      c.rgba[3] === brush.color[3],
  );
  return (
    <>
      <div style={sectionTitle}>
        Brush{masked ? " — clipped to the selection" : ""}
      </div>
      <Slider
        label="Size (px)"
        min={1}
        max={512}
        step={1}
        value={brush.size}
        disabled={disabled}
        onChange={(size) => onChange({ size })}
      />
      <Slider
        label="Hardness"
        min={0}
        max={1}
        step={0.05}
        value={brush.hardness}
        disabled={disabled}
        onChange={(hardness) => onChange({ hardness })}
      />
      <Slider
        label="Opacity"
        min={0}
        max={1}
        step={0.05}
        value={brush.opacity}
        disabled={disabled}
        onChange={(opacity) => onChange({ opacity })}
      />
      <Slider
        label="Flow"
        min={0}
        max={1}
        step={0.05}
        value={brush.flow}
        disabled={disabled}
        onChange={(flow) => onChange({ flow })}
      />
      <Slider
        label="Spacing (× diameter)"
        min={0.01}
        max={1}
        step={0.01}
        value={brush.spacing}
        disabled={disabled}
        onChange={(spacing) => onChange({ spacing })}
      />
      <div style={row}>
        <label htmlFor="pg-image-brush-blend">Blend</label>
        <select
          id="pg-image-brush-blend"
          data-image-brush-blend
          value={brush.blend}
          disabled={disabled || blendModes.length === 0}
          onChange={(e) => onChange({ blend: e.target.value })}
        >
          {blendModes.length > 0 ? (
            blendModes.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))
          ) : (
            <option
              value={brush.blend}
            >{`${brush.blend} (engine not booted)`}</option>
          )}
        </select>
      </div>
      <div style={row}>
        <label htmlFor="pg-image-brush-color">Colour</label>
        <select
          id="pg-image-brush-color"
          data-image-brush-color
          value={colorIndex < 0 ? 0 : colorIndex}
          disabled={disabled}
          style={{ background: stopCss(brush.color) }}
          onChange={(e) =>
            onChange({ color: [...BRUSH_COLORS[Number(e.target.value)].rgba] })
          }
        >
          {BRUSH_COLORS.map((c, i) => (
            <option key={c.name} value={i}>
              {c.name}
            </option>
          ))}
        </select>
      </div>
      <div style={row}>
        <label htmlFor="pg-image-brush-pressure">Pen pressure drives</label>
        <select
          id="pg-image-brush-pressure"
          data-image-brush-pressure
          value={brush.pressureTarget}
          disabled={disabled}
          onChange={(e) =>
            onChange({ pressureTarget: e.target.value as PressureTarget })
          }
        >
          {PRESSURE_TARGETS.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </div>
      {strokeActive ? (
        <div style={row}>
          <span>Stroke</span>
          <span style={mono} data-image-brush-stats>
            {strokeStats
              ? `${strokeStats.dabs} dabs · ${Math.round(strokeStats.x)},${Math.round(
                  strokeStats.y,
                )} ${Math.round(strokeStats.w)}×${Math.round(strokeStats.h)}`
              : "in progress"}
          </span>
        </div>
      ) : null}
      {!gpu ? (
        <div style={note}>
          Painting is GPU-only — the dab composite is a registered WGSL kernel
          dispatch and no CPU blend path ships. Without a WebGPU device the
          paint tools decline instead of painting differently.
        </div>
      ) : null}
      <div style={note} data-image-brush-note>
        {BRUSH_SCOPE_NOTE}
      </div>
      <div style={note}>
        Colours are a shortlist (the host contract wires no colour-picker door);
        the blend list is the engine&apos;s own `compose.*` kernel registry.
        Parameters are frozen into each stroke at pointer-down — a stroke whose
        size changed halfway through would not replay.
      </div>
    </>
  );
}

/** Bytes → a compact "12.3 MB" for the history readout. */
function mb(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** The LAYERS palette + the undo readout — a PURE component (props in,
 *  elements out, no hooks) so a spec can render it without a DOM and
 *  assert what it says. Rows read TOP-DOWN (a layers palette's
 *  convention); the engine's own order is bottom-first. */
export function LayersSection({
  layers,
  active,
  history,
  blendModes,
  layersNote,
  gpu,
  disabled,
  onSelect,
  onAdd,
  onDuplicate,
  onRemove,
  onMove,
  onVisible,
  onOpacity,
  onBlend,
  onLock,
  onAddAdjustment,
  onMaskFromSelection,
  onMaskToggle,
  onMaskClear,
  onBake,
  onUndo,
  onRedo,
}: {
  /** BOTTOM-first, as the engine holds them. */
  layers: readonly LayerInfo[];
  active: number;
  history: LayerHistory | null;
  blendModes: readonly string[];
  /** How the stack was opened (the PSD lane's honest one-liner). */
  layersNote: string | null;
  gpu: boolean;
  disabled: boolean;
  onSelect: (index: number) => void;
  onAdd: () => void;
  onDuplicate: (index: number) => void;
  onRemove: (index: number) => void;
  onMove: (from: number, to: number) => void;
  onVisible: (index: number, visible: boolean) => void;
  onOpacity: (index: number, opacity: number) => void;
  onBlend: (index: number, blend: string) => void;
  onLock: (index: number, locked: boolean) => void;
  /** Stack the chain as a non-destructive adjustment layer. */
  onAddAdjustment: () => void;
  /** Make the current selection this layer's mask. */
  onMaskFromSelection: (index: number) => void;
  /** Toggle whether the mask applies (the coverage is retained). */
  onMaskToggle: (index: number, enabled: boolean) => void;
  /** Delete the mask outright. */
  onMaskClear: (index: number) => void;
  onBake: () => void;
  onUndo: () => void;
  onRedo: () => void;
}) {
  // Top-down for reading; the engine's indices are preserved on each row.
  const rows = [...layers].reverse();
  const multi = layers.length > 1;
  return (
    <>
      <div style={sectionTitle} data-image-layers-title>
        Layers{layers.length > 0 ? ` (${layers.length})` : ""}
      </div>
      {layers.length === 0 ? (
        <div style={note}>Ingest an image to open its layer stack.</div>
      ) : null}
      {rows.map((l) => (
        <div
          key={l.id}
          data-image-layer-row={l.index}
          style={{
            ...row,
            background:
              l.index === active
                ? "var(--pg-accent-soft, rgba(127,127,255,0.12))"
                : undefined,
          }}
        >
          <span
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              minWidth: 0,
            }}
          >
            <input
              type="checkbox"
              title="Visible"
              data-image-layer-visible={l.index}
              checked={l.visible}
              disabled={disabled}
              onChange={(e) => onVisible(l.index, e.target.checked)}
            />
            <button
              type="button"
              data-image-layer-select={l.index}
              disabled={disabled}
              onClick={() => onSelect(l.index)}
              style={{
                font: "12px var(--font-sans, sans-serif)",
                fontWeight: l.index === active ? 600 : 400,
                background: "none",
                border: "none",
                padding: 0,
                cursor: "pointer",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {l.name}
              {l.index === active ? " ●" : ""}
            </button>
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <select
              data-image-layer-blend={l.index}
              value={l.blend}
              disabled={disabled || blendModes.length === 0}
              title="Blend mode"
              onChange={(e) => onBlend(l.index, e.target.value)}
              style={{
                font: "11px var(--font-sans, sans-serif)",
                maxWidth: 96,
              }}
            >
              {(blendModes.length > 0 ? blendModes : [l.blend]).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              title="Opacity"
              data-image-layer-opacity={l.index}
              value={l.opacity}
              disabled={disabled}
              onChange={(e) => onOpacity(l.index, Number(e.target.value))}
              style={{ width: 56 }}
            />
            <input
              type="checkbox"
              title="Lock pixels (paint, fills and bakes refuse; properties still move)"
              data-image-layer-lock={l.index}
              checked={l.locked}
              disabled={disabled}
              onChange={(e) => onLock(l.index, e.target.checked)}
            />
            {/* MASK. Three states in two controls, because "no mask",
                "masked" and "masked but disabled" are genuinely
                different and collapsing them would lose the one users
                rely on: a disabled mask KEEPS its coverage. */}
            {l.hasMask ? (
              <>
                <input
                  type="checkbox"
                  title={
                    l.maskEnabled
                      ? "Mask is applying — uncheck to disable it (the coverage is kept)"
                      : "Mask is disabled but retained — check to apply it again"
                  }
                  data-image-layer-mask={l.index}
                  checked={l.maskEnabled}
                  disabled={disabled}
                  onChange={(e) => onMaskToggle(l.index, e.target.checked)}
                />
                <button
                  type="button"
                  title="Delete this layer's mask (the coverage is lost)"
                  data-image-layer-mask-clear={l.index}
                  disabled={disabled}
                  onClick={() => onMaskClear(l.index)}
                >
                  ⌫
                </button>
              </>
            ) : (
              <button
                type="button"
                title="Make the current selection this layer's mask"
                data-image-layer-mask-add={l.index}
                disabled={disabled}
                onClick={() => onMaskFromSelection(l.index)}
              >
                ⬚
              </button>
            )}
            <button
              type="button"
              title="Move up"
              data-image-layer-up={l.index}
              disabled={disabled || l.index === layers.length - 1}
              onClick={() => onMove(l.index, l.index + 1)}
            >
              ↑
            </button>
            <button
              type="button"
              title="Move down"
              data-image-layer-down={l.index}
              disabled={disabled || l.index === 0}
              onClick={() => onMove(l.index, l.index - 1)}
            >
              ↓
            </button>
            <button
              type="button"
              title="Duplicate"
              data-image-layer-duplicate={l.index}
              disabled={disabled}
              onClick={() => onDuplicate(l.index)}
            >
              ⧉
            </button>
            <button
              type="button"
              title={
                multi
                  ? "Remove (NOT undoable — layer structure is not journaled)"
                  : "A document keeps at least one layer"
              }
              data-image-layer-remove={l.index}
              disabled={disabled || !multi}
              onClick={() => onRemove(l.index)}
            >
              ✕
            </button>
          </span>
        </div>
      ))}
      <div
        style={{ display: "flex", gap: 6, marginTop: "var(--space-1, 4px)" }}
      >
        <button
          type="button"
          data-image-layer-add
          disabled={disabled}
          onClick={onAdd}
        >
          Add layer
        </button>
        <button
          type="button"
          data-image-layer-add-adjustment
          disabled={disabled}
          title="Stack the adjustment chain as a non-destructive layer — it transforms everything beneath it, and deleting it restores the original exactly"
          onClick={onAddAdjustment}
        >
          Add adjustment layer
        </button>
        <button
          type="button"
          data-image-layer-bake
          disabled={disabled || !gpu}
          title="Bake the adjustment chain destructively into the active layer (undoable)"
          onClick={onBake}
        >
          Bake adjustments into layer
        </button>
      </div>
      <div
        style={{ display: "flex", gap: 6, marginTop: "var(--space-1, 4px)" }}
      >
        <button
          type="button"
          data-image-undo
          disabled={disabled || !history?.canUndo}
          onClick={onUndo}
        >
          Undo{history?.undoLabel ? ` ${history.undoLabel}` : ""}
        </button>
        <button
          type="button"
          data-image-redo
          disabled={disabled || !history?.canRedo}
          onClick={onRedo}
        >
          Redo{history?.redoLabel ? ` ${history.redoLabel}` : ""}
        </button>
      </div>
      {history ? (
        <div style={note} data-image-history-readout>
          History: {history.depth} undo / {history.redoDepth} redo,{" "}
          {mb(history.bytes)} of {mb(history.maxBytes)} and {history.maxEntries}{" "}
          steps.
          {history.dropped > 0
            ? ` ${history.dropped} older edit${
                history.dropped === 1 ? "" : "s"
              } fell past that bound and ${
                history.dropped === 1 ? "is" : "are"
              } now permanent.`
            : " Nothing has fallen past the bound yet."}
        </div>
      ) : null}
      {layersNote ? (
        <div style={note} data-image-layers-source-note>
          {layersNote}
        </div>
      ) : null}
      {!gpu && layers.length > 1 ? (
        <div style={note}>
          Compositing more than one layer is GPU-only — every blend is a
          registered WGSL kernel dispatch and no CPU blend path ships. Without a
          WebGPU device only a single visible layer at full opacity in normal
          mode can be shown (that fold is the identity).
        </div>
      ) : null}
      <div style={note} data-image-layers-note>
        {LAYERS_SCOPE_NOTE}
      </div>
    </>
  );
}

export function makeImagePanel(session: ImageSession) {
  return function ImagePanel() {
    const [, bump] = useReducer((n: number) => n + 1, 0);
    const [curvePoints, setCurvePoints] =
      useState<Array<[number, number]>>(IDENTITY_CURVE);
    const [aspect, setAspect] = useState<AspectPreset>("free");
    const [angle, setAngle] = useState(0);
    // Generate-section local state (the request shape, not engine state).
    const [gradKind, setGradKind] = useState<GradientKind>("linear");
    const [stop0, setStop0] = useState(0);
    const [stop1, setStop1] = useState(1);
    const [noiseAmount, setNoiseAmount] = useState(0.5);

    useEffect(() => {
      const sub = session.onDidChange(bump);
      return () => sub.dispose();
    }, []);

    const s = session.state();
    const p = s.params;
    const [resizeW, setResizeW] = useState(s.source?.width ?? 0);
    const [resizeH, setResizeH] = useState(s.source?.height ?? 0);
    const [resizeFilter, setResizeFilter] =
      useState<ResampleFilter>("lanczos3");
    // Track the natural size when the source changes (a new import).
    useEffect(() => {
      setResizeW(s.source?.width ?? 0);
      setResizeH(s.source?.height ?? 0);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [s.source?.handle]);
    const disabled = s.busy || !s.source;
    const engineLine =
      s.engine === "ready"
        ? s.gpu
          ? "ready (WebGPU)"
          : "ready (no WebGPU — adjustments disabled)"
        : s.engine;

    const setBase = (k: keyof AdjustParams, v: number) =>
      session.setParams({ [k]: v });

    // The extended stages are nested objects — patch them immutably and
    // hand the whole sub-object back to the session (shallow-merged).
    const setBalance = (
      range: "shadows" | "midtones" | "highlights",
      axis: 0 | 1 | 2,
      v: number,
    ) => {
      const next = { ...p.colorBalance };
      const row3 = [...next[range]] as [number, number, number];
      row3[axis] = v;
      next[range] = row3;
      session.setParams({ colorBalance: next });
    };
    const setMixer = (out: "r" | "g" | "b", i: 0 | 1 | 2 | 3, v: number) => {
      const next = { ...p.channelMixer };
      const r4 = [...next[out]] as [number, number, number, number];
      r4[i] = v;
      next[out] = r4;
      session.setParams({ channelMixer: next });
    };
    const setLevelsRgb = (
      ch: "r" | "g" | "b",
      key: keyof LevelsChannel,
      v: number,
    ) => {
      const next = { ...p.levelsRgb, [ch]: { ...p.levelsRgb[ch], [key]: v } };
      session.setParams({ levelsRgb: next });
    };
    const setBw = (i: number, v: number) => {
      const w = [...p.blackWhite.weights] as typeof p.blackWhite.weights;
      w[i] = v;
      session.setParams({ blackWhite: { ...p.blackWhite, weights: w } });
    };

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
            {s.source
              ? `${s.source.name} ${s.source.width}×${s.source.height}`
              : "none"}
          </span>
        </div>

        {/* CMS rung 1 — SAY which colour treatment the ingest applied.
            "sRGB assumed" is the honest majority case (most PNG/JPEG carry
            no profile) and a colour-critical user needs to see it, so it
            is stated rather than defaulted silently. */}
        {s.source ? (
          <div style={row}>
            <span>Colour</span>
            <span style={mono} data-image-colour-treatment={s.source.display}>
              {displayTreatmentLabel(s.source.display)}
            </span>
          </div>
        ) : null}

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

        {/* Histogram */}
        <div style={sectionTitle}>Histogram (R / G / B / luma)</div>
        {s.histogram ? (
          <HistogramView hist={s.histogram} />
        ) : (
          <div style={note}>Ingest an image to see its histogram.</div>
        )}
        <div
          style={{
            display: "flex",
            gap: "var(--space-2, 8px)",
            marginTop: "var(--space-2, 8px)",
          }}
        >
          <button
            type="button"
            data-image-auto-enhance
            disabled={disabled || !s.histogram}
            onClick={() => session.autoEnhance()}
          >
            Auto-enhance
          </button>
        </div>
        <div style={note}>
          Auto-enhance reads the histogram for auto-levels + a gray-world white
          balance and fills the sliders below; click Apply to composite.
        </div>

        {/* Selection readout — the engine-side coverage the marquee /
            lasso / wand tools build. When one exists, the committed Apply
            masks every adjust/filter dispatch (GPU mix(a, result, mask))
            + the CPU curves pass by it. */}
        <div style={sectionTitle}>Selection</div>
        {s.selection ? (
          <>
            <div style={row}>
              <span>Bounds</span>
              <span style={mono} data-image-selection-bounds>
                {`${Math.round(s.selection.x)},${Math.round(s.selection.y)} ` +
                  `${Math.round(s.selection.w)}×${Math.round(s.selection.h)}`}
              </span>
            </div>
            <div style={row}>
              <span>Coverage</span>
              <span style={mono} data-image-selection-coverage>
                {(s.selection.coverage * 100).toFixed(1)}%
              </span>
            </div>
            <div
              style={{
                display: "flex",
                gap: "var(--space-2, 8px)",
                marginTop: "var(--space-1, 4px)",
              }}
            >
              <button
                type="button"
                data-image-deselect
                disabled={s.busy}
                onClick={() => session.deselect()}
              >
                Deselect
              </button>
            </div>
          </>
        ) : (
          <div style={note}>
            No selection — adjustments apply to the whole image. Use the marquee
            / lasso / wand tools (shift = add, alt = subtract, shift+alt =
            intersect).
          </div>
        )}

        {/* Tone / color base */}
        <div style={sectionTitle}>
          Tone{s.selection ? " — applies to selection" : ""}
        </div>
        <Slider
          label="Exposure (EV)"
          min={-5}
          max={5}
          step={0.1}
          value={p.exposureEv}
          disabled={disabled}
          onChange={(v) => setBase("exposureEv", v)}
        />
        <Slider
          label="Brightness"
          min={-1}
          max={1}
          step={0.05}
          value={p.brightness}
          disabled={disabled}
          onChange={(v) => setBase("brightness", v)}
        />
        <Slider
          label="Contrast"
          min={0}
          max={4}
          step={0.05}
          value={p.contrast}
          disabled={disabled}
          onChange={(v) => setBase("contrast", v)}
        />
        <Slider
          label="Saturation"
          min={0}
          max={4}
          step={0.05}
          value={p.saturation}
          disabled={disabled}
          onChange={(v) => setBase("saturation", v)}
        />

        {/* White balance */}
        <div style={sectionTitle}>White balance</div>
        <Slider
          label="Temp"
          min={-1}
          max={1}
          step={0.02}
          value={p.temp}
          disabled={disabled}
          onChange={(v) => setBase("temp", v)}
        />
        <Slider
          label="Tint"
          min={-1}
          max={1}
          step={0.02}
          value={p.tint}
          disabled={disabled}
          onChange={(v) => setBase("tint", v)}
        />

        {/* Levels */}
        <div style={sectionTitle}>Levels (composite)</div>
        <Slider
          label="In black"
          min={0}
          max={1}
          step={0.01}
          value={p.levels.inBlack}
          disabled={disabled}
          onChange={(v) => session.setLevels({ inBlack: v })}
        />
        <Slider
          label="Gamma"
          min={0.1}
          max={4}
          step={0.05}
          value={p.levels.gamma}
          disabled={disabled}
          onChange={(v) => session.setLevels({ gamma: v })}
        />
        <Slider
          label="In white"
          min={0}
          max={1}
          step={0.01}
          value={p.levels.inWhite}
          disabled={disabled}
          onChange={(v) => session.setLevels({ inWhite: v })}
        />
        <Slider
          label="Out black"
          min={0}
          max={1}
          step={0.01}
          value={p.levels.outBlack}
          disabled={disabled}
          onChange={(v) => session.setLevels({ outBlack: v })}
        />
        <Slider
          label="Out white"
          min={0}
          max={1}
          step={0.01}
          value={p.levels.outWhite}
          disabled={disabled}
          onChange={(v) => session.setLevels({ outWhite: v })}
        />

        {/* Levels — PER CHANNEL (adjust.levels_rgb). Same Levels row
            idiom, three columns: in black / gamma / in white. The OUTPUT
            range stays composite above (the kernel has no output remap). */}
        <div style={sectionTitle}>Levels (per channel)</div>
        <NumRow label="&nbsp;">
          <span
            style={{
              ...mono,
              width: 46,
              fontSize: 10,
              opacity: 0.6,
              textAlign: "center",
            }}
          >
            blk
          </span>
          <span
            style={{
              ...mono,
              width: 46,
              fontSize: 10,
              opacity: 0.6,
              textAlign: "center",
            }}
          >
            gam
          </span>
          <span
            style={{
              ...mono,
              width: 46,
              fontSize: 10,
              opacity: 0.6,
              textAlign: "center",
            }}
          >
            wht
          </span>
        </NumRow>
        {(["r", "g", "b"] as const).map((ch) => (
          <NumRow key={ch} label={ch.toUpperCase()}>
            <Num
              testAttr={`data-image-levels-${ch}-black`}
              step={0.01}
              value={p.levelsRgb[ch].inBlack}
              disabled={disabled}
              onChange={(v) => setLevelsRgb(ch, "inBlack", v)}
              title={`${ch.toUpperCase()} in black`}
            />
            <Num
              testAttr={`data-image-levels-${ch}-gamma`}
              step={0.05}
              value={p.levelsRgb[ch].gamma}
              disabled={disabled}
              onChange={(v) => setLevelsRgb(ch, "gamma", v)}
              title={`${ch.toUpperCase()} gamma`}
            />
            <Num
              testAttr={`data-image-levels-${ch}-white`}
              step={0.01}
              value={p.levelsRgb[ch].inWhite}
              disabled={disabled}
              onChange={(v) => setLevelsRgb(ch, "inWhite", v)}
              title={`${ch.toUpperCase()} in white`}
            />
          </NumRow>
        ))}

        {/* COLOR — the grading stages (adjust.vibrance, .color_balance,
            .photo_filter, .channel_mixer). Chain order is fixed in the
            engine (documented on ingest::adjust_rgba8); this section only
            edits + forwards. */}
        <div style={sectionTitle}>Color</div>
        <Slider
          label="Vibrance"
          min={-1}
          max={1}
          step={0.05}
          value={p.vibrance}
          disabled={disabled}
          onChange={(v) => setBase("vibrance", v)}
        />

        <div style={{ ...kicker, marginTop: "var(--space-2, 8px)" }}>
          Color balance
        </div>
        <NumRow label="&nbsp;">
          <span
            style={{
              ...mono,
              width: 46,
              fontSize: 10,
              opacity: 0.6,
              textAlign: "center",
            }}
          >
            C↔R
          </span>
          <span
            style={{
              ...mono,
              width: 46,
              fontSize: 10,
              opacity: 0.6,
              textAlign: "center",
            }}
          >
            M↔G
          </span>
          <span
            style={{
              ...mono,
              width: 46,
              fontSize: 10,
              opacity: 0.6,
              textAlign: "center",
            }}
          >
            Y↔B
          </span>
        </NumRow>
        {(["shadows", "midtones", "highlights"] as const).map((range) => (
          <NumRow key={range} label={range[0].toUpperCase() + range.slice(1)}>
            {([0, 1, 2] as const).map((axis) => (
              <Num
                key={axis}
                testAttr={`data-image-balance-${range}-${axis}`}
                step={0.02}
                value={p.colorBalance[range][axis]}
                disabled={disabled}
                onChange={(v) => setBalance(range, axis, v)}
              />
            ))}
          </NumRow>
        ))}

        <div style={{ ...kicker, marginTop: "var(--space-2, 8px)" }}>
          Photo filter
        </div>
        <div style={row}>
          <label htmlFor="pg-image-photofilter">Gel</label>
          <select
            id="pg-image-photofilter"
            data-image-photo-filter
            disabled={disabled}
            value={
              PHOTO_FILTERS.findIndex(
                (f) =>
                  f.rgb[0] === p.photoFilter.color[0] &&
                  f.rgb[1] === p.photoFilter.color[1] &&
                  f.rgb[2] === p.photoFilter.color[2],
              ) < 0
                ? 0
                : PHOTO_FILTERS.findIndex(
                    (f) =>
                      f.rgb[0] === p.photoFilter.color[0] &&
                      f.rgb[1] === p.photoFilter.color[1] &&
                      f.rgb[2] === p.photoFilter.color[2],
                  )
            }
            onChange={(e) =>
              session.setParams({
                photoFilter: {
                  ...p.photoFilter,
                  color: [...PHOTO_FILTERS[Number(e.target.value)].rgb],
                },
              })
            }
          >
            {PHOTO_FILTERS.map((f, i) => (
              <option key={f.name} value={i}>
                {f.name}
              </option>
            ))}
          </select>
        </div>
        <Slider
          label="Density"
          min={0}
          max={1}
          step={0.05}
          value={p.photoFilter.density}
          disabled={disabled}
          onChange={(v) =>
            session.setParams({ photoFilter: { ...p.photoFilter, density: v } })
          }
        />
        <Gate
          label="Preserve luminosity"
          testAttr="data-image-photo-preserve"
          checked={p.photoFilter.preserveLuminosity}
          disabled={disabled}
          onChange={(preserveLuminosity) =>
            session.setParams({
              photoFilter: { ...p.photoFilter, preserveLuminosity },
            })
          }
        />

        <div style={{ ...kicker, marginTop: "var(--space-2, 8px)" }}>
          Channel mixer
        </div>
        <NumRow label="&nbsp;">
          {["R", "G", "B", "+"].map((h) => (
            <span
              key={h}
              style={{
                ...mono,
                width: 46,
                fontSize: 10,
                opacity: 0.6,
                textAlign: "center",
              }}
            >
              {h}
            </span>
          ))}
        </NumRow>
        {(["r", "g", "b"] as const).map((out) => (
          <NumRow key={out} label={`out ${out.toUpperCase()}`}>
            {([0, 1, 2, 3] as const).map((i) => (
              <Num
                key={i}
                testAttr={`data-image-mixer-${out}-${i}`}
                step={0.05}
                value={p.channelMixer[out][i]}
                disabled={disabled}
                onChange={(v) => setMixer(out, i, v)}
              />
            ))}
          </NumRow>
        ))}

        {/* EFFECTS — the range-destroying stages. Each sits behind an
            enable GATE (they look destructive, so they never ride a
            "neutral value" default the user could hit by accident). */}
        <div style={sectionTitle}>Effects</div>
        <Gate
          label="Black &amp; white"
          testAttr="data-image-bw-enable"
          checked={p.blackWhite.enabled}
          disabled={disabled}
          onChange={(enabled) =>
            session.setParams({ blackWhite: { ...p.blackWhite, enabled } })
          }
        />
        {p.blackWhite.enabled && (
          <>
            <NumRow label="R / Y / G">
              {[0, 1, 2].map((i) => (
                <Num
                  key={i}
                  testAttr={`data-image-bw-${i}`}
                  value={p.blackWhite.weights[i]}
                  disabled={disabled}
                  onChange={(v) => setBw(i, v)}
                />
              ))}
            </NumRow>
            <NumRow label="C / B / M">
              {[3, 4, 5].map((i) => (
                <Num
                  key={i}
                  testAttr={`data-image-bw-${i}`}
                  value={p.blackWhite.weights[i]}
                  disabled={disabled}
                  onChange={(v) => setBw(i, v)}
                />
              ))}
            </NumRow>
            <button
              type="button"
              disabled={disabled}
              onClick={() =>
                session.setParams({
                  blackWhite: {
                    enabled: true,
                    weights: [...DEFAULT_BW_WEIGHTS],
                  },
                })
              }
            >
              Default mix
            </button>
          </>
        )}
        <Gate
          label="Posterize"
          testAttr="data-image-posterize-enable"
          checked={p.posterizeLevels !== null}
          disabled={disabled}
          onChange={(on) =>
            session.setParams({ posterizeLevels: on ? 6 : null })
          }
        />
        {p.posterizeLevels !== null && (
          <Slider
            label="Levels"
            min={2}
            max={32}
            step={1}
            value={p.posterizeLevels}
            disabled={disabled}
            onChange={(v) => session.setParams({ posterizeLevels: v })}
          />
        )}
        <Gate
          label="Threshold"
          testAttr="data-image-threshold-enable"
          checked={p.threshold !== null}
          disabled={disabled}
          onChange={(on) => session.setParams({ threshold: on ? 0.5 : null })}
        />
        {p.threshold !== null && (
          <Slider
            label="Cut (luma)"
            min={0}
            max={1}
            step={0.01}
            value={p.threshold}
            disabled={disabled}
            onChange={(v) => session.setParams({ threshold: v })}
          />
        )}

        {/* GENERATE — the gen.* family. Unlike everything above this is
            NOT a re-runnable chain stage: it paints ONCE into the working
            image (composited through the selection mask on the GPU) and
            swaps the engine source, like a crop commit. Honest v0: a
            fixed TWO-STOP gradient whose geometry is derived from the
            selection bounds (no on-canvas drag handle yet). */}
        <div style={sectionTitle}>
          Generate
          {s.selection ? " — fills the selection" : " — fills the whole image"}
        </div>
        <div style={row}>
          <label htmlFor="pg-image-gradient">Gradient</label>
          <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <select
              id="pg-image-gradient"
              data-image-gradient-kind
              value={gradKind}
              disabled={disabled}
              onChange={(e) => setGradKind(e.target.value as GradientKind)}
            >
              {GRADIENT_KINDS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
            <select
              data-image-gradient-stop0
              value={stop0}
              disabled={disabled}
              onChange={(e) => setStop0(Number(e.target.value))}
              style={{ background: stopCss(STOPS[stop0].rgba) }}
              title="Start colour"
            >
              {STOPS.map((c, i) => (
                <option key={c.name} value={i}>
                  {c.name}
                </option>
              ))}
            </select>
            <select
              data-image-gradient-stop1
              value={stop1}
              disabled={disabled}
              onChange={(e) => setStop1(Number(e.target.value))}
              style={{ background: stopCss(STOPS[stop1].rgba) }}
              title="End colour"
            >
              {STOPS.map((c, i) => (
                <option key={c.name} value={i}>
                  {c.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              data-image-fill-gradient
              disabled={disabled || !s.gpu}
              title={
                s.gpu
                  ? "Paint the gradient into the selection"
                  : "Generators are GPU-only — no WebGPU device"
              }
              onClick={() =>
                void session.fillSelection({
                  kind: "gradient",
                  gradient: gradKind,
                  c0: [...STOPS[stop0].rgba],
                  c1: [...STOPS[stop1].rgba],
                })
              }
            >
              Fill
            </button>
          </span>
        </div>
        <div style={row}>
          <label htmlFor="pg-image-noise">Noise</label>
          <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <input
              id="pg-image-noise"
              data-image-noise-amount
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={noiseAmount}
              disabled={disabled}
              onChange={(e) => setNoiseAmount(Number(e.target.value))}
            />
            <span style={{ ...mono, minWidth: "3.5em", textAlign: "right" }}>
              {noiseAmount.toFixed(2)}
            </span>
            <button
              type="button"
              data-image-fill-noise
              disabled={disabled || !s.gpu}
              onClick={() =>
                void session.fillSelection({
                  kind: "noise",
                  amount: noiseAmount,
                })
              }
            >
              Fill
            </button>
          </span>
        </div>
        <div style={note}>
          A fill is DESTRUCTIVE into the engine source (the document and the
          placed file are untouched — re-ingest restores). Two stops only in v0;
          the gradient geometry follows the selection bounds.
        </div>

        {/* Filters — the T1/T2 kernels' first editor reach (blur, unsharp,
            hue rotation, invert); same GPU chain, same Apply commit. */}
        <div style={sectionTitle}>Filters</div>
        <Slider
          label="Blur (σ px)"
          min={0}
          max={8}
          step={0.1}
          value={p.blurSigma}
          disabled={disabled}
          onChange={(v) => setBase("blurSigma", v)}
        />
        <Slider
          label="Sharpen"
          min={0}
          max={3}
          step={0.05}
          value={p.sharpenAmount}
          disabled={disabled}
          onChange={(v) => setBase("sharpenAmount", v)}
        />
        <Slider
          label="Hue rotate (°)"
          min={-180}
          max={180}
          step={1}
          value={p.hueDegrees}
          disabled={disabled}
          onChange={(v) => setBase("hueDegrees", v)}
        />
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            font: "12px var(--font-sans, sans-serif)",
          }}
        >
          <input
            type="checkbox"
            data-image-invert
            checked={p.invert}
            disabled={disabled}
            onChange={(e) => session.setParams({ invert: e.target.checked })}
          />
          Invert colors
        </label>

        {/* LAYERS — the layer graph itself: order, visibility, opacity,
            blend and the active-layer choice paint/fill/bake land in,
            plus the journal's undo/redo and its stated bound. */}
        <LayersSection
          layers={s.layers.layers}
          active={s.layers.active}
          history={s.history}
          blendModes={s.blendModes}
          layersNote={s.layersNote}
          gpu={s.gpu}
          disabled={disabled}
          onSelect={(i) => session.setActiveLayer(i)}
          onAdd={() => void session.addLayer()}
          onDuplicate={(i) => void session.duplicateLayer(i)}
          onRemove={(i) => void session.removeLayer(i)}
          onMove={(from, to) => void session.reorderLayer(from, to)}
          onVisible={(i, v) => void session.setLayerVisible(i, v)}
          onOpacity={(i, v) => void session.setLayerOpacity(i, v)}
          onBlend={(i, b) => void session.setLayerBlend(i, b)}
          onLock={(i, v) => session.setLayerLocked(i, v)}
          onAddAdjustment={() => void session.addAdjustmentLayer()}
          onMaskFromSelection={(i) => void session.layerMaskFromSelection(i)}
          onMaskToggle={(i, v) => void session.setLayerMaskEnabled(i, v)}
          onMaskClear={(i) => void session.clearLayerMask(i)}
          onBake={() => void session.bakeAdjustToLayer()}
          onUndo={() => void session.undo()}
          onRedo={() => void session.redo()}
        />

        {/* BRUSH — the paint tools' frozen-at-pointer-down parameters
            (paged.image's RASTER brush/pencil/eraser, distinct from
            paged.draw's vector ones on the same rail). The section states
            the scope in the UI, not only in the code. */}
        <BrushSection
          brush={s.brush}
          blendModes={s.blendModes}
          strokeActive={s.strokeActive}
          strokeStats={s.strokeStats}
          masked={s.selection !== null}
          gpu={s.gpu}
          disabled={disabled}
          onChange={(patch) => session.setBrushParams(patch)}
        />

        {/* Resize — the T1 resample kernels (GPU-only; the button says so
            when no device). Swaps the engine source like a crop commit. */}
        <div style={sectionTitle}>Resize</div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            font: "12px var(--font-sans, sans-serif)",
          }}
        >
          <input
            type="number"
            min={1}
            data-image-resize-w
            value={resizeW}
            onChange={(e) =>
              setResizeW(Math.max(1, Number(e.target.value) | 0))
            }
            style={{ width: 64, font: "11px var(--font-mono, monospace)" }}
            disabled={disabled}
          />
          ×
          <input
            type="number"
            min={1}
            data-image-resize-h
            value={resizeH}
            onChange={(e) =>
              setResizeH(Math.max(1, Number(e.target.value) | 0))
            }
            style={{ width: 64, font: "11px var(--font-mono, monospace)" }}
            disabled={disabled}
          />
          <select
            data-image-resize-filter
            value={resizeFilter}
            onChange={(e) => setResizeFilter(e.target.value as ResampleFilter)}
            disabled={disabled}
          >
            <option value="lanczos3">Lanczos 3</option>
            <option value="mitchell">Mitchell</option>
            <option value="nearest">Nearest</option>
          </select>
          <button
            type="button"
            data-image-resize-apply
            disabled={disabled || !s.gpu}
            title={
              s.gpu
                ? "Resample the source"
                : "Resample is GPU-only — no WebGPU device"
            }
            onClick={() =>
              void session.resizeTo(resizeW, resizeH, resizeFilter)
            }
          >
            Resample
          </button>
        </div>

        {/* PSD layers — the structural session (mutatable tier): record
            edits accumulate on the retained parse; Export Center's "PSD
            (edited)" re-emits preservation-safe. The canvas composite
            stays the import-time flatten (re-flatten is a follow-up). */}
        {s.psd && (
          <>
            <div style={sectionTitle}>PSD layers ({s.psd.layers.length})</div>
            {s.psd.layers.map((l) => (
              <div
                key={l.index}
                data-image-psd-layer={l.index}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  font: "12px var(--font-sans, sans-serif)",
                  padding: "1px 0",
                }}
              >
                <span
                  style={{
                    flex: 1,
                    minWidth: 0,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    opacity: l.hidden ? 0.5 : 1,
                  }}
                  title={l.hidden ? `${l.name} (hidden in the PSD)` : l.name}
                  onDoubleClick={() => {
                    const name = window.prompt("Layer name", l.name);
                    if (name && name !== l.name)
                      session.psdRenameLayer(l.index, name);
                  }}
                >
                  {l.name}
                </span>
                <input
                  type="number"
                  min={0}
                  max={100}
                  step={1}
                  data-image-psd-opacity
                  value={Math.round((l.opacity / 255) * 100)}
                  style={{
                    width: 52,
                    font: "11px var(--font-mono, monospace)",
                  }}
                  title="Layer opacity (%)"
                  onChange={(e) =>
                    session.psdSetLayerOpacity(
                      l.index,
                      (Number(e.target.value) / 100) * 255,
                    )
                  }
                />
                <button
                  type="button"
                  data-image-psd-remove
                  title="Remove layer (in the exported PSD)"
                  style={{
                    border: "none",
                    background: "none",
                    cursor: "pointer",
                    color: "var(--pg-fg)",
                  }}
                  onClick={() => session.psdRemoveLayer(l.index)}
                >
                  ✕
                </button>
              </div>
            ))}
            <p
              style={{
                margin: "2px 0 0",
                font: "10px/1.5 var(--font-sans, sans-serif)",
                color: "var(--pg-muted-fg)",
              }}
            >
              Edits land in the exported PSD (preservation-safe — a zero-edit
              export is byte-identical). The canvas shows the import-time
              flatten.
            </p>
          </>
        )}

        {/* Curves */}
        <div style={sectionTitle}>Curves</div>
        <div
          style={{
            display: "flex",
            gap: "var(--space-2, 8px)",
            alignItems: "flex-start",
          }}
        >
          <CurveEditor
            points={curvePoints}
            onChange={pushCurve}
            disabled={disabled}
          />
          <button
            type="button"
            disabled={disabled}
            onClick={() =>
              pushCurve(IDENTITY_CURVE.map((q) => [...q] as [number, number]))
            }
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
        <div
          style={{
            display: "flex",
            gap: "var(--space-2, 8px)",
            marginTop: "var(--space-1, 4px)",
          }}
        >
          <button
            type="button"
            disabled={disabled || !machine}
            onClick={() => void session.commitCrop()}
          >
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

        {/* SAVE-BACK — bake the adjustments into the SOURCE FILE bytes.
            The button computes + STAGES them; the Export Center delivers
            them, because the host contract has no save-file door
            (shell.pickFile READS bytes in). Stated, not hidden. */}
        <div style={sectionTitle}>Apply to file</div>
        <div style={{ display: "flex", gap: "var(--space-2, 8px)" }}>
          <button
            type="button"
            data-image-apply-to-file
            disabled={disabled}
            onClick={() => void session.applyToFile()}
          >
            Apply to file
          </button>
        </div>
        {s.saveBack ? (
          <>
            <div style={row}>
              <span>Ready</span>
              <span style={mono} data-image-saveback-file>
                {s.saveBack.fileName} · {s.saveBack.bytes.length} B
              </span>
            </div>
            <div style={note} data-image-saveback-note>
              {s.saveBack.note}
            </div>
          </>
        ) : (
          <div style={note}>
            Bakes the adjustments into the source file&apos;s bytes at full
            resolution: a PSD writes its channel pixels back (single-layer; a
            MULTI-layer PSD is flattened into a new single-layer PSD and says
            so), a PNG/JPEG is re-encoded. The bytes are handed to the Export
            Center — the host wires no save-file door.
          </div>
        )}

        {/* Commit */}
        <div
          style={{
            display: "flex",
            gap: "var(--space-2, 8px)",
            marginTop: "var(--space-3, 12px)",
          }}
        >
          <button
            type="button"
            disabled={disabled}
            onClick={() => void session.apply()}
          >
            Apply
          </button>
          <button
            type="button"
            data-image-reset
            disabled={s.busy}
            onClick={() => {
              // session.reset() restores the FULL identity params — every
              // stage above, including the extended ones; the local
              // pickers below are the panel's own state.
              void session.reset();
              setCurvePoints(IDENTITY_CURVE);
              setAspect("free");
              setAngle(0);
              setGradKind("linear");
              setStop0(0);
              setStop1(1);
              setNoiseAmount(0.5);
            }}
          >
            Reset
          </button>
        </div>

        <div style={note}>{s.status}</div>
        <div style={note}>
          Apply composites an in-frame PREVIEW layer (C-1 Stage A) — the
          document and the placed file are unchanged. The crop commit cuts the
          engine source, straightening first when the angle is non-zero (a
          bilinear resample, so it needs the GPU; 0&deg; stays an exact
          axis-aligned cut). &ldquo;Apply to file&rdquo; is the save-back lane.
          Per-drag preview for the ADJUSTMENT sliders is still a later milestone
          (Stage B); the paint tools do stream a preview per pointer sample, at
          the cost of a whole-image byte payload each time.
        </div>
      </div>
    );
  };
}
