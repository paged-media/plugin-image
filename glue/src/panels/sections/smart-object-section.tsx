// SMART OBJECT — convert the active layer, and re-render it at a new
// scale FROM ITS PRESERVED SOURCE (never from the already-scaled pixels).

import type { ImageSession } from "../../session";
import { mono, note, row, sectionTitle } from "./styles";

export function SmartObjectSection({
  session,
  active,
  isSmart,
  gpu,
  disabled,
  scale,
  onScale,
}: {
  session: ImageSession;
  active: number;
  isSmart: boolean;
  gpu: boolean;
  disabled: boolean;
  scale: number;
  onScale: (v: number) => void;
}) {
  return (
    <>
      <div style={sectionTitle}>Smart object</div>
      <div style={row}>
        <button
          type="button"
          data-image-make-smart
          disabled={disabled || active < 0 || isSmart}
          onClick={() => void session.makeLayerSmart(active)}
        >
          Convert active layer
        </button>
        <span style={mono}>{isSmart ? "smart" : "pixels"}</span>
      </div>
      <div style={row}>
        <label htmlFor="pg-image-smart-scale">Scale</label>
        <input
          id="pg-image-smart-scale"
          type="range"
          min={0.1}
          max={2}
          step={0.05}
          value={scale}
          disabled={!isSmart}
          onChange={(e) => onScale(Number(e.target.value))}
        />
        <span style={mono}>{Math.round(scale * 100)}%</span>
        <button
          type="button"
          data-image-render-smart
          disabled={disabled || !isSmart || !gpu}
          onClick={() => void session.renderLayerSmart(active, scale)}
        >
          Render
        </button>
      </div>
      <div style={note}>
        A smart object keeps its original pixels. Scaling re-renders from them,
        so shrinking and then enlarging again loses nothing.
      </div>
    </>
  );
}
