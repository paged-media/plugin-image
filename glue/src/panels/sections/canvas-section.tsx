// IMAGE ▸ Rotate / Flip / Canvas Size — over the whole layer stack (every
// layer, mask and smart source moves together). Exact pixel moves; the
// undo history is cleared because its tiles addressed the old canvas.

import type { CanvasOpKind } from "../../engine";
import type { ImageSession } from "../../session";
import { mono, note, row, sectionTitle } from "./styles";

const OPS: Array<[CanvasOpKind, string]> = [
  ["rotate-ccw", "⟲ 90°"],
  ["rotate-cw", "⟳ 90°"],
  ["rotate-180", "180°"],
  ["flip-h", "Flip ↔"],
  ["flip-v", "Flip ↕"],
];

export interface CanvasSize {
  width: number;
  height: number;
  /** 0 = left/top, 1 = centre, 2 = right/bottom. */
  anchorX: number;
  anchorY: number;
}

export function CanvasSection({
  session,
  size,
  onSize,
  disabled,
}: {
  session: ImageSession;
  size: CanvasSize;
  onSize: (s: CanvasSize) => void;
  disabled: boolean;
}) {
  return (
    <>
      <div style={sectionTitle}>Canvas</div>
      <div style={row}>
        {OPS.map(([op, label]) => (
          <button
            key={op}
            type="button"
            data-image-canvas-op={op}
            disabled={disabled}
            onClick={() => void session.canvasOp(op)}
          >
            {label}
          </button>
        ))}
      </div>
      <div style={row}>
        <label htmlFor="pg-image-canvas-w">Size</label>
        <input
          id="pg-image-canvas-w"
          type="number"
          min={1}
          style={{ width: "5em" }}
          value={size.width}
          onChange={(e) => onSize({ ...size, width: Number(e.target.value) })}
        />
        <span style={mono}>×</span>
        <input
          type="number"
          min={1}
          style={{ width: "5em" }}
          value={size.height}
          onChange={(e) => onSize({ ...size, height: Number(e.target.value) })}
        />
      </div>
      <div style={row}>
        <span>Anchor</span>
        <span style={{ display: "grid", gridTemplateColumns: "repeat(3, 18px)", gap: 2 }}>
          {[0, 1, 2].flatMap((ay) =>
            [0, 1, 2].map((ax) => (
              <button
                key={`${ax}${ay}`}
                type="button"
                data-image-anchor={`${ax},${ay}`}
                aria-pressed={size.anchorX === ax && size.anchorY === ay}
                style={{ width: 18, height: 18, padding: 0 }}
                onClick={() => onSize({ ...size, anchorX: ax, anchorY: ay })}
              >
                {size.anchorX === ax && size.anchorY === ay ? "●" : ""}
              </button>
            )),
          )}
        </span>
        <button
          type="button"
          data-image-canvas-size
          disabled={disabled || size.width < 1 || size.height < 1}
          onClick={() => void session.canvasOp("canvas", size)}
        >
          Apply size
        </button>
      </div>
      <div style={note}>
        Rotate, flip and Canvas Size move every layer and mask exactly. Added
        canvas is transparent; cut-off canvas is gone. Each clears the undo
        history.
      </div>
    </>
  );
}
