// COLOUR — foreground / background, swap (X) and reset (D), and a
// picker. The picker is the browser's own colour input until the host
// offers a colour-picker widget to plugins; Alt-click with a paint tool
// samples the image into the foreground.

import { GRADIENT_KINDS, type GradientKind, type Rgba01 } from "../../engine";
import { fromHex, toHex, type ColorPair } from "../../color-state";
import type { ImageSession } from "../../session";
import { mono, note, row, sectionTitle } from "./styles";

const swatch = (c: Rgba01) => ({
  width: 28,
  height: 20,
  border: "1px solid var(--pg-border, rgba(127,127,127,0.5))",
  background: `rgba(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(
    c[2] * 255,
  )}, ${c[3]})`,
});

export function ColorSection({
  session,
  colors,
  gradientKind,
}: {
  session: ImageSession;
  colors: ColorPair;
  /** The gradient tool's shape. */
  gradientKind: GradientKind;
}) {
  const pick = (which: "fg" | "bg") => (hex: string) => {
    const c = fromHex(hex);
    if (!c) return;
    if (which === "fg") session.setForeground(c);
    else session.setBackground(c);
  };
  return (
    <>
      <div style={sectionTitle}>Colour</div>
      <div style={row}>
        <label htmlFor="pg-image-fg">Foreground</label>
        <span style={swatch(colors.fg)} data-image-fg-swatch />
        <input
          id="pg-image-fg"
          type="color"
          data-image-fg
          value={toHex(colors.fg)}
          onChange={(e) => pick("fg")(e.target.value)}
        />
        <span style={mono}>{toHex(colors.fg)}</span>
      </div>
      <div style={row}>
        <label htmlFor="pg-image-bg">Background</label>
        <span style={swatch(colors.bg)} data-image-bg-swatch />
        <input
          id="pg-image-bg"
          type="color"
          data-image-bg
          value={toHex(colors.bg)}
          onChange={(e) => pick("bg")(e.target.value)}
        />
        <span style={mono}>{toHex(colors.bg)}</span>
      </div>
      <div style={row}>
        <button type="button" data-image-swap-colors onClick={() => session.swapColors()}>
          Swap
        </button>
        <button type="button" data-image-reset-colors onClick={() => session.resetColors()}>
          Black / white
        </button>
        <button type="button" data-image-fill-fg onClick={() => void session.fillForeground()}>
          Fill
        </button>
      </div>
      <div style={row}>
        <label htmlFor="pg-image-gradient-kind">Gradient tool</label>
        <select
          id="pg-image-gradient-kind"
          data-image-gradient-kind
          value={gradientKind}
          onChange={(e) => session.setGradientKind(e.target.value as GradientKind)}
        >
          {GRADIENT_KINDS.map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </div>
      <div style={note}>
        The brush and pencil paint the foreground. Alt-click with either takes
        the image&apos;s colour under the pointer (a 3×3 average) as the new
        foreground.
      </div>
    </>
  );
}
