// Shared canvas sizing for the visualizers.
//
// Keeps the backing store at exactly the element's device-pixel size so
// strokes and text stay crisp at any DPR (and follow DPR changes when the
// window moves between monitors). Size comes from a ResizeObserver, so the
// per-frame `fit()` is a couple of property reads — no forced layout.
//
// `version` bumps whenever the backing store is resized; visualizers key
// their cached static layers (axes, labels, gradients) on it.

export function makeSurface(canvas) {
  let cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  let devW = 0, devH = 0;          // exact device-pixel box, when supported
  const ro = new ResizeObserver((entries) => {
    const e = entries[entries.length - 1];
    cssW = e.contentRect.width;
    cssH = e.contentRect.height;
    const dp = e.devicePixelContentBoxSize && e.devicePixelContentBoxSize[0];
    if (dp) { devW = dp.inlineSize; devH = dp.blockSize; }
    else { devW = 0; devH = 0; }
  });
  try { ro.observe(canvas, { box: "device-pixel-content-box" }); }
  catch { ro.observe(canvas); }

  const s = {
    w: canvas.width, h: canvas.height, dpr: 1, cssW, cssH, version: 0,
    /** Resize the backing store if needed. Returns the surface itself. */
    fit() {
      const dpr = window.devicePixelRatio || 1;
      // Prefer the exact device-pixel box, but only when it agrees with
      // css × dpr (it lags a DPR change by one observer tick, and some
      // emulated/headless setups report CSS px there).
      const ew = Math.round(cssW * dpr), eh = Math.round(cssH * dpr);
      const w = Math.max(1, devW && Math.abs(devW - ew) <= 2 ? devW : ew);
      const h = Math.max(1, devH && Math.abs(devH - eh) <= 2 ? devH : eh);
      if (canvas.width !== w || canvas.height !== h || dpr !== s.dpr) {
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        s.w = w; s.h = h; s.dpr = dpr;
        s.version++;
      }
      s.cssW = cssW; s.cssH = cssH;
      return s;
    },
  };
  return s;
}

/** Offscreen layer the size of the surface, for static content (axes,
 *  labels, backgrounds). Re-created only when the caller's key changes. */
export function makeLayer() {
  const c = document.createElement("canvas");
  return { canvas: c, ctx: c.getContext("2d"), key: "" };
}

export const FONT_UI = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
export const FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";
