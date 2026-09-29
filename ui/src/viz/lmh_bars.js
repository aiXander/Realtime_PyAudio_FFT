// Three vertical bars with peak-hold ticks, plus an onset indicator (square,
// same width as the bar) directly above each band that flashes on the
// corresponding onset (low / mid / high).
//
// Static content (background, bar tracks, onset-square wells, labels) is
// cached in an offscreen layer keyed on size + the onset toggle. Per frame:
// one drawImage, three bar rects, three peak ticks, up to three flashes.
// Onset flash colors come from a precomputed ramp — no per-frame strings.

import { store, recordVizPerf } from "../store.js";
import { LMH, LMH_HEX, LMH_ORDER } from "../colors.js";
import { makeSurface, makeLayer, FONT_UI, BG } from "./surface.js";

const COLORS = LMH_HEX;
const LABELS = LMH_ORDER;
const TRACK = "#1a1d22";

// Onset-flash decay time (seconds): time for the flash to fade from 1 → 0
// after an onset.
const ONSET_FLASH_DECAY_S = 0.20;
const FLASH_STEPS = 32;

// FLASH_RAMP[band][k] = band color mixed toward white by k / (FLASH_STEPS-1).
const FLASH_RAMP = LMH_ORDER.map((name) => {
  const [r, g, b] = LMH[name].rgb.split(",").map(Number);
  const out = new Array(FLASH_STEPS);
  for (let k = 0; k < FLASH_STEPS; k++) {
    const t = k / (FLASH_STEPS - 1);
    const mix = (c) => Math.round(c + (255 - c) * t);
    out[k] = `rgb(${mix(r)},${mix(g)},${mix(b)})`;
  }
  return out;
});

export function makeBars(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const surf = makeSurface(canvas);
  const layer = makeLayer();
  const peaks = new Float64Array(3);
  const vals = new Float64Array(3);
  const onsetTs = new Float64Array(3);
  let lastT = performance.now();

  // Geometry, recomputed in place each frame (no allocation).
  const g0 = { padX: 0, padTop: 0, slotW: 0, barW: 0, onsetSize: 0, barsTop: 0, baseY: 0, barsUsableH: 1 };
  function geometry(w, h, dpr, showOnsets) {
    const padTop = 12 * dpr, padBot = 16 * dpr;
    g0.padX = 16 * dpr;
    g0.padTop = padTop;
    g0.slotW = (w - 2 * g0.padX) / 3;
    g0.barW = Math.min(g0.slotW * 0.6, 120 * dpr);
    // The onset square is bar-wide, but never eats more than 30% of the
    // height (wide + short cards would otherwise leave no room for bars).
    g0.onsetSize = showOnsets ? Math.min(g0.barW, (h - padTop - padBot) * 0.3) : 0;
    g0.barsTop = padTop + g0.onsetSize + (showOnsets ? 6 * dpr : 0);
    g0.baseY = h - padBot;
    g0.barsUsableH = Math.max(1, g0.baseY - g0.barsTop);
    return g0;
  }

  function buildStatic(w, h, dpr) {
    const c = layer.canvas, g = layer.ctx;
    c.width = w; c.height = h;
    g.fillStyle = BG;
    g.fillRect(0, 0, w, h);
    g.font = `${Math.round(10 * dpr)}px ${FONT_UI}`;
    g.textAlign = "center";
    g.textBaseline = "alphabetic";
    for (let i = 0; i < 3; i++) {
      const x = g0.padX + g0.slotW * i + (g0.slotW - g0.barW) / 2;
      g.fillStyle = TRACK;
      g.fillRect(x, g0.barsTop, g0.barW, g0.barsUsableH);
      if (g0.onsetSize > 0) g.fillRect(x, g0.padTop, g0.barW, g0.onsetSize);
      g.fillStyle = "#8b939c";
      g.fillText(LABELS[i], x + g0.barW / 2, h - 3 * dpr);
    }
  }

  function draw() {
    const t0 = performance.now();
    const { w, h, dpr, version } = surf.fit();
    const now = performance.now();
    const dt = Math.min(0.25, (now - lastT) / 1000);
    lastT = now;
    const showOnsets = !!store.show_onsets;
    geometry(w, h, dpr, showOnsets);
    const key = showOnsets ? version * 2 + 1 : version * 2;
    if (layer.key !== key) { layer.key = key; buildStatic(w, h, dpr); }
    ctx.drawImage(layer.canvas, 0, 0);

    vals[0] = store.low; vals[1] = store.mid; vals[2] = store.high;
    onsetTs[0] = store.low_onset_pulse_t;
    onsetTs[1] = store.mid_onset_pulse_t;
    onsetTs[2] = store.high_onset_pulse_t;
    const decay = (store.peak_decay_per_s ?? 0.6) * dt;
    const tickH = Math.max(2, Math.round(2 * dpr));

    for (let i = 0; i < 3; i++) {
      const v = Math.max(0, Math.min(1, vals[i]));
      if (v > peaks[i]) peaks[i] = v;
      else peaks[i] = Math.max(0, peaks[i] - decay);

      const x = g0.padX + g0.slotW * i + (g0.slotW - g0.barW) / 2;

      if (showOnsets) {
        const since = (now - onsetTs[i]) / 1000;
        if (since >= 0 && since < ONSET_FLASH_DECAY_S) {
          const f = 1 - since / ONSET_FLASH_DECAY_S;
          ctx.fillStyle = FLASH_RAMP[i][Math.round(f * (FLASH_STEPS - 1))];
          ctx.fillRect(x, g0.padTop, g0.barW, g0.onsetSize);
        }
      }

      const barH = g0.barsUsableH * v;
      ctx.fillStyle = COLORS[i];
      ctx.fillRect(x, g0.baseY - barH, g0.barW, barH);
      const peakY = g0.baseY - g0.barsUsableH * peaks[i];
      ctx.fillStyle = "#fff";
      ctx.fillRect(x, peakY - tickH / 2, g0.barW, tickH);
    }
    recordVizPerf("bars", performance.now() - t0);
  }

  function reset() { peaks.fill(0); }
  return { draw, reset };
}
