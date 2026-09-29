// Spatially-separated L/M/H scene:
//   low  -> central glowing disc (radius + alpha grow with low)
//   mid  -> full-screen background color hue (alpha grows with mid)
//   high -> bright random noise sprinkled across the screen (count + alpha
//           grow with high), resampled fresh every frame (every ~320 ms
//           and dimmer under prefers-reduced-motion)
// Layering (bottom -> top): dark base, mid hue tint, low disc, high noise.

import { store, recordVizPerf } from "../store.js";
import { LMH } from "../colors.js";
import { makeSurface } from "./surface.js";

// ─────────────────────────────────────────────────────────────────────────
// Tuning. All visual knobs live here — tweak freely.
// ─────────────────────────────────────────────────────────────────────────
const CONFIG = {
  bgColor: "#0a0b0d",

  low: {
    // Radius scales with low: lowR = baseR * (radiusBase + radiusGain*lo) * radiusScale
    // baseR = 0.5 * min(w, h). radiusScale=1.25 makes the disc 25% larger.
    radiusScale: 1.25,
    radiusBase:  0.15,
    radiusGain:  0.40,
    // Outer alpha of the disc gradient. lo=0 -> alphaMin (invisible),
    // lo=1 -> alphaMax. Inner stop scales by midStopAlphaScale.
    alphaMin: 0.1,
    alphaMax: 0.95,
    saturation:    80,   // %
    lightnessMin:  25,   // % at lo=0
    lightnessMax:  85,   // % at lo=1
    midStop:           0.7,  // gradient stop position [0..1]
    midStopAlphaScale: 0.5,
  },

  mid: {
    // Full-screen tint behind everything else. Alpha = alphaMin..alphaMax
    // mapped from md=0..1. Even at alpha=1, low/high paint over it so it
    // never visually dominates.
    alphaMin: 0.0,
    alphaMax: 0.95,
    saturation: 45,   // %
    lightness:  20,   // % — kept dim so a full-alpha tint doesn't blind
  },

  high: {
    // Random sample count scales with hi.
    minPoints: 100,
    maxPoints: 750,
    // Per-point alpha scales with hi (alphaMin at hi=0, alphaMax at hi=1).
    alphaMin: 0.0,
    alphaMax: 1.0,
    // Visual point size in CSS pixels (multiplied by devicePixelRatio).
    pointSize: 2.0,
    // 0 = uniform across screen, 1 = strong push toward edges.
    // We bias outward by rejection sampling against an acceptance prob
    // that grows with normalized distance from center.
    edgeBias: 1.5,
    // Color: bright/whitish on the high hue so they read as sparkles.
    saturation: 50,  // %
    lightness:  90,   // %
  },
};
// ─────────────────────────────────────────────────────────────────────────

// Low-disc sprites: a radial gradient (alpha 1 → midStopAlphaScale → 0) at
// one of LIGHT_STEPS lightness levels, rendered once and scaled + alpha'd
// with drawImage/globalAlpha each frame instead of building a new gradient.
const LIGHT_STEPS = 24;
const SPRITE_PX = 256;

const reducedMotionMq = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;

export function makeScene(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const surf = makeSurface(canvas);

  const sprites = new Array(LIGHT_STEPS).fill(null);
  function sprite(k) {
    let c = sprites[k];
    if (c) return c;
    c = document.createElement("canvas");
    c.width = c.height = SPRITE_PX;
    const g = c.getContext("2d");
    const r = SPRITE_PX / 2;
    const light = CONFIG.low.lightnessMin + (CONFIG.low.lightnessMax - CONFIG.low.lightnessMin) * (k / (LIGHT_STEPS - 1));
    const col = (a) => `hsla(${LMH.low.hue}, ${CONFIG.low.saturation}%, ${light}%, ${a})`;
    const grad = g.createRadialGradient(r, r, 0, r, r, r);
    grad.addColorStop(0, col(1));
    grad.addColorStop(CONFIG.low.midStop, col(CONFIG.low.midStopAlphaScale));
    grad.addColorStop(1, col(0));
    g.fillStyle = grad;
    g.fillRect(0, 0, SPRITE_PX, SPRITE_PX);
    sprites[k] = c;
    return c;
  }

  const MID_FILL  = `hsl(${LMH.mid.hue}, ${CONFIG.mid.saturation}%, ${CONFIG.mid.lightness}%)`;
  const HIGH_FILL = `hsl(${LMH.high.hue}, ${CONFIG.high.saturation}%, ${CONFIG.high.lightness}%)`;

  // Sparkle positions. Normally resampled every frame (the flicker is the
  // effect); with prefers-reduced-motion they're resampled at ~3 Hz and drawn
  // dimmer, so the field shimmers gently instead of strobing.
  const px = new Float32Array(CONFIG.high.maxPoints);
  const py = new Float32Array(CONFIG.high.maxPoints);
  let lastResample = -Infinity, sampledW = 0, sampledH = 0;

  function resample(w, h, n) {
    const cx = w / 2, cy = h / 2, halfW = w * 0.5, halfH = h * 0.5;
    const bias = CONFIG.high.edgeBias;
    for (let i = 0; i < n; i++) {
      // Rejection sample with edge-biased acceptance, capped attempts.
      let x = 0, y = 0;
      for (let attempt = 0; attempt < 4; attempt++) {
        x = Math.random() * w;
        y = Math.random() * h;
        const dx = (x - cx) / halfW;
        const dy = (y - cy) / halfH;
        const d = Math.min(1, Math.sqrt(dx * dx + dy * dy));
        if (Math.random() < (1 - bias) + bias * d) break;
      }
      px[i] = x | 0; py[i] = y | 0;
    }
    sampledW = w; sampledH = h;
  }

  function draw() {
    const t0 = performance.now();
    const { w, h, dpr } = surf.fit();
    const reduced = !!(reducedMotionMq && reducedMotionMq.matches);

    const lo = Math.max(0, Math.min(1, store.low));
    const md = Math.max(0, Math.min(1, store.mid));
    const hi = Math.max(0, Math.min(1, store.high));

    const cx = w / 2, cy = h / 2;
    const baseR = Math.min(w, h) * 0.5;

    // --- Layer 0: solid dark background. ---
    ctx.globalAlpha = 1;
    ctx.fillStyle = CONFIG.bgColor;
    ctx.fillRect(0, 0, w, h);

    // --- Layer 1: MID full-screen tint (base hue, behind everything). ---
    const midAlpha = CONFIG.mid.alphaMin + (CONFIG.mid.alphaMax - CONFIG.mid.alphaMin) * md;
    if (midAlpha > 0.001) {
      ctx.globalAlpha = midAlpha;
      ctx.fillStyle = MID_FILL;
      ctx.fillRect(0, 0, w, h);
    }

    // --- Layer 2: LOW central disc (prerendered gradient sprite). ---
    const lowR = baseR * (CONFIG.low.radiusBase + CONFIG.low.radiusGain * lo) * CONFIG.low.radiusScale;
    const lowAlpha = CONFIG.low.alphaMin + (CONFIG.low.alphaMax - CONFIG.low.alphaMin) * lo;
    if (lowAlpha > 0.001 && lowR > 0.5) {
      ctx.globalAlpha = lowAlpha;
      ctx.drawImage(sprite(Math.round(lo * (LIGHT_STEPS - 1))), cx - lowR, cy - lowR, 2 * lowR, 2 * lowR);
    }

    // --- Layer 3: HIGH bright random noise. ---
    let hiAlpha = CONFIG.high.alphaMin + (CONFIG.high.alphaMax - CONFIG.high.alphaMin) * hi;
    if (reduced) hiAlpha *= 0.55;
    if (hiAlpha > 0.001) {
      const nPoints = Math.round(CONFIG.high.minPoints + (CONFIG.high.maxPoints - CONFIG.high.minPoints) * hi);
      const sz = Math.max(1, Math.round(CONFIG.high.pointSize * dpr));
      const interval = reduced ? 320 : 0;
      if (t0 - lastResample >= interval || sampledW !== w || sampledH !== h) {
        lastResample = t0;
        resample(w, h, reduced ? CONFIG.high.maxPoints : nPoints);
      }
      ctx.globalAlpha = hiAlpha;
      ctx.fillStyle = HIGH_FILL;
      ctx.beginPath();
      for (let i = 0; i < nPoints; i++) ctx.rect(px[i], py[i], sz, sz);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    recordVizPerf("scene", performance.now() - t0);
  }
  return { draw };
}
