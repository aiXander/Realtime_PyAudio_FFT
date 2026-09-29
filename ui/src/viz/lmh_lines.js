// Three rolling polylines for low/mid/high.
//
// - Horizontal alpha gradient: line goes 1.0 (right, newest) → 0 (left, oldest).
// - Subtle per-band area fill at ~0.10 alpha on the right, fading to 0 on the left.
// - Faint y-axis labels (0..1) on the left edge.
// - History window (2..30s, log-scale) via store.history_s. Each sample is
//   plotted at x = w * (1 - age_ms / window_ms), so the time axis is absolute:
//   a half-empty buffer leaves the left portion of the canvas blank, and new
//   samples flow in from the right rather than re-stretching the existing curve.
//
// Static content (background, mid grid line, axis labels) lives in an
// offscreen layer rebuilt only on resize / DPR change; each frame is one
// drawImage + three fill/stroke pairs. No per-frame allocation.

import { store, recordVizPerf } from "../store.js";
import { LMH, theme } from "../colors.js";
import { makeSurface, makeLayer, FONT_MONO } from "./surface.js";

// Ring big enough to cover 30s of history at the top UI refresh rate (120 fps)
// with comfortable headroom.
const N_MAX = 4096;

const FILL_ALPHA_RIGHT   = 0.10;
const FILL_ALPHA_LEFT    = 0.0;
const STROKE_ALPHA_RIGHT = 1.00;
const STROKE_ALPHA_LEFT  = 0.0;

export function makeLines(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const surf = makeSurface(canvas);
  const layer = makeLayer();
  const buf = {
    low:  new Float32Array(N_MAX),
    mid:  new Float32Array(N_MAX),
    high: new Float32Array(N_MAX),
  };
  // Float64: ms since epoch0. Float32 has only ~24 bits of mantissa, so after
  // a few hours of uptime timestamps quantize to several ms and the x
  // positions visibly jitter.
  const ts = new Float64Array(N_MAX);
  const epoch0 = performance.now();
  let head = 0, count = 0;

  let gradVersion = -1, gradTheme = -1;
  let strokeGrads = null, fillGrads = null;

  // rAF pauses while the tab is hidden, so the ring's newest sample is
  // timestamped from when we last drew. On refocus, drop the stale history
  // and let the curve fill in from the right edge again.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) reset();
  });

  function reset() { head = 0; count = 0; }

  function buildGradients(w) {
    const mk = (rgb, a0, a1) => {
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, `rgba(${rgb},${a0})`);
      grad.addColorStop(1, `rgba(${rgb},${a1})`);
      return grad;
    };
    strokeGrads = {
      low:  mk(LMH.low.rgb,  STROKE_ALPHA_LEFT, STROKE_ALPHA_RIGHT),
      mid:  mk(LMH.mid.rgb,  STROKE_ALPHA_LEFT, STROKE_ALPHA_RIGHT),
      high: mk(LMH.high.rgb, STROKE_ALPHA_LEFT, STROKE_ALPHA_RIGHT),
    };
    fillGrads = {
      low:  mk(LMH.low.rgb,  FILL_ALPHA_LEFT, FILL_ALPHA_RIGHT),
      mid:  mk(LMH.mid.rgb,  FILL_ALPHA_LEFT, FILL_ALPHA_RIGHT),
      high: mk(LMH.high.rgb, FILL_ALPHA_LEFT, FILL_ALPHA_RIGHT),
    };
  }

  function buildStatic(w, h, dpr) {
    const c = layer.canvas, g = layer.ctx;
    c.width = w; c.height = h;
    g.fillStyle = theme.bg;
    g.fillRect(0, 0, w, h);
    // Quarter grid, faint; mid line slightly stronger.
    g.lineWidth = Math.max(1, Math.round(dpr));
    for (const [f, a] of [[0.25, 0.025], [0.5, 0.05], [0.75, 0.025]]) {
      const y = Math.round(f * h) + 0.5;
      g.strokeStyle = `rgba(255,255,255,${a})`;
      g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
    }
    g.fillStyle = "rgba(255,255,255,0.32)";
    g.font = `${Math.round(10 * dpr)}px ${FONT_MONO}`;
    g.textBaseline = "middle";
    g.textAlign = "left";
    const padL = 4 * dpr;
    for (const [f, label] of [[0.0, "1.0"], [0.5, "0.5"], [1.0, "0"]]) {
      const y = f * h;
      const yy = f === 0.0 ? y + 7 * dpr : f === 1.0 ? y - 7 * dpr : y;
      g.fillText(label, padL, yy);
    }
  }

  function update() {
    buf.low[head]  = store.low;
    buf.mid[head]  = store.mid;
    buf.high[head] = store.high;
    ts[head] = performance.now() - epoch0;
    head = (head + 1) % N_MAX;
    if (count < N_MAX) count++;
  }

  // Number of most-recent samples that fall within [now - histS, now].
  function effectiveK(histMs, nowRel) {
    if (count === 0) return 0;
    const cutoff = nowRel - histMs;
    const latest = (head - 1 + N_MAX) % N_MAX;
    let k = 1;
    while (k < count) {
      const idx = (latest - k + N_MAX) % N_MAX;
      if (ts[idx] < cutoff) break;
      k++;
    }
    return k;
  }

  // Scratch arrays for one series' (x,y).
  const _xs = new Float32Array(N_MAX);
  const _ys = new Float32Array(N_MAX);

  function drawSeries(arr, fillGrad, strokeGrad, K, w, h, dpr, histMs, nowRel) {
    if (K < 2) return;
    const latest = (head - 1 + N_MAX) % N_MAX;
    for (let i = 0; i < K; i++) {
      const idx = (latest - (K - 1 - i) + N_MAX) % N_MAX;
      const v = arr[idx];
      _xs[i] = w * (1 - (nowRel - ts[idx]) / histMs);
      _ys[i] = h - (v < 0 ? 0 : v > 1 ? 1 : v) * h;
    }
    const firstX = _xs[0];
    const lastX  = _xs[K - 1];

    // Fill first (closed to baseline), stroke on top.
    ctx.beginPath();
    ctx.moveTo(_xs[0], _ys[0]);
    for (let i = 1; i < K; i++) ctx.lineTo(_xs[i], _ys[i]);
    ctx.lineTo(lastX,  h);
    ctx.lineTo(firstX, h);
    ctx.closePath();
    ctx.fillStyle = fillGrad;
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(_xs[0], _ys[0]);
    for (let i = 1; i < K; i++) ctx.lineTo(_xs[i], _ys[i]);
    ctx.strokeStyle = strokeGrad;
    ctx.lineWidth = 1.5 * dpr;
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  function draw() {
    const t0 = performance.now();
    const { w, h, dpr, version } = surf.fit();
    if (version !== gradVersion || theme.version !== gradTheme) {
      gradVersion = version;
      gradTheme = theme.version;
      buildGradients(w);
      buildStatic(w, h, dpr);
    }
    update();
    ctx.drawImage(layer.canvas, 0, 0);

    const histS = Math.max(2, Math.min(30, store.history_s ?? 5));
    const histMs = histS * 1000;
    const nowRel = performance.now() - epoch0;
    const K = effectiveK(histMs, nowRel);
    drawSeries(buf.low,  fillGrads.low,  strokeGrads.low,  K, w, h, dpr, histMs, nowRel);
    drawSeries(buf.mid,  fillGrads.mid,  strokeGrads.mid,  K, w, h, dpr, histMs, nowRel);
    drawSeries(buf.high, fillGrads.high, strokeGrads.high, K, w, h, dpr, histMs, nowRel);

    recordVizPerf("lines", performance.now() - t0);
  }
  return { draw, reset };
}
