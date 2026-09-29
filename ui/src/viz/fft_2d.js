// FFT bars — thin renderer.
//
// All post-processing happens server-side (server/dsp/fft_postprocess.py).
// The wire payload is whatever the server is sending RIGHT NOW, which is
// also what's going out over OSC on /audio/fft:
//
//   meta.fft_send_raw_db === false  (default):
//     post-processed [0..1] values — server has already done sentinel
//     interpolation, smoothing, peak normalization, gate, tanh and the
//     strength blend. Render directly as bar height; y-axis is 0..1.
//
//   meta.fft_send_raw_db === true:
//     raw wire dB. Sentinel bins (-1000) render as gaps. Y-axis is in dB
//     between [meta.fft_db_floor, meta.fft_db_ceiling].
//
// X-axis is log frequency in either mode. Tick labels at decade-ish anchors.
//
// Interactive band overlay (when setInteractive(true)): click a colored band
// to select it; drag the body to shift, drag an edge handle to move that
// edge. Mirrors the freq_axis side-panel widget — same set_band messages,
// same selection/drag model — so when FFT is enabled it replaces the
// side-panel "Bandpass edges" UI.

import { store, recordVizPerf } from "../store.js";
import { LMH, LMH_ORDER, theme, hexRgb } from "../colors.js";
import { send } from "../ws.js";
import { makeSurface, makeLayer, FONT_UI } from "./surface.js";

// Bar colors: the old per-bin 256-entry LUT, quantized to N_BUCKETS so the
// bar pass is one path + one fill per bucket instead of a fillStyle switch
// per bin (sampled at bucket centers — visually indistinguishable).
// The ramp comes from the palette (theme.ramp: evenly spaced hex stops; null
// = the original blue → green → red formula) and rebuilds when it changes.
const N_BUCKETS = 32;
const BUCKET_STR = new Array(N_BUCKETS);
function buildBuckets() {
  const stops = theme.ramp ? theme.ramp.map(hexRgb) : null;
  for (let k = 0; k < N_BUCKETS; k++) {
    const t = (k + 0.5) / N_BUCKETS;
    let r, g, b;
    if (stops) {
      const p = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), f = p - i;
      [r, g, b] = stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * f));
    } else {
      r = Math.round(255 * Math.min(1, Math.max(0, -0.2 + 1.6 * t)));
      g = Math.round(255 * (0.1 + 0.85 * Math.sin(Math.PI * t)));
      b = Math.round(255 * Math.max(0, 1 - 1.6 * t + 0.5 * Math.pow(t, 4)));
    }
    BUCKET_STR[k] = `rgb(${r},${g},${b})`;
  }
}

const SENTINEL_THRESHOLD = -500;
const MIN_BAR_PX = 1;          // CSS px, so silent bins stay visible in 0..1 mode

const X_TICK_CANDIDATES = [30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

// Band-edit constants — match server validators / freq_axis.js.
const F_AXIS_MIN = 20;
const MIN_GAP_HZ = 50;
const HANDLE_W_PX = 10;        // CSS-px hit-width for edge handles (mouse)
const HANDLE_W_TOUCH_PX = 24;  // wider for fingers
const ORDER = ["low", "mid", "high"];

// Plot padding, CSS px.
const PAD_L = 38, PAD_R = 6, PAD_T = 4, PAD_B = 16;

// Precomputed overlay colors per band (rebuilt with the bucket ramp).
const BAND_STYLE = {};
let colorVersion = -1;
function buildColors() {
  buildBuckets();
  for (const name of LMH_ORDER) {
    const c = LMH[name].rgb;
    BAND_STYLE[name] = {
      fillSel: `rgba(${c},0.30)`, fillEdit: `rgba(${c},0.10)`, fillPassive: `rgba(${c},0.15)`,
      edgeSel: `rgba(${c},1)`, edge: `rgba(${c},0.55)`, handle: `rgba(${c},0.95)`,
    };
  }
  colorVersion = theme.version;
}
buildColors();

// Mirrors snapHz in controls.js / freq_axis.js so the overlay and sliders
// agree on values.
function snapHz(f) {
  const step = Math.min(128, Math.max(8, Math.pow(2, Math.round(Math.log2(f / 40)))));
  return Math.max(step, Math.round(f / step) * step);
}

function fmtHz(v) {
  return v >= 1000 ? `${(v / 1000).toFixed(1)} kHz` : `${Math.round(v)} Hz`;
}

export function makeFft(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const surf = makeSurface(canvas);
  const layer = makeLayer();
  let peaks = null, vCache = null, binBucket = null, order = null;
  const bucketStart = new Int32Array(N_BUCKETS + 1);
  const bucketCursor = new Int32Array(N_BUCKETS);
  let lastT = performance.now();

  // Band-edit state — mirrors meta.bands; only updated from meta when not
  // mid-drag (so server echoes don't fight the user).
  const bandState = {
    low:  { lo_hz: 30,   hi_hz: 250   },
    mid:  { lo_hz: 250,  hi_hz: 4000  },
    high: { lo_hz: 4000, hi_hz: 16000 },
  };
  let interactive = false;
  let selected = null;
  let drag = null;
  // CSS-px plot geometry from the most recent draw, used by pointer
  // hit-testing. Mutated in place; `valid` is false until a spectrum draws.
  const layout = {
    valid: false,
    cssPlotX: 0, cssPlotY: 0, cssPlotW: 1, cssPlotH: 1,
    fMin: 30, fMax: 24000, logFmin: 0, logSpan: 1, sr: 48000, fMaxHard: 21600,
  };

  // Static layer: background, grid, axis labels — or the empty-state
  // message. Rebuilt only when this key changes.
  function staticKey(version, rawDb, floor, ceiling, fMin, sr, empty) {
    return `${version}|${theme.version}|${rawDb ? 1 : 0}|${floor}|${ceiling}|${fMin}|${sr}|${empty}`;
  }
  // Cheap pre-check so the key string is only built when an input changed.
  let kVersion = -1, kRaw = null, kFloor = NaN, kCeil = NaN, kFmin = NaN, kSr = NaN, kEmpty = null;

  function buildStatic(w, h, dpr, rawDb, floor, ceiling, fMin, fMax, empty) {
    const c = layer.canvas, g = layer.ctx;
    c.width = w; c.height = h;
    g.fillStyle = theme.bg;
    g.fillRect(0, 0, w, h);
    if (empty) {
      g.fillStyle = "#5a6068";
      g.font = `${Math.round(12 * dpr)}px ${FONT_UI}`;
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(empty, w / 2, h / 2);
      return;
    }
    const padL = Math.round(PAD_L * dpr), padR = Math.round(PAD_R * dpr);
    const padT = Math.round(PAD_T * dpr), padB = Math.round(PAD_B * dpr);
    const plotX = padL, plotY = padT;
    const plotW = Math.max(1, w - padL - padR);
    const plotH = Math.max(1, h - padT - padB);

    // Y axis: grid lines + labels.
    g.font = `${Math.round(10 * dpr)}px ${FONT_UI}`;
    g.fillStyle = "#7a8088";
    g.textAlign = "right";
    g.textBaseline = "middle";
    const labels = rawDb
      ? [[0, `${ceiling}`], [0.5, `${Math.round((floor + ceiling) / 2)}`], [1, `${floor}`]]
      : [[0, "1.0"], [0.5, "0.5"], [1, "0.0"]];
    const xLabel = plotX - Math.round(4 * dpr);
    g.strokeStyle = "rgba(255,255,255,0.06)";
    g.lineWidth = Math.max(1, Math.round(dpr));
    for (const [v, text] of labels) {
      const y = Math.round(plotY + v * plotH) + 0.5;
      g.fillText(text, xLabel, y);
      g.beginPath();
      g.moveTo(plotX, y);
      g.lineTo(w - padR, y);
      g.stroke();
    }
    // No corner unit label: it collided with the "0.0" / first-Hz ticks, and
    // the card title already names the mode ("raw dB" / "scaled 0..1").
    g.textBaseline = "top";

    // X axis ticks.
    const logFmin = Math.log10(fMin);
    const logSpan = Math.max(1e-6, Math.log10(fMax) - logFmin);
    g.fillStyle = "#7a8088";
    g.textAlign = "center";
    const yLabel = plotY + plotH + Math.round(2 * dpr);
    const minSpacingPx = Math.round(40 * dpr);
    let lastTickPx = -Infinity;
    for (const f of X_TICK_CANDIDATES) {
      if (f < fMin || f > fMax) continue;
      const x = Math.round(plotX + ((Math.log10(f) - logFmin) / logSpan) * plotW);
      if (x - lastTickPx < minSpacingPx) continue;
      lastTickPx = x;
      g.fillText(f >= 1000 ? `${f / 1000}k` : `${f}`, x, yLabel);
      g.fillRect(x, plotY + plotH, Math.max(1, Math.round(dpr)), Math.round(3 * dpr));
    }
  }

  function draw() {
    const t0 = performance.now();
    const { w: W, h: H, dpr, version, cssW, cssH } = surf.fit();
    const now = t0;
    const dt = Math.max(1e-3, Math.min(0.1, (now - lastT) / 1000));
    lastT = now;
    const peakDecay = store.peak_decay_per_s ?? 0.6;

    const meta = store.meta || {};
    const rawDb = !!meta.fft_send_raw_db;
    const floor = meta.fft_db_floor ?? -60;
    const ceiling = meta.fft_db_ceiling ?? 0;
    const fMin = meta.fft_f_min ?? 30;
    const sr = meta.sr ?? 48000;
    const fMax = sr / 2;

    const bins = store.fft_bins;
    const n = bins ? bins.length : 0;
    const empty = n > 0 ? "" : (meta.fft_enabled ? "waiting for data…" : "FFT disabled");

    if (colorVersion !== theme.version) { buildColors(); kVersion = -1; }
    if (version !== kVersion || rawDb !== kRaw || floor !== kFloor || ceiling !== kCeil
        || fMin !== kFmin || sr !== kSr || empty !== kEmpty) {
      kVersion = version; kRaw = rawDb; kFloor = floor; kCeil = ceiling; kFmin = fMin; kSr = sr; kEmpty = empty;
      const key = staticKey(version, rawDb, floor, ceiling, fMin, sr, empty);
      if (key !== layer.key) {
        layer.key = key;
        buildStatic(W, H, dpr, rawDb, floor, ceiling, fMin, fMax, empty);
      }
    }
    ctx.drawImage(layer.canvas, 0, 0);

    if (n === 0) {
      layout.valid = false;
      recordVizPerf("fft", performance.now() - t0);
      return;
    }
    if (!peaks || peaks.length !== n) {
      peaks = new Float32Array(n);
      vCache = new Float32Array(n);
      binBucket = new Uint8Array(n);
      order = new Uint16Array(n);
    }

    const padL = Math.round(PAD_L * dpr), padR = Math.round(PAD_R * dpr);
    const padT = Math.round(PAD_T * dpr), padB = Math.round(PAD_B * dpr);
    const plotX = padL, plotY = padT;
    const plotW = Math.max(1, W - padL - padR);
    const plotH = Math.max(1, H - padT - padB);
    const logFmin = Math.log10(fMin);
    const logSpan = Math.max(1e-6, Math.log10(fMax) - logFmin);

    // ---------------- bars ----------------
    // Pass 1: value, peak-hold, color bucket per bin (sentinels → NaN, no
    // bar, no tick). Pass 2: counting sort bins by bucket. Pass 3: one path
    // + one fill per non-empty bucket.
    const span = Math.max(1, ceiling - floor);
    const decay = peakDecay * dt;
    bucketStart.fill(0);
    for (let i = 0; i < n; i++) {
      const raw = bins[i];
      let v;
      if (rawDb) {
        if (raw < SENTINEL_THRESHOLD) {
          peaks[i] = Math.max(0, peaks[i] - decay);
          vCache[i] = NaN;
          binBucket[i] = 255;
          continue;
        }
        v = (raw - floor) / span;
      } else {
        v = raw;
      }
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      if (v > peaks[i]) peaks[i] = v;
      else peaks[i] = Math.max(0, peaks[i] - decay);
      vCache[i] = v;
      const b = Math.min(N_BUCKETS - 1, (v * N_BUCKETS) | 0);
      binBucket[i] = b;
      bucketStart[b + 1]++;
    }
    for (let b = 0; b < N_BUCKETS; b++) {
      bucketStart[b + 1] += bucketStart[b];
      bucketCursor[b] = bucketStart[b];
    }
    for (let i = 0; i < n; i++) {
      const b = binBucket[i];
      if (b !== 255) order[bucketCursor[b]++] = i;
    }
    const barW = plotW / n;
    const gap = barW > 3 ? Math.min(dpr, barW * 0.25) : 0;
    const drawW = Math.max(1, barW - gap);
    const minBarPx = rawDb ? 0 : MIN_BAR_PX * dpr;
    const baseY = plotY + plotH;
    for (let b = 0; b < N_BUCKETS; b++) {
      const s0 = bucketStart[b], s1 = bucketStart[b + 1];
      if (s1 === s0) continue;
      ctx.fillStyle = BUCKET_STR[b];
      ctx.beginPath();
      for (let j = s0; j < s1; j++) {
        const i = order[j];
        const barH = Math.max(minBarPx, vCache[i] * plotH);
        ctx.rect(plotX + i * barW, baseY - barH, drawW, barH);
      }
      ctx.fill();
    }

    // Peak ticks — one path, one fill.
    const tickH = Math.max(2, Math.round(1.5 * dpr));
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      if (vCache[i] !== vCache[i]) continue; // NaN: sentinel
      ctx.rect(plotX + i * barW, baseY - peaks[i] * plotH - tickH / 2, drawW, tickH);
    }
    ctx.fill();

    // ---------------- band overlay (interactive when `interactive`) ---------
    layout.valid = true;
    layout.cssPlotX = PAD_L;
    layout.cssPlotY = PAD_T;
    layout.cssPlotW = Math.max(1, cssW - PAD_L - PAD_R);
    layout.cssPlotH = Math.max(1, cssH - PAD_T - PAD_B);
    layout.fMin = fMin; layout.fMax = fMax;
    layout.logFmin = logFmin; layout.logSpan = logSpan;
    layout.sr = sr; layout.fMaxHard = Math.min(22000, 0.45 * sr);

    drawBandsOverlay(meta, plotX, plotY, plotW, plotH, dpr, fMin, fMax, logFmin, logSpan);

    recordVizPerf("fft", performance.now() - t0);
  }

  function freqToCanvasX(f, plotX, plotW, fMin, fMax, logFmin, logSpan) {
    if (f <= fMin) return plotX;
    if (f >= fMax) return plotX + plotW;
    return plotX + ((Math.log10(f) - logFmin) / logSpan) * plotW;
  }

  function drawChip(cx, ty, text, dpr) {
    const w = ctx.measureText(text).width + Math.round(8 * dpr);
    const h = Math.round(13 * dpr);
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(cx - w / 2, ty - 1, w, h);
    ctx.fillStyle = "#d6d9dc";
    ctx.fillText(text, cx, ty + 1);
  }

  function drawBandsOverlay(meta, plotX, plotY, plotW, plotH, dpr, fMin, fMax, logFmin, logSpan) {
    if (fMax <= fMin) return;
    // Live drag state when interactive, server's meta.bands otherwise.
    const src = interactive ? bandState : (meta.bands || null);
    if (!src) return;

    for (const name of LMH_ORDER) {
      const b = src[name];
      if (!b) continue;
      const x0 = freqToCanvasX(b.lo_hz, plotX, plotW, fMin, fMax, logFmin, logSpan);
      const x1 = freqToCanvasX(b.hi_hz, plotX, plotW, fMin, fMax, logFmin, logSpan);
      if (x1 <= x0) continue;
      const st = BAND_STYLE[name];
      const isSel = interactive && (name === selected);
      ctx.fillStyle = isSel ? st.fillSel : (interactive ? st.fillEdit : st.fillPassive);
      ctx.fillRect(x0, plotY, x1 - x0, plotH);
      const ew = (isSel ? 2 : 1) * Math.max(1, Math.round(dpr));
      ctx.fillStyle = isSel ? st.edgeSel : st.edge;
      ctx.fillRect(x0, plotY, ew, plotH);
      ctx.fillRect(x1 - ew, plotY, ew, plotH);

      if (isSel) {
        const handleW = Math.max(2, Math.round(4 * dpr));
        ctx.fillStyle = st.handle;
        ctx.fillRect(x0 - handleW / 2, plotY, handleW, plotH);
        ctx.fillRect(x1 - handleW / 2, plotY, handleW, plotH);
        ctx.font = `${Math.round(10 * dpr)}px ${FONT_UI}`;
        ctx.textBaseline = "top";
        ctx.textAlign = "center";
        const ty = plotY + Math.round(2 * dpr);
        drawChip(x0, ty, fmtHz(b.lo_hz), dpr);
        drawChip(x1, ty, fmtHz(b.hi_hz), dpr);
      }
    }
  }

  // ---------------- Interactive band-edit ----------------
  // Pointer coords via offsetX/Y (relative to the canvas; no layout read).
  // With pointer capture the canvas stays the event target, so offsets stay
  // canvas-relative even when the pointer leaves it mid-drag.

  function cssXToFreq(cssX) {
    const t = (cssX - layout.cssPlotX) / layout.cssPlotW;
    return Math.pow(10, layout.logFmin + t * layout.logSpan);
  }

  function cssFreqToX(f) {
    if (f <= layout.fMin) return layout.cssPlotX;
    if (f >= layout.fMax) return layout.cssPlotX + layout.cssPlotW;
    return layout.cssPlotX + ((Math.log10(f) - layout.logFmin) / layout.logSpan) * layout.cssPlotW;
  }

  function clampEdges(lo, hi) {
    const fmaxHard = layout.fMaxHard;
    if (lo < F_AXIS_MIN) lo = F_AXIS_MIN;
    if (hi > fmaxHard) hi = fmaxHard;
    if (hi < lo + MIN_GAP_HZ) hi = lo + MIN_GAP_HZ;
    return [lo, hi];
  }

  function inPlotY(cssY) {
    return cssY >= layout.cssPlotY && cssY <= layout.cssPlotY + layout.cssPlotH;
  }

  function hitTest(cssX, cssY, handleW) {
    if (!layout.valid || !inPlotY(cssY)) return null;
    // Selected band wins: check its handles, then its body.
    if (selected) {
      const s = bandState[selected];
      const xLo = cssFreqToX(s.lo_hz);
      const xHi = cssFreqToX(s.hi_hz);
      if (Math.abs(cssX - xLo) <= handleW / 2) return { name: selected, role: "lo" };
      if (Math.abs(cssX - xHi) <= handleW / 2) return { name: selected, role: "hi" };
      if (cssX >= xLo && cssX <= xHi) return { name: selected, role: "body" };
    }
    // Otherwise walk in reverse paint order so the visually-on-top band wins.
    for (let i = ORDER.length - 1; i >= 0; i--) {
      const name = ORDER[i];
      if (name === selected) continue;
      const s = bandState[name];
      if (cssX >= cssFreqToX(s.lo_hz) && cssX <= cssFreqToX(s.hi_hz)) return { name, role: "body" };
    }
    return null;
  }

  function onPointerDown(evt) {
    if (!interactive || !layout.valid) return;
    if (evt.pointerType === "mouse" && evt.button !== 0) return;
    const x = evt.offsetX, y = evt.offsetY;
    const hw = evt.pointerType === "mouse" ? HANDLE_W_PX : HANDLE_W_TOUCH_PX;
    const hit = hitTest(x, y, hw);
    if (!hit) {
      selected = null; // click outside any band → deselect
      return;
    }
    evt.preventDefault();
    if (hit.role === "body" && selected !== hit.name) {
      selected = hit.name; // first click only selects
      return;
    }
    drag = {
      name: hit.name, mode: hit.role, startX: x,
      startLo: bandState[hit.name].lo_hz, startHi: bandState[hit.name].hi_hz,
      pointerId: evt.pointerId,
    };
    if (hit.role === "body") canvas.style.cursor = "grabbing";
    try { canvas.setPointerCapture(evt.pointerId); } catch {}
  }

  function onPointerMove(evt) {
    if (!interactive || !layout.valid) return;
    const x = evt.offsetX, y = evt.offsetY;
    if (drag) {
      if (evt.pointerId !== drag.pointerId) return;
      let lo = drag.startLo;
      let hi = drag.startHi;
      if (drag.mode === "lo") {
        lo = snapHz(cssXToFreq(x));
        if (lo > hi - MIN_GAP_HZ) lo = hi - MIN_GAP_HZ;
      } else if (drag.mode === "hi") {
        hi = snapHz(cssXToFreq(x));
        if (hi < lo + MIN_GAP_HZ) hi = lo + MIN_GAP_HZ;
      } else {
        // Body drag preserves log-width.
        const ratio = cssXToFreq(x) / cssXToFreq(drag.startX);
        lo = drag.startLo * ratio;
        hi = drag.startHi * ratio;
        const fmin = F_AXIS_MIN, fmax = layout.fMaxHard;
        if (lo < fmin) { const k = fmin / lo; lo *= k; hi *= k; }
        if (hi > fmax) { const k = fmax / hi; lo *= k; hi *= k; }
        lo = snapHz(lo);
        hi = snapHz(hi);
      }
      [lo, hi] = clampEdges(lo, hi);
      const cur = bandState[drag.name];
      if (cur.lo_hz === lo && cur.hi_hz === hi) return; // snapped: nothing new to send
      bandState[drag.name] = { lo_hz: lo, hi_hz: hi };
      send({ type: "set_band", band: drag.name, lo_hz: lo, hi_hz: hi, commit: false });
      return;
    }
    if (evt.pointerType !== "mouse") return;
    // Hover: update cursor based on what we're over.
    const hit = hitTest(x, y, HANDLE_W_PX);
    let cursor = "default";
    if (hit) {
      if (hit.role === "lo" || hit.role === "hi") cursor = "ew-resize";
      else cursor = (hit.name === selected) ? "grab" : "pointer";
    }
    if (canvas.style.cursor !== cursor) canvas.style.cursor = cursor;
  }

  // Ends a drag on pointerup, pointercancel, or lost capture (e.g. the
  // window lost focus mid-drag) — the drag can never get stuck.
  function endDrag(evt) {
    if (!drag) return;
    if (evt && evt.pointerId !== undefined && evt.pointerId !== drag.pointerId) return;
    const { name, pointerId } = drag;
    const s = bandState[name];
    drag = null;
    canvas.style.cursor = "grab";
    try { if (canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId); } catch {}
    send({ type: "set_band", band: name, lo_hz: s.lo_hz, hi_hz: s.hi_hz, commit: true });
  }

  canvas.addEventListener("pointerdown",   onPointerDown);
  canvas.addEventListener("pointermove",   onPointerMove);
  canvas.addEventListener("pointerup",     endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("lostpointercapture", endDrag);
  canvas.addEventListener("pointerleave",  () => { if (!drag) canvas.style.cursor = "default"; });
  window.addEventListener("blur", () => endDrag());

  // Page-level deselect when clicking outside the canvas.
  document.addEventListener("pointerdown", (evt) => {
    if (!interactive || selected === null) return;
    if (evt.target === canvas) return;
    selected = null;
  });

  return {
    draw,
    /** Forget peak-hold state (used on disconnect). */
    reset() { if (peaks) peaks.fill(0); },
    /** Mirror server-confirmed band geometry. Ignored mid-drag so the user's
     *  in-flight values aren't clobbered by the echo. */
    syncBands(metaBands) {
      if (drag) return;
      if (!metaBands) return;
      for (const name of ORDER) {
        const b = metaBands[name];
        if (b && typeof b.lo_hz === "number" && typeof b.hi_hz === "number") {
          const cur = bandState[name];
          if (cur.lo_hz !== b.lo_hz || cur.hi_hz !== b.hi_hz) {
            bandState[name] = { lo_hz: b.lo_hz, hi_hz: b.hi_hz };
          }
        }
      }
    },
    setInteractive(on) {
      on = !!on;
      if (on === interactive) return;
      interactive = on;
      // touch-action: none only while bands are editable, so a finger drag on
      // a passive spectrum still scrolls the page on touch devices.
      canvas.style.touchAction = interactive ? "none" : "";
      if (interactive) {
        // Mirror the sidebar "Bandpass edges" tooltip onto the canvas so
        // users discover the bands are clickable.
        const src = document.getElementById("freq-axis");
        const tip = (src && src.getAttribute("data-tooltip"))
          || "Click a colored band to select it. Once selected, drag its body to shift the whole band, or drag its left/right handle to move just that edge. Click outside any band to deselect. Bands may overlap.";
        canvas.setAttribute("data-tooltip", tip);
      } else {
        selected = null;
        if (drag) endDrag();
        canvas.style.cursor = "default";
        canvas.removeAttribute("data-tooltip");
      }
    },
  };
}
