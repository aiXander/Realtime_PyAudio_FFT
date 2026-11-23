// FFT "ocean" — 3D view of the recent spectrogram, drawn in a pluggable style.
//
// Data (`ui.fft3d.source`), one of two:
//   - "spectro" (default): the server's spectro stream (`set_spectro`; see
//     server/dsp/spectro.py or engine/server/src/spectro.rs), NOT the
//     LED-oriented FFT the 2D view shows. It is a multi-resolution STFT made
//     for looking: a frame every ~5 ms, ~1024 log bins from 20 Hz to 20 kHz,
//     long windows in the bass and short ones in the treble, raw dB, no
//     smoothing or per-bin autoscale. Frames arrive in batches (one WS
//     message per broadcast tick), u8-quantised.
//   - "fft": the /audio/fft frames the 2D view draws (makeFftRows), one row
//     per frame holding the 2D graph's own height. These rows are "linear":
//     no averaging, blur, tonal lift, tilt or dB range, so the history is the
//     2D picture over time, byte for byte what the LEDs got.
//
// Everything stays on the GPU at full resolution:
//   - History: one RG8 texture (R = level in dB, G = how tonal the bin is;
//     see makeIntegrator), n_bins wide, TEX_ROWS tall, used as a ring
//     (row = seq mod TEX_ROWS, REPEAT wrap). Each frame uploads only the new
//     rows. The dB → height mapping (range, tilt, level) is uniforms, so
//     changing it re-maps the whole history at once. A second texture, one
//     RGBA texel per row, holds which bands were sounding in that frame
//     (makeMoment): the styles turn it into the row's hue (Color picker).
//   - Geometry: a grid of (rows × cols) vertices whose heights the vertex
//     shader reads from the texture (pooled over the data rows each grid row
//     stands for). Grid rows are anchored to data rows and glide with the
//     playhead, so nothing crawls.
//   - Shading: fragment shaders read the texture again per pixel (`SPEC_FN`
//     in common.js), pooled over the data rows / bins under the pixel.
//     Colour, contours and peak lines are therefore at data resolution, not
//     mesh resolution.
//
// The data side lives in fft3d_data.js (no DOM, runs in node): the frames are
// averaged in power over the last ~40 ms before they enter the ring (the raw
// stream is speckle wherever the sound is noise-like), the front edge follows
// a constant-speed playhead behind the bursty arrivals, and the dB mapped to
// full height holds the loudest level of the last 20 s.
//
// Rendering is WebGL2 on its own canvas, overlaid on the card's 2D canvas
// (a .viz-stack sibling, transparent, no pointer events) and composited by
// the browser. The 2D canvas only holds the static layer (background, style
// art, axis labels) and is repainted only when that changes. Don't go back
// to drawImage'ing the GL canvas into the 2D one: that per-frame copy
// stalled the pipeline and dropped the whole UI from 60 to ~35 fps.

import { store, recordVizPerf } from "../store.js";
import { LMH, LMH_ORDER, theme } from "../colors.js";
import { makeSurface, makeLayer, FONT_UI } from "./surface.js";
import { hex01, MAX_POOL } from "./fft3d_styles/common.js";
import { styleFor } from "./fft3d_styles/index.js";
import { makeIntegrator, makeMoment, makePlayhead, makeLevel, toneLift } from "./fft3d_data.js";

export const TEX_ROWS = 8192;          // history ring rows (≥ 30 s at ~190 frames/s)
const MAX_COLS = 2048;                 // grid columns cap (= server MAX_BINS)
// Geometry only carries the relief; colour and light are per pixel, so a
// modest grid looks the same as a dense one at a fraction of the GPU cost.
const GRID_COLS = 512;                 // default grid columns (a style may ask for others)
const GRID_ROWS = 256;                 // default grid rows across the full depth
const MAX_GRID_ROWS = 1024;

// Camera defaults (a style's `camera` overrides them; the user's camera
// settings scale them: perspective × the narrowing, height × the amplitude).
const BACK_SCALE = 0.3;                // width/height of the oldest row vs the front one
const AMP_FRONT = 0.4;                 // front edge full-scale height, fraction of plot height

// Crossovers between low / mid / high for the moment's colour when the bands don't give one, Hz.
const MOMENT_SPLITS = [200, 1500];

const TONE_LIFT = 0.7;                 // tonal strength 1 lifts a tonal bin this far toward the top

const X_TICK_CANDIDATES = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

// Plot padding, CSS px.
const PAD_X = 10, PAD_T = 4, PAD_B = 16;

const rgb01 = (s) => s.split(",").map((v) => Number(v) / 255);

export function makeFft3d(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const surf = makeSurface(canvas);
  const layer = makeLayer();

  // ---- display mapping + camera (server-persisted `ui.fft3d`, set by main.js via setDisplay) ----
  const disp = {
    rangeDb: 60, tiltDbOct: 0, integrateMs: 40, blurOct: 0, gamma: 1, tonal: 0, moment: 0, glow: 0.5, bloom: 0,
    camera: { overscan: 0.05, perspective: 1, skew: 0, height: 1, horizon: 0 },
  };
  let camKey = "";

  // ---- history ring: integrated frames (CPU mirror of the texture: the front edge reads it) ----
  let nBins = 0, hopS = 0.005, fMin = 20, fMax = 20000, db0 = -127.5, dbStep = 0.5;
  let kind = "spectro";                  // which encoding the ring holds (a change reconfigures)
  let linear = false;                    // level byte = height 0..1 (source "fft"), not dB
  let ring = null;                       // Uint8Array(TEX_ROWS * nBins * 2): [level, tonalness] per bin
  let lastSeq = -1;                      // newest frame stored (absolute seq)
  let firstSeq = -1;                     // oldest seq still valid in the ring
  let dirtyLo = Infinity;                // oldest seq not yet uploaded
  let fullUpload = false;
  let tiltLut = null;                    // Float32Array(nBins): tilt in dB per bin
  let tiltKey = "";
  const tintRing = new Uint8Array(TEX_ROWS * 4);   // per row: [low, mid, high activity, 255] (makeMoment)
  let splitKey = "";
  const moment = makeMoment();
  const integ = makeIntegrator();
  const playhead = makePlayhead();
  const level = makeLevel();

  function reset() {
    if (ring) ring.fill(0);
    tintRing.fill(0);
    moment.reset();
    lastSeq = -1; firstSeq = -1; dirtyLo = Infinity; fullUpload = true;
    integ.clear(); playhead.reset(); level.reset();
  }

  // Linear rows pass through the integrator untouched (one row, no blur, no tonal).
  const syncIntegrate = () => {
    integ.setRows(linear ? 1 : disp.integrateMs / 1000 / hopS);
    // blurOct is the Gaussian's full width at half maximum, in octaves.
    integ.setBlur(linear ? 0 : (disp.blurOct * nBins) / Math.log2(fMax / fMin) / 2.355);
    integ.setTonal(!linear && disp.tonal > 0);
  };
  const tiltOf = () => (linear ? 0 : disp.tiltDbOct);

  function configure(m) {
    nBins = m.nBins; hopS = m.hopS; fMin = m.fMin; fMax = m.fMax; db0 = m.db0; dbStep = m.dbStep;
    kind = m.kind ?? "spectro"; linear = !!m.linear;
    ring = new Uint8Array(TEX_ROWS * nBins * 2);
    integ.configure(nBins, dbStep, nBins / Math.log2(fMax / fMin), hopS);
    syncIntegrate();
    const qDb = m.qDb ?? Float64Array.from({ length: 256 }, (_, q) => db0 + q * dbStep);
    moment.configure(nBins, qDb, fMin, fMax, hopS);
    splitKey = "";
    texW = 0;                            // reallocate the texture
    tiltKey = "";
    reset();
  }

  function syncTilt() {
    const tilt = tiltOf();
    const key = `${nBins}|${fMin}|${fMax}|${tilt}`;
    if (key === tiltKey) return;
    tiltKey = key;
    tiltLut = new Float32Array(nBins);
    const oct = Math.log2(fMax / fMin);
    for (let b = 0; b < nBins; b++) {
      const f = fMin * Math.pow(2, ((b + 0.5) / nBins) * oct);
      tiltLut[b] = tilt * Math.log2(f / 1000);
    }
    level.reset();                       // re-find the top under the new tilt
  }

  /** The moment's colour splits the spectrum where the user's L/M/H bands meet. */
  function syncSplits() {
    const b = store.meta?.bands;
    let f1 = MOMENT_SPLITS[0], f2 = MOMENT_SPLITS[1];
    if (b?.low && b.mid && b.high) {
      f1 = Math.sqrt(b.low.hi_hz * b.mid.lo_hz);
      f2 = Math.sqrt(b.mid.hi_hz * b.high.lo_hz);
    }
    const key = `${f1}|${f2}`;
    if (key === splitKey) return;
    splitKey = key;
    if (f1 < f2) moment.setSplits(f1, f2); else moment.setSplits(...MOMENT_SPLITS);
  }

  /** One spectro message, or one makeFftRows row (called on arrival, from ws.js via main.js). */
  function push(m) {
    if (m.nBins !== nBins || m.hopS !== hopS || m.fMin !== fMin || m.fMax !== fMax
        || (m.kind ?? "spectro") !== kind || !ring) configure(m);
    syncTilt();
    syncSplits();
    const n = nBins;
    const s0 = m.firstSeq;
    // A seq far behind what we hold = the server restarted its count.
    if (lastSeq >= 0 && s0 < lastSeq - TEX_ROWS / 2) reset();
    // Gap (frames dropped on the way): clear the rows we never got.
    if (lastSeq >= 0 && s0 > lastSeq + 1) {
      const from = Math.max(lastSeq + 1, s0 - TEX_ROWS);
      for (let s = from; s < s0; s++) {
        ring.fill(0, (s % TEX_ROWS) * n * 2, (s % TEX_ROWS + 1) * n * 2);
        tintRing.fill(0, (s % TEX_ROWS) * 4, (s % TEX_ROWS + 1) * 4);
      }
      dirtyLo = Math.min(dirtyLo, from);
      integ.clear();
    }
    const prevSeq = lastSeq;
    let peak = -Infinity;
    for (let k = 0; k < m.nFrames; k++) {
      const seq = s0 + k;
      if (seq <= lastSeq) continue;      // duplicate
      const o = (seq % TEX_ROWS) * n * 2;
      integ.push(m.data.subarray(k * n, k * n + n), ring, o);
      moment.push(ring, o, tintRing, (seq % TEX_ROWS) * 4);
      for (let b = 0; b < n; b++) {
        const v = ring[o + 2 * b] * dbStep + db0 + tiltLut[b];
        if (v > peak) peak = v;
      }
      if (firstSeq < 0) firstSeq = seq;
      dirtyLo = Math.min(dirtyLo, seq);
      lastSeq = seq;
    }
    if (lastSeq === prevSeq) return;     // nothing new
    if (lastSeq - firstSeq >= TEX_ROWS) firstSeq = lastSeq - TEX_ROWS + 1;
    const now = performance.now();
    level.push(peak, now);
    playhead.arrive(now, prevSeq, lastSeq, hopS * 1000, lastSeq - Math.max(prevSeq, s0 - 1));
  }

  const styleOf = () => styleFor(store.fft3d_style);

  // ---- WebGL ----
  const glCanvas = document.createElement("canvas");
  glCanvas.className = "viz-overlay";
  glCanvas.setAttribute("aria-hidden", "true");
  glCanvas.hidden = true;
  canvas.after(glCanvas);
  let gl = null, glOk = false;
  let gridVao, gridIbo, iboCols = 0;
  let tex = null, texW = 0;
  let tintTex = null;                    // 1 × TEX_ROWS RGBA8: the moment's band activity per row (unit 1)
  let active = null, activeName = "";   // the current style's instance (one at a time)
  // GPU time per frame (EXT_disjoint_timer_query_webgl2), recorded as viz perf "fft_gpu".
  let timerExt = null;
  const queries = [];

  const edgeX = new Float32Array(MAX_COLS), edgeY = new Float32Array(MAX_COLS);
  const frontH = new Float32Array(MAX_COLS);

  // Color stops: each band's color at its geometric-center frequency.
  const stopPos = new Float32Array(3);
  const stopRgb = [new Float32Array(3), new Float32Array(3), new Float32Array(3)];
  let stopVersion = -1;

  // The per-frame bundle handed to style.render(); mutated in place.
  const f = {
    W: 1, H: 1, dpr: 1, geo: null,
    R: 0, mc: 0, K: 1, backScale: BACK_SCALE, rowDz: 0.002,
    now: 0, t: 0, depthS: 5, fMin: 20, fMax: 20000,
    frontH, edgeX, edgeY, yBaseF: 0, sF: 1,
    stopPos, stopRgb, bgRgb: new Float32Array(3),
    // Grid VAO (no attributes: the vertex shader works from gl_VertexID) +
    // index buffer, rows back to front; draw f.triCount indices.
    surfVao: null, triCount: 0,
    // Spectro uniforms (setCommonUniforms sets them; see SPEC_FN in common.js).
    spec: {
      tex: null, nBins: 1, texRows: TEX_ROWS,
      rowA: 0,          // texture row of grid row "A" (the newest anchored row), 0..TEX_ROWS
      frontOff: 0,      // play − seqA: the front row's offset past row A, data rows
      step: 1,          // data rows per grid row
      pool: 1,          // data rows the vertex shader pools per grid row
      bodyRows: 1,      // anchored grid rows (the front row comes on top)
      nView: 1,         // data rows across the full depth
      keyA: 0,          // grid-row key of row A (stable per grid row as it recedes, mod 65536)
      dbScale: 127.5, db0: -127.5, floorDb: -60, invRange: 1 / 60, tilt: 0,
      gamma: 1,         // height = (dB-linear 0..1) ^ gamma
      tonal: 0,         // how strongly styles light tonal bins, 0..1
      toneLift: 0,      // how far a tonal bin is lifted toward the top (toneLift in fft3d_data.js)
      glow: 0.5,        // cloud: 0 = lit matte terrain … 1 = a dark body that emits by level
      bloom: 0,         // cloud: bloom gain (0 = no bloom pass)
      moment: 0,        // how far colour follows the moment's bands instead of the style's own, 0..1
      logOct: 10,       // log2(fMax / fMin)
      log2FminK: 0,     // log2(fMin / 1000)
      gridCols: 1,
    },
  };

  function initGL() {
    glOk = false;
    iboCols = 0; texW = 0; tintTex = null;
    active = null; activeName = "";
    queries.length = 0;
    fullUpload = true;
    gl = glCanvas.getContext("webgl2", {
      alpha: true, premultipliedAlpha: true, antialias: true, depth: true, preserveDrawingBuffer: false,
    });
    if (!gl) return;
    timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    gridVao = gl.createVertexArray();
    gl.bindVertexArray(gridVao);
    gridIbo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gridIbo);   // captured by gridVao
    gl.bindVertexArray(null);
    f.surfVao = gridVao;
    glOk = true;
  }
  glCanvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); glOk = false; });
  glCanvas.addEventListener("webglcontextrestored", initGL);
  initGL();

  function instance(style) {
    if (activeName === style.name) return active;
    if (active?.dispose) active.dispose();
    active = null;
    activeName = style.name;
    try {
      active = style.create(gl, f);
    } catch (e) {
      console.error(`fft_3d: style "${style.name}" failed to build`, e);
    }
    return active;
  }

  function collectGpuTime() {
    while (queries.length) {
      const q = queries[0];
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
      queries.shift();
      if (!gl.getParameter(timerExt.GPU_DISJOINT_EXT)) {
        recordVizPerf("fft_gpu", gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6);
      }
      gl.deleteQuery(q);
    }
  }

  /** Row-major grid triangles, oldest row first (= back to front). */
  function buildIndices(mc) {
    const rows = MAX_GRID_ROWS + 1;
    const idx = new Uint32Array((rows - 1) * (mc - 1) * 6);
    let o = 0;
    for (let r = 0; r < rows - 1; r++) {
      for (let m = 0; m < mc - 1; m++) {
        const a = r * mc + m, c = a + mc;
        idx[o++] = a; idx[o++] = c; idx[o++] = a + 1;
        idx[o++] = a + 1; idx[o++] = c; idx[o++] = c + 1;
      }
    }
    gl.bindVertexArray(gridVao);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    iboCols = mc;
  }

  /** Upload ring rows [a, b] (absolute seqs) to both textures (level on unit 0, tint on unit 1), splitting at the wrap. */
  function uploadRows(a, b) {
    const n = nBins;
    let s = a;
    while (s <= b) {
      const r = s % TEX_ROWS;
      const len = Math.min(b - s + 1, TEX_ROWS - r);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, r, n, len, gl.RG, gl.UNSIGNED_BYTE, ring, r * n * 2);
      gl.activeTexture(gl.TEXTURE1);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, r, 1, len, gl.RGBA, gl.UNSIGNED_BYTE, tintRing, r * 4);
      gl.activeTexture(gl.TEXTURE0);
      s += len;
    }
  }

  function syncTexture() {
    if (!ring) return;
    gl.activeTexture(gl.TEXTURE1);
    if (!tintTex) {
      tintTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tintTex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 1, TEX_ROWS);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      fullUpload = true;
    }
    gl.bindTexture(gl.TEXTURE_2D, tintTex);
    gl.activeTexture(gl.TEXTURE0);
    if (texW !== nBins || !tex) {
      if (tex) gl.deleteTexture(tex);
      tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RG8, nBins, TEX_ROWS);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      texW = nBins;
      fullUpload = true;
    }
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    if (fullUpload) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, nBins, TEX_ROWS, gl.RG, gl.UNSIGNED_BYTE, ring);
      gl.activeTexture(gl.TEXTURE1);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 1, TEX_ROWS, gl.RGBA, gl.UNSIGNED_BYTE, tintRing);
      gl.activeTexture(gl.TEXTURE0);
      fullUpload = false;
    } else if (dirtyLo <= lastSeq) {
      uploadRows(Math.max(dirtyLo, lastSeq - TEX_ROWS + 1), lastSeq);
    }
    dirtyLo = Infinity;
    f.spec.tex = tex;
  }

  function showGl(on) {
    if (glCanvas.hidden === !on) return;
    glCanvas.hidden = !on;
  }

  // ---- static layer: background, style art, x-axis labels ----
  let painted = false;
  let kVersion = -1, kFmin = NaN, kFmax = NaN, kEmpty = null, kDepth = NaN, kStyle = "", kTheme = -1, kCam = "";

  /**
   * Screen layout + camera. The front row is `halfW` either side of `cx`
   * (overscan: wider than the plot, so it reaches the edges a moment back);
   * rows shrink by 1/(1 + K·z) toward the back, sit `ground(z)` of the way up
   * to `yBack` and shift sideways by `skewPx` at the horizon (PROJ_FN).
   */
  function geometry(W, H, dpr, style) {
    const cam = disp.camera;
    const K = (1 / (style.camera?.backScale ?? BACK_SCALE) - 1) * cam.perspective;
    const backScale = 1 / (1 + K);
    const padX = Math.round(PAD_X * dpr), padT = Math.round(PAD_T * dpr), padB = Math.round(PAD_B * dpr);
    const plotX = padX, plotW = Math.max(1, W - 2 * padX);
    const y0 = H - padB;                            // front baseline
    const plotH = Math.max(1, y0 - padT);
    const ampFront = (style.camera?.ampFront ?? AMP_FRONT) * cam.height * plotH;
    // At horizon 0 the back baseline sits just low enough that a full-scale
    // back row still fits; horizon lowers it toward the front.
    const yFit = padT + ampFront * backScale + Math.round(6 * dpr);
    const yBack = Math.min(y0 - 1, yFit + cam.horizon * (y0 - yFit));
    return {
      plotX, plotW, cx: plotX + plotW / 2, halfW: (plotW / 2) * (1 + 2 * cam.overscan),
      y0, yBack, ampFront, padT, backScale, K, skewPx: cam.skew * (plotW / 2),
    };
  }

  const bgOf = (style) => (typeof style.background === "function" ? style.background() : style.background);

  function buildStatic(W, H, dpr, fLo, fHi, empty, depthS, style) {
    const c = layer.canvas, g = layer.ctx;
    c.width = W; c.height = H;
    const text = style.text || {};
    g.fillStyle = bgOf(style);
    g.fillRect(0, 0, W, H);
    g.font = `${Math.round(empty ? 12 * dpr : 10 * dpr)}px ${FONT_UI}`;
    if (empty) {
      g.fillStyle = text.dim || "#5a6068";
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(empty, W / 2, H / 2);
      return;
    }
    const geo = geometry(W, H, dpr, style);
    const logFmin = Math.log10(fLo);
    const logSpan = Math.max(1e-6, Math.log10(fHi) - logFmin);
    if (style.drawStatic) {
      g.save();
      style.drawStatic(g, geo, W, H, dpr, { depthS, logFmin, logSpan, fMin: fLo, fMax: fHi });
      g.restore();
      g.font = `${Math.round(10 * dpr)}px ${FONT_UI}`;
    }
    // X axis under the front edge (log), at the front row's positions; ticks
    // that the overscan pushes off the card are skipped.
    g.fillStyle = text.axis || "#7a8088";
    g.textAlign = "center";
    g.textBaseline = "top";
    const yLabel = geo.y0 + Math.round(3 * dpr);
    const minSpacingPx = Math.round(40 * dpr);
    const edgePx = Math.round(10 * dpr);
    let lastTickPx = -Infinity;
    for (const fq of X_TICK_CANDIDATES) {
      if (fq < fLo || fq > fHi) continue;
      const x = Math.round(geo.cx + (((Math.log10(fq) - logFmin) / logSpan) - 0.5) * 2 * geo.halfW);
      if (x < geo.plotX + edgePx || x > geo.plotX + geo.plotW - edgePx) continue;
      if (x - lastTickPx < minSpacingPx) continue;
      lastTickPx = x;
      g.fillText(fq >= 1000 ? `${fq / 1000}k` : `${fq}`, x, yLabel);
    }
    g.fillStyle = text.dim || "#5a6068";
    g.textAlign = "right";
    g.textBaseline = "middle";
    g.fillText(`horizon −${depthS < 10 ? depthS.toFixed(1) : Math.round(depthS)} s`, geo.plotX + geo.plotW, geo.yBack);
  }

  /** Height (0..1) of absolute seq s, bin b, from the CPU ring (same mapping as SPEC_FN). */
  function levelAt(s, b) {
    const i = ((s % TEX_ROWS) * nBins + b) * 2;
    const v = ring[i] * dbStep + db0 + tiltLut[b];
    const d = (v - f.spec.floorDb) * f.spec.invRange;
    const h = toneLift(d <= 0 ? 0 : d >= 1 ? 1 : d, ring[i + 1] / 255, f.spec.toneLift);
    return h <= 0 ? 0 : h >= 1 ? 1 : Math.pow(h, f.spec.gamma);
  }

  /** `now`: the animation frame's timestamp (steadier than the time this draw happens to start). */
  function draw(frameT) {
    const t0 = performance.now();
    const { w: W, h: H, dpr, version } = surf.fit();
    const style = styleOf();
    const meta = store.meta || {};
    const depthS = store.history_s;

    const inst = glOk ? instance(style) : null;
    const empty = !glOk ? "3D view needs WebGL2"
      : !inst ? `3D style "${style.name}" failed to build`
      : store.fft3d.source === "spectro" && store.spectro_supported === false
        ? "3D view needs an audio-server with the spectro stream (update it)"
      : !meta.fft_enabled ? "FFT disabled"
      : lastSeq < 1 ? "waiting for data…" : "";

    if (version !== kVersion || fMin !== kFmin || fMax !== kFmax || empty !== kEmpty || depthS !== kDepth
        || style.name !== kStyle || theme.version !== kTheme || camKey !== kCam) {
      kVersion = version; kFmin = fMin; kFmax = fMax; kEmpty = empty; kDepth = depthS;
      kStyle = style.name; kTheme = theme.version; kCam = camKey;
      buildStatic(W, H, dpr, fMin, fMax, empty, depthS, style);
      f.bgRgb.set(hex01(bgOf(style)));
      painted = false;
    }
    if (!painted) {
      ctx.drawImage(layer.canvas, 0, 0);
      painted = true;
    }
    if (empty) {
      showGl(false);
      recordVizPerf("fft", performance.now() - t0);
      return;
    }

    const now = frameT ?? t0;
    syncTilt();
    const P = playhead.advance(now, lastSeq, hopS * 1000);
    const topDb = level.value(now);
    const nView = Math.max(2, depthS / hopS);           // data rows across the depth
    const geo = geometry(W, H, dpr, style);

    // Grid: columns (≤ bins) × anchored rows + one front row exactly at the playhead.
    const mc = Math.max(2, Math.min(nBins, style.grid?.cols ?? GRID_COLS, MAX_COLS));
    const wantRows = Math.min(MAX_GRID_ROWS - 1, style.grid?.rows ?? GRID_ROWS);
    const step = Math.max(1, Math.ceil(nView / wantRows));  // whole data rows per grid row: rows stay anchored
    const seqA = Math.floor(P / step) * step;            // newest anchored row
    const bodyRows = Math.min(MAX_GRID_ROWS, Math.ceil(nView / step) + 1);
    const R = bodyRows + 1;

    // Display mapping.
    const sp = f.spec;
    sp.nBins = nBins;
    sp.rowA = seqA % TEX_ROWS;
    sp.frontOff = P - seqA;
    sp.step = step;
    sp.pool = Math.min(step, MAX_POOL);
    sp.bodyRows = bodyRows;
    sp.nView = nView;
    sp.keyA = (seqA / step) % 65536;
    sp.dbScale = 255 * dbStep;
    sp.db0 = db0;
    // Linear rows: the byte is the height (db0 0, dbStep 1/255), so floor 0, range 1, no curve.
    sp.floorDb = linear ? 0 : topDb - disp.rangeDb;
    sp.invRange = linear ? 1 : 1 / disp.rangeDb;
    sp.tilt = tiltOf();
    sp.gamma = linear ? 1 : disp.gamma;
    sp.tonal = linear ? 0 : Math.min(1, 1.5 * disp.tonal);
    sp.toneLift = linear ? 0 : TONE_LIFT * disp.tonal;
    sp.moment = disp.moment;
    sp.glow = disp.glow;
    sp.bloom = disp.bloom;
    sp.logOct = Math.log2(fMax / fMin);
    sp.log2FminK = Math.log2(fMin / 1000);
    sp.gridCols = mc;

    const toU = (hz) => {
      const p = Math.log2(hz / fMin) / sp.logOct;
      return p < 0 ? 0 : p > 1 ? 1 : p;
    };
    const bands = meta.bands || {};
    for (let k = 0; k < 3; k++) {
      const b = bands[LMH_ORDER[k]];
      stopPos[k] = toU(b ? Math.sqrt(b.lo_hz * b.hi_hz) : [100, 1000, 8000][k]);
    }
    if (stopVersion !== theme.version) {
      stopVersion = theme.version;
      for (let k = 0; k < 3; k++) stopRgb[k].set(rgb01(LMH[LMH_ORDER[k]].rgb));
    }

    // Front edge at the playhead (CPU, same mapping as the shaders), max-pooled
    // over the bins each grid column stands for.
    const s0 = Math.floor(P), fr = P - s0;
    const sLo = Math.max(firstSeq, s0), sHi = Math.min(lastSeq, s0 + 1);
    for (let m = 0; m < mc; m++) {
      const b0 = Math.floor((m / (mc - 1)) * (nBins - 1));
      const b1 = Math.min(nBins - 1, Math.max(b0, Math.floor(((m + 1) / (mc - 1)) * (nBins - 1)) - 1));
      let v = 0;
      for (let b = b0; b <= b1; b++) {
        const h = levelAt(sLo, b) + (levelAt(sHi, b) - levelAt(sLo, b)) * fr;
        if (h > v) v = h;
      }
      frontH[m] = v;
    }
    const sF = 1;                                        // front row sits at z = 0
    const yBaseF = geo.y0;
    const ampF = geo.ampFront * sF;
    for (let m = 0; m < mc; m++) {
      const u = m / (mc - 1);
      edgeX[m] = geo.cx + (u - 0.5) * 2 * geo.halfW * sF;
      edgeY[m] = yBaseF - Math.min(1, frontH[m]) * ampF;
    }

    f.W = W; f.H = H; f.dpr = dpr; f.geo = geo;
    f.R = R; f.mc = mc; f.K = geo.K; f.backScale = geo.backScale;
    f.rowDz = step / nView;
    f.now = now; f.t = now / 1000; f.depthS = depthS; f.fMin = fMin; f.fMax = fMax;
    f.yBaseF = yBaseF; f.sF = sF;
    f.triCount = (R - 1) * (mc - 1) * 6;

    if (glCanvas.width !== W) glCanvas.width = W;
    if (glCanvas.height !== H) glCanvas.height = H;
    if (iboCols !== mc) buildIndices(mc);
    syncTexture();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);

    let q = null;
    if (timerExt && queries.length < 4) {
      q = gl.createQuery();
      gl.beginQuery(timerExt.TIME_ELAPSED_EXT, q);
    }
    inst.render(f);
    if (q) {
      gl.endQuery(timerExt.TIME_ELAPSED_EXT);
      queries.push(q);
    }
    if (timerExt) collectGpuTime();

    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(true);
    showGl(true);
    recordVizPerf("fft", performance.now() - t0);
  }

  /** The 2D view is drawing on the shared canvas: hide the overlay, repaint the static layer on return. */
  function hide() {
    showGl(false);
    painted = false;
  }

  /**
   * Display settings: { rangeDb, tiltDbOct, gamma, moment, glow, bloom, integrateMs, blurOct, tonal,
   * camera: { overscan, perspective, skew, height, horizon } } (any may be
   * omitted). Integration, blur and tonal apply to new frames; everything else
   * re-maps the whole history.
   */
  function setDisplay(d) {
    if (d.rangeDb) disp.rangeDb = d.rangeDb;
    if (d.tiltDbOct !== undefined) disp.tiltDbOct = d.tiltDbOct;
    if (d.gamma) disp.gamma = d.gamma;
    if (d.moment !== undefined) disp.moment = d.moment;
    if (d.glow !== undefined) disp.glow = d.glow;
    if (d.bloom !== undefined) disp.bloom = d.bloom;
    if ((d.integrateMs !== undefined && d.integrateMs !== disp.integrateMs)
        || (d.blurOct !== undefined && d.blurOct !== disp.blurOct)
        || (d.tonal !== undefined && d.tonal !== disp.tonal)) {
      disp.integrateMs = d.integrateMs ?? disp.integrateMs;
      disp.blurOct = d.blurOct ?? disp.blurOct;
      disp.tonal = d.tonal ?? disp.tonal;
      if (nBins) syncIntegrate();
    }
    if (d.camera) {
      Object.assign(disp.camera, d.camera);
      const c = disp.camera;
      camKey = `${c.overscan}|${c.perspective}|${c.skew}|${c.height}|${c.horizon}`;
    }
  }

  return { draw, push, reset, hide, setDisplay };
}
