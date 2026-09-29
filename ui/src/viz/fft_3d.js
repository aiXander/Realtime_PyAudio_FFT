// FFT "ocean" — 3D view of the recent FFT history, drawn in a pluggable style.
//
// Thin renderer, but deliberately NOT exact: this view is for watching, so
// on top of the spectra the server sent it applies display-only smoothing to
// turn the stack of spectra into a continuous sheet. The 2D view stays the
// byte-exact picture of the OSC payload.
//
//   - Across frequency: a Gaussian whose width follows the FFT's real
//     resolution (wide at the low end, where log bins are finer than the
//     rfft bins and the server only interpolates between them; a fixed
//     fraction of an octave above), then Catmull-Rom upsampling. Needle
//     peaks and bin-to-bin jitter don't survive; broad spectral shape does.
//   - Across time: a peak-preserving blur (max(value, blurred)) + Catmull-Rom
//     subdivision. A row's height is final once its two newer neighbours
//     exist, so nothing shrinks as it recedes.
//
// The look is pluggable: the core (this file) owns the history ring, the
// smoothing, the mesh + vertex buffer, the WebGL context (and its loss /
// restore), the static layer and compositing; a style module in
// fft3d_styles/ decides what to draw from the mesh (programs, blending,
// post-processing) plus its camera and background. store.fft3d_style picks
// the style (server-persisted as ui.fft3d_style). See fft3d_styles/common.js
// for the style contract.
//
// Rendering is WebGL2 on an offscreen canvas, composited into the card's 2D
// canvas each frame (drawImage), on top of the static layer (background,
// style art, axis labels).
//
// Rows are placed by arrival time (z = age / depth), not by row index, so the
// scroll speed is constant in seconds whatever the WS frame rate is, and rows
// glide smoothly between frames.
//
// History ring. Frames arrive via push() at the WS rate (15–60 Hz). To keep
// the drawn row count bounded (≈ MAX_ROWS across the full depth) consecutive
// frames that fall within one row slot (depth / MAX_ROWS) are merged with an
// element-wise max, so a one-frame transient is never dropped. Very wide
// spectra are max-pooled down to MAX_COLS columns on the way in. Each ring
// row keeps a smoothed, upsampled copy (meshCols wide), refreshed on push.

import { store, recordVizPerf } from "../store.js";
import { LMH, LMH_ORDER, theme } from "../colors.js";
import { makeSurface, makeLayer, FONT_UI } from "./surface.js";
import { hex01 } from "./fft3d_styles/common.js";
import { STYLES, DEFAULT_STYLE } from "./fft3d_styles/index.js";

const MAX_ROWS = 96;           // history rows across the full depth
const RING_ROWS = 160;         // ring capacity (headroom over MAX_ROWS for depth changes)
const MAX_COLS = 480;          // wider spectra are max-pooled to this many columns
const SENTINEL_THRESHOLD = -500;

// Smoothing (display only). All tunable by eye.
const T_SUB = 2;               // mesh rows per history row (Catmull-Rom in time)
const MESH_COLS_MIN = 256;     // mesh columns (Catmull-Rom in frequency) …
const MESH_COLS_MAX = 384;     // … ≈ 2× the bin count, clamped to this range
const FREQ_SMOOTH_OCT = 0.08;  // Gaussian σ across frequency, octaves (FWHM ≈ 1/5 octave) …
const FREQ_RES_SIGMA = 0.6;    // … at least this many rfft-bin spacings where the FFT is coarser …
const FREQ_SMOOTH_MAX_OCT = 0.35; // … capped here (the far low end)
const TIME_BLUR_PASSES = 2;    // peak-preserving [¼ ½ ¼] passes across history rows

// Camera defaults (a style's `camera` overrides them).
const BACK_SCALE = 0.3;        // width/height of the oldest row vs the front one
const AMP_FRONT = 0.4;         // front edge full-scale height, fraction of plot height
// Lighting-only world scale (surface width = 2): how steep slopes look to the light.
const WORLD_HEIGHT = 0.5, WORLD_DEPTH = 3.0;

const X_TICK_CANDIDATES = [30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

// Plot padding, CSS px.
const PAD_X = 10, PAD_T = 4, PAD_B = 16;

const MAX_MESH_ROWS = (RING_ROWS - 1) * T_SUB + 1;
const VERT_FLOATS = 5;                // h, z, slope_x, slope_z, row key

const rgb01 = (s) => s.split(",").map((v) => Number(v) / 255);

/**
 * Per-column Gaussian kernels across frequency, width following the FFT's
 * real resolution. Columns are log-spaced from fMin to fMax. Returns
 * { lo, len, off, w }: column c averages src[lo[c] .. lo[c]+len[c]) with
 * weights w[off[c] ..]. Kernels are truncated at the spectrum edges and
 * renormalized.
 */
export function buildFreqKernel(cols, fMin, fMax, sr, windowSize) {
  const octs = Math.max(1e-6, Math.log2(fMax / fMin));
  const colsPerOct = cols / octs;
  const binHz = sr / windowSize;                     // rfft bin spacing
  const widthFrac = Math.pow(2, 1 / colsPerOct) - 1; // column width / its frequency
  const lo = new Int32Array(cols), len = new Int32Array(cols), off = new Int32Array(cols);
  const sig = new Float64Array(cols);
  let total = 0;
  for (let c = 0; c < cols; c++) {
    const f = fMin * Math.pow(2, ((c + 0.5) / cols) * octs);
    const rfftPerCol = (f * widthFrac) / binHz;      // < 1: the server interpolates between rfft bins here
    let s = Math.max(FREQ_SMOOTH_OCT * colsPerOct, FREQ_RES_SIGMA / Math.max(1e-6, rfftPerCol));
    s = Math.min(s, FREQ_SMOOTH_MAX_OCT * colsPerOct);
    sig[c] = s;
    const r = Math.ceil(3 * s);
    lo[c] = Math.max(0, c - r);
    len[c] = Math.min(cols - 1, c + r) - lo[c] + 1;
    off[c] = total;
    total += len[c];
  }
  const w = new Float32Array(total);
  for (let c = 0; c < cols; c++) {
    let sum = 0;
    for (let k = 0; k < len[c]; k++) {
      const d = lo[c] + k - c;
      sum += w[off[c] + k] = Math.exp(-(d * d) / (2 * sig[c] * sig[c]));
    }
    for (let k = 0; k < len[c]; k++) w[off[c] + k] /= sum;
  }
  return { lo, len, off, w };
}

/** Apply a buildFreqKernel() kernel: dst[c] = Σ w · src. */
export function smoothFreq(kernel, src, dst) {
  const { lo, len, off, w } = kernel;
  for (let c = 0; c < lo.length; c++) {
    let acc = 0;
    const o = off[c], l = lo[c];
    for (let k = 0; k < len[c]; k++) acc += w[o + k] * src[l + k];
    dst[c] = acc;
  }
}

export function makeFft3d(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const surf = makeSurface(canvas);
  const layer = makeLayer();

  // ---- history ring ----
  let n = 0, cols = 0, rawDbMode = null;
  let ring = null;                                  // Float32Array(RING_ROWS * cols), NaN = no data
  // When the row's slot closes (first frame's arrival + one slot). Depth is
  // measured from this fixed time, so rows glide smoothly: keying on the
  // latest merged frame instead made them twitch back on every merge.
  const rowT = new Float64Array(RING_ROWS);
  const rowSeq = new Float64Array(RING_ROWS);       // monotonic row number (keys the point noise)
  let seqN = 0;
  let rowStartT = -Infinity;                        // arrival time of the open row's first frame
  let head = -1, count = 0;
  let colOf = null;                                 // Uint16Array(n): bin → column

  // Smoothed + upsampled copy of each ring row, meshCols wide.
  let meshCols = 0;
  let sRing = null;                                 // Float32Array(RING_ROWS * meshCols)
  let fillBuf = null, blurBuf = null;               // Float32Array(cols) scratch
  let kernel = null, kernelKey = "";                // frequency smoothing (depends on FFT geometry)
  let upI = null, upT = null;                       // mesh col → source col + fraction

  function reset() { head = -1; count = 0; rowStartT = -Infinity; }

  function configure(nBins, rawDb) {
    n = nBins;
    rawDbMode = rawDb;
    cols = Math.min(n, MAX_COLS);
    ring = new Float32Array(RING_ROWS * cols);
    colOf = new Uint16Array(n);
    for (let i = 0; i < n; i++) colOf[i] = Math.min(cols - 1, Math.floor((i * cols) / n));

    meshCols = Math.max(MESH_COLS_MIN, Math.min(MESH_COLS_MAX, 2 * cols));
    sRing = new Float32Array(RING_ROWS * meshCols);
    fillBuf = new Float32Array(cols);
    blurBuf = new Float32Array(cols);
    kernelKey = "";
    // Mesh column m sits at u = m / (meshCols-1); ring column c is centered at (c + 0.5) / cols.
    upI = new Int32Array(meshCols);
    upT = new Float32Array(meshCols);
    for (let m = 0; m < meshCols; m++) {
      let p = (m / (meshCols - 1)) * cols - 0.5;
      p = p < 0 ? 0 : p > cols - 1 ? cols - 1 : p;
      const i = Math.min(cols - 2, Math.floor(p));
      upI[m] = Math.max(0, i);
      upT[m] = cols > 1 ? p - upI[m] : 0;
    }
    reset();
  }

  /** Rebuild the smoothed copy of one ring row: fill gaps → blur → upsample. */
  function smoothRow(idx) {
    const base = idx * cols;
    // 1. Fill no-data columns (raw-dB sentinels) by linear interpolation, clamped at the ends.
    let prev = -1;
    for (let c = 0; c < cols; c++) {
      const v = ring[base + c];
      if (v !== v) continue;
      if (prev < 0) fillBuf.fill(v, 0, c);
      else if (c - prev > 1) {
        const pv = fillBuf[prev];
        for (let k = prev + 1; k < c; k++) fillBuf[k] = pv + ((v - pv) * (k - prev)) / (c - prev);
      }
      fillBuf[c] = v;
      prev = c;
    }
    if (prev < 0) fillBuf.fill(0);
    else fillBuf.fill(fillBuf[prev], prev + 1);
    // 2. Resolution-aware Gaussian across frequency.
    smoothFreq(kernel, fillBuf, blurBuf);
    // 3. Catmull-Rom upsample to the mesh columns.
    const last = cols - 1;
    const out = idx * meshCols;
    for (let m = 0; m < meshCols; m++) {
      const i = upI[m], t = upT[m];
      const p1 = blurBuf[i];
      const p0 = blurBuf[i > 0 ? i - 1 : 0];
      const p2 = blurBuf[i + 1 <= last ? i + 1 : last];
      const p3 = blurBuf[i + 2 <= last ? i + 2 : last];
      const v = p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
      sRing[out + m] = v < 0 ? 0 : v;
    }
  }

  /** Record one FFT frame (called on arrival, from ws.js via main.js). */
  function push(bins) {
    const meta = store.meta || {};
    const rawDb = !!meta.fft_send_raw_db;
    if (bins.length !== n || rawDb !== rawDbMode) configure(bins.length, rawDb);
    const fMin = meta.fft_f_min ?? 30, sr = meta.sr ?? 48000, win = meta.fft_window_size ?? 1024;
    const key = `${fMin}|${sr}|${win}`;
    if (key !== kernelKey) {
      kernel = buildFreqKernel(cols, fMin, sr / 2, sr, win);
      kernelKey = key;
    }
    const floor = meta.fft_db_floor ?? -60;
    const span = Math.max(1, (meta.fft_db_ceiling ?? 0) - floor);
    const t = performance.now();
    const slotMs = (store.history_s * 1000) / MAX_ROWS;

    const open = count > 0 && t - rowStartT < slotMs;
    if (!open) {
      head = (head + 1) % RING_ROWS;
      if (count < RING_ROWS) count++;
      rowStartT = t;
      rowT[head] = t + slotMs;
      rowSeq[head] = seqN++;
      ring.fill(NaN, head * cols, head * cols + cols);
    }
    const base = head * cols;
    for (let i = 0; i < n; i++) {
      const raw = bins[i];
      let v;
      if (rawDb) {
        if (raw < SENTINEL_THRESHOLD) continue;   // empty log bin: no data
        v = (raw - floor) / span;
      } else {
        v = raw;
      }
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      const j = base + colOf[i];
      const cur = ring[j];
      if (!(cur >= v)) ring[j] = v;               // NaN-aware max
    }
    smoothRow(head);
  }

    // ---- WebGL: shared mesh buffers; the style draws ----
  const glCanvas = document.createElement("canvas");
  let gl = null, glOk = false;
  let surfVao, surfVbo, surfIbo, hTex, rowTex;
  let iboCols = 0;                                  // meshCols the index buffer was built for
  let active = null, activeName = "";               // the current style's instance (one at a time)
  // GPU time per frame (EXT_disjoint_timer_query_webgl2), recorded as viz perf "fft_gpu":
  // the CPU paint cost alone hides fill-rate / post-processing cost.
  let timerExt = null;
  const queries = [];                               // in flight, oldest first

  const vtx = new Float32Array(MAX_MESH_ROWS * MESH_COLS_MAX * VERT_FLOATS);
  const meshH = new Float32Array(MAX_MESH_ROWS * MESH_COLS_MAX);
  const meshZ = new Float64Array(MAX_MESH_ROWS);
  const meshSeq = new Float64Array(MAX_MESH_ROWS);
  const rowZK = new Float32Array(MAX_MESH_ROWS * 2);   // per mesh row: z, key (heightTex styles)
  const dIdx = new Int32Array(RING_ROWS);           // history rows in view, oldest first
  const dZ = new Float64Array(RING_ROWS);
  const dSeq = new Float64Array(RING_ROWS);
  const tA = new Float32Array(RING_ROWS * MESH_COLS_MAX);  // time-blurred rows in view, ping-pong
  const tB = new Float32Array(RING_ROWS * MESH_COLS_MAX);
  const edgeX = new Float32Array(MESH_COLS_MAX), edgeY = new Float32Array(MESH_COLS_MAX);

  // Color stops: each band's color at its geometric-center frequency.
  const stopPos = new Float32Array(3);
  const stopRgb = [new Float32Array(3), new Float32Array(3), new Float32Array(3)];
  let stopVersion = -1;

  // The per-frame bundle handed to style.render(); mutated in place.
  const f = {
    W: 1, H: 1, dpr: 1, geo: null,
    R: 0, mc: 0, rowDz: 1 / (MAX_ROWS * T_SUB), K: 1, backScale: BACK_SCALE,
    now: 0, t: 0, dt: 0, depthS: 5, fMin: 30, fMax: 24000,
    meshH, meshZ, meshSeq, edgeX, edgeY, frontOff: 0, yBaseF: 0, sF: 1,
    bloomScale: 1,                                  // > 1 while a bloom capture redraws the scene small
    key0: 0,                                        // row key of mesh row 0 (keys count up by 1 per mesh row)
    stopPos, stopRgb, bgRgb: new Float32Array(3),
    bandU: new Float32Array(6),                     // [lo, hi] u of low / mid / high
    audio: { low: 0, mid: 0, high: 0, bpm: 0, onsetAge: new Float64Array(3) },
    // Surface VAO (with index buffer; draw f.triCount indices, back to front) and,
    // for heightTex styles, meshH as an R32F texture (mc × R) + per-row (z, key) as RG32F (R × 1).
    surfVao: null, triCount: 0, hTex: null, rowTex: null,
  };

  function initGL() {
    glOk = false;
    iboCols = 0;
    active = null; activeName = "";
    queries.length = 0;
    gl = glCanvas.getContext("webgl2", {
      alpha: true, premultipliedAlpha: true, antialias: true, depth: true, preserveDrawingBuffer: false,
    });
    if (!gl) return;
    timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    const F = 4;
    surfVao = gl.createVertexArray();
    gl.bindVertexArray(surfVao);
    surfVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, surfVbo);
    gl.bufferData(gl.ARRAY_BUFFER, vtx.byteLength, gl.DYNAMIC_DRAW);
    const stride = VERT_FLOATS * F;
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 1, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, stride, 1 * F);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, stride, 2 * F);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 1, gl.FLOAT, false, stride, 4 * F);
    surfIbo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, surfIbo);   // captured by surfVao
    gl.bindVertexArray(null);
    // Mesh as textures, for styles that read it outside the surface grid (heightTex: true).
    const floatTex = (w, h, fmt) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    hTex = floatTex(MESH_COLS_MAX, MAX_MESH_ROWS, gl.R32F);
    rowTex = floatTex(MAX_MESH_ROWS, 1, gl.RG32F);
    gl.bindTexture(gl.TEXTURE_2D, null);
    f.surfVao = surfVao; f.hTex = hTex; f.rowTex = rowTex;
    glOk = true;
  }
  glCanvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); glOk = false; });
  glCanvas.addEventListener("webglcontextrestored", initGL);
  initGL();

  /**
   * The active style's instance. Only one lives at a time: switching styles
   * disposes the old one first, so an unused style never holds GPU memory
   * (bloom buffers are large at full-screen size). null if its shaders fail.
   */
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
    const idx = new Uint32Array((MAX_MESH_ROWS - 1) * (mc - 1) * 6);
    let o = 0;
    for (let r = 0; r < MAX_MESH_ROWS - 1; r++) {
      for (let m = 0; m < mc - 1; m++) {
        const a = r * mc + m, c = a + mc;
        idx[o++] = a; idx[o++] = c; idx[o++] = a + 1;
        idx[o++] = a + 1; idx[o++] = c; idx[o++] = c + 1;
      }
    }
    gl.bindVertexArray(surfVao);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    iboCols = mc;
  }

  // ---- static layer: background, style art, x-axis labels ----
  let kVersion = -1, kFmin = NaN, kSr = NaN, kEmpty = null, kDepth = NaN, kStyle = "", kTheme = -1;

  function geometry(W, H, dpr, style) {
    const backScale = style.camera?.backScale ?? BACK_SCALE;
    const padX = Math.round(PAD_X * dpr), padT = Math.round(PAD_T * dpr), padB = Math.round(PAD_B * dpr);
    const plotX = padX, plotW = Math.max(1, W - 2 * padX);
    const y0 = H - padB;                            // front baseline
    const plotH = Math.max(1, y0 - padT);
    const ampFront = (style.camera?.ampFront ?? AMP_FRONT) * plotH;
    // Back baseline sits low enough that a full-scale back row still fits,
    // plus the style's optional sky above the horizon (fraction of plot height).
    const yBack = padT + ampFront * backScale + Math.round(6 * dpr) + (style.camera?.horizon ?? 0) * plotH;
    return { plotX, plotW, cx: plotX + plotW / 2, y0, yBack, ampFront, padT, backScale };
  }

  const bgOf = (style) => (typeof style.background === "function" ? style.background() : style.background);

  function buildStatic(W, H, dpr, fMin, fMax, empty, depthS, style) {
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
    const logFmin = Math.log10(fMin);
    const logSpan = Math.max(1e-6, Math.log10(fMax) - logFmin);
    if (style.drawStatic) {
      g.save();
      style.drawStatic(g, geo, W, H, dpr, { depthS, logFmin, logSpan, fMin, fMax });
      g.restore();
      g.font = `${Math.round(10 * dpr)}px ${FONT_UI}`;
    }

    // X axis under the front edge — same log mapping as the 2D view.
    g.fillStyle = text.axis || "#7a8088";
    g.textAlign = "center";
    g.textBaseline = "top";
    const yLabel = geo.y0 + Math.round(3 * dpr);
    const minSpacingPx = Math.round(40 * dpr);
    let lastTickPx = -Infinity;
    for (const fq of X_TICK_CANDIDATES) {
      if (fq < fMin || fq > fMax) continue;
      const x = Math.round(geo.plotX + ((Math.log10(fq) - logFmin) / logSpan) * geo.plotW);
      if (x - lastTickPx < minSpacingPx) continue;
      lastTickPx = x;
      g.fillText(fq >= 1000 ? `${fq / 1000}k` : `${fq}`, x, yLabel);
    }

    // Depth label at the horizon, right edge (clear of the narrow back rows).
    g.fillStyle = text.dim || "#5a6068";
    g.textAlign = "right";
    g.textBaseline = text.horizonAbove ? "bottom" : "middle";
    g.fillText(`horizon −${depthS < 10 ? depthS.toFixed(1) : Math.round(depthS)} s`, geo.plotX + geo.plotW,
      text.horizonAbove ? geo.yBack - Math.round(3 * dpr) : geo.yBack);
  }

  /**
   * Copy the N rows in view into tA/tB and blur them across time, keeping
   * max(value, blurred) so peaks keep their height. Returns the result buffer.
   */
  function blurTime(N) {
    const mc = meshCols;
    for (let i = 0; i < N; i++) tA.set(sRing.subarray(dIdx[i] * mc, dIdx[i] * mc + mc), i * mc);
    let src = tA, dst = tB;
    for (let pass = 0; pass < TIME_BLUR_PASSES; pass++) {
      for (let i = 0; i < N; i++) {
        const o = i * mc;
        const op = (i > 0 ? i - 1 : 0) * mc, on = (i < N - 1 ? i + 1 : N - 1) * mc;
        for (let m = 0; m < mc; m++) {
          const v = src[o + m];
          const b = 0.5 * v + 0.25 * (src[op + m] + src[on + m]);
          dst[o + m] = b > v ? b : v;
        }
      }
      const t = src; src = dst; dst = t;
    }
    return src;
  }

  /** Fill meshH / meshZ / meshSeq from the N history rows in view. Returns the mesh row count. */
  function buildMesh(N) {
    const mc = meshCols;
    const rows = blurTime(N);
    const R = (N - 1) * T_SUB + 1;
    for (let r = 0; r < R; r++) {
      const i = Math.min(N - 2, Math.floor(r / T_SUB));
      const t = r / T_SUB - i;
      const i0 = i > 0 ? i - 1 : 0, i2 = i + 1, i3 = i + 2 < N ? i + 2 : N - 1;
      // Catmull-Rom weights at t.
      const t2 = t * t, t3 = t2 * t;
      const w0 = 0.5 * (-t3 + 2 * t2 - t);
      const w1 = 0.5 * (3 * t3 - 5 * t2 + 2);
      const w2 = 0.5 * (-3 * t3 + 4 * t2 + t);
      const w3 = 0.5 * (t3 - t2);
      const a0 = i0 * mc, a1 = i * mc, a2 = i2 * mc, a3 = i3 * mc;
      const o = r * mc;
      for (let m = 0; m < mc; m++) {
        const v = w0 * rows[a0 + m] + w1 * rows[a1 + m] + w2 * rows[a2 + m] + w3 * rows[a3 + m];
        meshH[o + m] = v < 0 ? 0 : v > 1.25 ? 1.25 : v;
      }
      meshZ[r] = dZ[i] + (dZ[i2] - dZ[i]) * t;
      meshSeq[r] = dSeq[i] + (dSeq[i2] - dSeq[i]) * t;
    }
    return R;
  }

  /** Interleave height, depth, world-space slopes and the row key into vtx. */
  function fillVertices(R) {
    const mc = meshCols;
    const dX = 2 / (mc - 1);
    let o = 0;
    for (let r = 0; r < R; r++) {
      const rp = r > 0 ? r - 1 : 0, rn = r < R - 1 ? r + 1 : R - 1;
      const z = meshZ[r];
      // Rows run oldest (far) → newest (near), so rp is the farther neighbor.
      const dZw = (meshZ[rp] - meshZ[rn]) * WORLD_DEPTH;
      const invDz = dZw > 1e-6 ? WORLD_HEIGHT / dZw : 0;
      const key = Math.round(meshSeq[r] * T_SUB) % 65536;   // stable per mesh row as it recedes
      rowZK[2 * r] = z; rowZK[2 * r + 1] = key;
      const row = r * mc, rowP = rp * mc, rowN = rn * mc;
      for (let m = 0; m < mc; m++) {
        const ml = m > 0 ? m - 1 : 0, mr = m < mc - 1 ? m + 1 : mc - 1;
        vtx[o++] = meshH[row + m];
        vtx[o++] = z;
        vtx[o++] = ((meshH[row + mr] - meshH[row + ml]) * WORLD_HEIGHT) / ((mr - ml) * dX);
        vtx[o++] = (meshH[rowP + m] - meshH[rowN + m]) * invDz;
        vtx[o++] = key;
      }
    }
  }

  let lastDrawT = performance.now();

  function draw() {
    const t0 = performance.now();
    const { w: W, h: H, dpr, version } = surf.fit();
    const style = STYLES[store.fft3d_style] || STYLES[DEFAULT_STYLE];

    const meta = store.meta || {};
    const fMin = meta.fft_f_min ?? 30;
    const sr = meta.sr ?? 48000;
    const fMax = sr / 2;
    const depthS = store.history_s;
    const depthMs = depthS * 1000;

    // FFT off (or no frame yet): drop the history so a stale ocean never shows.
    if (!store.fft_bins) reset();
    const inst = glOk ? instance(style) : null;
    const empty = !glOk ? "3D view needs WebGL2"
      : !inst ? `3D style "${style.name}" failed to build`
      : count > 0 ? "" : (meta.fft_enabled ? "waiting for data…" : "FFT disabled");

    if (version !== kVersion || fMin !== kFmin || sr !== kSr || empty !== kEmpty || depthS !== kDepth
        || style.name !== kStyle || theme.version !== kTheme) {
      kVersion = version; kFmin = fMin; kSr = sr; kEmpty = empty; kDepth = depthS;
      kStyle = style.name; kTheme = theme.version;
      buildStatic(W, H, dpr, fMin, fMax, empty, depthS, style);
      f.bgRgb.set(hex01(bgOf(style)));                 // fog / face color (style + palette)
    }
    ctx.drawImage(layer.canvas, 0, 0);
    if (empty) {
      recordVizPerf("fft", performance.now() - t0);
      return;
    }

    // History rows inside the depth window, oldest first. The open row's slot
    // hasn't closed yet (z < 0 → 0): it is the live front edge.
    const now = performance.now();
    let N = 0;
    for (let k = count - 1; k >= 0; k--) {
      const idx = (head - k + RING_ROWS) % RING_ROWS;
      const z = (now - rowT[idx]) / depthMs;
      if (z > 1) continue;
      dIdx[N] = idx; dZ[N] = z < 0 ? 0 : z; dSeq[N] = rowSeq[idx]; N++;
    }
    if (N === 0) {
      recordVizPerf("fft", performance.now() - t0);
      return;
    }

    const mc = meshCols;
    const geo = geometry(W, H, dpr, style);
    const logFmin = Math.log10(fMin);
    const logSpan = Math.max(1e-6, Math.log10(fMax) - logFmin);
    const toU = (hz) => { const p = (Math.log10(hz) - logFmin) / logSpan; return p < 0 ? 0 : p > 1 ? 1 : p; };
    const bands = meta.bands || {};
    for (let k = 0; k < 3; k++) {
      const b = bands[LMH_ORDER[k]];
      stopPos[k] = toU(b ? Math.sqrt(b.lo_hz * b.hi_hz) : [100, 1000, 8000][k]);
      f.bandU[2 * k] = toU(b ? b.lo_hz : [30, 250, 4000][k]);
      f.bandU[2 * k + 1] = toU(b ? b.hi_hz : [250, 4000, 16000][k]);
    }
    if (stopVersion !== theme.version) {
      stopVersion = theme.version;
      for (let k = 0; k < 3; k++) stopRgb[k].set(rgb01(LMH[LMH_ORDER[k]].rgb));
    }

    // Mesh (a single history row still gets a front edge).
    let R = 1;
    if (N >= 2) {
      R = buildMesh(N);
      fillVertices(R);
    } else {
      const src = dIdx[0] * mc;
      for (let m = 0; m < mc; m++) meshH[m] = sRing[src + m];
      meshZ[0] = dZ[0];
      meshSeq[0] = dSeq[0];
    }

    // Front edge (the newest mesh row) in device px.
    const K = 1 / geo.backScale - 1;
    const zF = meshZ[R - 1];
    const sF = 1 / (1 + K * zF);
    const yBaseF = geo.y0 - ((1 - sF) / (1 - geo.backScale)) * (geo.y0 - geo.yBack);
    const ampF = geo.ampFront * sF;
    const front = (R - 1) * mc;
    const halfW = geo.plotW / 2;
    for (let m = 0; m < mc; m++) {
      const u = m / (mc - 1);
      edgeX[m] = geo.cx + (u - 0.5) * 2 * halfW * sF;
      edgeY[m] = yBaseF - Math.min(1, meshH[front + m]) * ampF;
    }

    // Fill the frame bundle.
    f.W = W; f.H = H; f.dpr = dpr; f.geo = geo;
    f.R = N >= 2 ? R : 0; f.mc = mc; f.K = K; f.backScale = geo.backScale;
    f.dt = Math.min(0.1, Math.max(0, (now - lastDrawT) / 1000));
    lastDrawT = now;
    f.now = now; f.t = now / 1000; f.depthS = depthS; f.fMin = fMin; f.fMax = fMax;
    f.frontOff = front; f.yBaseF = yBaseF; f.sF = sF;
    f.key0 = Math.round(meshSeq[0] * T_SUB) % 65536;
    f.triCount = f.R >= 2 ? (f.R - 1) * (mc - 1) * 6 : 0;
    const a = f.audio;
    a.low = store.low; a.mid = store.mid; a.high = store.high; a.bpm = store.bpm;
    a.onsetAge[0] = (now - store.low_onset_pulse_t) / 1000;
    a.onsetAge[1] = (now - store.mid_onset_pulse_t) / 1000;
    a.onsetAge[2] = (now - store.high_onset_pulse_t) / 1000;

    if (glCanvas.width !== W) glCanvas.width = W;
    if (glCanvas.height !== H) glCanvas.height = H;
    if (f.R >= 2) {
      if (iboCols !== mc) buildIndices(mc);
      gl.bindVertexArray(surfVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, surfVbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, vtx, 0, R * mc * VERT_FLOATS);
      gl.bindVertexArray(null);
      if (style.heightTex) {
        gl.bindTexture(gl.TEXTURE_2D, hTex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, mc, R, gl.RED, gl.FLOAT, meshH);
        gl.bindTexture(gl.TEXTURE_2D, rowTex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, R, 1, gl.RG, gl.FLOAT, rowZK);
        gl.bindTexture(gl.TEXTURE_2D, null);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

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
    ctx.drawImage(glCanvas, 0, 0);
    recordVizPerf("fft", performance.now() - t0);
  }

  return { draw, push, reset };
}
