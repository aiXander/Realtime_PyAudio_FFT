// Shared pieces for the 3D FFT styles: shader snippets (projection, the
// spectro texture + grid, band colors, magnitude colormap, AA grid lines,
// hashing), program building, the surface uniforms and screen-space strips
// for the front edge.
//
// A style module (see cloud.js for the reference) default-exports:
//
//   {
//     name, label, description,
//     camera: { backScale, ampFront },         // optional: the look at perspective ×1 / height ×1
//     background: "#hex" | () => "#hex",       // static-layer fill (+ fog color)
//     text: { axis, dim },                     // static-layer label colors
//     drawStatic(g, geo, W, H, dpr, info),     // optional extra static art
//     grid: { cols, rows },                    // optional: geometry density caps (defaults 1024 × 480);
//                                              //   per-pixel shading (SPEC_FN) is at data resolution anyway
//     create(gl, core) → { render(f), dispose() }, // compile programs, own GL objects; dispose frees them
//   }
//
// Only the active style has an instance: `create` runs when a style becomes
// active (and again after a WebGL context restore), `dispose` when another
// style replaces it. The
// frame `f` handed to render() is preallocated and mutated by the core each
// frame (see the `f` bundle in fft_3d.js for the fields). Nothing in render()
// may allocate: preallocate typed arrays in create().

import { LMH, theme } from "../../colors.js";
// Lighting-only world scale (surface width = 2): how steep slopes look to the light.
export const WORLD_HEIGHT = 0.5, WORLD_DEPTH = 3.0;
/** Vertex shader: data rows pooled per grid row (cap). */
export const MAX_POOL = 6;

export const glf = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));
export const hex01 = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);

// ---- palette helpers (static art in JS; shaders read uC0/uC1/uC2 = low/mid/high) ----

/** Blend two #rrggbb colors → #rrggbb. */
export function mixHex(a, b, t) {
  const pa = hex01(a), pb = hex01(b);
  return "#" + pa.map((v, i) => Math.round(255 * (v + (pb[i] - v) * t)).toString(16).padStart(2, "0")).join("");
}
/** The palette's band color: "low" | "mid" | "high". */
export const pal = (k) => LMH[k].hex;
/** The palette's background, tinted toward a band color. */
export const palBg = (k = "low", t = 0.08) => mixHex(theme.bg, LMH[k].hex, t);

// ---- shader snippets ----

// Projection shared by every style (fft_3d.js geometry() sets the camera).
// Row depth z ∈ [0 front, 1 horizon] shrinks by s(z) = 1/(1 + K·z) toward
// the horizon (K = 0: parallel rows). ground(z) ∈ [0, 1] is how far up the
// row's baseline sits between the front (uY0) and the horizon (uYBack); the
// skew shifts rows sideways by the same share. w = 1/s so varyings
// interpolate perspective-correctly; depth (NDC −0.9 + 1.8·ground) is affine
// in w, so it is exact for planes and still orders rows when K = 0.
// Mirrored on the CPU by projectRow() below.
export const PROJ_FN = `
uniform vec2 uRes;
uniform float uCx, uHalfW, uY0, uYBack, uAmp, uDpr, uRowDz, uK, uSkew;
float persp(float z) { return 1.0 / (1.0 + uK * z); }
float ground(float z) { return z * (1.0 + uK) / (1.0 + uK * z); }
/** Device-px position of (u, h) at depth z; s = its perspective scale. */
vec2 toPx(float u, float h, float z, out float s) {
  s = persp(z);
  float g = ground(z);
  return vec2(uCx + uSkew * g + (u - 0.5) * 2.0 * uHalfW * s, uY0 - g * (uY0 - uYBack) - h * uAmp * s);
}
vec4 clipPos(vec2 px, float s, float z) {
  float w = 1.0 / s;
  return vec4((px.x / uRes.x * 2.0 - 1.0) * w, (1.0 - px.y / uRes.y * 2.0) * w, (-0.9 + 1.8 * ground(z)) * w, w);
}`;

/** CPU twin of toPx for a row's baseline at depth z: { s, y, cx } (row center x). */
export function projectRow(geo, z) {
  const s = 1 / (1 + geo.K * z);
  const g = z * (1 + geo.K) * s;
  return { s, y: geo.y0 - g * (geo.y0 - geo.yBack), cx: geo.cx + geo.skewPx * g };
}

export const HASH_FN = `
uint hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
float hash01(uint x) { return float(hash(x) & 0xFFFFu) / 65535.0; }`;

// ---- the spectro texture (see fft_3d.js) ----

/**
 * SPEC_FN: read the history texture. Rows are absolute data rows mod the
 * ring (REPEAT wrap, so any float row works); u is 0..1 across frequency.
 *   specH(u, row)            height 0..1 (dB → range, tilt, level)
 *   specPeak(u, row, r, b)   pooled height over a footprint of r rows (older from row) × b bins
 *   specSurface(u, row, h, n) per-pixel height (pooled over the pixel's footprint) + normal (SPEC_FS)
 * setCommonUniforms sets every uniform from f.spec.
 *
 * Pooling is halfway between the mean and the max of the taps (poolMix).
 * The mean alone is the steadier picture as it scrolls, but a hat far back,
 * where one pixel covers many rows, sinks into it; the max alone keeps the
 * hat and shimmers twice as much there (the taps slide over the rows).
 */
export const SPEC_FN = `
uniform sampler2D uSpec;
uniform float uNBins, uTexRows, uDbScale, uDb0, uFloorDb, uInvRange, uTilt, uLogOct, uLog2FminK, uNView;
float poolMix(float mx, float mean) { return 0.5 * (mx + mean); }
float specH(float u, float row) {
  vec2 tc = vec2((u * (uNBins - 1.0) + 0.5) / uNBins, (row + 0.5) / uTexRows);
  float db = texture(uSpec, tc).r * uDbScale + uDb0 + uTilt * (uLog2FminK + u * uLogOct);
  return clamp((db - uFloorDb) * uInvRange, 0.0, 1.0);
}
float specPeak(float u, float row, float rows, float bins) {
  int nr = int(clamp(ceil(rows), 1.0, 6.0));
  int nc = int(clamp(ceil(bins), 1.0, 3.0));
  float dr = rows / float(nr), du = bins / float(nc) / (uNBins - 1.0);
  float m = 0.0, sum = 0.0;
  for (int i = 0; i < 6; i++) {
    if (i >= nr) break;
    for (int j = 0; j < 3; j++) {
      if (j >= nc) break;
      float h = specH(u + (float(j) - 0.5 * float(nc - 1)) * du, row - float(i) * dr);
      m = max(m, h);
      sum += h;
    }
  }
  return poolMix(m, sum / float(nr * nc));
}`;

/** SPEC_FN + specSurface (fragment stage only: it needs fwidth). */
export const SPEC_FS = SPEC_FN + `
void specSurface(float u, float row, out float h, out vec3 n) {
  float rows = max(fwidth(row), 1.0);              // data rows under this pixel
  float bins = fwidth(u) * (uNBins - 1.0);         // bins under this pixel
  h = specPeak(u, row, rows, bins);
  float du = max(bins, 1.0) / (uNBins - 1.0);
  float sx = ${glf(WORLD_HEIGHT)} * (specH(u + du, row) - specH(u - du, row)) / (4.0 * du);
  float sz = ${glf(WORLD_HEIGHT)} * (specH(u, row - rows) - specH(u, row + rows)) / (2.0 * rows / uNView * ${glf(WORLD_DEPTH)});
  n = vec3(-sx, 1.0, -sz);
}`;

/**
 * GRID_FN: the grid vertex → (u, texture row, depth z, row key, pool).
 * Rows 0 … uBodyRows−1 are anchored to data rows uStep apart (the newest,
 * "row A", at texture row uRowA); row uBodyRows is the live front edge at the
 * playhead, uFrontOff data rows past row A. Needs SPEC_FN before it.
 */
export const GRID_FN = `
uniform int uCols;
uniform float uRowA, uFrontOff, uStep, uPool, uBodyRows, uKeyA;
void gridVertex(int vid, out float u, out float row, out float z, out float key, out float pool) {
  int c = vid % uCols, r = vid / uCols;
  u = float(c) / float(uCols - 1);
  float back = uBodyRows - 1.0 - float(r);         // grid rows behind row A
  float rel = back < 0.0 ? uFrontOff : -back * uStep;
  key = mod(back < 0.0 ? uKeyA + 1.0 : uKeyA - back, 65536.0);
  pool = back < 0.0 ? 1.0 : uPool;
  rel = max(rel, uFrontOff - uNView);              // the back row sits on the horizon
  z = clamp((uFrontOff - rel) / uNView, 0.0, 1.0);
  row = uRowA + rel;
}
/** Height pooled over the data rows one grid row stands for (same pooling as specPeak). */
float gridHeight(float u, float row, float pool) {
  float m = 0.0, sum = 0.0, d = uStep / pool;
  for (int i = 0; i < ${MAX_POOL}; i++) {
    if (float(i) >= pool) break;
    float h = specH(u, row - float(i) * d);
    m = max(m, h);
    sum += h;
  }
  return poolMix(m, sum / pool);
}`;

/** Surface vertex stage: one vertex per grid point, heights from the texture. Fragment
 *  stages shade per pixel from the texture (specSurface), so there's no vertex normal. */
export const SURF_VS = `#version 300 es
${PROJ_FN}
${SPEC_FN}
${GRID_FN}
out float vU, vH, vZ, vKey, vS, vT;
out vec2 vPx;
void main() {
  float u, row, z, key, pool;
  gridVertex(gl_VertexID, u, row, z, key, pool);
  float h = gridHeight(u, row, pool);
  float s;
  vec2 px = toPx(u, h, z, s);
  gl_Position = clipPos(px, s, z);
  vU = u; vH = h; vZ = z; vKey = key; vS = s; vPx = px; vT = row;
}`;

/** The palette's band colors as uniforms (set by setCommonUniforms). Don't combine with bandFn(), which declares them too. */
export const PAL_UNIFORMS = `uniform vec3 uC0, uC1, uC2;   // palette low / mid / high`;

/** band(u): the palette's L/M/H colors at their band centers, blended across frequency. */
export const bandFn = (desaturate = 0) => `
uniform vec3 uStopPos;
uniform vec3 uC0, uC1, uC2;
vec3 band(float u) {
  vec3 c;
  if (u <= uStopPos.x) c = uC0;
  else if (u <= uStopPos.y) c = mix(uC0, uC1, (u - uStopPos.x) / max(1e-4, uStopPos.y - uStopPos.x));
  else if (u <= uStopPos.z) c = mix(uC1, uC2, (u - uStopPos.y) / max(1e-4, uStopPos.z - uStopPos.y));
  else c = uC2;
  return mix(c, vec3(dot(c, vec3(0.299, 0.587, 0.114))), ${glf(desaturate)});
}`;

/**
 * ramp(h): a magnitude colormap built from the palette, ordered by lightness
 * like a scientific heat scale (viridis / inferno): hue walks low → mid →
 * high, luminance rises monotonically from near-black to near-white whatever
 * the palette's own colors are, so equal steps in h read as equal steps in
 * brightness. Needs PAL_UNIFORMS (or bandFn) declared before it.
 */
export const RAMP_FN = `
vec3 ramp(float h) {
  h = clamp(h, 0.0, 1.0);
  vec3 c = h < 0.5 ? mix(uC0, uC1, h * 2.0) : mix(uC1, uC2, h * 2.0 - 1.0);
  const vec3 Y = vec3(0.2126, 0.7152, 0.0722);
  float target = mix(0.025, 0.93, pow(h, 0.9));
  c *= target / max(dot(c, Y), 1e-3);
  float m = max(c.r, max(c.g, c.b));
  if (m > 1.0) {
    // Out of gamut: keep the hue, make up the missing luminance with white.
    c /= m;
    float y = dot(c, Y);
    c = mix(c, vec3(1.0), clamp((target - y) / max(1.0 - y, 1e-3), 0.0, 1.0));
  }
  return c;
}`;

/**
 * gridLine(q, halfPx): AA coverage of lines at integer q, halfPx wide on
 * screen (fwidth-based, so constant pixel width at any depth or slope),
 * faded to nothing where they crowd closer than a few px (no moiré).
 */
export const LINE_FN = `
float gridLine(float q, float halfPx) {
  float w = max(fwidth(q), 1e-5);
  float d = abs(fract(q + 0.5) - 0.5) / w;
  return (1.0 - smoothstep(halfPx - 0.5, halfPx + 0.5, d)) * (1.0 - smoothstep(0.12, 0.3, w));
}`;

/**
 * logPaper(u): log graph paper across frequency, as (major, minor) line
 * coverage: majors at the decades (100 Hz, 1k, 10k), minors at 2…9 × each
 * decade. Needs LINE_FN before it; set uLog0 / uLogSpan with setLogUniforms.
 */
export const LOG_PAPER_FN = `
uniform float uLog0, uLogSpan;   // log10(fMin), log10(fMax / fMin)
vec2 logPaper(float u, float halfPx) {
  float dec = uLog0 + u * uLogSpan;
  return vec2(gridLine(dec, halfPx), gridLine(pow(10.0, fract(dec)), halfPx));
}`;
export function setLogUniforms(gl, u, f) {
  gl.uniform1f(u.uLog0, Math.log10(f.fMin));
  gl.uniform1f(u.uLogSpan, Math.log10(f.fMax / f.fMin));
}

/** Static-layer labels for the 1 s parallels (−1s, −2s …) at the right edge, thinned where they crowd. */
export function drawSecondLabels(g, geo, dpr, depthS, color) {
  g.font = `${Math.round(9 * dpr)}px ui-monospace, Menlo, monospace`;
  g.fillStyle = color;
  g.textAlign = "left";
  g.textBaseline = "middle";
  let lastY = Infinity;
  for (let sec = 1; sec < depthS; sec++) {
    const { s, y, cx } = projectRow(geo, sec / depthS);
    if (lastY - y < 14 * dpr) continue;
    lastY = y;
    // Rows wider than the plot (overscan) put their edge off screen: pin the label inside.
    g.fillText(`−${sec}s`, Math.min(cx + geo.halfW * s + 4 * dpr, geo.plotX + geo.plotW - 22 * dpr), y);
  }
}

// Screen-space strips (curtain / rim): positions are device px from the CPU.
export const FLAT_VS = `#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
uniform vec2 uRes;
out vec2 vUV;
void main() {
  vUV = aUV;
  gl_Position = vec4(aPos.x / uRes.x * 2.0 - 1.0, 1.0 - aPos.y / uRes.y * 2.0, 0.0, 1.0);
}`;

// ---- program building ----

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}

/** Compile + link. Returns { prog, u } with u[name] = location for every active uniform. */
export function program(gl, vs, fs) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const u = {};
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(prog, i).name.replace(/\[0\]$/, "");
    u[name] = gl.getUniformLocation(prog, name);
  }
  return { prog, u };
}

/** Delete programs, and anything else with a dispose() (strips), from a style's instance. */
export function disposeAll(gl, ...things) {
  for (const t of things) {
    if (!t) continue;
    if (t.prog) gl.deleteProgram(t.prog);
    else if (t.dispose) t.dispose();
  }
}

/** Set the projection / band / background uniforms a program declares (missing ones are skipped). */
export function setCommonUniforms(gl, u, f) {
  const geo = f.geo;
  if (u.uCols) gl.uniform1i(u.uCols, f.mc);
  if (u.uRes) gl.uniform2f(u.uRes, f.W, f.H);
  if (u.uCx) gl.uniform1f(u.uCx, geo.cx);
  if (u.uHalfW) gl.uniform1f(u.uHalfW, geo.halfW);
  if (u.uSkew) gl.uniform1f(u.uSkew, geo.skewPx);
  if (u.uY0) gl.uniform1f(u.uY0, geo.y0);
  if (u.uYBack) gl.uniform1f(u.uYBack, geo.yBack);
  if (u.uAmp) gl.uniform1f(u.uAmp, geo.ampFront);
  if (u.uDpr) gl.uniform1f(u.uDpr, f.dpr);
  if (u.uRowDz) gl.uniform1f(u.uRowDz, f.rowDz);
  if (u.uK) gl.uniform1f(u.uK, f.K);
  if (u.uBg) gl.uniform3fv(u.uBg, f.bgRgb);
  if (u.uStopPos) gl.uniform3fv(u.uStopPos, f.stopPos);
  if (u.uC0) gl.uniform3fv(u.uC0, f.stopRgb[0]);
  if (u.uC1) gl.uniform3fv(u.uC1, f.stopRgb[1]);
  if (u.uC2) gl.uniform3fv(u.uC2, f.stopRgb[2]);
  const sp = f.spec;
  if (u.uSpec) gl.uniform1i(u.uSpec, 0);
  if (u.uNBins) gl.uniform1f(u.uNBins, sp.nBins);
  if (u.uTexRows) gl.uniform1f(u.uTexRows, sp.texRows);
  if (u.uDbScale) gl.uniform1f(u.uDbScale, sp.dbScale);
  if (u.uDb0) gl.uniform1f(u.uDb0, sp.db0);
  if (u.uFloorDb) gl.uniform1f(u.uFloorDb, sp.floorDb);
  if (u.uInvRange) gl.uniform1f(u.uInvRange, sp.invRange);
  if (u.uTilt) gl.uniform1f(u.uTilt, sp.tilt);
  if (u.uLogOct) gl.uniform1f(u.uLogOct, sp.logOct);
  if (u.uLog2FminK) gl.uniform1f(u.uLog2FminK, sp.log2FminK);
  if (u.uNView) gl.uniform1f(u.uNView, sp.nView);
  if (u.uRowA) gl.uniform1f(u.uRowA, sp.rowA);
  if (u.uFrontOff) gl.uniform1f(u.uFrontOff, sp.frontOff);
  if (u.uStep) gl.uniform1f(u.uStep, sp.step);
  if (u.uPool) gl.uniform1f(u.uPool, sp.pool);
  if (u.uBodyRows) gl.uniform1f(u.uBodyRows, sp.bodyRows);
  if (u.uKeyA) gl.uniform1f(u.uKeyA, sp.keyA);
}

// ---- screen-space strips ----

/** A dynamic [x, y, u, v] vertex buffer for FLAT_VS strips. */
export function makeStrips(gl, maxVerts) {
  const data = new Float32Array(maxVerts * 4);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, data.byteLength, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
  gl.bindVertexArray(null);
  return {
    data,
    dispose() { gl.deleteBuffer(vbo); gl.deleteVertexArray(vao); },
    /** Upload the first `floats` floats and leave the VAO bound. */
    upload(floats) {
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data, 0, floats);
    },
  };
}

/** Curtain under the front edge: 2·mc verts, uv = (u, height 0..1 at the edge, 0 at the baseline). Returns the new offset. */
export function fillCurtain(f, out, o) {
  const mc = f.mc;
  for (let m = 0; m < mc; m++) {
    const u = m / (mc - 1), hv = Math.min(1, f.frontH[m]);
    out[o++] = f.edgeX[m]; out[o++] = f.edgeY[m]; out[o++] = u; out[o++] = hv;
    out[o++] = f.edgeX[m]; out[o++] = f.yBaseF;   out[o++] = u; out[o++] = 0;
  }
  return o;
}

/** Rim along the front edge, offset ±halfPx along the screen normal: uv = (u, side −1..1). Returns the new offset. */
export function fillRim(f, out, o, halfPx) {
  const mc = f.mc, ex = f.edgeX, ey = f.edgeY;
  for (let m = 0; m < mc; m++) {
    const ml = m > 0 ? m - 1 : 0, mr = m < mc - 1 ? m + 1 : mc - 1;
    const tx = ex[mr] - ex[ml], ty = ey[mr] - ey[ml];
    const inv = halfPx / (Math.hypot(tx, ty) || 1);
    const nx = -ty * inv, ny = tx * inv;
    const u = m / (mc - 1);
    out[o++] = ex[m] + nx; out[o++] = ey[m] + ny; out[o++] = u; out[o++] = 1;
    out[o++] = ex[m] - nx; out[o++] = ey[m] - ny; out[o++] = u; out[o++] = -1;
  }
  return o;
}
