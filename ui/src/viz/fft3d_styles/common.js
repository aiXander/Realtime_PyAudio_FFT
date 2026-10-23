// Shared pieces for the 3D FFT styles: shader snippets (projection, band
// colors, magnitude colormap, AA grid lines, hashing), program building, the
// surface uniforms and screen-space strips for the front edge.
//
// A style module (see cloud.js for the reference) default-exports:
//
//   {
//     name, label, description,
//     camera: { backScale, ampFront },         // optional
//     background: "#hex" | () => "#hex",       // static-layer fill (+ fog color)
//     text: { axis, dim },                     // static-layer label colors
//     drawStatic(g, geo, W, H, dpr, info),     // optional extra static art
//     smoothing: { freqOct, timePasses },      // optional: lighter display smoothing for detail styles
//                                              //   (defaults 0.08 oct σ across frequency, 2 time passes)
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

// Projection shared by every style. Mirrors the 2D ridge layout: row depth
// z ∈ [0 front, 1 horizon] shrinks by s(z) = 1/(1 + K·z) toward the
// horizon; w = 1/s so varyings interpolate perspective-correctly, and depth
// (NDC 0.9 − 1.8·s) is linear in 1/w, so it is exact for planes.
export const PROJ_FN = `
uniform vec2 uRes;
uniform float uCx, uHalfW, uY0, uYBack, uAmp, uDpr, uRowDz, uK, uBackScale;
float persp(float z) { return 1.0 / (1.0 + uK * z); }
/** Device-px position of (u, h) at depth z; s = its perspective scale. */
vec2 toPx(float u, float h, float z, out float s) {
  s = persp(z);
  float yBase = uY0 - (1.0 - s) / (1.0 - uBackScale) * (uY0 - uYBack);
  return vec2(uCx + (u - 0.5) * 2.0 * uHalfW * s, yBase - h * uAmp * s);
}
vec4 clipPos(vec2 px, float s) {
  float w = 1.0 / s;
  return vec4((px.x / uRes.x * 2.0 - 1.0) * w, (1.0 - px.y / uRes.y * 2.0) * w, (0.9 - 1.8 * s) * w, w);
}`;

export const HASH_FN = `
uint hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
float hash01(uint x) { return float(hash(x) & 0xFFFFu) / 65535.0; }`;

/** Surface vertex stage: one vertex per mesh point, row-major, u from the vertex index. */
export const SURF_VS = `#version 300 es
layout(location = 0) in float aH;
layout(location = 1) in float aZ;
layout(location = 2) in vec2 aSlope;
layout(location = 3) in float aKey;
uniform int uCols;
${PROJ_FN}
uniform float uRow0;   // optional: f.key0 mod a multiple of f.tSub (row-aligned lines; see docs/reference/fft_3d_styles.md)
out float vU, vH, vZ, vKey, vS, vRow;
out vec3 vN;
out vec2 vPx;
void main() {
  float u = float(gl_VertexID % uCols) / float(uCols - 1);
  float s;
  vec2 px = toPx(u, aH, aZ, s);
  gl_Position = clipPos(px, s);
  vU = u; vH = aH; vZ = aZ; vKey = aKey; vS = s; vPx = px;
  vRow = float(gl_VertexID / uCols) + uRow0;   // mesh row, stable per row as it recedes (wrap-free, unlike vKey)
  vN = vec3(-aSlope.x, 1.0, -aSlope.y);
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
  const K = 1 / geo.backScale - 1;
  g.font = `${Math.round(9 * dpr)}px ui-monospace, Menlo, monospace`;
  g.fillStyle = color;
  g.textAlign = "left";
  g.textBaseline = "middle";
  let lastY = Infinity;
  for (let sec = 1; sec < depthS; sec++) {
    const s = 1 / (1 + K * (sec / depthS));
    const y = geo.y0 - ((1 - s) / (1 - geo.backScale)) * (geo.y0 - geo.yBack);
    if (lastY - y < 14 * dpr) continue;
    lastY = y;
    g.fillText(`−${sec}s`, geo.cx + (geo.plotW / 2) * s + 4 * dpr, y);
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
  if (u.uHalfW) gl.uniform1f(u.uHalfW, geo.plotW / 2);
  if (u.uY0) gl.uniform1f(u.uY0, geo.y0);
  if (u.uYBack) gl.uniform1f(u.uYBack, geo.yBack);
  if (u.uAmp) gl.uniform1f(u.uAmp, geo.ampFront);
  if (u.uDpr) gl.uniform1f(u.uDpr, f.dpr);
  if (u.uRowDz) gl.uniform1f(u.uRowDz, f.rowDz);
  if (u.uK) gl.uniform1f(u.uK, f.K);
  if (u.uBackScale) gl.uniform1f(u.uBackScale, f.backScale);
  if (u.uBg) gl.uniform3fv(u.uBg, f.bgRgb);
  if (u.uStopPos) gl.uniform3fv(u.uStopPos, f.stopPos);
  if (u.uC0) gl.uniform3fv(u.uC0, f.stopRgb[0]);
  if (u.uC1) gl.uniform3fv(u.uC1, f.stopRgb[1]);
  if (u.uC2) gl.uniform3fv(u.uC2, f.stopRgb[2]);
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
    const u = m / (mc - 1), hv = Math.min(1, f.meshH[f.frontOff + m]);
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
