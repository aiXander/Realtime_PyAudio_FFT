// Shared pieces for the 3D FFT styles: shader snippets (projection, band
// colors, hashing), program building, the surface uniforms, screen-space
// strips for the front edge, and an opt-in bloom post-process.
//
// A style module (see cloud.js for the reference) default-exports:
//
//   {
//     name, label, description,
//     camera: { backScale, ampFront, horizon }, // optional; horizon = sky above the back row, fraction of plot height
//     background: "#hex" | () => "#hex",       // static-layer fill (+ fog color)
//     text: { axis, dim, horizonAbove },       // static-layer label colors (+ horizon label above the line)
//     drawStatic(g, geo, W, H, dpr, info),     // optional extra static art
//     heightTex: true,                         // optional: core uploads meshH as a texture
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

/** Seeded PRNG (mulberry32) for reproducible static art. */
export function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

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
out float vU, vH, vZ, vKey, vS;
out vec3 vN;
out vec2 vPx;
void main() {
  float u = float(gl_VertexID % uCols) / float(uCols - 1);
  float s;
  vec2 px = toPx(u, aH, aZ, s);
  gl_Position = clipPos(px, s);
  vU = u; vH = aH; vZ = aZ; vKey = aKey; vS = s; vPx = px;
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

// Fullscreen triangle (no buffers): vUV in [0,1], vPx in device px (y down).
export const FULL_VS = `#version 300 es
uniform vec2 uRes;
out vec2 vUV;
out vec2 vPx;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUV = p;
  vPx = vec2(p.x, 1.0 - p.y) * uRes;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
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

/** Delete programs, and anything else with a dispose() (strips, bloom), from a style's instance. */
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
  if (u.uTime) gl.uniform1f(u.uTime, f.t);
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

// ---- bloom ----

const BRIGHT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
uniform float uThreshold, uKnee;
out vec4 o;
void main() {
  // Soft-knee bright pass.
  vec3 c = texture(uTex, vUV).rgb;
  float br = max(c.r, max(c.g, c.b));
  float soft = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  soft = soft * soft / (4.0 * uKnee + 1e-4);
  float w = max(soft, br - uThreshold) / max(br, 1e-4);
  o = vec4(c * w, 1.0);
}`;

// 9-tap Gaussian using linear filtering (5 fetches).
const BLUR_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
uniform vec2 uDir;
out vec4 o;
void main() {
  vec3 c = texture(uTex, vUV).rgb * 0.2270270270;
  c += (texture(uTex, vUV + uDir * 1.3846153846).rgb + texture(uTex, vUV - uDir * 1.3846153846).rgb) * 0.3162162162;
  c += (texture(uTex, vUV + uDir * 3.2307692308).rgb + texture(uTex, vUV - uDir * 3.2307692308).rgb) * 0.0702702703;
  o = vec4(c, 1.0);
}`;

const COPY_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
out vec4 o;
void main() { o = vec4(texture(uTex, vUV).rgb, 1.0); }`;

// Glow levels, added onto the frame. Alpha rises with the glow's brightest
// channel so the premultiplied result stays valid (color ≤ alpha) and the
// glow composites over the static layer as light.
const COMPOSITE_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uB0, uB1, uB2;
uniform vec3 uWeights;
uniform float uStrength;
out vec4 o;
void main() {
  vec3 b = (texture(uB0, vUV).rgb * uWeights.x + texture(uB1, vUV).rgb * uWeights.y
          + texture(uB2, vUV).rgb * uWeights.z) * uStrength;
  o = vec4(b, max(b.r, max(b.g, b.b)));
}`;

/**
 * Opt-in bloom. The style draws its scene normally (full resolution, the
 * default framebuffer's MSAA), then redraws it through capture() into a
 * 1/SCALE-resolution buffer that only feeds the glow — 1/16 of the pixels,
 * no multisampling, so the glow costs little even full screen:
 *
 *   drawScene(f);                        // → default framebuffer
 *   bloom.capture(f, drawScene)          // same calls, small target; f.bloomScale = SCALE,
 *                                        //   f.dpr divided by SCALE so point sizes stay put
 *   bloom.apply(opts)                    // bright-pass, blur 3 levels, add onto the frame
 *
 * opts: { threshold, knee, strength, weights: [w0, w1, w2] } (level 0 = 1/SCALE resolution).
 * Screen-space strips drawn in the capture should be at least f.bloomScale px wide.
 */
export function makeBloom(gl) {
  const SCALE = 4;
  const bright = program(gl, FULL_VS, BRIGHT_FS);
  const blur = program(gl, FULL_VS, BLUR_FS);
  const copy = program(gl, FULL_VS, COPY_FS);
  const comp = program(gl, FULL_VS, COMPOSITE_FS);
  const emptyVao = gl.createVertexArray();
  let W = 0, H = 0, SW = 0, SH = 0;
  let sceneFbo = null, sceneTex = null, sceneDepth = null;
  const lv = [];   // { w, h, texA, fboA, texB, fboB }

  function tex(w, h) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  function fboFor(t) {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    return fb;
  }
  function freeTargets() {
    if (sceneFbo) gl.deleteFramebuffer(sceneFbo);
    if (sceneTex) gl.deleteTexture(sceneTex);
    if (sceneDepth) gl.deleteRenderbuffer(sceneDepth);
    sceneFbo = sceneTex = sceneDepth = null;
    for (const l of lv) {
      gl.deleteTexture(l.texA); gl.deleteTexture(l.texB);
      gl.deleteFramebuffer(l.fboA); gl.deleteFramebuffer(l.fboB);
    }
    lv.length = 0;
    W = H = 0;
  }
  function resize(w, h) {
    freeTargets();
    W = w; H = h;
    SW = Math.max(1, Math.round(w / SCALE)); SH = Math.max(1, Math.round(h / SCALE));
    sceneTex = tex(SW, SH);
    sceneFbo = fboFor(sceneTex);
    sceneDepth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, sceneDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, SW, SH);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, sceneDepth);
    let lw = SW, lh = SH;
    for (let i = 0; i < 3; i++) {
      if (i > 0) { lw = Math.max(1, lw >> 1); lh = Math.max(1, lh >> 1); }
      const texA = tex(lw, lh), texB = tex(lw, lh);
      lv.push({ w: lw, h: lh, texA, fboA: fboFor(texA), texB, fboB: fboFor(texB) });
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function pass(p, fbo, w, h) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, w, h);
    gl.useProgram(p.prog);
    if (p.u.uRes) gl.uniform2f(p.u.uRes, w, h);
  }
  function bindTex(unit, t, loc) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.uniform1i(loc, unit);
  }
  function blurLevel(l, radius) {
    pass(blur, l.fboB, l.w, l.h);
    bindTex(0, l.texA, blur.u.uTex);
    gl.uniform2f(blur.u.uDir, radius / l.w, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    pass(blur, l.fboA, l.w, l.h);
    bindTex(0, l.texB, blur.u.uTex);
    gl.uniform2f(blur.u.uDir, 0, radius / l.h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  return {
    capture(f, draw) {
      if (f.W !== W || f.H !== H) resize(f.W, f.H);
      gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFbo);
      gl.viewport(0, 0, SW, SH);
      gl.clearColor(0, 0, 0, 0);
      gl.clearDepth(1);
      gl.depthMask(true);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const dpr = f.dpr;
      f.dpr = dpr / SCALE;
      f.bloomScale = SCALE;
      draw(f);
      f.dpr = dpr;
      f.bloomScale = 1;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, W, H);
    },
    apply(opts) {
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.bindVertexArray(emptyVao);
      pass(bright, lv[0].fboA, lv[0].w, lv[0].h);
      bindTex(0, sceneTex, bright.u.uTex);
      gl.uniform1f(bright.u.uThreshold, opts.threshold);
      gl.uniform1f(bright.u.uKnee, opts.knee ?? 0.2);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      blurLevel(lv[0], 1);
      for (let i = 1; i < 3; i++) {
        pass(copy, lv[i].fboA, lv[i].w, lv[i].h);
        bindTex(0, lv[i - 1].texA, copy.u.uTex);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        blurLevel(lv[i], 1.25);
      }
      pass(comp, null, W, H);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      bindTex(0, lv[0].texA, comp.u.uB0);
      bindTex(1, lv[1].texA, comp.u.uB1);
      bindTex(2, lv[2].texA, comp.u.uB2);
      const w = opts.weights;
      gl.uniform3f(comp.u.uWeights, w[0], w[1], w[2]);
      gl.uniform1f(comp.u.uStrength, opts.strength);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
      gl.activeTexture(gl.TEXTURE0);
    },
    dispose() {
      freeTargets();
      for (const p of [bright, blur, copy, comp]) gl.deleteProgram(p.prog);
      gl.deleteVertexArray(emptyVao);
    },
  };
}
