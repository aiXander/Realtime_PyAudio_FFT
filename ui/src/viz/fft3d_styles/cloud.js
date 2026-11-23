// cloud — an opaque surface in the palette's band colors (by frequency, or,
// with the Color picker, by which bands are sounding in each moment) under a
// dust of tiny points. Near ridges hide what is behind them; older rows
// recede into fog. Follows the palette.
//
// Two dials (the camera panel's Light sliders, `ui.fft3d.glow` / `.bloom`):
//   glow   how the surface gets its light. 0 = matte terrain lit from the
//          front left, shade in the valleys. 1 = a near-black body that emits
//          by level: the sound is the light.
//   bloom  the bright parts (peaks, notes, beginnings, the front rim) drawn
//          again at quarter resolution, blurred and added on top. 0 = no
//          bloom pass at all.
//
// The light says what kind of sound it is:
//   - a note (tonal: narrow in frequency, steady in time) is a white thread
//     along the crest of its ridge, running into the depth for as long as it
//     sounds, on a warm glow. Threads outlast the fog, so a melody stays
//     readable further back than the drums around it.
//   - where a sound begins, the crest is lit in its band color: a streak
//     across frequency for a hit, a dot at the head of a note.
//   - the dust sparkles on those two and is only a faint grain elsewhere.
// "Now" is the front edge: the newest rows are lit brighter, and the rim
// turns white where a note is sounding.
//
// The surface is drawn back to front and writes depth, so points hidden
// behind nearer swells are culled. A depth pre-pass (DEPTH_FS) comes first, so
// hidden ridges aren't shaded, and the bloom pass draws a thinned grid
// (makeCoarseGrid). Why and how much: docs/reference/fft_3d_styles.md.

import { theme, PALETTES, hexRgb } from "../../colors.js";
import {
  glf, PROJ_FN, HASH_FN, SURF_VS, SPEC_FN, SPEC_FS, GRID_FN, FLAT_VS, MOMENT_FN, bandFn, program, setCommonUniforms,
  makeStrips, fillCurtain, fillRim, disposeAll,
} from "./common.js";

// Look. All tunable by eye.
const FOG_START = 0.2;         // depth (0 front … 1 horizon) where things start sinking into fog
const FOG_MAX = 0.8;           // share of the background the far rows take
const FADE_START = 0.65;       // depth where they start going transparent (gone at 1)
const BACKDROP = 0.1;          // light behind the horizon (static layer), alpha of the palette accent
const POINT_PX = 1.6;          // point size at the front, CSS px (squares)
const POINT_JITTER = 1.0;      // point position noise, fraction of the grid spacing (frequency and time)
const POINT_COPIES = 2;        // points per grid vertex, each jittered independently
const GRID = { cols: 384, rows: 200 };   // dust density (the surface under it shades per pixel)
const RIM_PX = 2.0;            // front-edge rim width, CSS px
const CURTAIN = 0.85;          // front curtain brightness
// "Now" is the front edge; everything behind it is memory. The newest rows
// are lit brighter and the light dies off with age, so the eye rests on what
// is sounding, and a hit flashes at the front before it recedes.
const LIVE_GAIN = 0.75;        // extra brightness at the front edge (× the normal one)
const LIVE_TAU_S = 0.4;        // how fast that light dies with age, seconds
// Notes. A melody is quiet next to the drums, so height alone hides it.
const THREAD_PX = 1.1;         // half width of a note's thread, CSS px at the front
const THREAD_GAIN = 1.15;      // its brightness
const THREAD_FOG = 0.45;       // share of the fog a thread takes (the rest of the picture takes all of it)
// Dust: a faint grain everywhere, bright only on notes and beginnings.
const DUST_BASE = 0.25;        // grain brightness (× height)
const DUST_TONE = 2.2;         // sparkle on a tonal ridge
const DUST_ONSET = 1.6;        // sparkle where a sound begins
// Bloom (the bloom dial is its gain).
const BLOOM_DIV = 4;                 // the bloom targets are 1/4 of the canvas each way
const BLOOM_STEPS = [1.0, 2.5];      // blur passes (H + V each), tap spacing in texels
const BLOOM_UNIT = 2;                // texture unit (0 / 1 are the core's)
const BLOOM_GRID = 3;                // the bloom pass draws every Nth grid row and column (see makeCoarseGrid)

const FOG_FN = `
float live(float z) { return 1.0 + ${glf(LIVE_GAIN)} * exp(-z * uDepthS / ${glf(LIVE_TAU_S)}); }
float fog(float z) { return smoothstep(${glf(FOG_START)}, 1.0, z) * ${glf(FOG_MAX)}; }
float fade(float z) { return 1.0 - smoothstep(${glf(FADE_START)}, 1.0, z); }`;

// Surface: lit terrain and emitting body, mixed by the glow dial, plus the
// light for notes (thread + glow) and beginnings. In the bloom pass it writes
// only what should bloom, opaque, so near ridges hide far light.
const SURF_FS = `#version 300 es
precision highp float;
in float vU, vZ, vT, vS;
out vec4 o;
uniform vec3 uBg;
uniform float uDpr, uGlow, uBloomPass;
${SPEC_FS}
${bandFn(0)}
${MOMENT_FN}
${FOG_FN}
const vec3 VIEW = vec3(0.0, 0.4472, -0.8944);   // surface → camera: in front, a little above
void main() {
  vec3 base = momentMix(band(vU), vT, fwidth(vT));
  float h; vec3 n;
  specSurface(vU, vT, h, n);
  n = normalize(n);
  float up = smoothstep(0.02, 0.2, h);             // nothing lights the floor
  float tone = specTone(vU, vT) * up;
  float thread = specThread(vU, vT, ${glf(THREAD_PX)} * uDpr * sqrt(vS));
  float onset = specOnset(vU, vT) * up;
  vec3 hot = mix(base, vec3(1.0), 0.75);
  float lv = live(vZ), fg = fog(vZ);
  float key = smoothstep(-0.45, 0.9, dot(n, normalize(vec3(-0.62, 0.5, -0.6))));
  // Lit: shade is deeper and more saturated, not grey; valleys sit in shade.
  vec3 lit = mix(base * base * 0.2, base, key) * (0.3 + 0.7 * smoothstep(0.0, 0.5, h)) + hot * pow(h, 4.0) * 0.3;
  // Emitting: brightness is level; a little form from the key light and a rim on the crests.
  float rim = pow(1.0 - max(dot(n, VIEW), 0.0), 2.0) * up;
  vec3 emit = base * ((0.025 + 0.95 * pow(h, 1.5)) * (0.45 + 0.55 * key) + 0.22 * rim);
  vec3 on = mix(base, vec3(1.0), 0.25) * onset * lv * mix(0.6, 0.9, uGlow);
  vec3 col = mix(lit, emit, uGlow) * lv + on + hot * tone * mix(0.25 * key, 0.3, uGlow);
  col = mix(col, uBg, fg);
  vec3 th = hot * thread * ${glf(THREAD_GAIN)} * (1.0 - ${glf(THREAD_FOG)} * fg);
  col += th;
  float a = smoothstep(0.0, 0.06, h);
  if (uBloomPass > 0.5) {
    col = (base * pow(h, 3.0) * 0.45 + on * 0.6 + hot * tone * 0.4) * (1.0 - fg) + th;   // not × lv: the front would burn out
    a = 1.0;
  }
  a = min(1.0, a + thread) * fade(vZ);
  o = vec4(col * a, a);
}`;

// Depth pre-pass. The surface blends, so the GPU can't skip hidden fragments
// on its own and every ridge behind a nearer one ran the full SURF_FS (~25
// texture reads) only to be painted over. Laying down depth first lets the
// shading pass run once per pixel. Rows that fade out are left out of it:
// there a far ridge shows through a nearer one, so they keep the plain
// back-to-front blend. (Near the floor the surface is also see-through, but
// nothing lies behind it there: farther rows sit higher on screen.)
const DEPTH_FS = `#version 300 es
precision highp float;
in float vZ;
out vec4 o;
void main() {
  if (vZ > ${glf(FADE_START)}) discard;
  o = vec4(0.0);
}`;

// Points ride the grid vertices, jittered by a stable per-point hash keyed
// on (column, grid row, copy) so the noise travels with the data; each reads
// its own height, tonalness and onset from the texture at its jittered spot.
const POINT_VS = `#version 300 es
${PROJ_FN}
${HASH_FN}
${SPEC_FN}
${GRID_FN}
${bandFn(0)}
${MOMENT_FN}
out float vH, vZ, vRand, vLit;
out vec3 vCol;
void main() {
  float u, row, z, key, pool;
  gridVertex(gl_VertexID, u, row, z, key, pool);
  int col = gl_VertexID % uCols;
  uint k = hash(uint(col) * 0x9E3779B1u ^ hash(uint(key) * 4u + uint(gl_InstanceID) + 0x632BE5ABu));
  float r1 = float(k & 0xFFFFu) / 65535.0, r2 = float(k >> 16) / 65535.0;
  u += (r1 - 0.5) * ${glf(POINT_JITTER)} / float(uCols - 1);
  float dz = (r2 - 0.5) * ${glf(POINT_JITTER)} * uRowDz;
  float zz = max(0.0, z + dz);
  float rr = row - dz * uNView;
  float h = specH(u, rr);
  vRand = float(hash(k) & 0xFFFFu) / 65535.0;
  float s;
  vec2 px = toPx(u, h, zz, s);
  gl_Position = clipPos(px, s, zz);
  gl_Position.z -= 0.004 * gl_Position.w;   // sit just in front of the surface they ride on
  gl_PointSize = max(1.0, ${glf(POINT_PX)} * uDpr * sqrt(s));
  vCol = momentMix(band(u), rr, 1.0);
  vH = h; vZ = zz;
  vLit = ${glf(DUST_TONE)} * specTone(u, rr) + ${glf(DUST_ONSET)} * specOnset(u, rr);
}`;

// Points: additive dust, whiter where lit, per-point random brightness.
const POINT_FS = `#version 300 es
precision highp float;
in float vH, vZ, vRand, vLit;
in vec3 vCol;
out vec4 o;
uniform float uDepthS;
${FOG_FN}
void main() {
  float h = clamp(vH, 0.0, 1.0);
  float lit = vLit * smoothstep(0.02, 0.2, h);
  vec3 col = mix(vCol, vec3(1.0), min(0.6, 0.3 * lit));
  float a = min(1.0, (${glf(DUST_BASE)} * h * live(vZ) + lit) * (0.25 + 0.75 * vRand)) * fade(vZ) * (1.0 - fog(vZ));
  o = vec4(col * a, a);
}`;

// Front edge: curtain down to the baseline + rim, both whiter where a note is sounding.
const FLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform int uMode;       // 0 = curtain (v = height, 0 at baseline), 1 = rim (v = side, -1..1)
uniform float uCore;     // rim: solid fraction of the half width (rest is AA falloff)
uniform float uRowA, uFrontOff;
out vec4 o;
${SPEC_FN}
${bandFn(0)}
${MOMENT_FN}
void main() {
  vec3 base = momentMix(band(vUV.x), uRowA + uFrontOff, 1.0);
  float tone = specTone(vUV.x, uRowA + uFrontOff);
  if (uMode == 0) {
    float v = clamp(vUV.y, 0.0, 1.0);
    float a = 0.5 + 0.4 * v;
    o = vec4(mix(base, vec3(1.0), 0.35 * tone) * (0.08 + (0.6 + 0.5 * tone) * v) * ${glf(CURTAIN)} * a, a);
  } else {
    float a = 1.0 - smoothstep(uCore, 1.0, abs(vUV.y));
    o = vec4(mix(base, vec3(1.0), 0.25 + 0.65 * tone) * (0.85 + 0.15 * tone) * a, a);
  }
}`;

// ---- bloom ----

const TRI_VS = `#version 300 es
out vec2 vUV;
void main() {
  vUV = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}`;
const BLUR_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
uniform vec2 uDir;       // one tap step, in uv
out vec4 o;
void main() {
  vec4 c = texture(uTex, vUV) * 0.227027;
  c += (texture(uTex, vUV + uDir * 1.0) + texture(uTex, vUV - uDir * 1.0)) * 0.1945946;
  c += (texture(uTex, vUV + uDir * 2.0) + texture(uTex, vUV - uDir * 2.0)) * 0.1216216;
  c += (texture(uTex, vUV + uDir * 3.0) + texture(uTex, vUV - uDir * 3.0)) * 0.054054;
  c += (texture(uTex, vUV + uDir * 4.0) + texture(uTex, vUV - uDir * 4.0)) * 0.016216;
  o = c;
}`;
const ADD_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
uniform float uGain;
out vec4 o;
void main() {
  vec3 c = texture(uTex, vUV).rgb * uGain;
  o = vec4(c, min(1.0, max(c.r, max(c.g, c.b))));
}`;

/**
 * The core's grid thinned to every `BLOOM_GRID`th row and column, for the
 * bloom pass. At quarter resolution the full grid's triangles are smaller
 * than a pixel, and a GPU shades at least a 2×2 pixel quad per triangle, so
 * most of that pass's cost was shading the same pixels again. Only the
 * blurred silhouette gets coarser: SURF_FS still reads color and height per
 * pixel from the texture. Same vertex ids as the core's grid (gridVertex),
 * so it needs only its own index buffers.
 *
 * The kept rows must stay pinned to the same data rows as the history
 * scrolls, like the core's grid rows: they are picked by row key (key ≡ 0
 * mod BLOOM_GRID), not by index. Picking by index (every Nth from the back)
 * moved the coarse surface onto different data rows at every grid step, and
 * the bloom flickered white across the whole picture. The back row, the
 * newest anchored row and the live front row are always kept. One index
 * buffer per phase (keyA mod BLOOM_GRID), rebuilt only when the grid's size
 * changes (history depth or style), never per frame.
 */
function makeCoarseGrid(gl) {
  const sets = [];                                  // per phase: { vao, ibo, count }
  for (let i = 0; i < BLOOM_GRID; i++) {
    const vao = gl.createVertexArray(), ibo = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);   // captured by vao
    sets.push({ vao, ibo, count: -1 });
  }
  gl.bindVertexArray(null);
  let R = 0, mc = 0;

  function build(set, phase) {
    const body = R - 1;                             // anchored rows 0 … body−1, then the front row
    const rows = [0];
    for (let r = 1; r < body - 1; r++) if (r % BLOOM_GRID === phase) rows.push(r);
    if (body - 1 > 0) rows.push(body - 1);
    rows.push(R - 1);
    const cols = [];
    for (let c = 0; c < mc - 1; c += BLOOM_GRID) cols.push(c);
    cols.push(mc - 1);
    const idx = new Uint32Array((rows.length - 1) * (cols.length - 1) * 6);
    let o = 0;
    for (let i = 0; i < rows.length - 1; i++) {     // oldest row first (= back to front)
      for (let j = 0; j < cols.length - 1; j++) {
        const a = rows[i] * mc + cols[j], b = a - cols[j] + cols[j + 1];
        const c = rows[i + 1] * mc + cols[j], d = c - cols[j] + cols[j + 1];
        idx[o++] = a; idx[o++] = c; idx[o++] = b;
        idx[o++] = b; idx[o++] = c; idx[o++] = d;
      }
    }
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    set.count = o;
  }

  return {
    /** Bind the thinned grid for this frame; returns the index count to draw. */
    bind(f) {
      if (f.R !== R || f.mc !== mc) {
        R = f.R; mc = f.mc;
        for (const set of sets) set.count = -1;
      }
      // Row r's key is keyA − (bodyRows − 1 − r): key ≡ 0 ⇔ r ≡ bodyRows − 1 − keyA.
      const n = BLOOM_GRID;
      const phase = (((f.spec.bodyRows - 1 - f.spec.keyA) % n) + n) % n;
      const set = sets[phase];
      gl.bindVertexArray(set.vao);
      if (set.count < 0) build(set, phase);
      return set.count;
    },
    dispose() {
      for (const set of sets) { gl.deleteBuffer(set.ibo); gl.deleteVertexArray(set.vao); }
    },
  };
}

/** Two quarter-resolution targets: `begin` binds the first to draw the bright parts into,
 *  `end` blurs them (ping-pong) and adds the result onto the canvas. */
function makeBloom(gl) {
  const blur = program(gl, TRI_VS, BLUR_FS);
  const add = program(gl, TRI_VS, ADD_FS);
  const vao = gl.createVertexArray();
  const tex = [null, null], fbo = [null, null];
  let depth = null, w = 0, h = 0;

  function free() {
    for (let i = 0; i < 2; i++) { gl.deleteTexture(tex[i]); gl.deleteFramebuffer(fbo[i]); }
    gl.deleteRenderbuffer(depth);
  }
  function resize(W, H) {
    const nw = Math.max(1, Math.round(W / BLOOM_DIV)), nh = Math.max(1, Math.round(H / BLOOM_DIV));
    if (nw === w && nh === h) return;
    if (w) free();
    w = nw; h = nh;
    gl.activeTexture(gl.TEXTURE0 + BLOOM_UNIT);
    depth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
    for (let i = 0; i < 2; i++) {
      tex[i] = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex[i]);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      fbo[i] = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex[i], 0);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[0]);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    gl.activeTexture(gl.TEXTURE0);
  }

  return {
    begin(f) {
      resize(f.W, f.H);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[0]);
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 0);
      gl.depthMask(true);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    },
    end(f, gain) {
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.bindVertexArray(vao);
      gl.activeTexture(gl.TEXTURE0 + BLOOM_UNIT);
      gl.useProgram(blur.prog);
      gl.uniform1i(blur.u.uTex, BLOOM_UNIT);
      for (let i = 0; i < BLOOM_STEPS.length; i++) {
        const s = BLOOM_STEPS[i];
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[1]);
        gl.bindTexture(gl.TEXTURE_2D, tex[0]);
        gl.uniform2f(blur.u.uDir, s / w, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[0]);
        gl.bindTexture(gl.TEXTURE_2D, tex[1]);
        gl.uniform2f(blur.u.uDir, 0, s / h);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, f.W, f.H);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(add.prog);
      gl.uniform1i(add.u.uTex, BLOOM_UNIT);
      gl.uniform1f(add.u.uGain, gain);
      gl.bindTexture(gl.TEXTURE_2D, tex[0]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.activeTexture(gl.TEXTURE0);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    },
    dispose() {
      if (w) free();
      gl.deleteVertexArray(vao);
      disposeAll(gl, blur, add);
    },
  };
}

export default {
  name: "cloud",
  label: "Cloud",
  description: "Opaque terrain and dust in the palette's band colors, lit or glowing (Light dials); notes are white threads, beginnings light up",
  grid: GRID,
  background: () => theme.bg,
  text: { axis: "#7a8088", dim: "#5a6068" },

  // Light behind the horizon, for the far rows to stand against.
  drawStatic(g, geo) {
    const a = hexRgb(PALETTES[theme.name].accent).join(",");
    const x = geo.cx + geo.skewPx;
    const glow = g.createRadialGradient(x, geo.yBack, 0, x, geo.yBack, geo.plotW * 0.6);
    glow.addColorStop(0, `rgba(${a},${BACKDROP})`);
    glow.addColorStop(1, `rgba(${a},0)`);
    g.fillStyle = glow;
    g.fillRect(0, 0, geo.plotX * 2 + geo.plotW, geo.y0);
  },

  create(gl) {
    const surf = program(gl, SURF_VS, SURF_FS);
    const depth = program(gl, SURF_VS, DEPTH_FS);
    const points = program(gl, POINT_VS, POINT_FS);
    const flat = program(gl, FLAT_VS, FLAT_FS);
    const strips = makeStrips(gl, 4 * 2048);
    const bloom = makeBloom(gl);
    const coarse = makeCoarseGrid(gl);

    // Depth pre-pass (DEPTH_FS), then back to front, writing depth so points
    // and far light behind nearer swells are culled.
    function surface(f, bloomPass) {
      let count = f.triCount;
      if (bloomPass) count = coarse.bind(f);
      else gl.bindVertexArray(f.surfVao);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.depthMask(true);
      gl.colorMask(false, false, false, false);
      gl.useProgram(depth.prog);
      setCommonUniforms(gl, depth.u, f);
      gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, 0);
      gl.colorMask(true, true, true, true);
      gl.useProgram(surf.prog);
      setCommonUniforms(gl, surf.u, f);
      gl.uniform1f(surf.u.uBloomPass, bloomPass);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // premultiplied
      gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, 0);
    }
    // Front edge: curtain down to its baseline + rim.
    function frontEdge(f, curtain) {
      const rimHalf = (RIM_PX / 2) * f.dpr + 1;       // + 1 px of AA falloff
      let o = fillCurtain(f, strips.data, 0);
      o = fillRim(f, strips.data, o, rimHalf);
      gl.disable(gl.DEPTH_TEST);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(flat.prog);
      setCommonUniforms(gl, flat.u, f);
      gl.uniform1f(flat.u.uCore, Math.max(0, 1 - 1 / rimHalf));
      strips.upload(o);
      if (curtain) {
        gl.uniform1i(flat.u.uMode, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 2 * f.mc);
      }
      gl.uniform1i(flat.u.uMode, 1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 2 * f.mc, 2 * f.mc);
    }

    return {
      render(f) {
        gl.enable(gl.BLEND);
        if (f.R >= 2) {
          surface(f, 0);
          // Dust: additive points on every mesh vertex.
          gl.useProgram(points.prog);
          setCommonUniforms(gl, points.u, f);
          gl.blendFunc(gl.ONE, gl.ONE);
          gl.depthMask(false);
          gl.drawArraysInstanced(gl.POINTS, 0, f.R * f.mc, POINT_COPIES);
          gl.depthMask(true);
        }
        frontEdge(f, true);
        if (f.spec.bloom > 0 && f.R >= 2) {
          bloom.begin(f);
          surface(f, 1);
          frontEdge(f, false);
          bloom.end(f, f.spec.bloom);
        }
      },
      dispose() { disposeAll(gl, surf, depth, points, flat, strips, bloom, coarse); },
    };
  },
};
