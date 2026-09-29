// synthwave — 80s outrun. The spectrum is a neon wireframe mountain range
// (near-black faces hide the lines behind them) standing on an endless grid
// plane that rolls toward a striped sun on a purple horizon. Grid lines are
// anchored to arrival time, so the plane and the mountains scroll together.
// Line color runs magenta (low) → cyan (high). Bloom on everything, faint
// scanlines on top. Palette: grid low → mid by height, sun high → low, haze low
// (the Neon palette gives the classic magenta / cyan / yellow outrun).
//
// Reactivity: a low-band onset pulses the sun and flashes the grid; while a
// tempo is locked the scanlines roll one line per beat.

import {
  glf, rng, PAL_UNIFORMS, palBg, SURF_VS, FULL_VS, FLAT_VS, PROJ_FN, program, setCommonUniforms,
  makeStrips, fillCurtain, fillRim, makeBloom, disposeAll,
} from "./common.js";

// Look. All tunable by eye.
const GRID_U = 36;             // frequency grid lines across the mesh width
const GRID_T = 18;             // time grid lines across the depth
const LINE_PX = 0.85;          // grid line half-width, device px
const SUN_PULSE = 0.05, FLASH = 0.8, ONSET_DECAY_S = 0.22;
const RIM_PX = 2.6;
const SCANLINE_PX = 3;         // scanline period, CSS px
const SUN_GAIN = 0.8;
const BLOOM = { threshold: 0.3, knee: 0.3, strength: 1.05, weights: [0.6, 0.8, 1.0] };


const LINE_FN = `
float lineAt(float q, float halfPx) {
  float w = fwidth(q);
  float d = abs(fract(q + 0.5) - 0.5) / max(w, 1e-5);
  return (1.0 - smoothstep(halfPx - 0.6, halfPx + 0.6, d)) * (1.0 - smoothstep(0.08, 0.3, w));
}
uniform float uPhase, uFlash;
uniform vec3 uBg;               // faces / ground: the palette background, tinted
${PAL_UNIFORMS}
float grid(float u, float z) {
  return max(lineAt(u * ${glf(GRID_U)}, ${glf(LINE_PX)}), lineAt(uPhase - z * ${glf(GRID_T)}, ${glf(LINE_PX)}));
}`;

// Backdrop: the grid plane below the horizon, the sun above it.
const BACK_FS = `#version 300 es
precision highp float;
in vec2 vPx;
${PROJ_FN}
uniform float uSunR, uTime, uSunShift;
out vec4 o;
${LINE_FN}
void main() {
  vec2 p = vPx;
  if (p.y > uY0) { o = vec4(0.0); return; }
  if (p.y >= uYBack) {
    // Invert the projection: screen y → perspective scale → depth; x → u (may leave [0, 1]).
    float s = 1.0 - (uY0 - p.y) * (1.0 - uBackScale) / (uY0 - uYBack);
    float z = (1.0 / s - 1.0) / uK;
    float u = (p.x - uCx) / (2.0 * uHalfW * s) + 0.5;
    float fog = smoothstep(0.35, 1.0, z);
    vec3 c = mix(uBg, uC0 * 0.3, pow(fog, 3.0));
    c += uC0 * grid(u, z) * (1.0 - 0.85 * fog) * (0.7 + uFlash);
    o = vec4(c, 1.0);
    return;
  }
  // Sun: a disc sitting on the horizon, cut by horizontal stripes that widen toward the bottom.
  vec2 c0 = vec2(uCx, uYBack + uSunShift);
  float d = length(p - c0) / uSunR;
  float t = (c0.y - p.y) / uSunR;               // 0 at the horizon, 1 at the top
  vec3 col = mix(uC0, uC2, smoothstep(0.0, 0.9, t));
  float st = t * 11.0 + uTime * 0.35;
  float gap = mix(0.6, 0.0, smoothstep(0.05, 0.62, t));
  float fw = fwidth(st);
  float cut = smoothstep(gap - fw, gap + fw, fract(st));
  float disc = (1.0 - smoothstep(1.0 - 1.5 / uSunR, 1.0, d)) * cut;
  float glow = exp(-max(d - 1.0, 0.0) * 5.0) * 0.18 * (1.0 - disc);
  vec3 outc = col * disc * ${glf(SUN_GAIN)} + uC0 * glow;
  float a = max(disc, glow);
  o = vec4(outc, a);
}`;

// Mountains: opaque dark faces carrying the height-colored grid.
const SURF_FS = `#version 300 es
precision highp float;
in float vU, vH, vZ;
out vec4 o;
${LINE_FN}
void main() {
  float h = clamp(vH, 0.0, 1.0);
  float fog = smoothstep(0.35, 1.0, vZ);
  vec3 c = mix(uBg, uC0 * 0.3, pow(fog, 3.0));
  vec3 lc = mix(uC0, uC1, smoothstep(0.05, 0.7, h));
  c += lc * grid(vU, vZ) * (1.0 - 0.85 * fog) * (0.7 + uFlash + 0.3 * h);
  o = vec4(c, 1.0);
}`;

const FLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform int uMode;
uniform float uCore, uFlash;
uniform vec3 uBg;
${PAL_UNIFORMS}
out vec4 o;
void main() {
  if (uMode == 0) { o = vec4(uBg, 1.0); return; }
  float a = 1.0 - smoothstep(uCore, 1.0, abs(vUV.y));
  vec3 c = mix(uC1, vec3(1.0), 0.35) * (1.0 + 0.5 * uFlash);
  o = vec4(c * a, a);
}`;

// Scanlines: multiply the frame down on every Nth device row.
const SCAN_FS = `#version 300 es
precision highp float;
in vec2 vPx;
uniform float uPeriod, uOffset;
out vec4 o;
void main() {
  float k = 1.0 - 0.22 * step(fract((vPx.y + uOffset) / uPeriod), 0.34);
  o = vec4(k);
}`;

export default {
  name: "synthwave",
  label: "Synthwave",
  description: "Neon wireframe mountains rolling toward a striped sun",
  camera: { backScale: 0.22, ampFront: 0.42, horizon: 0.3 },
  background: () => palBg("low", 0.05),
  text: { axis: "#b58ad8", dim: "#9a74bd", horizonAbove: true },

  drawStatic(g, geo, W, H, dpr) {
    const sky = g.createLinearGradient(0, 0, 0, geo.yBack);
    sky.addColorStop(0, palBg("low", 0.1));
    sky.addColorStop(0.65, palBg("low", 0.22));
    sky.addColorStop(1, palBg("low", 0.4));
    g.fillStyle = sky;
    g.fillRect(0, 0, W, geo.yBack + 1);
    const rnd = rng(3);
    for (let i = 0; i < 140; i++) {
      const x = rnd() * W, y = rnd() * geo.yBack * 0.8, b = rnd();
      g.fillStyle = `rgba(255,220,255,${(0.1 + 0.5 * b * b).toFixed(3)})`;
      g.fillRect(x, y, dpr, dpr);
    }
  },

  create(gl) {
    const back = program(gl, FULL_VS, BACK_FS);
    const surf = program(gl, SURF_VS, SURF_FS);
    const flat = program(gl, FLAT_VS, FLAT_FS);
    const scan = program(gl, FULL_VS, SCAN_FS);
    const strips = makeStrips(gl, 4 * 384);
    const bloom = makeBloom(gl);
    const emptyVao = gl.createVertexArray();
    let flash = 0, phase = 0, sunR = 1;

    function draw(f) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(emptyVao);
      gl.useProgram(back.prog);
      setCommonUniforms(gl, back.u, f);
      gl.uniform1f(back.u.uPhase, phase);
      gl.uniform1f(back.u.uFlash, flash);
      gl.uniform1f(back.u.uSunR, sunR);
      gl.uniform1f(back.u.uSunShift, sunR * 0.08);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (f.R >= 2) {
        gl.disable(gl.BLEND);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.useProgram(surf.prog);
        setCommonUniforms(gl, surf.u, f);
        gl.uniform1f(surf.u.uPhase, phase);
        gl.uniform1f(surf.u.uFlash, flash);
        gl.bindVertexArray(f.surfVao);
        gl.drawElements(gl.TRIANGLES, f.triCount, gl.UNSIGNED_INT, 0);
        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
      }
      // Front face + glowing rim (at least one pixel of the small bloom target wide).
      const aa = f.bloomScale;
      const rimHalf = (RIM_PX / 2) * f.dpr * f.bloomScale + aa;
      let o = fillCurtain(f, strips.data, 0);
      o = fillRim(f, strips.data, o, rimHalf);
      gl.useProgram(flat.prog);
      setCommonUniforms(gl, flat.u, f);
      gl.uniform1f(flat.u.uCore, Math.max(0, 1 - aa / rimHalf));
      gl.uniform1f(flat.u.uFlash, flash);
      strips.upload(o);
      gl.uniform1i(flat.u.uMode, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 2 * f.mc);
      gl.uniform1i(flat.u.uMode, 1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 2 * f.mc, 2 * f.mc);
    }

    return {
      render(f) {
        const age = f.audio.onsetAge[0];
        const pulse = age >= 0 && age < 4 * ONSET_DECAY_S ? Math.exp(-age / ONSET_DECAY_S) : 0;
        flash = FLASH * pulse;
        // Time-grid phase in lines, from the absolute clock (so lines ride the data).
        phase = (f.t * GRID_T / f.depthS) % 1;
        const geo = f.geo;
        sunR = Math.min(0.2 * geo.plotW, 0.9 * (geo.yBack - geo.padT)) * (1 + SUN_PULSE * pulse);

        draw(f);
        bloom.capture(f, draw);
        bloom.apply(BLOOM);

        // Scanlines over the frame; roll one line per beat when a tempo is locked.
        const period = SCANLINE_PX * f.dpr;
        const roll = f.audio.bpm > 0 ? ((f.t * f.audio.bpm) / 60) % 1 : 0;
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.ZERO, gl.SRC_COLOR, gl.ZERO, gl.SRC_ALPHA);
        gl.bindVertexArray(emptyVao);
        gl.useProgram(scan.prog);
        gl.uniform2f(scan.u.uRes, f.W, f.H);
        gl.uniform1f(scan.u.uPeriod, period);
        gl.uniform1f(scan.u.uOffset, roll * period);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      },
      dispose() {
        disposeAll(gl, back, surf, flat, scan, strips, bloom);
        gl.deleteVertexArray(emptyVao);
      },
    };
  },
};
