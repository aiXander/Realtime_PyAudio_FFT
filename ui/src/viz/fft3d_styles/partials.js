// partials — peak tracking: a dim, colormapped surface with every spectral
// peak (local maximum across frequency) traced as a thin bright line, so
// harmonics, formants and tonal components show up as tracks running back
// through time and you can read pitch glides, vibrato and onsets off them.
//
// Detection is free (per fragment, no extra pass): the mesh's frequency
// slope is interpolated across each triangle, so a peak is where that slope
// crosses zero going down; the line is drawn at that zero crossing with a
// constant pixel width. Each peak's width is estimated from height / curvature
// (σ ≈ √(h / |h''|)), so narrow tonal peaks light up fully and broad humps
// or small ripples riding a high floor stay dark.
// One opaque draw + the front-edge strips; no post-processing.
// Palette: magnitude colormap from low → mid → high (lightness-ordered).

import { theme } from "../../colors.js";
import {
  glf, SURF_VS, FLAT_VS, PAL_UNIFORMS, RAMP_FN, LINE_FN, LOG_PAPER_FN, program, setCommonUniforms,
  setLogUniforms, makeStrips, fillCurtain, fillRim, disposeAll,
} from "./common.js";

// Look. All tunable by eye.
const LINE_PX = 0.6;           // peak-line half-width, CSS px
const PEAK_OCT = [0.12, 0.45]; // estimated peak width (σ, octaves): full brightness below, gone above
const PEAK_MIN_H = [0.03, 0.12]; // ignore peaks below this height (fade-in range)
const BODY = 0.4;              // surface brightness under the lines
const GRID_ALPHA = [0.16, 0.05]; // log paper: decades, subdivisions
const FOG_START = 0.72;

// The mesh slope attribute is WORLD_HEIGHT · dh/dx over a surface 2 wide
// (fft_3d.js: WORLD_HEIGHT = 0.5): sx = 0.25 · dh/du.
const SLOPE_PER_DHDU = 0.25;

const SURF_FS = `#version 300 es
precision highp float;
in float vU, vH, vZ;
in vec3 vN;
uniform vec3 uBg;
uniform float uDpr;
out vec4 o;
${PAL_UNIFORMS}
${RAMP_FN}
${LINE_FN}
${LOG_PAPER_FN}
void main() {
  float h = clamp(vH, 0.0, 1.0);
  float fog = smoothstep(${glf(FOG_START)}, 1.0, vZ);
  // Body: dim colormap with soft light, so the energy stays readable under the tracks.
  float lit = dot(normalize(vN), normalize(vec3(-0.6, 1.0, 0.4)));
  vec3 col = ramp(h) * ${glf(BODY)} * (0.75 + 0.35 * clamp(lit, 0.0, 1.0));
  vec2 grid = logPaper(vU, 0.5);
  col = mix(col, vec3(0.8), max(grid.x * ${glf(GRID_ALPHA[0])}, grid.y * ${glf(GRID_ALPHA[1])}) * (1.0 - fog));
  // Peaks: the frequency slope crosses zero going down (u grows with screen x).
  float sx = -vN.x;
  vec2 gs = vec2(dFdx(sx), dFdy(sx));
  float curv = -gs.x / max(dFdx(vU), 1e-7) / ${glf(SLOPE_PER_DHDU)};   // −h''(u)
  float d = abs(sx) / max(length(gs), 1e-7);                             // px to the crossing
  float line = 1.0 - smoothstep(${glf(LINE_PX)} * uDpr - 0.25, ${glf(LINE_PX)} * uDpr + 0.75, d);
  float sigmaOct = sqrt(h / max(curv, 1e-6)) * uLogSpan * 3.321928;
  float strength = (curv > 0.0 ? 1.0 : 0.0)
                 * (1.0 - smoothstep(${glf(PEAK_OCT[0])}, ${glf(PEAK_OCT[1])}, sigmaOct))
                 * smoothstep(${glf(PEAK_MIN_H[0])}, ${glf(PEAK_MIN_H[1])}, h);
  vec3 lineCol = mix(ramp(0.4 + 0.6 * h), vec3(1.0), 0.15);
  col = mix(col, lineCol, line * strength * (1.0 - fog));
  col = mix(col, uBg, fog * 0.85);
  o = vec4(col, 1.0);
}`;

const FLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform int uMode;
uniform float uCore;
uniform vec3 uBg;
out vec4 o;
${PAL_UNIFORMS}
${RAMP_FN}
void main() {
  if (uMode == 0) {
    float v = clamp(vUV.y, 0.0, 1.0);
    o = vec4(mix(uBg, ramp(v) * 0.3, v), 1.0);
  } else {
    float a = 1.0 - smoothstep(uCore, 1.0, abs(vUV.y));
    o = vec4(vec3(0.9) * a, a);
  }
}`;

export default {
  name: "partials",
  label: "Partials",
  description: "Every spectral peak traced as a line through time: harmonics, formants, glides",
  camera: { backScale: 0.5, ampFront: 0.26 },
  smoothing: { freqOct: 0.025, timePasses: 1 },
  background: () => theme.bg,
  text: { axis: "#8a9098", dim: "#5a6068" },

  create(gl) {
    const surf = program(gl, SURF_VS, SURF_FS);
    const flat = program(gl, FLAT_VS, FLAT_FS);
    const strips = makeStrips(gl, 4 * 384);
    return {
      render(f) {
        gl.disable(gl.BLEND);
        if (f.R >= 2) {
          gl.enable(gl.DEPTH_TEST);
          gl.depthFunc(gl.LEQUAL);
          gl.useProgram(surf.prog);
          setCommonUniforms(gl, surf.u, f);
          setLogUniforms(gl, surf.u, f);
          gl.bindVertexArray(f.surfVao);
          gl.drawElements(gl.TRIANGLES, f.triCount, gl.UNSIGNED_INT, 0);
          gl.disable(gl.DEPTH_TEST);
        }
        const rimHalf = 0.6 * f.dpr + 1;
        let o = fillCurtain(f, strips.data, 0);
        o = fillRim(f, strips.data, o, rimHalf);
        gl.useProgram(flat.prog);
        setCommonUniforms(gl, flat.u, f);
        gl.uniform1f(flat.u.uCore, Math.max(0, 1 - 1 / rimHalf));
        strips.upload(o);
        gl.uniform1i(flat.u.uMode, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 2 * f.mc);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.uniform1i(flat.u.uMode, 1);
        gl.drawArrays(gl.TRIANGLE_STRIP, 2 * f.mc, 2 * f.mc);
      },
      dispose() { disposeAll(gl, surf, flat, strips); },
    };
  },
};
