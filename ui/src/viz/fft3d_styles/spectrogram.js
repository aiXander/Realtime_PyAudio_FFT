// spectrogram — a spectrogram sheet in shallow perspective: seen from high
// up with little foreshortening, so old rows stay almost as large as new
// ones and the whole history reads like a 2D spectrogram with relief.
// Magnitude is a lightness-ordered colormap (the main carrier of detail),
// lighting is kept gentle so it never fights the colors, and the graticule
// is log graph paper across frequency plus 1 s parallels (labelled).
// Colour and shading are read per pixel from the history texture (max over
// the rows / bins under the pixel), so the sheet is at data resolution.
// One opaque draw + the front-edge strips; no post-processing.
// Palette: magnitude colormap from low → mid → high (lightness-ordered).

import { theme } from "../../colors.js";
import {
  glf, SURF_VS, SPEC_FS, FLAT_VS, PAL_UNIFORMS, RAMP_FN, LINE_FN, LOG_PAPER_FN, program, setCommonUniforms,
  setLogUniforms, makeStrips, fillCurtain, fillRim, disposeAll, drawSecondLabels,
} from "./common.js";

// Look. All tunable by eye.
const SHADE = 0.28;            // hill-shading depth (0 = flat color, exact colormap)
// Colormap: level = h^GAMMA. h is already dB-linear over the display range
// (the loudest bin ≈ 1), so the map needs little help; GAMMA < 1 lifts quiet
// detail a touch.
const GAMMA = 0.9;
const GRID_ALPHA = [0.22, 0.08, 0.14]; // decades, subdivisions, 1 s parallels
const FOG_START = 0.85;

const LEVEL_FN = `
float level(float h) { return pow(clamp(h, 0.0, 1.0), ${glf(GAMMA)}); }`;

const SURF_FS = `#version 300 es
precision highp float;
in float vU, vZ, vT;
uniform vec3 uBg;
uniform float uDepthS;
out vec4 o;
${SPEC_FS}
${PAL_UNIFORMS}
${RAMP_FN}
${LINE_FN}
${LOG_PAPER_FN}
${LEVEL_FN}
void main() {
  float h; vec3 n;
  specSurface(vU, vT, h, n);
  vec3 col = ramp(level(h));
  float lit = dot(normalize(n), normalize(vec3(-0.6, 1.0, 0.5)));
  col *= 1.0 - ${glf(SHADE)} + ${glf(SHADE)} * 1.4 * clamp(lit, 0.0, 1.0);
  vec2 grid = logPaper(vU, 0.5);
  float g = max(max(grid.x * ${glf(GRID_ALPHA[0])}, grid.y * ${glf(GRID_ALPHA[1])}),
                gridLine(vZ * uDepthS, 0.5) * ${glf(GRID_ALPHA[2])});
  col = mix(col, vec3(0.85), g);
  col = mix(col, uBg, smoothstep(${glf(FOG_START)}, 1.0, vZ));
  o = vec4(col, 1.0);
}`;

// Under the front edge: the current spectrum as a colormapped fill; a light edge line.
const FLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform int uMode;
uniform float uCore;
uniform vec3 uBg;
out vec4 o;
${PAL_UNIFORMS}
${RAMP_FN}
${LEVEL_FN}
void main() {
  if (uMode == 0) {
    float v = level(vUV.y);
    o = vec4(mix(uBg, ramp(v) * 0.55, v), 1.0);
  } else {
    float a = 1.0 - smoothstep(uCore, 1.0, abs(vUV.y));
    o = vec4(vec3(0.92) * a, a);
  }
}`;

export default {
  name: "spectrogram",
  label: "Spectrogram",
  description: "Spectrogram sheet in shallow perspective, lightness-ordered colormap, log graph paper",
  camera: { backScale: 0.72, ampFront: 0.15 },
  background: () => theme.bg,
  text: { axis: "#8a9098", dim: "#5a6068" },

  drawStatic(g, geo, W, H, dpr, info) {
    drawSecondLabels(g, geo, dpr, info.depthS, "#6a7078");
  },

  create(gl) {
    const surf = program(gl, SURF_VS, SURF_FS);
    const flat = program(gl, FLAT_VS, FLAT_FS);
    const strips = makeStrips(gl, 4 * 2048);
    return {
      render(f) {
        gl.disable(gl.BLEND);
        if (f.R >= 2) {
          gl.enable(gl.DEPTH_TEST);
          gl.depthFunc(gl.LEQUAL);
          gl.useProgram(surf.prog);
          setCommonUniforms(gl, surf.u, f);
          setLogUniforms(gl, surf.u, f);
          gl.uniform1f(surf.u.uDepthS, f.depthS);
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
