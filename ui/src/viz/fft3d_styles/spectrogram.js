// spectrogram — the score: the whole history laid flat, seen from straight
// above, so nothing hides anything. Cloud shows how the sound feels; this
// one shows the pattern: a hit is a line across frequency, a note is a
// white thread along time whose length is how long it sounded, and a
// rhythm or an arpeggio reads as evenly spaced marks, like a piano roll.
// Level is a lightness-ordered colormap (linear in dB, so quiet layers stay
// readable under loud ones); gentle hill shading embosses it; the graticule
// is log graph paper across frequency plus 1 s parallels (labelled). The
// live spectrum stands as a curve at the front edge, as on an analyser.
// Colour and shading are read per pixel from the history texture, so the
// sheet is at data resolution. One opaque draw + the front-edge strips.
// Palette: colormap from low → mid → high (lightness-ordered). With the
// Color picker the hue is the moment's instead (which bands are sounding)
// and level is brightness alone.

import { theme } from "../../colors.js";
import {
  glf, SURF_VS, SPEC_FN, SPEC_FS, FLAT_VS, PAL_UNIFORMS, MOMENT_FN, RAMP_FN, LINE_FN, LOG_PAPER_FN, program, setCommonUniforms,
  setLogUniforms, makeStrips, fillCurtain, fillRim, setEdgeAmp, disposeAll, drawSecondLabels,
} from "./common.js";

// Look. All tunable by eye.
const SHADE = 0.2;             // hill-shading depth (0 = flat color, exact colormap)
// Colormap: level = (dB-linear 0..1)^GAMMA, whatever the Curve picker says
// (that one shapes height, and this sheet has none).
const GAMMA = 1.25;
const RAMP_TOP = 0.8;          // the colormap stops short of white: white is for notes
const THREAD_PX = 1.0;         // half width of a note's thread, CSS px
const TONE_TINT = 0.18;        // how far a tonal ridge is pulled toward white, under its thread
const ONSET_GLOW = 0.8;        // light added where a sound begins (sharpens the line a hit makes)
const GRID_ALPHA = [0.2, 0.07, 0.16]; // decades, subdivisions, 1 s parallels
const FOG_START = 0.9;
const LIVE_AMP = 0.16;         // the live curve's full-scale height, fraction of the plot height
const RELIEF = 0.02;           // what is left of the relief (the camera's height scales it)

const LEVEL_FN = `
float level(float d) { return pow(clamp(d, 0.0, 1.0), ${glf(GAMMA)}); }`;

const SURF_FS = `#version 300 es
precision highp float;
in float vU, vZ, vT;
uniform vec3 uBg;
uniform float uDpr;
out vec4 o;
${SPEC_FS}
${PAL_UNIFORMS}
${MOMENT_FN}
${RAMP_FN}
${LINE_FN}
${LOG_PAPER_FN}
${LEVEL_FN}
void main() {
  float h; vec3 n;
  specSurface(vU, vT, h, n);
  float d = pow(h, 1.0 / uGamma);                  // back to dB-linear
  vec3 col = momentMix(ramp(level(d)) * ${glf(RAMP_TOP)}, vT, fwidth(vT));
  float lit = dot(normalize(n), normalize(vec3(-0.6, 1.0, 0.5)));
  col *= 1.0 - ${glf(SHADE)} + ${glf(SHADE)} * 1.4 * clamp(lit, 0.0, 1.0);
  float up = smoothstep(0.05, 0.3, d);
  col += (col + 0.25 * d) * ${glf(ONSET_GLOW)} * specOnset(vU, vT) * up;
  col = mix(col, vec3(1.0), ${glf(TONE_TINT)} * specTone(vU, vT) * up);
  vec2 grid = logPaper(vU, 0.5);
  float g = max(max(grid.x * ${glf(GRID_ALPHA[0])}, grid.y * ${glf(GRID_ALPHA[1])}),
                gridLine(vZ * uDepthS, 0.5) * ${glf(GRID_ALPHA[2])});
  col = mix(col, vec3(0.85), g);
  col = mix(col, vec3(1.0), specThread(vU, vT, ${glf(THREAD_PX)} * uDpr));
  col = mix(col, uBg, smoothstep(${glf(FOG_START)}, 1.0, vZ));
  o = vec4(col, 1.0);
}`;

// The live spectrum at the front edge: a faint fill and a line, white where a note is sounding.
const FLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform int uMode;
uniform float uCore;
uniform float uRowA, uFrontOff;
out vec4 o;
${SPEC_FN}
${PAL_UNIFORMS}
${MOMENT_FN}
${RAMP_FN}
void main() {
  float tone = specTone(vUV.x, uRowA + uFrontOff);
  if (uMode == 0) {
    float v = clamp(vUV.y, 0.0, 1.0);
    float a = 0.25 + 0.45 * v;
    o = vec4(momentMix(ramp(0.25 + 0.6 * v), uRowA + uFrontOff, 1.0) * (0.5 + 0.5 * tone) * a, a);
  } else {
    float a = 1.0 - smoothstep(uCore, 1.0, abs(vUV.y));
    o = vec4(mix(momentMix(vec3(0.8), uRowA + uFrontOff, 1.0), vec3(1.0), tone) * a, a);
  }
}`;

export default {
  name: "spectrogram",
  label: "Spectrogram",
  description: "The score: the history laid flat, nothing hidden. Hits are lines across, notes are threads along time",
  camera: { backScale: 0.86, ampFront: RELIEF },
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
          gl.bindVertexArray(f.surfVao);
          gl.drawElements(gl.TRIANGLES, f.triCount, gl.UNSIGNED_INT, 0);
          gl.disable(gl.DEPTH_TEST);
        }
        const rimHalf = 0.75 * f.dpr + 1;
        setEdgeAmp(f, LIVE_AMP * (f.geo.y0 - f.geo.padT));
        let o = fillCurtain(f, strips.data, 0);
        o = fillRim(f, strips.data, o, rimHalf);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(flat.prog);
        setCommonUniforms(gl, flat.u, f);
        gl.uniform1f(flat.u.uCore, Math.max(0, 1 - 1 / rimHalf));
        strips.upload(o);
        gl.uniform1i(flat.u.uMode, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 2 * f.mc);
        gl.uniform1i(flat.u.uMode, 1);
        gl.drawArrays(gl.TRIANGLE_STRIP, 2 * f.mc, 2 * f.mc);
      },
      dispose() { disposeAll(gl, surf, flat, strips); },
    };
  },
};
