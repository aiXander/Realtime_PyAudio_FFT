// topo — a survey map of the sound, seen from higher up. Calm, precise,
// cartographic: an opaque surface with a hypsometric tint (sea → lowland →
// ochre → rock → snow), iso-height contour lines (every 5th an index
// contour), classic top-left hill shading, and a faint graticule — meridians
// at every octave, parallels every second of history (labelled at the right
// edge). The front edge is cut open like a block diagram, showing strata.
// No audio reactivity: this is the calm one.
// Palette: sea = low, lowlands = mid, highlands = high, then rock and snow.

import {
  glf, SURF_VS, PAL_UNIFORMS, palBg, FLAT_VS, program, setCommonUniforms, makeStrips, fillCurtain, fillRim, disposeAll,
} from "./common.js";

// Look. All tunable by eye.
const N_LEVELS = 20;           // contour interval = 1/N of full scale
const INDEX_EVERY = 5;         // every Nth contour is an index contour (thicker)
const LINE_PX = 0.9;           // contour half-width, device px
const INDEX_PX = 1.5;
const GRID_ALPHA = 0.16;       // graticule strength
// Hypsometric tint stops (height, GLSL color from the palette's uC0/uC1/uC2 = low/mid/high).
const TINT = [
  [0.00, "uC0 * 0.22"], [0.05, "uC0 * 0.42"], [0.09, "uC1 * 0.42"], [0.28, "uC1 * 0.68"],
  [0.46, "uC2 * 0.78"], [0.64, "mix(uC2, vec3(0.55), 0.6) * 0.8"], [0.82, "vec3(0.79, 0.77, 0.74)"],
  [0.95, "vec3(0.95, 0.94, 0.92)"],
];
const TINT_FN = `
${PAL_UNIFORMS}
vec3 tint(float h) {
  vec3 c = ${TINT[0][1]};
${TINT.slice(1).map(([p, c], i) => `  c = mix(c, ${c}, smoothstep(${glf(TINT[i][0])}, ${glf(p)}, h));`).join("\n")}
  return c;
}`;

const SURF_FS = `#version 300 es
precision highp float;
in float vU, vH, vZ, vKey, vS;
in vec3 vN;
uniform vec3 uBg;
uniform float uOct0, uOctSpan, uDepthS;
out vec4 o;
${TINT_FN}
/** Antialiased line coverage at integer values of q, halfPx wide on screen; fades out where lines crowd. */
float lineAt(float q, float halfPx) {
  float w = fwidth(q);
  float d = abs(fract(q + 0.5) - 0.5) / max(w, 1e-5);
  return (1.0 - smoothstep(halfPx - 0.5, halfPx + 0.5, d)) * (1.0 - smoothstep(0.12, 0.35, w));
}
void main() {
  float h = clamp(vH, 0.0, 1.2);
  vec3 col = tint(h);
  // Hill shading: light from the top-left (low frequencies, far away).
  vec3 n = normalize(vN);
  float lit = dot(n, normalize(vec3(-0.75, 1.0, 0.6)));
  col *= 0.62 + 0.55 * clamp(lit, 0.0, 1.2);
  // Contours: dark on light ground, light on dark (sea) ground.
  float q = h * ${glf(N_LEVELS)};
  float idx = lineAt(q / ${glf(INDEX_EVERY)}, ${glf(INDEX_PX)});
  float minor = lineAt(q, ${glf(LINE_PX)});
  float c = max(minor * 0.75, idx);
  float lum = dot(col, vec3(0.3, 0.55, 0.15));
  vec3 ink = lum > 0.28 ? vec3(0.04, 0.05, 0.06) : vec3(0.75, 0.86, 0.9);
  col = mix(col, ink, c * (h > 0.02 ? 0.6 : 0.0));
  // Graticule: meridians every octave, parallels every second of history.
  float oct = uOct0 + vU * uOctSpan;
  float g = max(lineAt(oct, 0.6), lineAt(vZ * uDepthS, 0.6));
  col = mix(col, vec3(0.85, 0.93, 1.0), g * ${glf(GRID_ALPHA)});
  // Map edge: fade into the table toward the horizon.
  col = mix(col, uBg, smoothstep(0.82, 1.0, vZ));
  o = vec4(col, 1.0);
}`;

// Cut face under the front edge (strata) + a crisp edge line.
const FLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform int uMode;
uniform float uCore;
out vec4 o;
void main() {
  if (uMode == 0) {
    float v = clamp(vUV.y, 0.0, 1.0);
    float band = 0.5 + 0.5 * sin(v * 40.0 + sin(vUV.x * 30.0) * 0.8);
    vec3 c = mix(vec3(0.10, 0.08, 0.07), vec3(0.23, 0.18, 0.13), v) * (0.85 + 0.15 * band);
    o = vec4(c, 1.0);
  } else {
    float a = 1.0 - smoothstep(uCore, 1.0, abs(vUV.y));
    o = vec4(vec3(0.02, 0.03, 0.035) * a, a);
  }
}`;

export default {
  name: "topo",
  label: "Topo",
  description: "Survey map with contour lines and hill shading, seen from above",
  camera: { backScale: 0.55, ampFront: 0.22 },
  background: () => palBg("low", 0.1),
  text: { axis: "#8ea3ad", dim: "#5d7682" },

  // 1 s parallels, labelled at the right edge (thinned where they crowd).
  drawStatic(g, geo, W, H, dpr, info) {
    const K = 1 / geo.backScale - 1;
    g.font = `${Math.round(9 * dpr)}px ui-monospace, Menlo, monospace`;
    g.fillStyle = "#6f8d99";
    g.textAlign = "left";
    g.textBaseline = "middle";
    let lastY = Infinity;
    for (let sec = 1; sec < info.depthS; sec++) {
      const s = 1 / (1 + K * (sec / info.depthS));
      const y = geo.y0 - ((1 - s) / (1 - geo.backScale)) * (geo.y0 - geo.yBack);
      if (lastY - y < 14 * dpr) continue;
      lastY = y;
      const x = geo.cx + geo.plotW / 2 * s + 4 * dpr;
      g.fillText(`−${sec}s`, x, y);
    }
  },

  create(gl) {
    const surf = program(gl, SURF_VS, SURF_FS);
    const flat = program(gl, FLAT_VS, FLAT_FS);
    const strips = makeStrips(gl, 4 * 384);
    return {
      render(f) {
        if (f.R >= 2) {
          gl.disable(gl.BLEND);
          gl.enable(gl.DEPTH_TEST);
          gl.depthFunc(gl.LEQUAL);
          gl.useProgram(surf.prog);
          setCommonUniforms(gl, surf.u, f);
          gl.uniform1f(surf.u.uOct0, Math.log2(f.fMin / 1000));
          gl.uniform1f(surf.u.uOctSpan, Math.log2(f.fMax / f.fMin));
          gl.uniform1f(surf.u.uDepthS, f.depthS);
          gl.bindVertexArray(f.surfVao);
          gl.drawElements(gl.TRIANGLES, f.triCount, gl.UNSIGNED_INT, 0);
          gl.disable(gl.DEPTH_TEST);
        }
        const rimHalf = 0.8 * f.dpr + 1;
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
