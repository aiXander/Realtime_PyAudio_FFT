// engraving — an old etching / banknote print: dark ink on warm paper, the
// only light style. The surface is paper-white and opaque (so it hides what
// lies behind it); all shading is done with engraved lines. The main family
// follows each spectrum across the frequency axis (so every moment of
// history is its own ridge line, anchored to arrival time) and swells from
// hairline to fat stroke with shadow; a crossing family appears only in the
// deepest shadow (cross-hatching). Where lines crowd toward the horizon they
// dissolve into their average tone instead of shimmering, and the distance
// fades to paper like atmospheric perspective in a print.
//
// Reactivity: a low-band onset makes the ink bleed (lines swell) briefly.
// Palette: the ink is tinted by low, the wash on the peaks is high; paper stays paper.

import {
  glf, rng, PAL_UNIFORMS, SURF_VS, FLAT_VS, program, setCommonUniforms, makeStrips, fillRim, disposeAll,
} from "./common.js";

// Look. All tunable by eye.
const PAPER = "#efe6d2";
const INK = "mix(uC0, vec3(0.07, 0.065, 0.06), 0.84)";   // near-black, tinted by the palette's low color
const WASH = "uC2";                                         // wash on the peaks: the palette's high color
const LINES_T = 150;           // ridge lines across the depth (time)
const LINES_U = 150;           // cross-hatch lines across the width (frequency)
const MIN_PX = 0.35;           // thinnest stroke half-width, device px
const BLEED = 0.14, ONSET_DECAY_S = 0.3;
const RIM_PX = 1.6;

const vec3 = (h) => `vec3(${[1, 3, 5].map((i) => glf(parseInt(h.slice(i, i + 2), 16) / 255)).join(", ")})`;

const SURF_FS = `#version 300 es
precision highp float;
in float vU, vH, vZ;
in vec3 vN;
uniform float uPhase, uBleed;
${PAL_UNIFORMS}
out vec4 o;
/** Coverage of a line family at integer q whose stroke fills 'frac' of each period (AA, crowd-safe). */
float hatch(float q, float frac) {
  float w = max(fwidth(q), 1e-5);                     // periods per device px
  float d = abs(fract(q + 0.5) - 0.5) / w;            // distance to the line, px
  float hw = max(${glf(MIN_PX)}, 0.5 * frac / w);     // stroke half-width, px
  float cov = (1.0 - smoothstep(hw - 0.5, hw + 0.5, d)) * smoothstep(0.0, 0.08, frac);
  return mix(cov, frac, smoothstep(0.22, 0.45, w));   // crowded → flat tone
}
void main() {
  float h = clamp(vH, 0.0, 1.2);
  vec3 n = normalize(vN);
  float diff = max(dot(n, normalize(vec3(-0.55, 0.75, -0.35))), 0.0);
  float tone = clamp(1.0 - (0.2 + 0.85 * diff), 0.0, 1.0);   // 0 lit … 1 deep shadow
  tone = tone * (1.0 + uBleed);
  float ridge = hatch(uPhase - vZ * ${glf(LINES_T)}, mix(0.1, 0.8, pow(tone, 1.3)));
  float cross = hatch(vU * ${glf(LINES_U)}, 0.55 * smoothstep(0.5, 0.85, tone));
  float ink = max(ridge, cross);
  float fog = smoothstep(0.25, 1.0, vZ);
  vec3 paper = mix(${vec3(PAPER)}, ${WASH}, 0.15 * smoothstep(0.3, 1.0, h));
  vec3 c = mix(paper, ${INK}, ink * (1.0 - 0.85 * fog));
  c = mix(c, ${vec3(PAPER)}, fog * 0.5);
  o = vec4(c, 1.0);
}`;

const FLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform float uCore;
${PAL_UNIFORMS}
out vec4 o;
void main() {
  float a = 1.0 - smoothstep(uCore, 1.0, abs(vUV.y));
  o = vec4(${INK} * a, a);
}`;

export default {
  name: "engraving",
  label: "Engraving",
  description: "Ink hatching on warm paper, like an old etching",
  background: PAPER,
  text: { axis: "#5b5346", dim: "#8a8070" },

  // Paper grain (±3 % luminance) and a faint darkening toward the edges.
  drawStatic(g, geo, W, H) {
    const img = g.getImageData(0, 0, W, H);
    const d = img.data, rnd = rng(11);
    for (let i = 0; i < d.length; i += 4) {
      const k = 1 + (rnd() - 0.5) * 0.06;
      d[i] *= k; d[i + 1] *= k; d[i + 2] *= k;
    }
    g.putImageData(img, 0, 0);
    const v = g.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.hypot(W, H) * 0.6);
    v.addColorStop(0, "rgba(120,90,50,0)");
    v.addColorStop(1, "rgba(120,90,50,0.16)");
    g.fillStyle = v;
    g.fillRect(0, 0, W, H);
  },

  create(gl) {
    const surf = program(gl, SURF_VS, SURF_FS);
    const flat = program(gl, FLAT_VS, FLAT_FS);
    const strips = makeStrips(gl, 2 * 384);
    return {
      render(f) {
        const age = f.audio.onsetAge[0];
        const bleed = age >= 0 && age < 4 * ONSET_DECAY_S ? BLEED * Math.exp(-age / ONSET_DECAY_S) : 0;
        if (f.R >= 2) {
          gl.disable(gl.BLEND);
          gl.enable(gl.DEPTH_TEST);
          gl.depthFunc(gl.LEQUAL);
          gl.useProgram(surf.prog);
          setCommonUniforms(gl, surf.u, f);
          gl.uniform1f(surf.u.uPhase, (f.t * LINES_T / f.depthS) % 1);
          gl.uniform1f(surf.u.uBleed, bleed);
          gl.bindVertexArray(f.surfVao);
          gl.drawElements(gl.TRIANGLES, f.triCount, gl.UNSIGNED_INT, 0);
          gl.disable(gl.DEPTH_TEST);
        }
        // Front edge: a crisp ink line, no curtain.
        const rimHalf = (RIM_PX / 2) * f.dpr + 0.75;
        const o = fillRim(f, strips.data, 0, rimHalf);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(flat.prog);
        setCommonUniforms(gl, flat.u, f);
        gl.uniform1f(flat.u.uCore, Math.max(0, 1 - 0.75 / rimHalf));
        strips.upload(o);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 2 * f.mc);
      },
      dispose() { disposeAll(gl, surf, flat, strips); },
    };
  },
};
