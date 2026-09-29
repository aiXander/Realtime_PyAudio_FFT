// aurora — northern lights. No solid surface: every few spectra hang a
// curtain of light. Like a real aurora, each curtain has a bright, sharp
// lower hem (the spectrum line itself) and fades upward in vertical rays;
// louder regions throw taller rays, and very loud hems blush pink. Rays
// shimmer slowly along the curtain. Additive light over a starry sky, with
// bloom. A low-band onset makes the whole sky swell briefly.
// Palette: hem = mid, rays = low, blush on loud hems = high.
//
// Curtains are anchored to history rows (row key % CURTAIN_STEP), so they
// travel with the data instead of flickering between frames. Drawn as one
// instanced triangle strip per curtain, reading heights from the core's
// height texture.

import {
  glf, rng, PROJ_FN, PAL_UNIFORMS, palBg, program, setCommonUniforms, makeBloom, disposeAll,
} from "./common.js";

// Look. All tunable by eye.
const CURTAIN_STEP = 4;        // one curtain every N mesh rows (2 mesh rows per history row)
const RAY_BASE = 0.10;         // ray height above the hem at silence (fraction of full scale)
const RAY_GAIN = 0.75;         // … plus this × the local level
const HEM_BELOW = 0.035;       // soft glow under the hem
const GAIN = 0.55;             // overall brightness
const SILENT_GLOW = 0.1;       // brightness floor so quiet curtains still read
const ONSET_BOOST = 0.3;       // low-band onset swell …
const ONSET_DECAY_S = 0.25;    // … and its decay
const BLOOM = { threshold: 0.12, knee: 0.25, strength: 1.25, weights: [0.55, 0.8, 1.0] };


const VS = `#version 300 es
uniform sampler2D uHTex;
uniform sampler2D uRowTex;
uniform int uCols, uFirst, uStep;
${PROJ_FN}
out float vU, vV, vH, vZ, vKey, vS;
void main() {
  int r = uFirst + gl_InstanceID * uStep;
  int col = gl_VertexID >> 1;
  float top = float(gl_VertexID & 1);
  vec2 zk = texelFetch(uRowTex, ivec2(r, 0), 0).rg;
  float h = min(texelFetch(uHTex, ivec2(col, r), 0).r, 1.1);
  float u = float(col) / float(uCols - 1);
  float len = ${glf(RAY_BASE)} + ${glf(RAY_GAIN)} * h;
  // Bottom vertex a little under the hem (soft underglow), top at the ray tips.
  float y = top > 0.5 ? h + len : max(0.0, h - ${glf(HEM_BELOW)});
  float s;
  vec2 px = toPx(u, y, zk.x, s);
  gl_Position = clipPos(px, s);
  vV = top > 0.5 ? 1.0 : -(h - max(0.0, h - ${glf(HEM_BELOW)})) / len;
  vU = u; vH = h; vZ = zk.x; vKey = zk.y; vS = s;
}`;

const FS = `#version 300 es
precision highp float;
in float vU, vV, vH, vZ, vKey, vS;
uniform float uTime, uBoost, uHemPx;
${PAL_UNIFORMS}
out vec4 o;
void main() {
  // Hem profile: sharp bright line at v = 0, soft glow below, rays fading above.
  float hem = vV < 0.0 ? exp(-pow(vV * 30.0, 2.0)) : 1.0;
  float up = pow(1.0 - clamp(vV, 0.0, 1.0), 1.7);
  float line = exp(-pow(vV * uHemPx, 2.0));                  // crisp hem core
  // Vertical rays: a few octaves of slowly drifting stripes along the curtain.
  float x = vU * 140.0 + vKey * 0.037;
  float rays = 0.55 + 0.25 * sin(x + uTime * 0.6 + 2.0 * sin(vU * 23.0 - uTime * 0.21))
                    + 0.20 * sin(x * 2.7 - uTime * 1.1 + vKey * 0.11);
  float h = clamp(vH, 0.0, 1.0);
  float lvl = ${glf(SILENT_GLOW)} + h;
  vec3 hemCol = mix(mix(uC1, vec3(1.0), 0.15), uC2, smoothstep(0.65, 1.0, h));
  vec3 col = mix(hemCol, uC0, smoothstep(0.05, 0.75, vV));
  col = mix(col, vec3(1.0), 0.55 * line * smoothstep(0.2, 0.9, h));
  float fade = 1.0 - smoothstep(0.6, 1.0, vZ);
  // Far curtains bunch up on screen; thin them so the horizon doesn't saturate.
  float bunch = vS * vS;
  float a = ${glf(GAIN)} * uBoost * lvl * hem * up * rays * fade * bunch * (1.0 + 1.5 * line);
  o = vec4(col * a, a);
}`;

export default {
  name: "aurora",
  label: "Aurora",
  description: "Curtains of light rising from each spectrum over a starry sky",
  background: () => palBg("low", 0.03),
  text: { axis: "#7c8aa0", dim: "#4c5870" },
  heightTex: true,

  drawStatic(g, geo, W, H, dpr) {
    const sky = g.createLinearGradient(0, 0, 0, geo.y0);
    sky.addColorStop(0, palBg("low", 0.02));
    sky.addColorStop(0.7, palBg("low", 0.07));
    sky.addColorStop(1, palBg("mid", 0.1));
    g.fillStyle = sky;
    g.fillRect(0, 0, W, geo.y0);
    const rnd = rng(7);
    const n = Math.round((W * H) / (2600 * dpr * dpr));
    for (let i = 0; i < n; i++) {
      const x = rnd() * W, y = rnd() * geo.y0;
      const b = rnd();
      g.fillStyle = `rgba(210,225,255,${(0.15 + 0.6 * b * b).toFixed(3)})`;
      const sz = b > 0.93 ? 2 * dpr : dpr;
      g.fillRect(x, y, sz, sz);
    }
  },

  create(gl) {
    const p = program(gl, VS, FS);
    const bloom = makeBloom(gl);
    const vao = gl.createVertexArray();
    let boost = 1;

    function draw(f) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(p.prog);
      setCommonUniforms(gl, p.u, f);
      gl.uniform1f(p.u.uBoost, boost);
      gl.uniform1f(p.u.uHemPx, 60);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, f.hTex);
      gl.uniform1i(p.u.uHTex, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, f.rowTex);
      gl.uniform1i(p.u.uRowTex, 1);
      gl.uniform1i(p.u.uStep, CURTAIN_STEP);
      gl.bindVertexArray(vao);
      // Anchored curtains (oldest first), then the live front edge.
      const first = (CURTAIN_STEP - (f.key0 % CURTAIN_STEP)) % CURTAIN_STEP;
      const n = first < f.R - 1 ? Math.floor((f.R - 2 - first) / CURTAIN_STEP) + 1 : 0;
      if (n > 0) {
        gl.uniform1i(p.u.uFirst, first);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 2 * f.mc, n);
      }
      gl.uniform1i(p.u.uFirst, f.R - 1);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 2 * f.mc, 1);
      gl.activeTexture(gl.TEXTURE0);
    }

    return {
      render(f) {
        if (f.R < 2) return;
        const age = f.audio.onsetAge[0];
        boost = 1 + (age >= 0 && age < 3 * ONSET_DECAY_S ? ONSET_BOOST * Math.exp(-age / ONSET_DECAY_S) : 0);
        draw(f);
        bloom.capture(f, draw);
        bloom.apply(BLOOM);
      },
      dispose() {
        disposeAll(gl, p, bloom);
        gl.deleteVertexArray(vao);
      },
    };
  },
};
