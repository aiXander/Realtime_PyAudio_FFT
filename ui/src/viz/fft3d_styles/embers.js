// embers — a fire. The spectrum is a smouldering ridge of coals (dim smoky
// surface, glowing where it's loud); the live front edge throws off sparks
// in proportion to its level², and every onset bursts a shower of sparks out
// of that band's frequency range. Sparks ride back with the history (at
// exactly the history speed), rise, drift on a lazy curl-ish breeze and cool
// along a heat ramp — white → the palette's high → low → dim → gone. The
// only style with state that evolves on its own: particles are simulated on
// the CPU in a preallocated pool and uploaded once per frame. Bloom on top.
//
// Simulation runs on wall-clock dt (clamped), never on frame count, so the
// UI refresh-rate slider doesn't change how the fire behaves.

import {
  glf, PROJ_FN, SURF_VS, FLAT_VS, program, setCommonUniforms, makeStrips,
  fillCurtain, fillRim, makeBloom, disposeAll,
  PAL_UNIFORMS, pal, palBg, hex01,
} from "./common.js";

// Look / physics. All tunable by eye.
const MAX_PARTICLES = 24000;
const SPAWN_RATE = 2600;       // sparks / s across the whole front edge at full level (∝ level^1.5)
const BURST = 220;             // sparks per onset
const RISE = [0.12, 0.55];     // initial rise speed range (full-scale heights / s)
const BURST_RISE = [0.35, 1.0];
const LIFE = [0.9, 3.2];       // seconds
const DRIFT = 0.05;            // sideways breeze strength (u / s²)
const BUOYANCY_DECAY = 0.7;    // rise speed decays at this rate (1/s)
const POINT_PX = 4.2;          // spark size at the front when fresh, CSS px
const SURFACE_ALPHA = 0.9;
const BLOOM = { threshold: 0.18, knee: 0.25, strength: 1.3, weights: [0.5, 0.8, 1.0] };

const F = 8;                   // floats per particle: u, h, z, age | life, vu, vh, seed

// Heat ramp from the palette: dim low → low → high → white hot (the Ember
// palette gives a real blackbody; others give "fires" in their own colors).
const BLACKBODY = `
${PAL_UNIFORMS}
vec3 blackbody(float x) {   // 1 = white hot … 0 = cold
  vec3 c = mix(uC0 * 0.3, uC0, smoothstep(0.0, 0.35, x));
  c = mix(c, uC2, smoothstep(0.35, 0.7, x));
  return mix(c, mix(uC2, vec3(1.0), 0.85), smoothstep(0.75, 1.0, x));
}`;

const SPARK_VS = `#version 300 es
layout(location = 0) in vec4 aA;   // u, h, z, age
layout(location = 1) in vec4 aB;   // life, vu, vh, seed
${PROJ_FN}
out float vK, vFade, vSeed, vAge;
void main() {
  float s;
  vec2 px = toPx(aA.x, aA.y, aA.z, s);
  gl_Position = clipPos(px, s);
  gl_Position.z -= 0.002 * gl_Position.w;
  float k = clamp(aA.w / aB.x, 0.0, 1.0);
  gl_PointSize = max(1.0, uDpr * ${glf(POINT_PX)} * mix(1.0, 0.3, k) * s * (0.6 + 0.8 * aB.w));
  vK = k; vFade = 1.0 - smoothstep(0.75, 1.0, aA.z); vSeed = aB.w; vAge = aA.w;
}`;

const SPARK_FS = `#version 300 es
precision highp float;
in float vK, vFade, vSeed, vAge;
out vec4 o;
${BLACKBODY}
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float r = dot(q, q);
  if (r > 1.0) discard;
  float heat = pow(1.0 - vK, 1.3);
  float flicker = 0.75 + 0.25 * sin(vAge * 31.0 + vSeed * 57.0);
  float a = (1.0 - r) * (0.25 + 1.1 * heat) * flicker * vFade;
  o = vec4(blackbody(heat) * a, a);
}`;

// Coals: a dim smoky body that glows where the spectrum is loud.
const SURF_FS = `#version 300 es
precision highp float;
in float vH, vZ;
in vec3 vN;
uniform vec3 uBg;
out vec4 o;
${BLACKBODY}
void main() {
  float h = clamp(vH, 0.0, 1.0);
  vec3 n = normalize(vN);
  float diff = 0.5 + 0.5 * max(dot(n, normalize(vec3(0.0, 0.8, -0.6))), 0.0);
  vec3 smoke = uC0 * 0.16 * diff * 0.28;
  // Coals: dark body, hot where loud — and hotter on the crests than in the valleys.
  float crest = 0.6 + 0.4 * smoothstep(0.2, 1.0, n.y);
  vec3 glow = blackbody(0.1 + 0.8 * h) * pow(h, 2.4) * 0.9 * crest;
  float fog = smoothstep(0.3, 1.0, vZ);
  vec3 c = mix(smoke + glow, uBg, fog);
  float a = ${glf(SURFACE_ALPHA)} * (1.0 - smoothstep(0.8, 1.0, vZ));
  o = vec4(c * a, a);
}`;

const FLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform int uMode;
uniform float uCore;
out vec4 o;
${BLACKBODY}
void main() {
  if (uMode == 0) {
    float v = clamp(vUV.y, 0.0, 1.0);
    float a = 0.25 + 0.5 * v;
    o = vec4(uC0 * 0.35 * v * v * a, a);
  } else {
    float a = 1.0 - smoothstep(uCore, 1.0, abs(vUV.y));
    o = vec4(blackbody(0.8) * a, a);
  }
}`;

const lerp = (r, t) => r[0] + (r[1] - r[0]) * t;

export default {
  name: "embers",
  label: "Embers",
  description: "Peaks throw sparks that drift back and cool, like a fire",
  background: () => palBg("low", 0.03),
  text: { axis: "#a07a66", dim: "#6e5043" },

  drawStatic(g, geo, W) {
    const glow = g.createRadialGradient(W / 2, geo.y0 + (geo.y0 - geo.yBack) * 0.3, 0, W / 2, geo.y0, (geo.y0 - geo.yBack) * 1.4);
    const c = hex01(pal("low")).map((v) => Math.round(v * 255)).join(",");
    glow.addColorStop(0, `rgba(${c},0.10)`);
    glow.addColorStop(1, `rgba(${c},0)`);
    g.fillStyle = glow;
    g.fillRect(0, 0, W, geo.y0);
  },

  create(gl) {
    const surf = program(gl, SURF_VS, SURF_FS);
    const sparkP = program(gl, SPARK_VS, SPARK_FS);
    const flat = program(gl, FLAT_VS, FLAT_FS);
    const strips = makeStrips(gl, 4 * 384);
    const bloom = makeBloom(gl);

    const pool = new Float32Array(MAX_PARTICLES * F);
    let n = 0;
    const lastAge = new Float64Array(3).fill(Infinity);   // detects new onsets (age resets)
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, pool.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, F * 4, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, F * 4, 16);
    gl.bindVertexArray(null);

    function spawn(f, u, rise) {
      if (n >= MAX_PARTICLES) return;
      const mc = f.mc;
      const m = Math.max(0, Math.min(mc - 1, Math.round(u * (mc - 1))));
      const o = n * F;
      pool[o] = u;
      pool[o + 1] = Math.min(1, f.meshH[f.frontOff + m]);
      pool[o + 2] = 0;
      pool[o + 3] = 0;
      pool[o + 4] = lerp(LIFE, Math.random());
      pool[o + 5] = (Math.random() - 0.5) * 0.03;
      pool[o + 6] = lerp(rise, Math.random());
      pool[o + 7] = Math.random();
      n++;
    }

    function simulate(f) {
      const dt = f.dt, dz = dt / f.depthS, t = f.t;
      const decay = Math.exp(-BUOYANCY_DECAY * dt);
      for (let i = 0; i < n; ) {
        const o = i * F;
        const age = pool[o + 3] + dt;
        const z = pool[o + 2] + dz;
        let u = pool[o];
        if (age > pool[o + 4] || z > 1 || u < -0.05 || u > 1.05) {
          // Swap-remove: move the last particle into this slot.
          n--;
          if (i < n) pool.copyWithin(o, n * F, n * F + F);
          continue;
        }
        const seed = pool[o + 7];
        let vu = pool[o + 5] + DRIFT * Math.sin(z * 17 + seed * 6.283 + t * 1.7) * dt;
        vu *= 0.985;
        u += vu * dt;
        const vh = pool[o + 6] * decay;
        pool[o] = u;
        pool[o + 1] += vh * dt;
        pool[o + 2] = z;
        pool[o + 3] = age;
        pool[o + 5] = vu;
        pool[o + 6] = vh;
        i++;
      }
      // Steady emission from the live edge, ∝ level².
      const mc = f.mc;
      const perCol = (SPAWN_RATE * dt) / mc;
      for (let m = 0; m < mc; m++) {
        const h = Math.min(1, f.meshH[f.frontOff + m]);
        const expect = perCol * h * Math.sqrt(h);
        if (Math.random() < expect) spawn(f, (m + Math.random() - 0.5) / (mc - 1), RISE);
      }
      // Onset bursts from each band's frequency range.
      for (let b = 0; b < 3; b++) {
        const age = f.audio.onsetAge[b];
        if (age < lastAge[b] && age < 0.2) {
          const lo = f.bandU[2 * b], hi = f.bandU[2 * b + 1];
          for (let k = 0; k < BURST; k++) spawn(f, lo + (hi - lo) * Math.random(), BURST_RISE);
        }
        lastAge[b] = age;
      }
    }

    function draw(f) {
      gl.enable(gl.BLEND);
      if (f.R >= 2) {
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(surf.prog);
        setCommonUniforms(gl, surf.u, f);
        gl.bindVertexArray(f.surfVao);
        gl.drawElements(gl.TRIANGLES, f.triCount, gl.UNSIGNED_INT, 0);
      }
      // Front edge: dark-red glow below, hot rim on top.
      gl.disable(gl.DEPTH_TEST);
      const aa = f.bloomScale;
      const rimHalf = 0.9 * f.dpr * f.bloomScale + aa;
      let o = fillCurtain(f, strips.data, 0);
      o = fillRim(f, strips.data, o, rimHalf);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(flat.prog);
      setCommonUniforms(gl, flat.u, f);
      gl.uniform1f(flat.u.uCore, Math.max(0, 1 - aa / rimHalf));
      strips.upload(o);
      gl.uniform1i(flat.u.uMode, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 2 * f.mc);
      gl.uniform1i(flat.u.uMode, 1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 2 * f.mc, 2 * f.mc);
      // Sparks: additive, depth-tested against the coals (no depth write).
      if (n > 0) {
        if (f.R >= 2) gl.enable(gl.DEPTH_TEST);
        gl.depthMask(false);
        gl.blendFunc(gl.ONE, gl.ONE);
        gl.useProgram(sparkP.prog);
        setCommonUniforms(gl, sparkP.u, f);
        gl.bindVertexArray(vao);
        gl.drawArrays(gl.POINTS, 0, n);
        gl.depthMask(true);
        gl.disable(gl.DEPTH_TEST);
      }
    }

    return {
      render(f) {
        simulate(f);
        if (n > 0) {
          gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, pool, 0, n * F);
        }
        draw(f);
        bloom.capture(f, draw);
        bloom.apply(BLOOM);
      },
      dispose() {
        disposeAll(gl, surf, sparkP, flat, strips, bloom);
        gl.deleteBuffer(vbo);
        gl.deleteVertexArray(vao);
      },
    };
  },
};
