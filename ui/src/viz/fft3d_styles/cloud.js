// cloud — the original look: a faint matte surface (diffuse light only)
// carrying a jittered grid of tiny square points — a dust cloud more than a
// solid — colored by frequency (the palette's L/M/H band colors, faded) and
// brightened by height. A soft rim and a tinted curtain hang under the front
// edge; older rows recede into fog. Follows the palette.
//
// The surface is drawn back to front and writes depth, so points hidden
// behind nearer swells are culled.

import { theme, PALETTES, hexRgb } from "../../colors.js";
import {
  glf, PROJ_FN, HASH_FN, SURF_VS, FLAT_VS, bandFn, program, setCommonUniforms,
  makeStrips, fillCurtain, fillRim, disposeAll,
} from "./common.js";

// Look. All tunable by eye.
const FOG_START = 0.2;         // depth (0 front … 1 horizon) where things start sinking into fog
const FADE_START = 0.65;       // depth where they start going transparent (gone at 1)
const DESATURATE = 0.3;        // fade the band colors toward grey
const SURFACE_ALPHA = 0.5;     // matte surface opacity (the "body" under the points)
const POINT_PX = 1.4;          // point size at the front, CSS px (squares)
const POINT_GAIN = 0.9;        // point brightness
const POINT_JITTER = 1.0;      // point position noise, fraction of the grid spacing (frequency and time)
const POINT_COPIES = 2;        // points per mesh vertex, each jittered independently
const RIM_PX = 1.3;            // front-edge rim width, CSS px

const FOG_FN = `
uniform vec3 uBg;
float fog(float z) { return smoothstep(${glf(FOG_START)}, 1.0, z) * 0.9; }
float fade(float z) { return 1.0 - smoothstep(${glf(FADE_START)}, 1.0, z); }`;

// Matte surface: soft diffuse light, dark floor → band color with height.
const SURF_FS = `#version 300 es
precision highp float;
in float vU, vH, vZ;
in vec3 vN;
out vec4 o;
${bandFn(DESATURATE)}
${FOG_FN}
void main() {
  vec3 base = band(vU);
  float h = clamp(vH, 0.0, 1.0);
  vec3 n = normalize(vN);
  float diff = 0.55 + 0.45 * max(dot(n, normalize(vec3(-0.4, 0.9, -0.35))), 0.0);
  vec3 col = base * (0.06 + 0.6 * smoothstep(0.0, 0.9, h)) * diff;
  col = mix(col, uBg, fog(vZ));
  float a = ${glf(SURFACE_ALPHA)} * fade(vZ) * (0.5 + 0.5 * smoothstep(0.0, 0.4, h));
  o = vec4(col * a, a);
}`;

// Points ride the surface vertices, jittered by a stable per-point hash
// keyed on (column, history row, copy) so the noise travels with the data.
const POINT_VS = `#version 300 es
layout(location = 0) in float aH;
layout(location = 1) in float aZ;
layout(location = 3) in float aKey;
uniform int uCols;
${PROJ_FN}
${HASH_FN}
out float vU, vH, vZ, vRand;
void main() {
  int col = gl_VertexID % uCols;
  float u = float(col) / float(uCols - 1);
  uint k = hash(uint(col) * 0x9E3779B1u ^ hash(uint(aKey) * 4u + uint(gl_InstanceID) + 0x632BE5ABu));
  float r1 = float(k & 0xFFFFu) / 65535.0, r2 = float(k >> 16) / 65535.0;
  u += (r1 - 0.5) * ${glf(POINT_JITTER)} / float(uCols - 1);
  float zz = max(0.0, aZ + (r2 - 0.5) * ${glf(POINT_JITTER)} * uRowDz);
  vRand = float(hash(k) & 0xFFFFu) / 65535.0;
  float s;
  vec2 px = toPx(u, aH, zz, s);
  gl_Position = clipPos(px, s);
  gl_Position.z -= 0.004 * gl_Position.w;   // sit just in front of the surface they ride on
  gl_PointSize = max(1.0, ${glf(POINT_PX)} * uDpr * sqrt(s));
  vU = u; vH = aH; vZ = zz;
}`;

// Points: additive dust, brighter and whiter with height, per-point random brightness.
const POINT_FS = `#version 300 es
precision highp float;
in float vU, vH, vZ, vRand;
out vec4 o;
${bandFn(DESATURATE)}
${FOG_FN}
void main() {
  float h = clamp(vH, 0.0, 1.0);
  vec3 col = mix(band(vU), vec3(1.0), 0.3 * h);
  float a = ${glf(POINT_GAIN)} * (0.12 + 0.88 * h) * (0.25 + 0.75 * vRand) * fade(vZ) * (1.0 - fog(vZ));
  o = vec4(col * a, a);
}`;

const FLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform int uMode;       // 0 = curtain (v = height, 0 at baseline), 1 = rim (v = side, -1..1)
uniform float uCore;     // rim: solid fraction of the half width (rest is AA falloff)
out vec4 o;
${bandFn(DESATURATE)}
void main() {
  vec3 base = band(vUV.x);
  if (uMode == 0) {
    float v = clamp(vUV.y, 0.0, 1.0);
    float a = 0.35 + 0.4 * v;
    o = vec4(base * (0.04 + 0.25 * v * v) * a, a);
  } else {
    float a = 0.8 * (1.0 - smoothstep(uCore, 1.0, abs(vUV.y)));
    o = vec4(mix(base, vec3(1.0), 0.15) * a, a);
  }
}`;

export default {
  name: "cloud",
  label: "Cloud",
  description: "Dust cloud on a faint matte sea, in the palette's band colors",
  background: () => theme.bg,
  text: { axis: "#7a8088", dim: "#5a6068" },

  // Faint glow at the horizon so the far rows have something to sink into.
  drawStatic(g, geo) {
    const a = hexRgb(PALETTES[theme.name].accent).join(",");
    const glow = g.createLinearGradient(0, geo.padT, 0, geo.yBack + (geo.y0 - geo.yBack) * 0.5);
    glow.addColorStop(0, `rgba(${a},0.00)`);
    glow.addColorStop(0.35, `rgba(${a},0.045)`);
    glow.addColorStop(1, `rgba(${a},0.00)`);
    g.fillStyle = glow;
    g.fillRect(0, 0, geo.plotX * 2 + geo.plotW, geo.y0);
  },

  create(gl, core) {
    const surf = program(gl, SURF_VS, SURF_FS);
    const points = program(gl, POINT_VS, POINT_FS);
    const flat = program(gl, FLAT_VS, FLAT_FS);
    const strips = makeStrips(gl, 4 * 384);
    return {
      render(f) {
        gl.enable(gl.BLEND);
        if (f.R >= 2) {
          gl.bindVertexArray(f.surfVao);
          gl.enable(gl.DEPTH_TEST);
          gl.depthFunc(gl.LEQUAL);
          // Matte body, back to front, writing depth so points behind nearer swells are culled.
          gl.useProgram(surf.prog);
          setCommonUniforms(gl, surf.u, f);
          gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // premultiplied
          gl.depthMask(true);
          gl.drawElements(gl.TRIANGLES, f.triCount, gl.UNSIGNED_INT, 0);
          // Dust: additive points on every mesh vertex.
          gl.useProgram(points.prog);
          setCommonUniforms(gl, points.u, f);
          gl.blendFunc(gl.ONE, gl.ONE);
          gl.depthMask(false);
          gl.drawArraysInstanced(gl.POINTS, 0, f.R * f.mc, POINT_COPIES);
          gl.depthMask(true);
          gl.disable(gl.DEPTH_TEST);
        }
        // Front edge: curtain down to its baseline + rim.
        const rimHalf = (RIM_PX / 2) * f.dpr + 1;   // + 1 px of AA falloff
        let o = fillCurtain(f, strips.data, 0);
        o = fillRim(f, strips.data, o, rimHalf);
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
      dispose() { disposeAll(gl, surf, points, flat, strips); },
    };
  },
};
