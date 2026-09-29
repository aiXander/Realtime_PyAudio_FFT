# FFT 3D view — five alternative looks, selectable by URL

**Status:** draft plan, nothing built. **Goal:** Xander wants to *play with the look* of the 3D FFT view. Build five distinct visual styles for it, each reachable on its own URL, so he can open them side by side and pick favourites. Pure eye candy — this view is explicitly *not* scientifically exact (see the "Deliberate exception to byte-identical" note in `CLAUDE.md` → UI → "FFT card: 2D / 3D").

Read first: `ui/src/viz/fft_3d.js` (the whole current renderer, ~650 lines, well commented) and the UI section of `CLAUDE.md`.

## What exists today (the baseline, keep it as style `cloud`)

`ui/src/viz/fft_3d.js` does, in order:

1. **History ring** (`push()`, called on every WS FFT frame): max-merges frames into ~96 rows across the History-slider depth, each row timestamped.
2. **Display smoothing per row** (`smoothRow()`): fills raw-dB sentinels, resolution-aware Gaussian across frequency (`buildFreqKernel` / `smoothFreq`, exported), Catmull-Rom upsample to `meshCols` (256–384).
3. **Per frame** (`draw()`): picks the rows inside the depth window, peak-preserving time blur (`blurTime`), Catmull-Rom subdivision in time (`buildMesh`, `T_SUB` mesh rows per history row), world-space slopes for lighting (`fillVertices`) → one interleaved vertex buffer `[h, z, slope_x, slope_z, rowKey]`.
4. **WebGL2 on an offscreen canvas**, composited onto the card's 2D canvas with `drawImage` over a cached static layer (background, horizon glow, x-axis labels). Passes: matte surface (triangles, back-to-front, writes depth) → jittered additive points (`drawArraysInstanced`, depth-tested) → front-edge curtain + rim (screen-space strips).
5. **Projection** is a pseudo-perspective in the vertex shader: `s(z) = 1/(1+K·z)` with `z ∈ [0,1]` = age/depth, front row full width at the bottom, back row `BACK_SCALE` wide near the top; `w = 1/s` for perspective-correct varyings; NDC depth `0.9 − 1.8·s`.

Steps 1–3 and the projection are style-independent. Everything in step 4 is "the look".

## Deliverable

### 1. Refactor into core + styles

- `ui/src/viz/fft_3d.js` keeps the **core**: ring, smoothing, mesh build, vertex buffer, GL context/loss handling, static layer, compositing, perf recording.
- New folder `ui/src/viz/fft3d_styles/`, one ES module per style: `cloud.js` (the current look, moved verbatim — it must look identical after the refactor), `aurora.js`, `topo.js`, `synthwave.js`, `engraving.js`, `embers.js`.
- A style module exports one object, roughly:
  ```js
  export default {
    name: "aurora",
    camera: { BACK_SCALE, AMP_FRONT, HORIZON_PAD },   // overrides; omitted = core defaults
    smoothing: { T_SUB, FREQ_SMOOTH_OCT, ... },          // optional overrides
    background: "#05060a",                               // static-layer fill (+ optional drawStatic(g, geo) hook)
    init(gl, core) { /* compile programs, create FBOs */ },
    render(gl, frame) { /* issue draw calls */ },
    resize(gl, W, H) {},                                 // optional, for FBO-based post-processing
  };
  ```
  `frame` carries what the core computed: mesh row count `R`, `meshCols`, the bound surface VAO, the front-edge arrays (`edgeX/edgeY/meshH` front row), geometry (`geo`), `dpr`, band color stops, `now`, and a small **audio-reactivity** bundle read from `store`: `low/mid/high` (scaled 0..1), seconds since each band's onset (`store.{low,mid,high}_onset_pulse_t`), `bpm`. Share shader snippets (projection vertex stage, `band()` color function, fog) from a `fft3d_styles/common.js` rather than copy-pasting.
- Loosen the core only where a style needs it (e.g. camera params, `T_SUB`); don't pre-generalize.

### 2. URL selection

- `?fft3d=<name>` picks the style, e.g. `http://127.0.0.1:8766/?fft3d=synthwave`. Missing or unknown → `cloud` (log a console warning for unknown).
- Resolve once at startup in `main.js` and pass the style name to `makeFft3d(canvas, style)`. Dynamic `import()` of just that module is fine (no build step).
- The style is **UI-only and not persisted** — do *not* add it to the server config/meta. (Whether the card is in 3D at all stays server-authoritative via the existing 2D/3D toggle; don't force it from the URL — the repo rule is that the UI never holds state that can drift from the server.)
- Show the active style name in the FFT card title mode text, e.g. `3D · synthwave · scaled 0..1`.
- Nice-to-have: a tiny `ui/styles.html` gallery page — six links, one line of description each.

## The five styles

Each must be **immediately distinguishable** from the others and from `cloud`. Hex colors below are starting points — tune by eye. All knobs go as named constants at the top of the style module.

### `aurora` — curtains of light
- **Vibe:** northern lights. No solid surface at all: every spectrum hangs a curtain of light *downward* from its ridge line, glowing brightest at the ridge and dissolving toward the floor.
- **Render:** for each mesh row, a vertical strip from ridge (`h`) down to the row's baseline (reuse the curtain idea from `cloud`'s front edge, but for every row). Additive blending, no depth. Fragment: alpha = `pow(v, 2.5)` along the curtain height × a slow vertical shimmer (`sin` of `u·40 + time·0.7` and a hash noise, a few % amplitude) so the light ripples like fabric.
- **Palette:** ignore the band colors; height-driven gradient deep violet `#3b1d6e` → teal `#1fd1b2` → pale green `#b8ffcf` at peaks. Background near-black blue `#04060d` with a faint star field in the static layer (a few hundred 1-px dots, seeded).
- **Post:** soft bloom (see "Shared bloom" below). This style needs it most.
- **Reactivity:** low-band onset → brief global brightness swell (+20 %, 250 ms decay).
- **Camera:** default.
- **Watch out:** ~190 overlapping additive curtains saturate to white at the back (rows bunch up) — scale alpha by `s(z)²` or by row spacing on screen.

### `topo` — survey map
- **Vibe:** a glowing topographic map of the sound, seen from higher up. Calm, precise, cartographic.
- **Render:** opaque surface with a **hypsometric tint** (elevation colors) plus **iso-height contour lines** computed in the fragment shader: `fract(h · N_LEVELS)` with `fwidth` antialiasing, every 5th line thicker (index contour). Faint lat/long grid: vertical lines at octave frequencies (30, 60, 125 … Hz), horizontal lines every 1 s of history. Tiny labels on the 1 s lines in the static layer (only at the right edge).
- **Palette:** sea `#0b2a3c` (h≈0) → lowland green `#2f6b4f` → ochre `#b89a5a` → rock `#8a7f78` → snow `#f2f0ea` at peaks; contour lines dark `#0a0d10` at 60 % on light areas, light on dark areas.
- **Camera:** much higher viewpoint — depth takes most of the screen height and peaks are shorter: e.g. `BACK_SCALE 0.55`, `AMP_FRONT 0.22`, horizon near the top. Soft hill-shading from the existing slopes (light from top-left, the classic map convention).
- **Reactivity:** none — this one is the calm baseline.
- **Watch out:** contour lines moiré at the back where rows compress; fade line contrast out as `fwidth` grows: `line *= 1 - smoothstep(0.15, 0.4, fwidth(q))` where `q` is the contour coordinate (lines closer than ~3–6 px vanish instead of shimmering).

### `synthwave` — outrun grid
- **Vibe:** 80s retro-futurism. Neon wireframe landscape rolling toward a striped sun on the horizon.
- **Render:** solid near-black faces (`#07030f`) with **hidden-line removal** (faces write depth, lines test it), and a neon grid drawn in the fragment shader on the surface: lines along frequency (every k mesh columns) and along time (every n history rows, anchored to the row key so they scroll with the data). Line color by height: magenta `#ff2bd6` low → cyan `#20e3ff` high. Front edge: thick glowing cyan rim.
- **Static layer:** gradient sky (`#12002b` → `#3a0a4f` at the horizon), a big half-sun at the horizon center with the classic horizontal cut stripes (gradient `#ffd23f` → `#ff3d81`), faint horizontal scanlines over the whole card.
- **Post:** bloom (strong, on the lines and sun).
- **Reactivity:** low onset → sun pulses (scale +4 %) and grid brightness flashes; `bpm` > 0 → a slow scanline roll synced to the beat period.
- **Camera:** slightly lower and wider than default (e.g. `BACK_SCALE 0.22`) for a more dramatic horizon.
- **Watch out:** the sun lives in the static layer but must pulse — either redraw it per frame on the 2D canvas under the GL composite (cheap), or draw it in GL.

### `engraving` — woodcut on paper
- **Vibe:** an old engraving / banknote print. Inverted: dark ink on warm paper. The only light-background style.
- **Render:** opaque surface whose shading is done purely with **hatching lines** running along the time direction (one line family, spaced in *screen* space ~3 px, or anchored to `u`), where line **thickness** = darkness from the diffuse light (lit slopes → hairlines, shadowed slopes → fat lines, merging to solid ink in deep shadow). Optional second, crossing hatch family only in the darkest areas (cross-hatching). Front edge: crisp solid ink line, no curtain.
- **Palette:** paper `#efe6d2` with subtle grain (hash noise in the static layer, ±3 % luminance), ink `#1c1a17`. One accent optional: a faint sepia wash `#b08a5a` at 15 % on peaks.
- **Fog:** instead of fading to dark, rows fade to paper color toward the horizon (like atmospheric perspective in etchings).
- **Reactivity:** none, or ink "bleed" (line thickness +10 %) on low onsets — try it and judge.
- **Camera:** default.
- **Watch out:** thin lines alias badly — compute line coverage with `fwidth`-based smoothstep; test at dpr 1 and 2. The axis labels in the static layer need dark text on this background.

### `embers` — sparks off the spectrum
- **Vibe:** a fire. The spectrum is a glowing ridge of heat; peaks throw off sparks that drift back and up with the history, cooling as they go. The only style with real **particle dynamics** (state that evolves over time, not just a function of the mesh).
- **Render:** (a) a dim, smoky surface (low-alpha, dark red-brown `#2a0f08`, no lighting) as the ground; (b) a particle system: each frame, spawn particles at the front edge with probability ∝ local height² (so peaks emit most), plus a burst of ~200 on each onset from that band's frequency range. Each particle: position (u, z, h), velocity (moves back with the history at exactly `1/depth` per second in z, rises slowly in h, small random sideways drift via curl-ish noise), age. Color by age (blackbody ramp): white `#fff4d6` → yellow `#ffc34a` → orange `#ff6a1a` → deep red `#7a1206` → gone. Point size shrinks with age and perspective. Additive blending + bloom.
- **Implementation:** CPU-simulated particles in a preallocated `Float32Array` ring (cap ~20–30k), uploaded as one `bufferSubData` per frame; simulate with frame `dt` (clamped). No per-frame allocation. (GPU transform feedback is overkill.)
- **Reactivity:** is the whole point — onsets per band emit bursts from that band's frequency range (use `meta.bands` edges to map to `u`).
- **Camera:** default.
- **Watch out:** particles must be keyed to real time, not frame count (the UI refresh-rate slider changes the draw rate). Kill particles past `z > 1`.

## Shared bloom (used by aurora, synthwave, embers)

Put it in `fft3d_styles/common.js` as an opt-in helper: render the scene into a multisampled renderbuffer → blit to a texture → bright-pass → 2–3 downsampled separable Gaussian blur levels → additive composite over the scene into the default framebuffer. Recreate FBOs on resize. Keep strength / threshold / radius as per-style parameters. If the helper costs > ~1 ms GPU at full screen on an M2, drop to fewer levels.

## Constraints (from the repo's rules — don't break these)

- **No build step, no framework, no new dependencies**: plain ES modules, raw WebGL2.
- **No DSP in the UI beyond display smoothing** (the existing core smoothing is the allowance). Styles only *draw*; they must not reshape what the 2D view or OSC shows.
- **No per-frame allocation** in hot paths (`draw`, `render`, particle sim): preallocate typed arrays, reuse buffers. Same discipline as the server hot paths (see `CLAUDE.md` → "Coding conventions").
- **Perf budget:** the FFT card's paint cost is shown live in the side panel (Performance → paint cost → `fft` row, CPU ms per `draw()`). Keep each style ≲ 3 ms average at full-screen size on the dev M2. Measure with the tab visible — a hidden/background tab throttles rAF and gives garbage numbers.
- **WebGL context loss** must still recover (the core already handles it; styles must re-run `init` on restore).
- Keep the static layer's axis labels legible in every style (text color per style).

## Verification

- `cloud` after the refactor is pixel-for-pixel the same as before (compare screenshots).
- Open all six URLs side by side with music playing; each is distinct at a glance.
- Check each at card size and full screen, dpr 1 and 2, with 64 / 256 / 1840 FFT bins, History slider at 2 s and 30 s, raw-dB on and off, FFT disabled (must show "FFT disabled"), and after a server restart (reconnect path).
- No console errors; `fft` paint cost within budget.

## Open questions for Xander (ask after building, don't block on them)

- Which styles to keep, and should the choice then become a persisted, server-side UI setting with a picker in the card title (like the 2D/3D toggle)?
- Should styles be allowed to override the band colors (aurora/topo/synthwave do), or should everything stay on the L/M/H palette for consistency with the rest of the UI?
