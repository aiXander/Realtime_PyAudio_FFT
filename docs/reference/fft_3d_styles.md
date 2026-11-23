# 3D FFT view: spectro stream, core, styles, and the UI palette

Map of the 3D FFT card and the UI-wide color palette. Read this before adding a style, changing the 3D core or the spectro stream, or touching colors anywhere in `ui/`.

The 3D view does **not** show the LED-oriented FFT the 2D view shows (`/audio/fft`, smeared, smoothed and autoscaled per bin). It has its own analysis made for looking, the **spectro stream**: a multi-resolution STFT on the server (a frame every ~5 ms, 1024 log bins 20 Hz–20 kHz, raw dB), sent only to clients that ask for it (`set_spectro`). The stream is raw: no smoothing on the server. The UI averages each frame in power over the last ~40 ms, keeps the history in one GPU texture, and every pixel of every style reads that texture directly.

## Files

| File | Role |
|---|---|
| `engine/dsp/src/spectro.rs`, `engine/server/src/spectro.rs` | Rust: the analysis (window ladder, log-bin taps, unit tests) and the worker thread + queue + wire encoder. |
| `server/dsp/spectro.py` | Python twin of both (numpy, same maths, same wire). `server/io/ws_server.py` handles `set_spectro` and sends the frames. |
| `ui/src/viz/fft3d_data.js` | **Data side, no DOM or WebGL (runs in node):** `decodeSpectro`, `makeIntegrator` (power average over time), `makePlayhead` (the front edge's clock), `makeLevel` (the dB mapped to full height). |
| `ui/src/viz/fft_3d.js` | **Core.** CPU ring + R8 ring texture of integrated frames, grid index buffer, WebGL context (loss/restore), static layer (background, style art, axis labels), GL overlay canvas, CPU + GPU perf. Knows nothing about looks. |
| `ui/src/viz/fft3d_styles/common.js` | The **style contract** (header comment) and shared shader snippets: `SPEC_FN` / `SPEC_FS` (texture reads), `GRID_FN` + `SURF_VS` (grid vertices), `PROJ_FN`, `bandFn`, `RAMP_FN`, `LINE_FN`, `LOG_PAPER_FN`, `HASH_FN`, `FLAT_VS`; `program()`, `setCommonUniforms()`, front-edge strips, `drawSecondLabels()`. |
| `ui/src/viz/fft3d_styles/index.js` | Registry: `STYLE_ORDER` (dropdown order), `STYLES`, `DEFAULT_STYLE`. |
| `ui/src/viz/fft3d_styles/<name>.js` | One module per look: `cloud` (palette-colored dust), `topo` (survey map), `spectrogram`, `partials` (peak tracks). Look constants sit at the top of each. |
| `ui/src/main.js` | `syncSpectro()`: subscribes while the 3D view shows live data; the 3D title-bar controls (resolution, smooth, range, tilt; `controls.js`, localStorage). |
| `ui/src/colors.js` | **Palettes** + the mutable `LMH` / `LMH_HEX` / `theme` objects every viz reads. |
| `server/control/validate.py` | `FFT3D_STYLES` / `PALETTES` name lists — **keep in sync** with the registry and `PALETTE_ORDER`. |

Wire format and message fields: [`engine_contracts.md`](engine_contracts.md) → WebSocket.

## The spectro analysis (server)

- **Window ladder.** Every hop runs Hann windows of 512 … 8192 samples (at 44.1–48 kHz; the ladder scales with sr), zero-padded 2×, all **centred on the same instant** (the middle of the longest window), so an onset lines up across frequency. The price is ~85 ms of latency, fine for watching.
- **Per log bin**, the wanted window is `sr / (f · rel_res)` (constant Q), clamped to the ladder; the bin crossfades in dB between the two nearest windows, so there are no seams. A log bin wider than a padded rfft bin takes the **max** of the bins it covers (tonal peaks survive); a narrower one interpolates.
- **Amplitude-normalised** (a sine reads its dBFS level in any window). With constant-Q windows, pink noise then reads flat between the clamps; the ladder is clamped above ~2 kHz, so treble falls off and the UI's default tilt is +3 dB/oct.
- **`res`** picks `rel_res`: `time` 0.08, `balanced` 0.045, `pitch` 0.02. Hann's main lobe is ~2.5 bins, so `pitch` splits a semitone above ~200 Hz (pinned by `resolves_a_semitone_at_440` in `spectro.rs`). One analysis serves every subscriber; the last request wins.
- **Hop** = the whole number of blocks nearest 5 ms (256 samples at 48 kHz / 256-block).
- **Cost** (M2, live): Rust ≈ 0.1–0.3 ms per hop (2–5 % of a core); Python ≈ 1 ms per hop (~20 %, mostly GIL contention), no dropped blocks. Only while a 3D view is subscribed; the thread idles otherwise.
- **Not realtime, not parity-tested.** The worker polls the slot ring every 4 ms at normal priority (the ring keeps ≥ 2 × 16384 samples), so the callback doesn't know it exists and the LED path (DSP, FFT, OSC, goldens) is untouched. It is deliberately outside `server_status` and `meta` (their key shapes are compared by the wire-parity tool); its cost goes out as a separate `spectro_perf` message.

## The UI core

- **Integration (`makeIntegrator`).** Each frame entering the ring is the power average of the last N raw frames per bin (a boxcar; the **Smooth** picker: raw / 20 / 40 / 80 ms, default 40). This is Welch averaging: a tone keeps its level, noise-like content stops being a new random draw every 5 ms. It is in the UI, not the server, because it is display taste: the picker changes it live, both engines stay untouched, and a uniform boxcar adds nothing visible to the bass (its windows are already 85–170 ms).
- **History = one R8 texture**, `n_bins` wide, `TEX_ROWS` (8192) tall, used as a ring (row = seq mod 8192, REPEAT wrap, so any float row samples correctly, negative ones too). Each draw uploads only the rows that arrived. The dB → height mapping (range, tilt, level) is uniforms: changing it re-maps the whole history at once.
- **Level (`makeLevel`):** the dB mapped to full height is the loudest bin of the last 20 s plus 3 dB, approached with a 0.15 s rise and a 4 s fall, never below −50 dBFS. Steady music keeps hitting the same peak, so the mapping stands still; a quiet passage shorter than 20 s stays low (it shows relative level).
- **Playhead (`makePlayhead`):** frames arrive in bursts (one message per broadcast tick). The largest "seq − arrival time" seen is the stream's clock; how far that clock had run past the newest held frame at each arrival is the lead the playhead needs. It advances by wall-clock time alone and trails the clock by the largest recent need, so its speed is constant. It never passes the newest frame, and snaps when more than 0.4 s off (reconnect, tab return). `draw(now)` takes the animation-frame timestamp.
- **Geometry vs. shading.** The grid (default 512 × 256 vertices; a style may set `grid`) only carries the relief: the vertex shader reads heights from the texture, pooling the data rows each grid row stands for. Grid rows are **anchored** to data rows (`step` whole rows apart) and glide with the playhead, plus one live front row at the playhead; sampling at moving positions instead made the surface crawl. Fragment shaders call `specSurface` (in `SPEC_FS`), which reads height and normal per pixel, pooled over the rows / bins under the pixel (`fwidth`), so detail is at data resolution. Pooling is halfway between mean and max of the taps (`poolMix`).
- **Subscription:** `syncSpectro` subscribes while connected, synced, in 3D view, FFT enabled and the tab visible. The FFT enable toggle stays the one "off" switch. The server forgets subscriptions on disconnect, so `spectroSent` resets then.

## Selection and persistence

Both choices are server-authoritative UI settings, same pattern as the 2D/3D toggle: dropdown → WS `set_fft3d_style {style}` / `set_palette {palette}` → `cfg.ui.fft3d_style` / `cfg.ui.palette` (persisted in YAML) → meta `ui_fft3d_style` / `ui_palette` → `controls.syncMeta` sets `store.fft3d_style` and calls `applyPalette()`. The style dropdown sits in the FFT card title and only shows in 3D view. The palette dropdown is in the top bar. Unknown names in YAML fall back to the default (`_or_default` in `server/config.py`), so removing a style never breaks config loading.

The 3D view's own settings (resolution, smooth, range, tilt) are UI-local: `localStorage` key `fft3d_display`. Resolution is a `set_spectro` field; smooth applies to frames from then on; range and tilt are display uniforms only.

## Style contract (short version, details in `common.js`)

A style default-exports `{ name, label, description, camera?, grid?, background, text?, drawStatic?, create(gl, f) }`. `create` returns `{ render(f), dispose() }`.

- **Only one instance is alive.** The core disposes the old instance on a switch, and recreates it after a context restore.
- `f` is one preallocated bundle mutated per frame: geometry (`geo`), grid (`R` rows, `mc` cols, `surfVao`, `triCount`, `rowDz`), front edge (`frontH`, `edgeX/edgeY`, `yBaseF`), band stops (palette), `t` (wall clock), and `spec` (every texture/grid uniform, set by `setCommonUniforms`). The texture is bound on unit 0 before `render()`. `render()` must not allocate.
- The grid VAO has **no attributes**: vertex shaders work from `gl_VertexID` through `gridVertex()` (u, texture row, z, row key, pool). `SURF_VS` outputs `vU, vH, vZ, vKey, vS, vT, vPx`; `vT` is the texture row to pass to `specSurface` / `specH`.
- `camera: { backScale, ampFront }` reshapes the projection (uniforms `uK`, `uBackScale`).
- `specSurface` uses `fwidth`, so it lives in `SPEC_FS` (fragment only). Including it in a vertex shader fails to compile ("no matching overloaded function").
- Lines on a uniform time grid use the depth: `gridLine(vZ · depthS)` (1 s parallels).

## Decisions and traps

- **Never draw the raw frames.** A short-window bin of noise-like sound is chi-square distributed (~±5 dB) and independent from one 5 ms frame to the next above ~2 kHz. Drawn raw it is speckle, a front edge that dances at the display rate, and shimmer as the picture scrolls (each pixel's taps slide over uncorrelated rows). Max-pooling raw rows made it worse: the max of six random draws is biased ~4 dB up and still random. Measured on a music file (40 ms vs raw): treble frame-to-frame change 3.2 → 0.5 dB, scroll shimmer down 3–8×, largest treble onset rise in 21 ms 40 → 34 dB.
- **A boxcar, not a one-pole, for the integration.** A one-pole in power rises fast but falls only 4.3 dB per time constant: 25 ms leaves a ~350 ms tail behind every click on a 60 dB scale.
- **The level must not follow single peaks.** The mapping is one uniform for the whole history, so every change re-maps everything on screen. An instant-attack / 6 dB/s-release top moved ~9 dB/s in steady music: the floor dropped on every kick and crept back. Any new scheme has to hold still for material that keeps its level.
- **The playhead must not read past the newest row.** Rows beyond it hold data from 8192 frames ago; with linear filtering even row `lastSeq + 0.9` is mostly stale.
- **Shaders can be compiled without a browser:** build each style against a fake `gl` that records `shaderSource`, swap `#version 300 es` for `310 es`, run `glslc --target-env=opengl`. The data module runs in node as is.
- **Styles were pruned to the ones that show data (2026-10-23).** Aurora, synthwave, engraving and embers (bloom, particles, audio-reactive effects) were removed as gimmicky and low on detail, and later `waterfall` (hidden-line traces per row) because it looked bad; the bloom post-process, `heightTex` texture upload, `audio` / `bandU` / `dt` frame fields and the `camera.horizon` sky went with them. If an effect comes back, git history has them (commit before this change). The bar for a new style: it must show *more* of the data than cloud/topo, in one opaque pass, no post-processing.
- **The detail-first styles** use `RAMP_FN`: a palette colormap with luminance forced to rise monotonically (like viridis/inferno), so magnitude reads correctly whatever the palette's own colors are. Height is dB-linear over the display range, so `spectrogram` maps `level = h^0.9` with no gain.
  - `partials`: per pixel, the slope over ±1 bin crosses zero going down at a peak; the line is drawn at that crossing (`|sx| / |∇sx|` = px distance). Raw spectra ripple everywhere, so a peak must also be narrow (`σ ≈ √(h / −h'')` below `PEAK_OCT`) **and prominent**: ≥ ~3 dB above both flanks at ±8 bins (`PROM_BINS`, about a main lobe). With the flanks at ±4 bins, real harmonics failed the gate and the noise didn't.
- **Perf panel:** an `fft_gpu` row (WebGL timer query around `render()`; the `fft` row is CPU only and hides fill-rate cost) and a `spectro` row (server worker, from `spectro_perf`). Card size (≈ 760×340 CSS px @2×, M2): spectrogram ≈ 4.5 ms GPU, partials ≈ 5–6, CPU ≈ 0.35 ms. A 1024 × 480 grid cost twice that with no visible gain: per-pixel shading carries the detail. These numbers double when another GPU app is busy.
- **UI frame-rate traps.**
  - *Compositing:* the GL canvas is a DOM overlay (`canvas.viz-overlay` in `.viz-stack`), not `drawImage`d into the 2D canvas. The per-frame copy stalled the GPU pipeline: 3D ran at ~35 fps, 2D at 60. The overlay is hidden (`showGl(false)`) whenever no GL frame is drawn (empty states, 2D view via `fft3d.hide()`), because a WebGL canvas keeps showing its last frame.
  - *Shared canvas:* 2D and 3D each have their own `makeSurface` on the same canvas, so `fit()` also compares against its own `s.w/s.h`. Otherwise the view that didn't do the resize draws at a stale size after full screen / tiling changes.
  - *Perf panel:* the bars animate `transform: scaleX`, not `width`, and `.perf-num` is a fixed box with `contain: strict`. A `width` transition re-laid-out the whole side panel on every frame it ran (5 Hz status × 11 rows): ~70 ms/s of layout + paint on the main thread, in both views.

## Palette

`applyPalette(name)` mutates `LMH` / `LMH_HEX` / `theme` in place, bumps `theme.version`, and sets CSS custom properties on `:root` (`--low/--mid/--high`, `--accent*`, `--bg-*`). DOM and CSS follow automatically: `freq_axis.js` uses inline `var(--low)` styles, and the brand logo does the same. Canvas viz that cache colors key their caches on `theme.version`: `lmh_bars` flash ramp, `lmh_lines` gradients, `lmh_scene` sprites/fills, `fft_2d` bucket ramp + band overlays, the 3D static layer and band stops. A palette sets the band colors, accent, background tint and the 2D spectrum ramp (`ramp: null` = the original formula). The palette also drives **every 3D style**. Shaders read `uC0/uC1/uC2` (low/mid/high, declared via `PAL_UNIFORMS`, set by `setCommonUniforms`). Static art uses `pal()` / `palBg()` from `common.js`. Each style maps its roles onto the palette: cloud = band colors across frequency; topo sea = low, lowland = mid, highland = high; spectrogram / partials = `ramp()` (hue low → mid → high, lightness-ordered). A style's `background()` is re-read only when the style or palette changes (`theme.version`), never per frame. The top-bar picker is a custom listbox (`setupPalettePicker` in `controls.js`) because a native `<select>` can't show swatches. The scene viz caps saturation at the palette color's own, so `mono` stays grey.
