# 3D FFT view: core + styles, and the UI palette

Map of how the 3D FFT card and the UI-wide color palette work. Read this before adding a style, changing the 3D core, or touching colors anywhere in `ui/`.

## Files

| File | Role |
|---|---|
| `ui/src/viz/fft_3d.js` | **Core.** History ring, display smoothing, mesh + vertex buffer, WebGL context (loss/restore), static layer (background, style art, axis labels), compositing, CPU + GPU perf recording. Knows nothing about looks. |
| `ui/src/viz/fft3d_styles/common.js` | The **style contract** (header comment), shared shader snippets (`PROJ_FN` projection, `SURF_VS`, `bandFn`, `HASH_FN`, `FULL_VS`, `FLAT_VS`), `program()`, `setCommonUniforms()`, front-edge strips (`fillCurtain` / `fillRim`), `makeBloom()`, `disposeAll()`, seeded `rng()`. |
| `ui/src/viz/fft3d_styles/index.js` | Registry: `STYLE_ORDER` (dropdown order), `STYLES`, `DEFAULT_STYLE`. |
| `ui/src/viz/fft3d_styles/<name>.js` | One module per look: `cloud` (the original, palette-colored), `aurora`, `topo`, `synthwave`, `engraving`, `embers`. Look constants sit at the top of each. |
| `ui/src/colors.js` | **Palettes** + the mutable `LMH` / `LMH_HEX` / `theme` objects every viz reads. |
| `server/control/validate.py` | `FFT3D_STYLES` / `PALETTES` name lists — **keep in sync** with the registry and `PALETTE_ORDER`. |

## Selection and persistence

Both choices are server-authoritative UI settings, same pattern as the 2D/3D toggle: dropdown → WS `set_fft3d_style {style}` / `set_palette {palette}` → `cfg.ui.fft3d_style` / `cfg.ui.palette` (persisted in YAML) → meta `ui_fft3d_style` / `ui_palette` → `controls.syncMeta` sets `store.fft3d_style` and calls `applyPalette()`. The style dropdown sits in the FFT card title and only shows in 3D view. The palette dropdown is in the top bar. Unknown names in YAML fall back to the default (`_or_default` in `server/config.py`), so removing a style never breaks config loading.

## Style contract (short version, details in `common.js`)

A style default-exports `{ name, label, description, camera?, background, text?, drawStatic?, heightTex?, create(gl, f) }`. `create` returns `{ render(f), dispose() }`.

- **Only one instance is alive.** The core disposes the old instance on a switch, so an unused style holds no GPU memory. It also recreates the instance after a context restore.
- `f` is one preallocated bundle mutated per frame: geometry (`geo`), mesh (`meshH`, `meshZ`, `R`, `mc`, `surfVao`, `triCount`), front edge (`edgeX/edgeY`, `yBaseF`), band stops (palette), `bandU` (band edges as u), `audio` (L/M/H, onset ages, bpm), `t` / `dt` (wall clock), `key0`. `render()` must not allocate: pass named functions to `bloom.capture`, not closures.
- `camera: { backScale, ampFront, horizon }` reshapes the projection. `horizon` adds sky above the back row (synthwave). The projection is a uniform (`uK`, `uBackScale`), not baked into shaders.
- `heightTex: true` makes the core upload `meshH` as an R32F texture plus per-row (z, key) as RG32F. Aurora uses this to draw curtains per row instead of the surface grid.
- Anything "anchored to the data" (grid lines, hatching) uses the absolute clock: `phase = (f.t · N / depthS) mod 1`, `line = phase − z·N`. Rows are placed at `z = (now − rowT)/depth`, so this equals the row's arrival time and lines ride the data.

## Decisions and traps

- **Row timing:** `rowT` = the row's *slot close time* (first frame + one slot), fixed when the row opens. It used to be the latest merged frame's arrival, which made rows twitch backward on every merge. Invisible in `cloud`'s dust, but it read as stutter in the crisp-line styles. The open row has `z < 0 → 0` and is the live front edge (no "pin while fresh" hack any more).
- **Bloom is a quarter-res redraw, not a post-process of the full frame.** The style draws its scene normally (MSAA default framebuffer), then `bloom.capture(f, draw)` redraws it into a 1/4-res FBO (`f.dpr` divided, `f.bloomScale = 4`), and `apply()` blurs 3 levels and adds them on. The first version rendered a full-res 4× MSAA offscreen copy: ~170 MB at full screen and visibly laggy. Screen-space strips in a capture must be ≥ `f.bloomScale` px wide or they drop out.
- **Premultiplied alpha over the static layer.** Glow is added with alpha = its brightest channel, so color ≤ alpha. Otherwise the 2D-canvas composite is undefined. Scanlines (synthwave) multiply color and alpha together.
- **Perf panel has an `fft_gpu` row** (WebGL timer query around `render()`). The `fft` row is CPU only and hid every fill-rate problem. Full-screen Retina on the M2, idle GPU: cloud ≈ 4.7 ms, aurora ≈ 5.5, topo ≈ 4.0, synthwave ≈ 6.5, engraving ≈ 2.2, embers ≈ 4.2. These numbers double when another GPU app (e.g. DaVinci Resolve) is busy, so compare against cloud measured at the same time.
- Crisp lines use `fwidth`-based AA and fade to their average tone where they crowd (`lineAt` in topo/synthwave, `hatch` in engraving). Otherwise the far rows moiré.
- `embers` simulates on wall-clock `dt` (clamped to 0.1 s) in a swap-remove `Float32Array` pool (24k), so the UI refresh-rate slider doesn't change the physics. Onsets are detected when a band's `onsetAge` resets.

## Palette

`applyPalette(name)` mutates `LMH` / `LMH_HEX` / `theme` in place, bumps `theme.version`, and sets CSS custom properties on `:root` (`--low/--mid/--high`, `--accent*`, `--bg-*`). DOM and CSS follow automatically: `freq_axis.js` uses inline `var(--low)` styles, and the brand logo does the same. Canvas viz that cache colors key their caches on `theme.version`: `lmh_bars` flash ramp, `lmh_lines` gradients, `lmh_scene` sprites/fills, `fft_2d` bucket ramp + band overlays, the 3D static layer and band stops. A palette sets the band colors, accent, background tint and the 2D spectrum ramp (`ramp: null` = the original formula). The palette also drives **every 3D style**. Shaders read `uC0/uC1/uC2` (low/mid/high, declared via `PAL_UNIFORMS`, set by `setCommonUniforms`). Static art uses `pal()` / `palBg()` from `common.js`. Each style maps its roles onto the palette: cloud = band colors across frequency; aurora hem = mid, rays = low, blush = high; topo sea = low, lowland = mid, highland = high; synthwave grid = low→mid, sun = high→low (Neon = classic outrun); engraving ink tinted by low, wash = high; embers heat ramp = low→high→white (Ember = real fire). A style's `background()` is re-read only when the style or palette changes (`theme.version`), never per frame. The top-bar picker is a custom listbox (`setupPalettePicker` in `controls.js`) because a native `<select>` can't show swatches. The scene viz caps saturation at the palette color's own, so `mono` stays grey.
