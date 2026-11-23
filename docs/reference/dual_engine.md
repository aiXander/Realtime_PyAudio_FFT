# Two engines: Python and Rust, kept in agreement

The repo ships two permanent audio-server engines with the same outward behaviour: `server/` (Python) and `engine/` (Rust, the default and what Xander runs day to day). They share one `ui/`, one `configs/`, one CLI, one WS protocol and one OSC protocol, so a consumer (WZRD, the LED controller, the browser UI) can't tell which engine it talks to. The engines differ only in their core. Python is the **DSP reference**: the parity goldens are generated from it and the Rust engine is tested against them. Someone cloning the repo installs one engine (`uv sync` or the Rust toolchain); `make start` runs Rust, `make start-python` runs Python.

## Where the code lives

| What | Where |
|---|---|
| Outward contract both engines implement (CLI, config, WS, OSC, HTTP, devices, timing) | [`engine_contracts.md`](engine_contracts.md) |
| Rust DSP crate / server binary | [`rust_dsp.md`](rust_dsp.md), [`rust_server.md`](rust_server.md) (both carry a Python-source ↔ Rust-module table) |
| DSP golden generator: drives the real Python `App` offline | `tools/parity/make_goldens.py`, `tools/parity/offline.py` |
| Rust replay of the goldens | `engine/dsp/tests/parity.rs` (+ the server crate's dispatcher/config tests) |
| Live wire capture and shape compare, engine-agnostic | `tools/parity/record_wire.py` (`--engines` = capture both, compare) |
| Golden format and replay contract | [`testdata/golden/README.md`](../../testdata/golden/README.md) |
| The one command that proves agreement | `make check` |

## The rule

- **A behaviour change lands in both engines in the same commit**, followed by `make check`. That covers DSP semantics, a new knob or WS/OSC message, config schema, and bug fixes with an outward effect. Only an engine-internal change (a Rust perf rewrite, a Python allocation fix) touches one side.
- **A change to Python DSP semantics regenerates the goldens.** `make check` does it every run (they are gitignored local files), and the Rust parity test then fails until Rust matches. Never hand-edit a golden, and never loosen a tolerance to make a diff go away.
- **UI and wire contracts exist once.** The UI must work unmodified against either engine; if it needs an engine-specific branch, one engine is wrong.
- **Deliberate deviations are documented, not silent.** Each Rust-only difference (real-time worker scheduling, OSC host-name resolution, WS admission/Origin check, robustness clamps) is listed with its reason in [`rust_server.md`](rust_server.md) and marked "Rust engine only" in [`engine_contracts.md`](engine_contracts.md). Add any new one there.
- **Test infra:** the parity harness (`tools/parity/`, `testdata/golden/`, `make check`) is the approved exception to "ask before scaffolding tests".

## What `make check` proves, and what it doesn't

1. `make_goldens.py` regenerates the DSP goldens from current Python: 27 cases (signals × config matrix + a scripted run of real WS messages) of per-block L/M/H raw/scaled/onsets/BPM/level, per-hop FFT raw dB/processed, derived internal state and a filter-design table.
2. `cargo test` replays them in Rust: FFT bit-identical, L/M/H within 1e-9 (actual ≤ 6.4e-13), onsets and BPM identical. It also runs the Rust unit tests.
3. `record_wire.py --engines` starts each engine on the BlackHole loopback (ports 18765/18766/19000, so a dev server on 8765 doesn't collide), plays a drum loop, captures every WS message type, both FFT binary flavours, all OSC packets and the persisted YAML, then compares **shapes**: JSON key sets and types, OSC address/type-tag/payload layout, binary headers. Values aren't compared (live audio differs run to run). Without the `BlackHole 2ch` device the step prints `SKIP` and passes.

**Not covered:** the **spectro stream** (the 3D view's STFT, `server/dsp/spectro.py` / `engine/dsp/src/spectro.rs` + `engine/server/src/spectro.rs`). It is opt-in (`set_spectro`) and not in the DSP goldens or wire captures, so its numbers can drift between engines with nothing failing. Also uncovered: device hot-plug, the audio watchdog, ALSA/Linux, and timing (latency, drops — that is `tools/bench/live_bench.py`, see [`perf_baseline.md`](perf_baseline.md)).

## Parity traps

Places where a "reasonable" reimplementation diverges from the Python reference. Read before changing a DSP stage in either engine.

1. **`np.hanning(ws)` is the *symmetric* Hann** (denominator `ws−1`), not the periodic one. `_win_power_corr = 2 / sum(hann)²`.
2. **Filter design:** `iirfilter(order, [lo/nyq, hi/nyq], btype="bandpass", ftype="butter", output="sos")`. Order N gives N biquad sections. Section pairing and ordering come from `zpk2sos`, and they affect rounding. Compare coefficients against scipy, not just the frequency response. Filter input is the **f32 block cast to f64**. State is DF2T `zi` per section.
3. **`block_rms` = `sqrt(dot(x,x)/n)` in f64** on the filter output.
4. **Log-bin map:** `freqs = linspace(0, sr/2, n_rfft)`, `edges = logspace(log10 f_min, log10 f_max, n+1)`, `searchsorted(side="right") − 1`, clamped to `n−1`. Valid iff `f_min ≤ f ≤ f_max` and idx ≥ 0, so the top edge folds into the last bin. A log bin's value is the *mean of power* over its rfft bins.
5. **FFT wire path order:** power × corr → per-bin mean → `max(EPS)` → `10·log10` → **cast to f32** → *then* write the −1000 sentinel into empty bins. The post-processor consumes that **f32** array, not the f64 bins, so replicate the quantisation.
6. **Sentinel-interp LUT** is built from the *first* frame after (re)allocation (valid iff `db ≥ −500`). Edge bins clamp to the nearest valid bin.
7. **`log_bin_centers`** (used for tilt and per-bin tau) uses `(k+0.5)/n` in log10 space up to `sr/2`. That is a different formula from the bin-map edges; port both. Per-bin tau = `np.interp` over anchors **sorted** by log-centre, clamped outside.
8. **Smear:** kernel radius `max(1, int(4σ+0.5))`, normalised. `scipy.ndimage.convolve1d(mode="reflect")` is *half-sample symmetric* (`d c b a | a b c d | d c b a`) and must work when radius > n_bins (repeated reflection).
9. **Warm-up:** FFT post: the first frame copies `smooth = lin` and resets peak to `nf`, then the peak step *still runs* on that frame. AutoScaler: the first `update` sets peak = `nf`, then follows.
10. **Two different noise-floor variables in AutoScaler:** the gate uses raw `noise_floor` (may be 0), the divisor uses `max(noise_floor, EPS)`. The FFT post uses `max(nf, EPS)` for both and gates the baseline on `interp_db < 20·log10(nf)`.
11. **Setter side effects:** `set_noise_floor` / `update_autoscale(noise_floor)` raise the existing peaks to the new floor. `set_bands` recomputes tilt, log-bin geometry, `_per_bin_factor` and the noise budget. `update_bands` on the FFT side re-anchors tau only. `reconfigure` reallocates and re-warms.
12. **Onset/BPM:** "median" is `sorted[len//2]` (the *upper* median). The outlier filter uses strict `0.5m < v < 2m`. Octave folding uses `while` loops. Silence decay keys off the low band's last fire. `advance()` moves the clock on dropped blocks. `onset_count` is **never** reset (resetting it emits spurious onsets). Comparisons are strict `>` / `<` exactly as written.
13. **Strength branches:** `s ≥ 1` skips the baseline entirely and `s ≤ 0` skips the tanh blend. Keep the branches.
14. **Master gain** multiplies only the *outputs* (OSC `/audio/lmh`, the WS snapshot `low/mid/high`, the processed FFT on both wires). It does not touch raw dB, the `*_raw` fields or the onset detector input.
15. **OSC raw-dB FFT** rewrites sentinels to `db_floor`. OSC is big-endian, WS binary is little-endian (`[1][0][n:u16][f32×n]`).
16. **JSON:** avoid NaN/inf (Python emits a non-standard `NaN` and serde errors). `meta.sr` is an int. Match key sets exactly; the UI reads by key.
17. **Band edges are clamped** to ≤ 0.45·sr at startup, `set_band`, preset load and every hot-switch, and the clamped values are written back to cfg.
18. **Both smoothers are asymmetric.** `ExpSmoother` has per-band `tau` (release) and `tau_attack`, and picks attack iff `rms − v > 0.0` (strict). The FFT post-processor's per-bin smoother does the same, using `tau_attack_per_bin` interpolated exactly like `tau_per_bin`. The first frame copies `smooth = lin` with no smoothing.
19. **Onset `abs_floor` is per band.** `thr_high = max(sens·thr_avg, abs_floor)`, `thr_low = max(K_LOW·thr_avg, 0.5·abs_floor)`.
20. **`tau_to_alpha` floors tau at 1e-3 s**; OnsetTracker uses its own `max(tau, 1e-3)` and `dt = blocksize / max(sr, 1)`.
21. **Geometry guards:** `log_bin_centers` uses `max(f, 1e-3)` and `log_span ≥ 1e-6`. AutoScaler log-bin geometry uses `log2(max(nyq/f_min, 1+1e-6))` and `max(oct, 1e-3)`. `build_log_bin_map` uses `f_max = 2·f_min` when `sr/2 ≤ f_min`. Smear is off when `f_max ≤ f_min`.
22. **`FFTPostProcessor.reset()` resets smooth + peak + warm-up only.** `peak_smoothed` and the sentinel LUT survive. `reconfigure()` rebuilds everything, and the LUT is rebuilt from the next frame.
23. **`load_preset` always reconfigures the FFT** through the `fft.n_bins` and `fft.window` sections, even with unchanged values, which resets all FFT post-processing state. The goldens' `script_drums` event at block 1100 records the exact call sequence.
24. **Unknown WS message types get an error with no snap-back meta.** Only handler `ValueError`s add the targeted meta (see `testdata/golden/wire/`).
25. **Legacy config migrations are Python-only.** Python still ports `beat:` → `onset:` and the old `./config.yaml` + `presets/` layout; Rust reads only the current schema (a file without `onset:` gets the onset defaults). Load and re-save an old config with Python once before pointing Rust at it.

## Runbook

- **`cargo test` parity failure after a Python change:** expected until Rust implements the same change. The failing case and the first diverging `derived_*` array say which stage.
- **Parity tests print `SKIP: goldens not generated`:** fresh clone; run `make check` (or `uv run python tools/parity/make_goldens.py`).
- **Wire compare `DIFF` on a JSON file:** a key was added or retyped in one engine only. Diff the two temp captures the run printed (`…/wire_engines_*/python` vs `…/rust`).
- **Wire capture `MISSING OSC` / timeout:** BlackHole exists but nothing plays into it, or another process holds port 19000.
