# Rust port of the audio server

**Status:** phases 0–3 done: the `dsp` crate matches every golden (FFT bit-exact) and the Rust `audio-server` binary runs the unmodified UI with the same wire contracts and passes the 10-minute zero-drop soak on the M2. **Next: phase 4 (Pi validation + cutover).** **Scope:** replace the whole Python server (`server/`) with a Rust binary that behaves the same. The browser UI (`ui/`), the config files (`configs/*.yaml`), the WS protocol and the OSC protocol stay unchanged.

## Why

Measured on the M2 in phase 0 (48 kHz, 256-sample blocks, example config, BlackHole drum loop; full numbers and the rerun commands are in [`reference/perf_baseline.md`](../reference/perf_baseline.md)):

| | Python, live server (avg / p95) | Same Python stage, offline single thread (mean) |
|---|---|---|
| DSP per block (5.33 ms budget) | 264 / 662 µs | 29 µs |
| FFT per hop (10.67 ms budget) | 403 / 893 µs | 53 µs |
| Whole process | 8.5% of one core idle, 11.8% with a UI client | — |

- **Where the live cost goes.** The numerics are already near their floor even in Python. Live, ~90% of the cost is contention: GIL hand-offs between the CoreAudio callback thread, the two workers and the asyncio loop, plus thread wake-ups.
- **Tail failures.** With a UI client streaming, the Python server reproducibly stalls for 50–75 ms a few times per 30 s, with ~25 DSP / ~10 FFT drops.
- **Pi 4 projection.** The same contention is projected to reach a large fraction of a core, with p95s past the block period, so dropouts are likely.
- **What Rust buys.** Real parallelism with no GIL, a few µs per stage, and tight tails. It also deletes a whole class of Python-only workarounds: the allocation audit traps, the private scipy `_sosfilt` probe, `sys.setswitchinterval`, the double-buffered no-copy tricks and the PortAudio re-init hack.

The Pi has no baseline yet; measure it in phase 4 (below).

## Decisions

- **Same repo, sibling directory, Python frozen as the reference and deleted at cutover.** Both versions live side by side only while the port is in flight. They share `ui/`, `configs/` and the wire contracts. Python is the *golden reference* that the parity harness compares Rust against. Do not maintain two engines long-term. After cutover the Python *engine* (`server/`) goes, and git history keeps it. `pip install` survives as the packaging for the Rust binary (phase 5).
- **Python feature freeze for the duration.** The Python server gets bug fixes only. Any change to DSP semantics lands in both engines, or waits. UI-only changes are fine as long as the wire format is untouched.
- **Pure port first, improvements after.** No behaviour changes, no new knobs, no DSP "fixes" during the port, even obvious ones. Every deviation from Python breaks parity testing. Log improvement ideas under "After cutover" below.
- **The UI must work unmodified.** That is the acceptance test for the control plane. If the UI needs a change, the Rust server is wrong.
- **Real-time worker scheduling in the Rust engine (decided 2026-10-24, deviation from the reference).** The pure port's QoS policy dropped blocks in every 10-minute soak (workers left unscheduled for 30–70 ms); the macOS time-constraint policy (Linux: `SCHED_FIFO` when permitted, else nice −5) gave 0 drops in 4 soaks at no CPU cost. Default, no knob. Parameters, fallbacks and why not `os_workgroup`: [`reference/rust_server.md`](../reference/rust_server.md) → Worker scheduling. Linux untested until phase 4.
- **Bug fixes that went into both engines during the port:** phantom `fft_drops` after an FFT reconfigure (the catch-up skip over the ring's history is no longer counted; hop alignment and goldens unchanged). Rust-only robustness fixes (WS liveness, panic-safe dispatch, `peak_smear_oct` ≤ 4, probe queueing, macOS overrun counting, preset list) are listed in `reference/rust_server.md`.
- **`--no-ws` persists (decided by Xander, 2026-10-24):** "fine if this gets saved into the config". The flag lands on `cfg.ws.enabled` in both engines, so a save during a `--no-ws` run writes `ws.enabled: false`. No change.
- **Legacy config migrations are dropped (decided by Xander).** The Rust server does not port the `beat:` → `onset:` schema migration (`server/config.py:_build_config` and the preset loader) or the old `./config.yaml` + `presets/` → `configs/` layout move (`server/main.py:_migrate_legacy_config_layout`). It reads only the current schema. Phase 4 converts the existing files first (see below). *Correction (phase 0 audit):* the Python loader is **not** per-field-fallback. A missing required key or an unparseable file counts as corrupt (back up + reseed from the example); an invalid value is a fatal startup error; a short list of optional keys (including the whole `onset:` block) take defaults. Port exactly that. The list is in [`reference/engine_contracts.md`](../reference/engine_contracts.md) → Config. Without the migration, a file that lacks `onset:` simply gets the onset defaults, as Python does today when there's no `beat:` either.
- **This plan introduces test infra** (a parity harness plus golden fixtures). CLAUDE.md says to ask before scaffolding tests. Approving this plan is that approval, but only for the harness described here.

## Layout

```
server/            Python, frozen reference → deleted at cutover
engine/            Cargo workspace (new)
  Cargo.toml
  dsp/             lib crate: pure DSP. No IO, no threads, no allocation after construction.
                   Every stage from FilterBank to FFTPostProcessor, plus OnsetTracker. Replay-testable.
  server/          bin crate `audio-server`: audio input, ring, workers, OSC, WS + HTTP,
                   config / validate / persist, dispatcher, device management, CLI
  dsp/tests/       parity tests against goldens
tools/signals.py   deterministic test signals (numpy only); kept
tools/parity/      make_goldens.py + offline.py (import `server`; deleted at cutover),
                   record_wire.py (engine-agnostic wire capture/compare; kept)
tools/bench/       live_bench.py (engine-agnostic; kept), stage_bench.py (Python-only; deleted),
                   results/ (committed baseline JSON)
testdata/golden/   goldens: DSP .npy + manifest.json (gitignored, ~20 MB, regenerate) + wire/ + README (tracked)
ui/  configs/      shared, unchanged
```

## Crate choices (verified in phase 0 on rustc 1.95, built on the pinned 1.99.1 — don't relitigate)

Checked 2026-10-24 with throwaway test projects. The versions are the ones that resolved; pin the major.

| Need | Pick | Notes |
|---|---|---|
| Audio input | `cpal` 0.18 | **API changed vs older docs:** `DeviceTrait::name()` is gone (use `to_string()` / `id()`), streams need an explicit `play()`, `build_input_stream` takes the config by value, and the error type is unified. Request `SampleFormat::F32` explicitly (the default can be I24/I32). CoreAudio honours `BufferSize::Fixed(256)` exactly, but it sets the buffer **device-wide**. On ALSA it's only a request (`set_period_size_near`), so keep the re-blocking accumulator. One xrun error fires at stream start; don't treat it as fatal. The device list isn't cached (live enumeration, so the PortAudio re-init hack goes away). There's no "devices changed" event, so keep the 1 s watchdog. Physical hot-plug is untested. |
| Real FFT | `realfft` 3.5 (`rustfft` 6.4, NEON on by default) | vs `numpy.fft.rfft`: ~3e-16 relative error, ~1e-12 dB. Not bit-exact, so after the f32 cast expect an occasional 1-ULP (~4e-6 dB) difference, well inside tolerance. `process_with_scratch` doesn't allocate, but it **overwrites its input**, so copy the window in every hop. 1024-point: 3.8 µs on the M2. |
| Butterworth SOS design | `sci-rs` 0.4.1 | **Matches scipy: no hand port.** 96/96 cases (sr 16k/44.1k/48k × order 1/2/4/8 × 8 bands) have the same section count and order; worst coefficient error 3.6e-15, worst effect on block RMS 3.3e-12. scipy's `iirfilter` uses `zpk2sos(pairing='nearest')`, and sci-rs ports it. Caveats: it panics on invalid Wn (validate/clamp first; it runs on the control thread anyway), allocates, and pulls in ~50 crates (nalgebra/ndarray). If it's ever dropped, the hand port is `buttap → lp2bp_zpk → bilinear_zpk → zpk2sos(nearest)`, ~250 lines; the traps are `_cplxreal`'s `100·eps·|z|` tolerance and first-index argmin tie-breaking. The goldens' `filter_design_sos.npy` is the check. |
| Async / WS / HTTP | `tokio` 1.54 (current-thread) + `axum` 0.8 (`ws`) + `tower-http` 0.7 `ServeDir` | Smoke-tested together: WS text/binary to a Python client, plus `SetResponseHeaderLayer::overriding(CACHE_CONTROL, "no-cache")`. Keep **two ports** (WS 8765, HTTP 8766). |
| JSON / YAML | `serde_json` 1.0 (features **`preserve_order`, `float_roundtrip`**) + **`serde-saphyr` 2.0** | serde_yaml is archived. `serde_yaml_ng` / `serde_norway` are rejected: they write floats like `1e16` / `1e-7`, which PyYAML reads back as **strings** (1442 of 6020 test floats). serde-saphyr had 0 failures and round-trips both configs bit-exactly. Gotcha: parsing into a dynamic `Value` rejects `.inf` by default (typed f64 fields accept it). Load into a `Value` first and validate field by field to reproduce the corrupt-vs-invalid split (Config in `reference/engine_contracts.md`). |
| Worker → WS stores | `triple_buffer` 9 for FFT frames; atomics / small `Mutex` for L/M/H | Replaces the Python double-buffer ref trick entirely. |
| Control → worker | per-worker SPSC command queue (`rtrb` 0.4) | crates.io's "newest" shows 0.3.5 (a backport); 0.4.0 is current. |
| Enforce no-alloc | `assert_no_alloc` 1.1 in debug builds around the callback and the worker hot paths | Its default `disable_release` feature makes it a no-op in release. `AllocDisabler` exists only under `debug_assertions`, and each test binary needs `#[global_allocator]`. Verified that it aborts on a deliberate allocation. |
| Thread priority | `libc` 0.2 (`pthread_set_qos_class_self_np` / `thread_policy_set` on macOS, `pthread_setschedparam` / nice on Linux) | Main thread: same policy as `server/priority.py`; workers: real-time class (Decisions). Avoid the `libc` 1.0 alpha. |
| CLI / logging / open browser | `clap` 4, `tracing` 0.1 + `tracing-subscriber` 0.3, `webbrowser` 1 | Same flags: `--open --device --no-ws --config --log-level`. |
| Parity tests / benches | `npyz` 0.9 (reads the goldens), `criterion` 0.8, `loom` 0.7 (slot-ring), `hound` 3.5 (replay bin WAV) | |
| Pi builds | `cross` or `cargo-zigbuild` → `aarch64-unknown-linux-gnu` | ALSA dev libs in the sysroot. The toolchain is rustup, pinned in `engine/rust-toolchain.toml` (1.99.1 + clippy/rustfmt); in phase 4 add `targets = ["aarch64-unknown-linux-gnu"]` there. |
| Packaging | `maturin` 1.15 (`bindings = "bin"`), `maturin-action`, `rust-embed` 8, `directories` 6 | Phase 5: the pip wheel carries the binary; `ui/` + the example config are embedded. |

## Threading model and module map

As built: [`reference/rust_server.md`](../reference/rust_server.md) (threads, command queues, module ↔ Python source table). The outward contract is [`reference/engine_contracts.md`](../reference/engine_contracts.md).

## Parity traps (read before porting a stage)

Each of these is a place where a "reasonable" reimplementation diverges from the Python reference:

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
18. **Both smoothers are asymmetric** (added after this plan was first written). `ExpSmoother` has per-band `tau` (release) and `tau_attack`, and picks attack iff `rms − v > 0.0` (strict). The FFT post-processor's per-bin smoother does the same, using `tau_attack_per_bin` interpolated exactly like `tau_per_bin`. The first frame copies `smooth = lin` with no smoothing.
19. **Onset `abs_floor` is per band.** `thr_high = max(sens·thr_avg, abs_floor)`, `thr_low = max(K_LOW·thr_avg, 0.5·abs_floor)`.
20. **`tau_to_alpha` floors tau at 1e-3 s**; OnsetTracker uses its own `max(tau, 1e-3)` and `dt = blocksize / max(sr, 1)`.
21. **Geometry guards:** `log_bin_centers` uses `max(f, 1e-3)` and `log_span ≥ 1e-6`. AutoScaler log-bin geometry uses `log2(max(nyq/f_min, 1+1e-6))` and `max(oct, 1e-3)`. `build_log_bin_map` uses `f_max = 2·f_min` when `sr/2 ≤ f_min`. Smear is off when `f_max ≤ f_min`.
22. **`FFTPostProcessor.reset()` resets smooth + peak + warm-up only.** `peak_smoothed` and the sentinel LUT survive. `reconfigure()` rebuilds everything, and the LUT is rebuilt from the next frame.
23. **`load_preset` always reconfigures the FFT** through the `fft.n_bins` and `fft.window` sections, even with unchanged values, which resets all FFT post-processing state. The goldens' `script_drums` event at block 1100 records the exact call sequence.
24. **Unknown WS message types get an error with no snap-back meta.** Only handler `ValueError`s add the targeted meta (see `testdata/golden/wire/`).

## Parity harness (built in phase 0)

The parity harness is how "same behaviour" gets proven. Everything below exists and runs. The format and replay contract are in [`testdata/golden/README.md`](../../testdata/golden/README.md); read it before writing the Rust parity test.

- **DSP goldens:** `uv run python tools/parity/make_goldens.py` writes `testdata/golden/` (27 cases, ~20 MB, deterministic; `--check` proves it byte for byte).
  - **Driver:** it drives the real engine offline through `tools/parity/offline.py`, which is `App` with the device/network edges stubbed. Each block goes through `DSPWorker.process_block`, each hop through `FFTWorker.process_hop`, and control changes go through the real `Dispatcher`.
  - **Recorded:** per block `raw`/`scaled`/`onsets`/`bpm`; per hop `raw_db`/`processed` (f32), plus a hop table; `derived_*` internal state (bin map, taus, alphas, kernel, noise budget, SOS) for localising divergence; and a standalone filter-design table (sr × order 1–8 × 9 band pairs).
  - **Matrix:**

    | Group | Cases |
    |---|---|
    | Default config × signal | 7 signals: drums, sweep, white, pink, silence, step, clip |
    | Strength | 0 / 0.5 / 1 |
    | Tilt, smear | tilt 0, smear 0 |
    | Log bins | 16 / 256 / 1024 |
    | Filter order | 1 / 4 / 8 |
    | FFT window | 512 / 4096 |
    | Blocksize | 128 / 512 |
    | Sample rate | 16 k (bands clamp), 44.1 k |
    | Noise floor | 0 (white), 0.01 (pink) |
    | Scripted run | 10 s, 19 real WS messages (incl. a partial `load_preset`, FFT off/on, `n_bins` changes) |

  - **Event calls:** each scripted event records the DSP-level setter calls it caused. The Rust dsp-crate replay therefore needs no dispatcher, and the server crate's dispatcher test can check message → calls separately.
- **Deviation from the original plan:** the input signals are stored next to the goldens as f32 `.npy` rather than regenerated in Rust, because numpy's RNG (PCG64 + ziggurat normals) can't be reproduced there.
- **The DSP goldens are gitignored** (decided by Xander: ~20 MB of binary data stays out of the repo). They live only on disk; regenerate them with `make_goldens.py` while the Python engine exists. Only `testdata/golden/README.md` and `wire/` are tracked. The Rust parity tests must **skip with a clear message** (not fail obscurely) when `manifest.json` is missing, e.g. on a fresh clone. **Before cutover deletes `server/`, archive the generated set** (e.g. a tarball attached to a GitHub Release), because it can't be regenerated afterwards. Generator: `tools/signals.py`.
- **Tolerances** (in the manifest): L/M/H raw + scaled + BPM ≤ 1e-9 abs; onsets identical (the only allowed exception is a documented threshold-edge case); FFT raw dB ≤ 1e-4 dB with identical sentinel positions; FFT processed ≤ 1e-6; SOS ≤ 1e-12.
- **Wire goldens:** `uv run tools/parity/record_wire.py` captures from a live server into `testdata/golden/wire/`: every WS message type (4 error flavours), both binary FFT modes, all 8 OSC packets, the persisted main.yaml and a saved preset. For the Rust engine: `--out <dir> --server-cmd <rust binary>`, then `--compare testdata/golden/wire <dir>` diffs JSON key sets/types, OSC layout, FFT headers and YAML shape.
- **Live bench:** `uv run tools/bench/live_bench.py` is engine-agnostic and runs a Rust binary via `--server-cmd`. See [`reference/perf_baseline.md`](../reference/perf_baseline.md).
- The generated goldens must be archived before cutover (above). `tools/parity/make_goldens.py`, `offline.py` and `tools/bench/stage_bench.py` import `server` and go with it. `record_wire.py`, `live_bench.py` and `tools/signals.py` are engine-agnostic PEP 723 scripts and stay.

## Phases

Order matters; each phase's acceptance gates the next.

### Phases 0–1: harness + `dsp` crate at parity (done)

- Python-side prep: the `process_block` / `process_hop` / `_build_fft_pipeline` extraction, DSP + wire goldens, benches and crate picks. The DSP goldens are gitignored; run `make_goldens.py` on a fresh clone.
- `engine/` is a Cargo workspace (`dsp` lib + `server` bin `audio-server`, a stub until phase 2). The toolchain is pinned in `engine/rust-toolchain.toml`.
- `dsp` passes the whole 27-case matrix: FFT raw dB / processed bit-identical, L/M/H ≤ 6.4e-13, onsets + BPM identical. `assert_no_alloc` is clean on `process_block`, `process_hop` and every realtime setter. Stage benches sit next to Python in `reference/perf_baseline.md`, and the `replay` example reproduces a golden from a WAV.
- **Constraint for phase 2:** allocating work is built on the control thread: `FilterDesign::new`, `Smear::build`, and a whole new `FftChain` for any FFT reconfigure (n_bins / window / hop / f_min / sr). All the other setters are alloc-free and may run on the worker.

Done — see [reference/rust_dsp.md](../reference/rust_dsp.md).

### Phases 2–3: realtime spine + control plane (done)

- `engine/server` = the `audio-server` binary: cpal input with a re-blocking callback, a correct-seqlock slot ring, DSP/FFT worker threads that own their chains and take commands over rtrb queues (allocating objects built on the control thread, returned for dropping), triple-buffer stores, byte-identical OSC, strict config load + atomic persister, the dispatcher with all handlers and section-by-section `load_preset`, three-lane WS clients, static HTTP, hot-switch with rollback, watchdog. One tokio current-thread control loop owns all mutable state and every timer.
- Verified: `record_wire.py --compare` 27/27 ok; Python and Rust statistically identical on the same live input; YAML loads both ways; full live bench with zero drops and CPU ~1/4 of Python's (numbers in `reference/perf_baseline.md`). **10-minute zero-drop soak: met** with real-time worker scheduling (Decisions); after a fresh-eyes review, robustness fixes and a UI re-check in Chrome.
- Constraints for phase 4: device indices are positions in cpal's device list (= PortAudio's on CoreAudio; unverified on ALSA, see open question 2); `BufferSize::Fixed(blocksize)` is requested only when the device's range allows it, and ALSA re-blocking is untested on hardware.
- Still to do by hand (Xander): unplug/replug a USB mic while running (watchdog → reopen by name → recovery).

Done — see [reference/rust_server.md](../reference/rust_server.md).

### Phase 4: Pi validation + cutover
- Cross-compile for `aarch64-unknown-linux-gnu` (the target's std is installed on the M2; `priority.rs` type-checks for it). Check on the Pi that the workers get `SCHED_FIFO` or fall back to nice −5 (startup log `priority:` lines). Run Python and Rust on a Pi 4 (and a Pi 5 if one is available) with the same config, and record `server_status.perf` plus `top` CPU for both in this doc.
- **Targets** (Pi 4, default config, UI open): total CPU < 10% of one core, DSP p95 < 0.5 ms, zero drops in a 10-minute soak.
- Pre-cutover: load and re-save `configs/main.yaml` and every preset once with the Python server, so all files are in the current schema before the Rust server (which has no migrations) becomes the only reader. Then `grep -l "^beat:" configs/*.yaml` must return nothing.
- Cutover: `make start` → `cargo run --release -p server -- --open` (or a built binary). Rewrite the README quick start for building from source (Rust toolchain instead of uv); phase 5 later puts `pip install` back on top. Rewrite the CLAUDE.md architecture section for the Rust layout. Promote the threading model + parity traps into `docs/reference/engine.md`. Archive the generated goldens (see the harness section), then delete `server/`, `tools/parity/make_goldens.py` + `offline.py`, `uv.lock`, `realtime_audio_server.egg-info`. Keep `pyproject.toml`; phase 5 rewrites it for maturin. Move this plan to `docs/finished/` once phase 5 is done too.

### Phase 5: distribution (pip + standalone binaries)
Decided by Xander: offer **both**. `pip install` / `uvx` stays the friendly "just Python" path, and standalone binaries go to people without Python. **One build serves both.** maturin with `bindings = "bin"` packages the compiled Rust `audio-server` binary *inside* a platform wheel. `pip install realtime-audio-server` then installs a ready-to-run `audio-server` command with no Rust toolchain and no Python engine. The same binaries are attached to a GitHub Release. Python users therefore keep the exact install story they have today (`pip install`, `uvx realtime-audio-server --open`) while running the Rust engine.

- `pyproject.toml` is **rewritten for maturin** (`build-backend = "maturin"`, `bindings = "bin"`, `manifest-path = "engine/server/Cargo.toml"`), not deleted. Keep the package name `realtime-audio-server` and the command `audio-server`, so existing users' commands don't change. Check that the name is free / owned on PyPI before the first upload.
- **Self-contained binary:** embed `ui/` and `configs/main.example.yaml` into the binary (`rust-embed` or `include_dir`). A wheel install has no repo next to it.
- **Config location:** a repo checkout keeps `<repo>/configs/main.yaml`, the current behaviour of `server/main.py:_resolve_config_path`. An installed binary uses the platform config dir (`directories` crate: `~/Library/Application Support/realtime-audio-server/`, `~/.config/realtime-audio-server/`), seeded from the embedded example on first run. `--config` overrides both. Detect "repo checkout" by a `configs/` dir beside the workspace root, resolved at build time via `CARGO_MANIFEST_DIR`, falling back to runtime lookup.
- **CI release matrix** (GitHub Actions, `maturin-action`): macOS arm64 + x86_64, Linux x86_64 + aarch64 (manylinux, which covers the Pi on 64-bit Raspberry Pi OS), Windows x86_64. Each job uploads the wheel to PyPI and the raw binary (tar/zip) to the GitHub Release. Optionally add `cargo-dist` for the Release side if it's simpler than hand-rolled steps.
- **Trap:** cpal on Linux links `libasound.so.2` dynamically. It isn't on the manylinux allowlist, so tell maturin/auditwheel to leave it external rather than vendor it (every Linux desktop and the Pi has ALSA). Document `sudo apt install libasound2` for minimal images. macOS and Windows have no extra system deps, unlike today's PortAudio requirement.
- **Done when:** on a clean machine with no Rust toolchain, `uvx realtime-audio-server --open` and `pip install realtime-audio-server && audio-server --open` both bring up the UI on macOS and on a Pi. The README quick start shows `pip install` / `uvx` first and the binary download second.

## Open questions (for Xander)

1. **Python library API?** The pip package above gives Python users the *server*: OSC/WS out, same as today. Should it *also* let people `import` the DSP from their own Python scripts, e.g. feed numpy blocks in and get L/M/H, onsets and FFT bins back? That would mean PyO3 bindings on the `dsp` crate, shipped in the same wheel. *Recommendation:* not in this sprint. Add it after cutover if someone asks; the `dsp` crate is designed pure and IO-free, so wrapping it later is cheap.

2. **Device identity on Linux.** Saved configs reopen the input **by name**. cpal's ALSA device names will likely differ from PortAudio's (PortAudio embeds `(hw:2,0)`; unverified), so on the Pi a Python-era `audio.device.name` may not match after cutover, and the server falls back to the system default once. cpal also offers a stable `id()` (CoreAudio UID / ALSA id). *Recommendation:* pure port: keep name matching; on a miss, fall back to default as today; the user re-picks once. Storing the `id` too would be a schema change, so do it after cutover if it bites. Not blocking before phase 4.

## After cutover (not this sprint)

Ideas the port unlocks, and reference quirks the port reproduced on purpose. Do not implement them during the port:
- Smaller `blocksize` (128/64) for lower latency, now that per-block overhead is ~µs.
- Multi-channel input / more bands.
