# Rust `dsp` crate: map, parity harness, design decisions

`engine/dsp/` is the pure-DSP library of the Rust engine: the Rust twin of `server/dsp/` (minus the worker loops). No IO, no threads, no allocation on the per-block / per-hop paths. It reproduces the Python goldens bit for bit on the FFT side and to < 1e-12 on L/M/H (tolerance 1e-9). How it is kept in agreement with Python, and the parity traps: [`dual_engine.md`](dual_engine.md).

## Toolchain

rustup, pinned in `engine/rust-toolchain.toml` (1.99.1 with clippy and rustfmt). `~/.zprofile` puts `~/.cargo/bin` ahead of Homebrew, but non-login shells (agent tool shells) may still find Homebrew's `cargo` first. Prefix commands with `export PATH=$HOME/.cargo/bin:$PATH` if `rustc --version` isn't 1.99.1.

## Files

| File | Python source | Notes |
|---|---|---|
| `src/lib.rs` | `App._build_pipeline_for_sr`, `_build_fft_pipeline`, `DSPWorker.process_block`, `FFTWorker.process_hop/reconfigure` | `EngineParams` (the cfg values the DSP reads, bands already clamped) → `DspChain` (level meter on the raw block, then filter → RMS → smoother → autoscaler → onset) and `FftChain` (analyzer + post-processor). These are the worker cores the server crate owns. |
| `src/math.rs` | `_math.py` + numpy internals | `tau_to_alpha`, `log_bin_center(s)`, `tilt_db`, plus exact replicas of `np.linspace`/`logspace`/`interp` and numpy's **pairwise summation** (`pairwise_sum` = `np.sum`; `reduce_segment` = one `np.add.reduceat` segment = `a[0] + pairwise(a[1:])`). |
| `src/filters.rs` | `filters.py` | `clamp_band_edges`; `design_bandpass` (sci-rs `iirfilter_dyn`, matches scipy to ~2e-15); `FilterDesign` is a fixed-capacity `Copy` struct (≤ 8 sections per band) so `FilterBank::install` never allocates. The kernel is scipy's DF2T `_sosfilt`, run as `run::<N>` (see Decisions). |
| `src/features.rs` | `features.py` | `block_rms`, `ExpSmoother`, `AutoScaler`: scalar n=3 code, line for line. `LevelMeter` (`/audio/level`): its scale is a wire contract, see [`engine_contracts.md`](engine_contracts.md) → OSC. |
| `src/onset.rs` | `onset.py` | The BPM ring is a fixed `[f64; 12]`. `onset_count` is never reset. |
| `src/fft.rs` | `fft.py` | `build_log_bin_map`, symmetric `hanning`, `FftAnalyzer` (realfft plan, contiguous valid-bin runs, f32 wire frame + sentinels). The sentinels are written once at construction; `process_hop` only computes (and takes the `log10` of) the non-empty bins. |
| `src/fft_post.rs` | `fft_postprocess.py` | `FftPostProcessor`. All setters are alloc-free except the smear. `Smear::build` is pure and runs on the control thread; `install_smear` swaps it in and returns the old one to drop off-thread. `prime_lut` builds the sentinel LUT ahead of the first frame. |
| `tests/golden_loader.rs` | | The phase-0 check that every golden array has the documented shape and dtype. |
| `tests/parity.rs` | | The filter-design table, `derived_*` internals per case, and the full replay of all 27 cases including the scripted events. Prints a per-case max-error table. |
| `tests/no_alloc.rs` | | `assert_no_alloc` around `process_block`, every `process_hop` of a fresh chain (first included) and every realtime setter. Needs no goldens; debug builds only (`cargo test --release` skips it). |
| `tests/edge_inputs.rs` | | What no golden contains: a NaN input sample (L/M/H recover) and an `f_min` just below Nyquist (bounded smear). |
| `tests/common/mod.rs` | | Golden loader (`npyz`), `EngineParams` from a manifest case config (or `example_params()`), error trackers. |
| `benches/stages.rs` | `tools/bench/stage_bench.py` | criterion, with the same stage names and config matrix as the Python bench. |
| `examples/replay.rs` | | WAV + config JSON → the golden per-case `.npy` dump, for diffing against Python on arbitrary audio. |

## Commands

```bash
cd engine
cargo test                                   # all; parity tests SKIP (pass) if goldens absent
cargo test -p dsp --test parity -- --nocapture       # per-case max-error table
PARITY_CASE=script_drums cargo test -p dsp --test parity replay -- --nocapture
GOLDEN_DIR=/other/golden cargo test          # point at another golden set
cargo clippy --all-targets                   # must be clean
cargo bench -p dsp --bench stages [-- <filter>]      # stage µs; numbers in perf_baseline.md
cargo run -p dsp --release --example replay -- in.wav config.json out_dir/
```

The DSP goldens are gitignored. Regenerate them with `uv run python tools/parity/make_goldens.py` (and verify with `--check`). The parity tests print `SKIP: goldens not generated …` and pass when `manifest.json` is missing.

## Decisions and why

- **Exact numpy arithmetic where it's cheap.** Matching numpy's pairwise summation (the Hann coherent gain, the per-log-bin power sums, the smear kernel normalisation), its `linspace` endpoint rule and `interp`'s slope formula made the FFT path **bit-identical** to Python after the f32 cast, not just within tolerance. Keep these replicas when touching those paths. A "simpler" sum is still within tolerance, but it loses the exact-match signal that makes divergences obvious.
- **Hann via libm `cos`** matches `np.hanning` exactly on macOS arm64 (checked: 1024/1024 identical).
- **The smear loop is interchanged** (offset outer, output inner) so it vectorises. Each output keeps scipy's summation order (centre, then offsets r…1), so it stays bit-identical. Order-preserving rewrites are fine; reordering a single output's accumulation is not.
- **Filter design on the control thread.** sci-rs allocates and panics on invalid Wn, so `FilterDesign::new` only ever runs after `clamp_band_edges`, off the realtime path.
- **Filter kernel: three bands interleaved, section count at compile time** (`FilterBank::run::<N>`, dispatched on the order). Coefficients and state stay in registers and the three independent recurrences overlap, ~4× faster than band by band at order 2 (8.5 → 2.1 µs per block); order 8 is a wash. Each band's arithmetic is unchanged: `filters::tests` pins it bit for bit against the plain `_sosfilt` loop for orders 1–8 (the L/M/H goldens alone only check 1e-9).
- **The sentinel LUT is primed at construction, so no hop allocates.** Python builds it from the first frame after a (re)allocation (parity trap 6). `FftChain::new` / `reconfigure` call `post.prime_lut(analyzer.wire())` instead: the analyzer's initial frame has sentinels in exactly the bins every real frame has them (valid bins are floored at −120 dB, far above the −500 threshold), so the LUT is the same. A bare `FftPostProcessor` (the parity replay) still builds it lazily from its first frame.
- **Non-finite input and degenerate geometry (both engines; no golden contains them).** A block whose band RMS isn't finite (a NaN/inf input sample) zeros the filter state and counts as silence; before, NaN stuck in the IIR state and `/audio/lmh` sent NaN until the next device switch. The smear σ is capped at `n_bins`: an `f_min` just below Nyquist (validation only requires > 0) used to make σ ~1e8 bins (GBs of kernel, or an abort); every geometry with a ≥ 4-octave span is below the cap even at the maximum smear.
- **Subnormals aren't flushed.** After loud input then digital silence, the DF2T state settles into a subnormal limit cycle and `*_raw` reads ~1e-323 instead of 0. Free on arm64 (M2, Pi: subnormals at full speed, measured); on x86 it would cost 10–100× in the filter bank. If x86 ever becomes a target, set FTZ/DAZ at worker-thread start in the server crate (L/M/H parity tolerance absorbs it).
- **`clamp`-style `min(1).max(0)`** instead of `f64::clamp` mirrors Python's `max(0, min(1, x))` on NaN. Clippy's `manual_clamp` is allowed crate-wide for that reason.

## Crate choices, whole workspace (verified on rustc 1.95, built on the pinned 1.99.1; don't relitigate)

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
| Thread priority | `libc` 0.2 (`pthread_set_qos_class_self_np` / `thread_policy_set` on macOS, `pthread_setschedparam` / nice on Linux) | Main thread: same policy as `server/priority.py`; workers: real-time class ([`rust_server.md`](rust_server.md) → Worker scheduling). Avoid the `libc` 1.0 alpha. |
| CLI / logging / open browser | `clap` 4, `tracing` 0.1 + `tracing-subscriber` 0.3, `webbrowser` 1 | Same flags: `--open --device --no-ws --config --log-level`. |
| Parity tests / benches | `npyz` 0.9 (reads the goldens), `criterion` 0.8, `loom` 0.7 (slot-ring), `hound` 3.5 (replay bin WAV) | |
| Pi builds | `cross` or `cargo-zigbuild` → `aarch64-unknown-linux-gnu` | ALSA dev libs in the sysroot. Runs on the Pi; one open check (real-time worker priority) is in the repo `CLAUDE.md`. |
| Packaging | `rust-embed` 8, `directories` 6 | Not built and not scheduled: Rust is built from source, pip ships the Python engine. See [`../TODO/engine_ideas.md`](../TODO/engine_ideas.md). |

## Known L/M/H residual

The L/M/H `raw` / `scaled` values differ from Python by ≤ 6.4e-13 (worst: `default_sweep` scaled), against a tolerance of 1e-9. The source is `block_rms`: Python's `np.dot` is a BLAS ddot with a different summation order. The onsets and BPM are identical in all 27 cases.
