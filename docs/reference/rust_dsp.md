# Rust `dsp` crate: map, parity harness, design decisions

`engine/dsp/` is the pure-DSP library of the Rust engine: a 1:1 port of `server/dsp/` (minus the worker loops). No IO, no threads, no allocation on the per-block / per-hop paths. It reproduces the Python goldens bit for bit on the FFT side and to < 1e-12 on L/M/H (tolerance 1e-9). The port plan, including the parity traps, is in [`../TODO/rust_port.md`](../TODO/rust_port.md).

## Toolchain

rustup, pinned in `engine/rust-toolchain.toml` (1.99.1 with clippy and rustfmt). `~/.zprofile` puts `~/.cargo/bin` ahead of Homebrew, but non-login shells (agent tool shells) may still find Homebrew's `cargo` first. Prefix commands with `export PATH=$HOME/.cargo/bin:$PATH` if `rustc --version` isn't 1.99.1.

## Files

| File | Python source | Notes |
|---|---|---|
| `src/lib.rs` | `App._build_pipeline_for_sr`, `_build_fft_pipeline`, `DSPWorker.process_block`, `FFTWorker.process_hop/reconfigure` | `EngineParams` (the cfg values the DSP reads, bands already clamped) → `DspChain` (filter → RMS → smoother → autoscaler → onset) and `FftChain` (analyzer + post-processor). These are the worker cores the server crate owns. |
| `src/math.rs` | `_math.py` + numpy internals | `tau_to_alpha`, `log_bin_center(s)`, `tilt_db`, plus exact replicas of `np.linspace`/`logspace`/`interp` and numpy's **pairwise summation** (`pairwise_sum` = `np.sum`; `reduce_segment` = one `np.add.reduceat` segment = `a[0] + pairwise(a[1:])`). |
| `src/filters.rs` | `filters.py` | `clamp_band_edges`; `design_bandpass` (sci-rs `iirfilter_dyn`, matches scipy to ~2e-15); `FilterDesign` is a fixed-capacity `Copy` struct (≤ 8 sections per band) so `FilterBank::install` never allocates. The kernel is scipy's DF2T `_sosfilt`. |
| `src/features.rs` | `features.py` | `block_rms`, `ExpSmoother`, `AutoScaler`: scalar n=3 code, line for line. |
| `src/onset.rs` | `onset.py` | The BPM ring is a fixed `[f64; 12]`. `onset_count` is never reset. |
| `src/fft.rs` | `fft.py` | `build_log_bin_map`, symmetric `hanning`, `FftAnalyzer` (realfft plan, contiguous valid-bin runs, f32 wire frame + sentinels). |
| `src/fft_post.rs` | `fft_postprocess.py` | `FftPostProcessor`. All setters are alloc-free except the smear. `Smear::build` is pure and runs on the control thread; `install_smear` swaps it in and returns the old one to drop off-thread. |
| `tests/golden_loader.rs` | | The phase-0 check that every golden array has the documented shape and dtype. |
| `tests/parity.rs` | | The filter-design table, `derived_*` internals per case, and the full replay of all 27 cases including the scripted events. Prints a per-case max-error table. |
| `tests/no_alloc.rs` | | `assert_no_alloc` around `process_block`, `process_hop` and every realtime setter. Needs no goldens. |
| `tests/common/mod.rs` | | Golden loader (`npyz`), `EngineParams` from a manifest case config, error trackers. |
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
- **First-hop LUT allocation is accepted.** The sentinel-interp LUT is built from the first frame after a (re)allocation, as in Python (parity trap 6: the LUT comes from that first frame), so that one hop allocates. Steady state is alloc-free. The server builds a fresh `FftChain` on the control thread for any reconfigure anyway.
- **`clamp`-style `min(1).max(0)`** instead of `f64::clamp` mirrors Python's `max(0, min(1, x))` on NaN. Clippy's `manual_clamp` is allowed crate-wide for that reason.

## Known L/M/H residual

The L/M/H `raw` / `scaled` values differ from Python by ≤ 6.4e-13 (worst: `default_sweep` scaled), against a tolerance of 1e-9. The source is `block_rms`: Python's `np.dot` is a BLAS ddot with a different summation order. The onsets and BPM are identical in all 27 cases.
