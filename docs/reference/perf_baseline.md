# Performance baseline + how to measure an engine

These are the measured load and latency of the Python engine on the M2. The same tools rerun unchanged against the Rust engine (`docs/TODO/rust_port.md`), so the two can be compared number for number. Raw JSON for every run is in `tools/bench/results/`.

## Tools

| Tool | What it measures | Engine-agnostic? |
|---|---|---|
| `uv run tools/bench/live_bench.py [--server-cmd CMD] [--set key=val …] [--phases idle,ui] [--out F]` | A real server process on private ports, fed a deterministic 120 BPM drum loop through **BlackHole 2ch** (installed on this Mac). Two phases: **idle** (no WS client) and **ui** (one client streaming like an open tab). Per phase: process CPU (% of one core, from CPU-time deltas), per-thread CPU (`ps -M`; key `ordinal:priority`, `97R` = CoreAudio realtime IO thread), OSC rates + inter-arrival jitter, the `server_status` stage timings / e2e / drops, and **kick → `/audio/onset/low` latency** (from the kick's first sample on the BlackHole DAC clock to OSC arrival). | yes (talks CLI/OSC/WS only; PEP 723 deps) |
| `uv run python tools/bench/stage_bench.py [--quick] [--out F]` | Offline, single-thread µs per stage of the Python classes (filter bank, RMS, smoother, autoscaler, onset, full block; window+rfft, power/log-bin/dB, post-processor, full hop; store/OSC publish; WS JSON encodes) plus the "math floor" (bare SOS kernel / rfft). | no (imports `server`; goes away with it) |

The stage names are the shared vocabulary: the Rust `criterion` benches should use the same ones.

## Live: Python vs Rust, M2, 48 kHz / 256-sample blocks, BlackHole drum loop (2026-10-24)

Median of 3 interleaved 30 s runs per cell (`compare_queue`-style: python default → rust default → python heavy → rust heavy, ×3). Rust = the shipped build (real-time workers). Heavy = `--set fft.n_bins=1024 --set dsp.filter_order=8 --set fft.window_size=4096 --set fft.hop=512 --set ws.snapshot_hz=120`. Machine load average 1.3–3.4 (other agents). Raw JSON: `tools/bench/results/compare_2026-10-24/`.

| Config | Phase | Engine | CPU (% 1 core) | RSS (MB) | DSP avg / p95 (µs) | FFT avg / p95 (µs) | lmh e2e avg / p95 (µs) | `/audio/lmh` gap p50 / p99 (ms) | kick→onset p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| default | idle | Python | 8.7 | 111 | 218 / 604 | 339 / 753 | 498 / 1006 | 1.20 / 10.96 | 18.5 / 23.0 |
| default | idle | Rust | 2.4 | 20 | 65 / 123 | 72 / 128 | 82 / 145 | 5.33 / 5.49 | 15.3 / 17.5 |
| default | ui | Python | **12.1** | 111 | 291 / 688 | 418 / 756 | 622 / 1129 | 1.49 / 10.95 | 18.3 / 23.2 |
| default | ui | Rust | **3.0** | 20 | 62 / 126 | 69 / 118 | 79 / 153 | 5.33 / 5.59 | 15.4 / 17.5 |
| heavy | idle | Python | 13.3 | 112 | 242 / 550 | 792 / 1662 | 506 / 1063 | 2.77 / 10.84 | 33.5 / 37.7 |
| heavy | idle | Rust | 4.3 | 21 | 54 / 116 | 185 / 421 | 68 / 136 | 5.33 / 5.48 | 29.0 / 31.3 |
| heavy | ui | Python | 20.3 | 112 | 289 / 619 | 920 / 1766 | 617 / 1222 | 1.86 / 10.95 | 32.8 / 37.3 |
| heavy | ui | Rust | 5.3 | 21 | 70 / 130 | 238 / 446 | 86 / 158 | 5.33 / 5.58 | 29.0 / 31.3 |

- **Rust vs Python, default + UI open:** CPU 3.0 vs 12.1% (4×), RSS 20 vs 111 MB, block→OSC 79 vs 622 µs avg (8×), DSP/FFT stage 4–6× lower live. Heavy: 5.3 vs 20.3% CPU.
- **The previous single-run Rust numbers (1.8 / 2.4% idle / UI) were a quieter moment, not a regression:** an interleaved A/B of the pre-review QoS build, the real-time build and the final build gave the same 2.1–2.5% idle / 2.6–3.1% UI for all three (`compare_2026-10-24/ab_cpu/`).
- **No drops in any 30 s run, either engine, this time.** Python's tails show in the soak below (and in 26/10 drops in an earlier 30 s UI run).
- **Delivery:** Rust gets one block per 5.33 ms from CoreAudio (fixed 256-frame buffer); Python gets bursts of 2 blocks every ~10.7 ms (gap p50 ~1.5 ms, p99 ~11 ms). That is also why kick→onset is ~3 ms lower in Rust; the heavy config's extra ~14 ms is the order-8 filters' group delay, identical in both.
- Thread keys (`ps -M` ordinal:priority): Rust workers show `97R` (real-time, same class as the CoreAudio IO thread), main `0:31T`.

**10-minute soaks (default config, UI streaming, 600 s).** JSON: `compare_2026-10-24/soak_*.json` (+ the earlier `live_rust_soak_qos.json`, `live_rust_soak_rt_experiment.json`).

| Engine / worker scheduling | Runs | drops dsp / fft per run | worst p95 block→OSC | CPU |
|---|---|---|---|---|
| Python (reference) | 1 | 31 / 14 (3 events) | 14.2 ms | 11.2% |
| Rust, QoS USER_INITIATED (the reference's policy) | 5 | 16 / 6, 6 / 2, 48 / 21, 35 / 14, 5 / 0 | up to 27 ms | 2.6–3.2% |
| **Rust, real-time (what ships)** | 4 | **0 / 0 every run** | 0.09–0.24 ms | 2.4–3.1% |

- The QoS and real-time A/B runs were interleaved (QoS, RT, Python, QoS, RT) at load average 1.5–3; the fourth real-time run is the shipped binary, the earlier one the first experiment.

- **Cause, instrumented:** the CoreAudio callback was perfectly regular (max inter-callback gap 5.46 ms, no bursts) while both workers occasionally went unscheduled for ~30–70 ms; the ported backlog policy then skips blocks and counts them. A scheduling-class problem, not DSP cost. Whether to adopt real-time worker scheduling is an open question in the port plan.
- **The bench's `/audio/lmh` "max gap"** includes stalls of its own Python receiver thread: the real-time run still printed a 67 ms max gap while every block was processed within 90 µs. Use the drop counters and `lmh_e2e` for server-side tails.

- **Python's tails** (max `/audio/lmh` gaps of 44–74 ms and ~25 / 10 drops in an earlier 30 s UI run, reproduced on a quiet machine and in `python_prerefactor_default.json`) are GIL/thread hand-off stalls; the backlog policy then skips blocks.
- **Pre/post refactor (Python).** Before vs after the `process_block` / `process_hop` extraction: 8.6 → 8.5% idle and 10.8 → 11.8% ui, within run-to-run noise; outputs bit-identical.

## Offline per stage, Python (µs, mean / p99), quiet machine

| Config | DSP block total | filter bank | autoscaler | onset | OSC lmh publish | FFT hop total | window+rfft | power→log-bin→dB | post-proc | OSC fft publish |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 28.9 / 38 | 15.2 | 4.7 | 5.4 | 14.7 | 53.0 / 72 | 9.3 | 15.2 | 31.3 | 8.8 |
| bins256 | 28.8 / 39 | 15.1 | 4.7 | 5.4 | 13.8 | 61.7 / 81 | 9.5 | 17.2 | 35.9 | 8.9 |
| bins1024 | 29.1 / 49 | 15.2 | 4.7 | 5.4 | 13.7 | 213.7 / 253 | 9.5 | 26.3 | 178.2 | 13.0 |
| order8 | 49.4 / 66 | 34.5 | 4.7 | 5.4 | 13.5 | 53.3 / 77 | 9.3 | 16.7 | 27.5 | 9.5 |
| win4096 | 31.2 / 83 | 15.2 | 4.6 | 5.4 | 13.7 | 81.8 / 109 | 29.0 | 25.6 | 27.3 | 9.3 |
| strength1 | 27.5 / 39 | 15.3 | 3.5 | 5.4 | 13.8 | 46.0 / 76 | 9.3 | 16.6 | 20.2 | 8.9 |
| block128 | 23.4 / 32 | 9.9 | 4.6 | 5.4 | 13.2 | 52.7 / 71 | 9.2 | 15.8 | 27.8 | 8.7 |

- **"Math floor":** the bare SOS kernel loop over the 3 bands plus RMS takes 17 µs (order 2) and 37 µs (order 8). The bare window+rfft takes 9.6 µs (1024) and 28.7 µs (4096).
- **WS encodes:** snapshot JSON 5 µs, meta JSON 19 µs, binary FFT 0.7 µs.

## Offline per stage, Rust `dsp` crate (µs, criterion mean)

Re-run on 2026-10-24 for the comparison (`compare_2026-10-24/stages_{python,rust}.json`, the same configs): every Rust cell within noise of the table below; Python's stage bench likewise (default DSP block 28.5 µs, FFT hop 52.5 µs).

`cd engine && cargo bench -p dsp --bench stages`: same stage names and configs as the Python table above. The configs and the drum signal come from the golden cases. Measured 2026-10-24 on the M2 under load from other agents (load average 5–20). Each cell is the best mean over 2–3 runs, so treat ±30% as noise; the strength1 post-proc number is one such outlier. `power→log-bin→dB` is derived as hop total − window+rfft − post-proc, as in Python.

| Config | DSP block total | filter bank | autoscaler | onset | FFT hop total | window+rfft | power→log-bin→dB | post-proc |
|---|---|---|---|---|---|---|---|---|
| default | 9.44 | 8.81 | 0.058 | 0.018 | 8.41 | 2.77 | 2.44 | 3.20 |
| bins256 | 9.45 | 8.54 | 0.057 | 0.018 | 13.35 | 2.76 | 3.62 | 6.97 |
| bins1024 | 9.45 | 8.53 | 0.057 | 0.018 | 78.82 | 2.76 | 19.19 | 56.87 |
| order8 | 12.31 | 11.37 | 0.057 | 0.018 | 7.97 | 2.76 | 1.99 | 3.22 |
| win4096 | 9.44 | 8.55 | 0.057 | 0.018 | 21.14 | 13.78 | 4.05 | 3.31 |
| strength1 | 9.39 | 8.54 | 0.036 | 0.018 | 9.16 | 2.76 | 1.21 | 5.20 |
| block128 | 4.65 | 4.25 | 0.058 | 0.017 | 7.77 | 2.76 | 1.99 | 3.01 |

- **vs Python, default config:** DSP block 29 → 9.4 µs (3×) and FFT hop 53 → 8.4 µs (6×). Glue overhead is gone: the autoscaler and onset stages drop from ~5 µs to < 0.1 µs.
- **What's left:** the filter bank (DF2T biquads, latency-bound per sample) is ~90% of the DSP block. The peak smear dominates the post-proc at high bin counts (1024 bins: σ ≈ 32 bins, 257 taps).

## What this means for the port

- **The work is cheap.** Single-threaded, a full Python DSP block is ~29 µs and a full FFT hop ~53 µs, about 0.5% of a core each. The per-stage numerics are already close to their floor.
- **Contention is the expensive part.** The live server spends ~10× that per stage (264 / 403 µs average) and 8.5–12% of a core in total. The extra goes to GIL hand-offs between the CoreAudio callback thread, the two workers and the asyncio thread; to the 0.5 ms switch interval; and to thread wake-ups. It also produces the rare 50–75 ms stalls that cause drops.
- **What Rust should fix.** The Rust engine's main win is real parallelism and no GIL, which should remove both the contention and those stalls. The per-stage speedup is secondary.
- **Targets (M2, default config, UI phase):** CPU 3.0% vs Python's 12.1% (interleaved medians): met. DSP p95 live is ~100 µs rather than single-digit because it includes the OSC sends and the thread wake-up; the pure DSP is 9.4 µs (criterion, above). **Zero drops over a 10-minute soak: met with real-time worker scheduling** (4 soaks, 0 drops; the pure port's QoS policy dropped in every soak, above).
- **Pi.** The Pi 4 has no baseline yet. It's measured in phase 4 of the plan.
