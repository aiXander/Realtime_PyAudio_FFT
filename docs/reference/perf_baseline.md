# Performance baseline + how to measure an engine

These are the measured load and latency of the Python engine on the M2. The same tools rerun unchanged against the Rust engine (`docs/TODO/rust_port.md`), so the two can be compared number for number. Raw JSON for every run is in `tools/bench/results/`.

## Tools

| Tool | What it measures | Engine-agnostic? |
|---|---|---|
| `uv run tools/bench/live_bench.py [--server-cmd CMD] [--set key=val …] [--phases idle,ui] [--out F]` | A real server process on private ports, fed a deterministic 120 BPM drum loop through **BlackHole 2ch** (installed on this Mac). Two phases: **idle** (no WS client) and **ui** (one client streaming like an open tab). Per phase: process CPU (% of one core, from CPU-time deltas), per-thread CPU (`ps -M`; key `ordinal:priority`, `97R` = CoreAudio realtime IO thread), OSC rates + inter-arrival jitter, the `server_status` stage timings / e2e / drops, and **kick → `/audio/onset/low` latency** (from the kick's first sample on the BlackHole DAC clock to OSC arrival). | yes (talks CLI/OSC/WS only; PEP 723 deps) |
| `uv run python tools/bench/stage_bench.py [--quick] [--out F]` | Offline, single-thread µs per stage of the Python classes (filter bank, RMS, smoother, autoscaler, onset, full block; window+rfft, power/log-bin/dB, post-processor, full hop; store/OSC publish; WS JSON encodes) plus the "math floor" (bare SOS kernel / rfft). | no (imports `server`; goes away with it) |

The stage names are the shared vocabulary: the Rust `criterion` benches should use the same ones.

## Live, M2 (8 cores), 48 kHz / 256-sample blocks, BlackHole drum loop, 30 s per phase (2026-10-24)

| Config | Phase | CPU (% 1 core) | Busiest threads (%) | DSP avg / p95 (µs) | FFT avg / p95 (µs) | lmh e2e avg / p95 (µs) | `/audio/lmh` gap p99 / max (ms) | kick→onset p50 / p95 (ms) | drops dsp / fft |
|---|---|---|---|---|---|---|---|---|---|
| default (example config: 128 bins, 1024/512, order 2) | idle | **8.5** | workers 3.5 + 3.3, CoreAudio IO 1.6 | 264 / 662 | 403 / 893 | 588 / 1094 | 11.0 / 44 | 18.3 / 22.8 | 0 / 0 |
| default | ui | **11.8** | workers 3.7 + 3.4, main (asyncio) 3.0, IO 1.7 | 313 / 719 | 480 / 847 | 717 / 1316 | 12.0 / **74** | 18.0 / 23.0 | **26 / 10** |
| heavy (1024 bins, 4096/512, order 8, WS 120 Hz) | idle | 12.8 | workers 6.8 + 4.2, IO 1.6 | 219 / 432 | 721 / 1476 | 455 / 796 | 10.9 / 72 | 33.2 / 37.8 | 0 / 0 |
| heavy | ui | 17.6 | workers 6.1 + 3.9, main 6.0, IO 1.6 | 252 / 469 | 759 / 1357 | 535 / 1023 | 11.0 / 55 | 33.2 / 37.7 | 9 / 4 |

**Rust engine** (`engine/target/release/audio-server`, same bench, same machine, same day; JSON `tools/bench/results/live_rust_{default,heavy}.json`). Thread keys are `ps -M` ordinal:priority; the two busiest `31T` are the DSP / FFT workers, `0:31T` the main (tokio) thread, `97R` the CoreAudio IO thread.

| Config | Phase | CPU (% 1 core) | Busiest threads (%) | DSP avg / p95 (µs) | FFT avg / p95 (µs) | lmh e2e avg / p95 (µs) | `/audio/lmh` gap p99 / max (ms) | kick→onset p50 / p95 (ms) | drops dsp / fft |
|---|---|---|---|---|---|---|---|---|---|
| default | idle | **1.8** | workers 1.0 + 0.6, IO 0.2 | 26 / 43 | 28 / 39 | 38 / 66 | 6.3 / 11.3 | 15.4 / 17.5 | 0 / 0 |
| default | ui | **2.4** | workers 1.1 + 0.6, main 0.6, IO 0.2 | 46 / 104 | 51 / 100 | 74 / 144 | 6.2 / 13.1 | 15.3 / 17.5 | 0 / 0 |
| heavy | idle | 3.0 | workers 1.7 + 1.2, IO 0.2 | 72 / 140 | 201 / 329 | 118 / 179 | 6.5 / 19.4 | 28.7 / 30.8 | 0 / 0 |
| heavy | ui | 4.3 | workers 1.8 + 1.3, main 1.0, IO 0.1 | 60 / 124 | 171 / 326 | 108 / 178 | 6.9 / 15.1 | 29.2 / 30.9 | 0 / 0 |

- **vs Python:** ~1/4 of the CPU (2.4% vs 11.8% with a UI client), stage times ~6–10× lower live, no tail stalls and no drops. The live DSP/FFT numbers include the OSC `sendto`s, which is why they sit above the offline criterion numbers below.
- **Delivery and latency:** CoreAudio honours the fixed 256-frame buffer for the Rust stream, so blocks arrive one per 5.33 ms (`/audio/lmh` p50 5.33 ms) instead of Python's 2-block bursts, and kick→onset is ~3 ms lower. The heavy config's extra ~14 ms is still the order-8 group delay.

**10-minute soaks (Rust, default config, ui phase, 600 s).** JSON: `live_rust_soak_qos.json`, `live_rust_soak_rt_experiment.json`.

| Worker scheduling | Runs | drops dsp / fft | worst block→processing lag | Notes |
|---|---|---|---|---|
| QoS USER_INITIATED (the reference's `priority.py` policy, what ships) | 3 | 16 / 6, 6 / 2, 48 / 21 | 27 ms | load avg 2–4 from other processes; each run had 1–5 drop events |
| time-constraint (macOS real-time), experiment only, not shipped | 1 | **0 / 0** | **0.09 ms** | same machine and load |

- **Cause, instrumented:** the CoreAudio callback was perfectly regular (max inter-callback gap 5.46 ms, no bursts) while both workers occasionally went unscheduled for ~30–70 ms; the ported backlog policy then skips blocks and counts them. A scheduling-class problem, not DSP cost. Whether to adopt real-time worker scheduling is an open question in the port plan.
- **The bench's `/audio/lmh` "max gap"** includes stalls of its own Python receiver thread: the real-time run still printed a 67 ms max gap while every block was processed within 90 µs. Use the drop counters and `lmh_e2e` for server-side tails.

How to read the table:

- **Normal delivery.** CoreAudio delivers BlackHole input as bursts of 2 blocks every ~10.7 ms. So an `/audio/lmh` inter-arrival p50 of ~2 ms and a p95 of ~10.7 ms is normal. The expected rate is 187.5 Hz (lmh) and 93.75 Hz (fft).
- **Tail failures.** The **max gaps (44–74 ms) and the drops in the ui phase are real Python tail failures**, reproduced on a quiet machine. The pre-refactor run, `python_prerefactor_default.json`, shows them too. They are rare stalls: a 74 ms gap means ~14 blocks late, and the backlog policy then skips blocks, which shows up as drops.
- **Kick → onset latency.** This includes the BlackHole/CoreAudio input buffering, the DSP and the onset detector's own rise time. The heavy config's +15 ms comes from the **group delay of the order-8 bandpass**. That's algorithmic and will be identical in Rust, so compare engines at the same config.
- **Pre/post refactor.** Before vs after the `process_block` / `process_hop` extraction: 8.6 → 8.5% idle and 10.8 → 11.8% ui, which is within run-to-run noise. The outputs are bit-identical.

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
- **Targets (M2, default config, UI phase):** CPU 2.4% vs Python's 11.8% and `/audio/lmh` max gap 13 ms in the 30 s runs: met. DSP p95 live is ~100 µs rather than single-digit because it includes the OSC sends and the thread wake-up; the pure DSP is 9.4 µs (criterion, below). **Zero drops over a 10-minute soak: not met by the pure port** (below).
- **Pi.** The Pi 4 has no baseline yet. It's measured in phase 4 of the plan.
