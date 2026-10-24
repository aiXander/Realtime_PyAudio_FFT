# Rust `audio-server` (`engine/server`): map, design, verification

The Rust binary that replaces the Python server (`server/`) with the same outward behaviour: same CLI, `configs/*.yaml`, WS protocol, OSC packets and static HTTP, so the **unmodified** UI runs against it. The outward contract it implements is [`engine_contracts.md`](engine_contracts.md); the DSP it drives is the `dsp` crate, [`rust_dsp.md`](rust_dsp.md). The port plan and what's left (Pi validation, cutover, packaging) is [`../TODO/rust_port.md`](../TODO/rust_port.md).

## Run

```bash
make start-rust                                 # cargo build --release -p server, then --open
engine/target/release/audio-server [--config P] [--no-ws] [--open] [--device N] [--log-level L]
cd engine && cargo test -p server && cargo clippy --all-targets
```

The binary finds the repo (for `configs/` and `ui/`) by walking up from its own path to the first directory holding `configs/main.example.yaml`, else the workspace it was built from (`CARGO_MANIFEST_DIR/../..`). Logs go to stderr (ANSI only on a TTY).

## Threads and who owns what

| Thread | Owns | Talks to others via |
|---|---|---|
| cpal callback (CoreAudio/ALSA RT thread) | a re-blocking accumulator per stream | `SlotRing::write_block`, `unpark()` both workers (through a `OnceLock<[Thread; 2]>`, set after the workers spawn) |
| `dsp-worker` | `DspChain` (filter bank, smoother, auto-scaler, onset tracker), `LmhPacker` | command queue in, return queue out, `triple_buffer` → FeatureSnap, OSC sends |
| `fft-worker` | `Box<FftChain>`, `FftPacker` | command queue in, return queue out, `triple_buffer` → FftFrame, OSC sends |
| main (tokio current-thread) | `App`: cfg, stream handle, WS clients, every timer, persister | the only mutator; axum WS/HTTP connection tasks run here too |

- **One control loop.** `App::run` is a single `select!` over {SIGINT/SIGTERM, inbound events, the earliest due timer}. Timers (broadcast tick, 5 Hz status, 0.25 s watchdog, OSC meta, persister, 50 ms filter-retune debounce, 250 ms drop re-baseline) are `Option<Instant>` deadlines on `App`, not tasks, so there's no shared state and no locks. Connection tasks only shovel frames: they send `Event::{Connect, Message, Disconnect}` over an mpsc and drain their client's lanes.
- **Workers own their DSP state; the control loop sends commands** (`DspCmd` / `FftCmd`, rtrb SPSC, drained before every block/hop). Anything that allocates is built on the control thread: a `FilterDesign` (fixed-size, sent by value), a `Smear`, a **whole new `FftChain` from the current cfg for any FFT reconfigure** (n_bins / window / hop / f_min / sr — equivalent to Python's `reconfigure`, including the state reset), and on a hot-switch a `DspCore` (filter + smoother + scaler for the new sr; the onset tracker is kept, `reconfigure` + `reset`). Replaced boxes go back over a return queue (`Garbage`) and are dropped on the control thread.
- **Command ordering traps.** Enabling the FFT pushes `ResetPost` + `Resync` *then* sets `fft_enabled`; the worker drains again after seeing the flag so those always land first. Commands that move a read pointer (`ResetReadPtr`, `ReplaceChain`, `Resync`) make the worker re-snapshot `write_idx` instead of continuing its drain.
- **Filter mirror.** `App.filter_installed` = (bands, order) of the design the worker runs. `set_filter_order` keeps the installed bands and the debounced retune keeps the installed order, exactly like Python's `filter_bank.set_order` / `retune` with a retune still pending.
- **Live values** the workers read per block/hop (master gain, `send_fft`, `send_raw_db`, `fft_enabled`, `db_floor`), drop counters and the six perf rings are atomics in `Shared`.

## Module map (`engine/server/src/`)

| File | Python source | Notes |
|---|---|---|
| `main.rs` | `main.py:main`, `_resolve_config_path` | CLI (clap), repo-root discovery, seeding, fatal `error:`/`hint:` lines; `AllocDisabler` global allocator in debug builds |
| `app.rs` | `main.py:App`, `ws_server.WSServer` (broadcast/status loops) | startup order, the event loop, meta / server_status builders, apply_* bus, hot-switch with rollback, watchdog, presets list |
| `dispatcher.rs` | `control/dispatcher.py` | all 22 handlers; `load_preset` deep-merges over main.yaml *as on disk* and applies section by section |
| `config.rs` | `config.py` | typed schema in dataclass field order; strict load (corrupt vs invalid), `.corrupt-<stamp>` backup + reseed, atomic write, `deep_merge` |
| `validate.rs`, `py.rs` | `control/validate.py`, Python builtins | error strings word for word; `py.rs` = `bool()`/`int()`/`float()`/`str()`/`repr()` semantics over `serde_json::Value` |
| `persist.rs` | `config.Persister` | deadline state machine (1 s drag / 50 ms commit / 250 ms cap) |
| `ring.rs` | `audio/ringbuffer.py` | slot ring, atomics + a correct seqlock (below); stress test |
| `audio.rs` | `audio/{devices,stream,callback}.py` | device list / lookup / probe, stream open (F32, Fixed blocksize when in range, I16/I32 fallback), the realtime callback |
| `workers.rs` | `dsp/worker.py`, `FFTWorker.run` | worker loops, backlog policies, commands, stores |
| `osc.rs` | `io/osc_sender.py`, `osc_publisher.py` | byte-identical templates; header tests against `testdata/golden/wire/osc_*.bin` |
| `ws.rs` | `io/ws_server.py` (`_Client`, connection handling) | the three lanes, axum upgrade handler, binary FFT encoder |
| `http.rs` | `io/http_server.py` | `ServeDir` + `Cache-Control: no-cache` + forced MIME types |
| `shared.rs` | perf arrays, `_errlog.py` | `now_ns` (CLOCK_MONOTONIC, the e2e clock), `PerfRing` + stats, rate-limited worker error log |
| `priority.rs` | `priority.py` | QoS USER_INITIATED (macOS) / nice −5 (Linux) |

## Decisions and why

- **Seqlock done right.** Samples are `AtomicU32` (f32 bits); the writer zeroes the slot's sequence, fences, writes, then publishes `idx+1`. Python's protocol could in principle accept a block lapped mid-copy; this can't. Transport robustness only, not a DSP change.
- **Device index = position in cpal's full device list** (inputs and outputs). On CoreAudio that is PortAudio's numbering (BlackHole 0, MacBook mic 1 on this Mac), so `--device`, `cfg.audio.device.index`, the `devices` list and `set_device` agree with Python-era configs. Names are what's matched on reopen. `hostapi` is reported PortAudio-style ("Core Audio", "ALSA").
- **No PortAudio re-init.** cpal enumerates live, so `reopen` just drops and reopens; the watchdog stays (no device-changed event).
- **Delivery is per block, not bursts.** CoreAudio honours `BufferSize::Fixed(256)`, so `/audio/lmh` arrives every 5.33 ms instead of Python's 2-block bursts every 10.7 ms; kick→onset latency drops ~3 ms. Note the fixed buffer is device-wide on CoreAudio.
- **First hop of a fresh `FftChain` runs under `permit_alloc`**: the sentinel-interp LUT is built from that frame (parity trap 6 in the plan). Every other callback / block / hop runs inside `assert_no_alloc` (aborts in debug builds).
- **Worker panics** are caught per wake (`catch_unwind`), logged at most once a second, and the read pointer advances like Python's `except` branch.
- **YAML** via serde-saphyr with non-finite floats parsed as strings, so `.inf` is "must be numeric" (invalid, fatal) rather than a parse error (corrupt). Output differs cosmetically from PyYAML (`0.00003213…` vs `3.21…e-05`, quoted `"127.0.0.1"`) and round-trips with identical values/types both ways.

## Reference quirks ported on purpose (pure port)

- **`--no-ws` lands on the cfg**, so anything persisted during such a run writes `ws.enabled: false`. Open question for Xander in the plan.
- **Any FFT reconfigure inflates `fft_drops`.** The read pointer resets to 0, and the next wake counts every hop since stream start as skipped (Python: 293 after one `set_n_fft_bins` at ~5 s uptime; it grows with uptime). Logged under "After cutover" in the plan.
- **`main.example` is listed as a preset** when the config dir is `configs/` (the reference globs every `*.yaml` except `main`).
- The startup "could not open input" hint still says `uv run python -m sounddevice`.

## How it was verified (2026-10-24, M2)

| Check | How | Result |
|---|---|---|
| Wire shapes | `record_wire.py --server-cmd engine/target/release/audio-server` then `--compare testdata/golden/wire` | 27/27 ok (also with the debug, alloc-checked binary) |
| Live numerics | Python and Rust side by side on the same BlackHole drum loop, 30 s of `/audio/lmh` each | same packet count; per-band mean / p10 / p50 / p90 within 0.001; BPM median 119.68 both |
| YAML both ways | Python `load_config` on Rust-written main.yaml + preset; Rust `load_config` on the Python goldens (`config::tests`) | identical values and types |
| Load / drops | `live_bench.py` default + heavy (30 s phases) | CPU ~1/4 of Python, zero drops; [`perf_baseline.md`](perf_baseline.md) |
| 10-min soak | `live_bench.py --phases ui --duration 600` | pure port: a few drop events per run from worker scheduling stalls; with an experimental real-time scheduling class: 0 drops (open question in the plan) |
| UI | the unmodified UI in Chrome: every slider, toggles, 2D/3D + styles, palette, raw dB, presets, device switch/refresh/probe, band-edge drag, layout drag, tab hide/show | all round-trip through meta; hidden tab idles the WS loop |

Not verified here (needs hardware hands): physical USB mic unplug/replug recovery, and Linux/ALSA.
