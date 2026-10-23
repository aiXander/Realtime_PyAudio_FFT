# Rust port of the audio server

**Status:** not started. This is the next major piece of work. **Scope:** replace the whole Python server (`server/`) with a Rust binary that behaves the same. The browser UI (`ui/`), the config files (`configs/*.yaml`), the WS protocol and the OSC protocol stay unchanged.

## Why

Measured on an M2 (default config: 44.1 kHz, 256-sample blocks, FFT window 1024 / hop 512, 256 log bins, OSC FFT on). The server uses about 10% of one core and has 0 drops. The interesting part is *where* the time goes:

| Stage | Python, full stage | Just the math (microbenchmark) | Overhead |
|---|---|---|---|
| DSP per block (5.8 ms budget) | 350 µs | ~18 µs (3 IIR bandpasses + RMS) | ~95% |
| FFT per hop (11.6 ms budget) | 460 µs | ~30–50 µs (rfft, smear, ~40 small numpy calls) | ~90% |

Almost all of the cost is interpreter overhead and the fixed per-call cost of numpy on tiny arrays. On a Pi 4 that overhead is projected to reach roughly 70–90% of one core. The GIL serialises it onto a single core. The p95 times are about 2.5× the average, which exceeds the 5.8 ms block period on a Pi 4, so dropouts are likely. The projected Rust load is about 3–5% of one core, with tight tail latency and stages that actually run in parallel. The port also gets rid of a whole class of Python-only workarounds: the allocation audit traps, the private scipy `_sosfilt` probe, `sys.setswitchinterval`, the double-buffered no-copy tricks and the PortAudio re-init hack.

These are estimates scaled from the M2. Measure on a real Pi in phase 4 (below).

## Decisions

- **Same repo, sibling directory, Python frozen as the reference and deleted at cutover.** Both versions live side by side only while the port is in flight. They share `ui/`, `configs/` and the wire contracts. Python is the *golden reference* that the parity harness compares Rust against. Do not maintain two engines long-term. After cutover the Python *engine* (`server/`) goes, and git history keeps it. `pip install` survives as the packaging for the Rust binary (phase 5).
- **Python feature freeze for the duration.** The Python server gets bug fixes only. Any change to DSP semantics lands in both engines, or waits. UI-only changes are fine as long as the wire format is untouched.
- **Pure port first, improvements after.** No behaviour changes, no new knobs, no DSP "fixes" during the port, even obvious ones. Every deviation from Python breaks parity testing. Log improvement ideas under "After cutover" below.
- **The UI must work unmodified.** That is the acceptance test for the control plane. If the UI needs a change, the Rust server is wrong.
- **Legacy config migrations are dropped (decided by Xander).** The Rust server does not port the `beat:` → `onset:` schema migration (`server/config.py:_build_config` and the preset loader) or the old `./config.yaml` + `presets/` → `configs/` layout move (`server/main.py:_migrate_legacy_config_layout`). It reads only the current schema. A file in the old shape falls back per field to defaults, like any other invalid value. Phase 4 converts the existing files first (see below).
- **This plan introduces test infra** (a parity harness plus committed golden fixtures). CLAUDE.md says to ask before scaffolding tests. Approving this plan is that approval, but only for the harness described here.

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
tools/parity/      Python golden generator (imports the `server` package); deleted at cutover
testdata/golden/   committed golden fixtures (.npy + manifest.json), kept after cutover
ui/  configs/      shared, unchanged
```

## Crate choices (confirm in phase 0, don't relitigate after)

| Need | Pick | Notes |
|---|---|---|
| Audio input | `cpal` | CoreAudio / ALSA / WASAPI. It enumerates devices live, so the `reinitialize_portaudio()` hack disappears. Callback buffer size is **not** guaranteed to equal `blocksize`; re-block inside the callback (see traps). |
| Real FFT | `realfft` (on `rustfft`) | Has NEON on aarch64. Plan the FFT once per window size. |
| Butterworth SOS design | `sci-rs` if its `iirfilter`/`butter` SOS output matches scipy to ~1e-12; else port scipy's `buttap → lp2bp_zpk → bilinear_zpk → zpk2sos` (~200 lines) | The filter *runtime* is a hand-written DF2T biquad cascade, which is trivial. |
| Async / WS / HTTP | `tokio` (current-thread runtime) + `axum` (WS) + `tower-http` `ServeDir` | Keep **two ports** (WS 8765, HTTP 8766) and `Cache-Control: no-cache` on static files. |
| JSON / YAML | `serde_json`; for YAML a maintained serde_yaml successor | `serde_yaml` is archived. Pick in phase 0 and check that round-tripping `configs/main.yaml` is stable. |
| Worker → WS stores | `triple_buffer` for FFT frames; atomics / small `Mutex` for L/M/H | Replaces the Python double-buffer ref trick entirely. |
| Control → worker | per-worker SPSC command queue (`rtrb`) | See threading below. |
| Enforce no-alloc | `assert_no_alloc` in debug builds around the callback and the worker hot paths | Turns the CLAUDE.md "no allocation" rule into a crash in tests. |
| Thread priority | `libc` (`pthread_set_qos_class_self_np` on macOS, absolute nice −5 on Linux) | Same policy as `server/priority.py`. |
| CLI / logging / open browser | `clap`, `tracing`, `webbrowser` | Same flags: `--open --device --no-ws --config --log-level`. |
| Pi builds | `cross` or `cargo-zigbuild` → `aarch64-unknown-linux-gnu` | ALSA dev libs in the sysroot. |
| Packaging | `maturin` (`bindings = "bin"`), `maturin-action`, `rust-embed`, `directories` | Phase 5: the pip wheel carries the binary; `ui/` + the example config are embedded. |

## Threading model in Rust

Keep the shape of the Python design (`CLAUDE.md` → "Architecture: the threading contract is the design"), with Rust-native plumbing:

- **Callback** (cpal thread): re-block into fixed `blocksize` blocks in a preallocated accumulator → mono mix → write into the slot ring → `unpark()` both workers. No locks, no alloc, no logging.
- **Slot ring:** port `SlotRing` (one producer, two consumers, seqlock). Keep slots rather than an SPSC FIFO, because the FFT reads *overlapping* windows (hop < window). **Trap:** a seqlock read of plain non-atomic data is UB in Rust. Store samples as `AtomicU32` (f32 bits) with `Relaxed` loads and Acquire/Release on the per-slot seq, or use `seqlock`-style volatile copies. Write a loom or stress test.
- **Workers own their DSP state outright.** The DSP worker owns FilterBank, ExpSmoother, AutoScaler and OnsetTracker. The FFT worker owns its FFT plan and FFTPostProcessor. Nothing is shared mutably. The control loop sends commands (`SetNoiseFloor`, `SetBands(prebuilt SOS)` …) over an SPSC queue that the worker drains at the top of each block or hop. Anything that allocates (filter redesign, `n_bins` reconfigure, smear kernel) is **built on the control thread** and sent as a boxed object. The worker sends the old one back to be dropped off-thread. This replaces the Python `_lock`s and "atomic attribute swap" setters. Command application order must reproduce the Python semantics: see the setter side effects in the traps list.
- **Control loop:** one current-thread tokio runtime owns config, dispatcher, WS, HTTP, persister, watchdog and the OSC meta loop. This is the same invariant as today: mutations only happen here.
- **OSC** is sent directly from the worker threads with pre-encoded packet templates, as in `server/io/osc_sender.py`.

## Python → Rust inventory

| Python | Rust home | Notes |
|---|---|---|
| `audio/callback.py`, `audio/ringbuffer.py` | `server/audio/` | Re-blocking is new (PortAudio gave fixed blocks). |
| `audio/stream.py`, `audio/devices.py` | `server/audio/` | Device probe (open each input briefly, check for signal), reopen by **name**. |
| `dsp/filters.py` | `dsp/filters.rs` | `clamp_band_edges` is pure; port it verbatim. |
| `dsp/features.py`, `dsp/_math.py` | `dsp/features.rs`, `dsp/math.rs` | Scalar n=3 code maps 1:1. |
| `dsp/onset.py` | `dsp/onset.rs` | Bounded BPM ring as a fixed array, no alloc. |
| `dsp/fft.py` (bin map + hop math) | `dsp/fft.rs` | Worker loop and backlog policy go in `server/`. |
| `dsp/fft_postprocess.py` | `dsp/fft_post.rs` | |
| `dsp/worker.py` | `server/workers.rs` | Backlog skip / drain-all / `advance()` on drops, as is. |
| `io/osc_sender.py`, `io/osc_publisher.py` | `server/osc.rs` | Byte-identical packets, including `send_silence`. |
| `io/stores.py` | `server/stores.rs` | Monotonic onset counters as `AtomicU64`. |
| `io/ws_server.py` | `server/ws.rs` | Three lanes (control FIFO / latest-only status / drop-oldest stream of 4), coalesced meta, `set_stream`, idle when there are no streamers. |
| `io/http_server.py` | `server/http.rs` | |
| `control/validate.py` | `server/validate.rs` | Single source of truth for ranges, as today. Includes the 9216-byte OSC FFT packet ceiling assert and `MAX_FFT_WINDOW`. |
| `control/dispatcher.py` | `server/dispatcher.rs` | All 22 handlers + `load_preset` section-by-section apply. |
| `config.py` | `server/config.rs` | Load → validate → fall back per field; debounced atomic persister (1 s drag / 250 ms cap / 50 ms commit). |
| `main.py` (App) | `server/app.rs` | Hot-switch with rollback, watchdog, `server_status` perf schema (the UI Performance panel reads it). |
| `priority.py` | `server/priority.rs` | |

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

## Parity harness

The parity harness is how "same behaviour" gets proven. Build it before porting any stage.

- **Python side (`tools/parity/make_goldens.py`):** drives the real `server.dsp` classes offline, block by block, from a deterministic signal plus a config plus a *parameter script* (e.g. "at block 400 set noise_floor=…, at block 900 set bands=…, at block 1200 set n_bins=64"). It dumps per block: `raw[3]`, `scaled[3]`, `onsets[3]`, `bpm`; and per hop: `raw_db[n]` (f32), `processed[n]` (f32). Requires one behaviour-preserving refactor of the reference: extract the loop bodies of `DSPWorker.run` and `FFTWorker.run` into `process_block(block)` / `process_hop(window)` methods, so the golden script calls exactly the production code.
- **Signals** are generated deterministically by the script, so no audio files are committed: a synthetic drum loop at 120 BPM (decaying sine kick, noise-burst snare, hi-passed noise hats), a log sine sweep, white and pink noise, silence, a silence→loud step, and full-scale clipping.
- **Matrix:** the default config (`configs/main.example.yaml`); strength 0 / 0.5 / 1; tilt 0; smear 0; `n_bins` 16 / 256 / 1024; filter order 1 / 4 / 8; sr 16000 / 44100 / 48000; window 512 / 4096; plus the parameter-script run. Keep clips to a few seconds so the fixtures stay at a few MB.
- **Rust side:** `cargo test -p dsp` replays the same signal, config and script through the `dsp` crate and compares with these tolerances:

| Output | Tolerance |
|---|---|
| L/M/H raw + scaled (f64) | max abs ≤ 1e-9 |
| Onset streams | identical. Any exception must be a documented threshold-edge case (value within 1e-9 of the threshold) |
| BPM | ≤ 1e-9 |
| FFT raw dB (f32) | ≤ 1e-4 dB, identical sentinel positions |
| FFT processed (f32) | ≤ 1e-6 abs |

- **Wire goldens:** record one of each WS JSON message type (`meta`, `snapshot`, `server_status`, `devices`, `presets`, `error`) and one OSC packet per address from the Python server. A Rust test compares **schemas** (key sets and value types) and OSC packet **layout** byte for byte, apart from payload values.
- Goldens are committed under `testdata/golden/`, so the Rust tests keep running after Python is deleted.

## Phases

Order matters; each phase's acceptance gates the next.

### Phase 0: scaffold + reference prep
- Cargo workspace `engine/` with `dsp` + `server` crates. CI-less is fine for now; `cargo test` and `cargo clippy` run locally.
- Confirm the crate picks above (`sci-rs` vs a hand port; the YAML crate).
- The Python `process_block` / `process_hop` extraction (behaviour-preserving; verify that the running server's perf and output are unchanged).
- `tools/parity/make_goldens.py` + committed goldens + wire goldens.
- **Done when:** goldens regenerate deterministically (same bytes twice) and an empty Rust parity test loads them.

### Phase 1: `dsp` crate at parity (offline, no audio device)
- Port in pipeline order and turn each golden green before the next: filter design + biquads → RMS → ExpSmoother → AutoScaler → OnsetTracker/BPM → log-bin map + FFT power → FFTPostProcessor. Then add the setters and the parameter script.
- `replay` bin: WAV in → the same dump format out (useful for debugging divergence against the Python script).
- Benchmark per stage with `criterion` on the M2.
- **Done when:** the whole golden matrix passes within tolerance, `assert_no_alloc` is clean on `process_block` / `process_hop`, and the benchmark numbers are recorded in this doc.

### Phase 2: realtime spine, headless
- cpal input + re-blocking, slot ring, DSP + FFT worker threads, command queues, stores, OSC publisher, perf rings and drop counters, priority boost, the backlog policies (DSP max backlog 4 blocks; FFT 2 hops), and the audio watchdog (1 s stall → `send_silence` → reopen by name every 3 s, falling back to the default device).
- `audio-server --no-ws` runs end-to-end.
- **Done when:** the OSC output is byte-layout-identical to Python. Live comparison: feed both servers the same loopback input (BlackHole) and diff `/audio/lmh` statistically; exactness is covered by phase 1. Unplugging and replugging a USB mic recovers. Zero drops in a 10-minute soak on the M2.

### Phase 3: control plane, UI works unmodified
- Config load/validate/fallback, the persister, presets (save/list/load with partial-section apply), the dispatcher (all handlers + `set_stream`), coalesced meta, the WS three-lane client queues + binary FFT, `server_status` (same perf schema), static HTTP, idle-when-nobody-watches, device list/probe/hot-switch with rollback, the 50 ms filter-retune debounce, band clamping, and the OSC meta loop (≤ 50 ms after a change, plus a 1 Hz resend).
- **Done when:** the wire-golden schema tests pass and a manual pass through the **unmodified** UI works: every slider, preset save/load, device switch, 2D/3D FFT, palette, raw-dB toggle, tab hide/show idling. `configs/main.yaml` written by Rust loads in Python and vice versa.

### Phase 4: Pi validation + cutover
- Cross-compile for `aarch64-unknown-linux-gnu`. Run Python and Rust on a Pi 4 (and a Pi 5 if one is available) with the same config, and record `server_status.perf` plus `top` CPU for both in this doc.
- **Targets** (Pi 4, default config, UI open): total CPU < 10% of one core, DSP p95 < 0.5 ms, zero drops in a 10-minute soak.
- Pre-cutover: load and re-save `configs/main.yaml` and every preset once with the Python server, so all files are in the current schema before the Rust server (which has no migrations) becomes the only reader. Then `grep -l "^beat:" configs/*.yaml` must return nothing.
- Cutover: `make start` → `cargo run --release -p server -- --open` (or a built binary). Rewrite the README quick start for building from source (Rust toolchain instead of uv); phase 5 later puts `pip install` back on top. Rewrite the CLAUDE.md architecture section for the Rust layout. Promote the threading model + parity traps into `docs/reference/engine.md`. Delete `server/`, `tools/parity/`, `uv.lock`, `realtime_audio_server.egg-info`. Keep `pyproject.toml`; phase 5 rewrites it for maturin. Move this plan to `docs/finished/` once phase 5 is done too.

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

## After cutover (not this sprint)

Ideas the port unlocks. Do not implement them during the port:
- Smaller `blocksize` (128/64) for lower latency, now that per-block overhead is ~µs.
- Multi-channel input / more bands.
- SCHED_FIFO workers on Linux when permitted.
