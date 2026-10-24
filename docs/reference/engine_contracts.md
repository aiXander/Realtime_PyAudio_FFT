# Engine contracts: what any audio-server engine must honour

This doc covers the outward-facing behaviour of the engine: CLI, config files, the WebSocket API, OSC, static HTTP, device handling, and the timing and backlog policies. The browser UI, downstream OSC consumers and saved configs all depend on it. Today the Python package `server/` implements it, and the Rust port (`docs/TODO/rust_port.md`) must reproduce it unchanged.

This is a map. **Exact message shapes** are the captured instances in `testdata/golden/wire/` (see its README). **DSP numerics** are the goldens in `testdata/golden/` plus the parity traps in the port plan. Where this doc and the code disagree, the code wins: fix this doc.

## CLI (`server/main.py:_parse_args`, `main`)

- **Flags:** `--config PATH`, `--no-ws`, `--open`, `--device N`, `--log-level L`.
- **Exit:** 0 on SIGINT/SIGTERM. On expected startup failures (bind error, no device, invalid config) it exits 1 after two stderr lines: `audio-server: error: …` and `audio-server: hint: …`, with no traceback.
- **Config path** (`_resolve_config_path`):
  - no `--config` → `<repo>/configs/main.yaml`;
  - absolute path → as given;
  - relative path → the cwd first, then the repo root.
  - If `main.yaml` is missing and no `--config` was given, it is seeded from `configs/main.example.yaml`.
- **`--open`** opens `http://<ws.host>:<http_port>/index.html`, with `?ws_port=<port>` appended when `ws.port != 8765`. It does nothing when WS or HTTP is disabled.
- **Startup order** (`App.start`): resolve device → open the stream → build the DSP for the stream's sr (band edges are clamped and persisted) → OSC sender → workers start → `/audio/meta` sent → persister → WS + HTTP bind → **stream started last** → open the browser.

## Config (`server/config.py`, `server/control/validate.py`)

- **Schema:** the `Config` dataclasses. `configs/main.example.yaml` is the canonical full example.
- **The loader is strict, not per-field-fallback:**
  - **missing section or key, YAML parse error, empty file** → *corrupt*: the file is moved to `<name>.corrupt-<YYYYmmdd-HHMMSS>[-n]`, re-seeded from the example, and reloaded;
  - **parseable file with an invalid value** → fatal startup error, because it's likely a hand edit worth fixing.
- **Optional keys** (absent → default): `dsp.filter_order` → 2; `ui.show_onsets` → false; `ui.fft_view` → `"2d"`; `ui.fft3d_style` → `"cloud"` (an unknown name also falls back); `ui.palette` → `"classic"` (likewise). Each `onset.{low,mid,high}.*` field defaults to:

  | band | sensitivity | refractory_s | slow_tau_s | abs_floor |
  |---|---|---|---|---|
  | low | 1.8 | 0.25 | 0.30 | 0.10 |
  | mid | 1.8 | 0.18 | 0.20 | 0.10 |
  | high | 1.8 | 0.12 | 0.15 | 0.10 |

  A missing `onset:` block uses these defaults. (The legacy `beat:` → `onset.low` migration also lives here; the Rust port drops it.)
- **Validators** are the single source of the value rules:
  - **Rejected** when non-finite or wrongly typed. `bool` is not accepted as a number.
  - **Clamped, not rejected:**

    | value | clamped to | how |
    |---|---|---|
    | `n_bins` | 16..1024 | |
    | `tilt_db_per_oct` | −6..12 | |
    | `peak_decay_per_s` | 0.05..3 | |
    | `ws.snapshot_hz` | 15..240 | rounded first |

  - **Band edges:** checked only structurally at load (`lo > 0`, `hi > lo`). Clamped to `1 ≤ lo ≤ 0.9·hi`, `hi ≤ 0.45·sr` once the sr is known, and the clamped values are written back to cfg and persisted.
  - **FFT geometry:** `window_size` / `hop` must be multiples of `blocksize` and ≤ 16384.
  - **Blocksize:** 16..16384.
  - **Accepted unclamped, clamped by the DSP:** `noise_floor` may be any finite value (the DSP uses `max(0, ·)`); `strength` is clamped to [0,1] inside the DSP, but cfg/meta keep the sent value.
- **Persister** (`Persister`): drag (`commit=false`) waits 1 s; `commit=true` waits 50 ms; both are capped at 250 ms after the first unsaved change. Writes are atomic: tmp + fsync + `os.replace` + directory fsync. The YAML is PyYAML `safe_dump(sort_keys=False)`, in dataclass field order. There's a sample in `testdata/golden/wire/persisted_main.yaml`. It does a final flush on shutdown.
- **Presets:**

  | aspect | behaviour |
  |---|---|
  | storage | `<config dir>/<name>.yaml`; full snapshots, same shape as main |
  | name rule | `^[A-Za-z0-9_\- ]+$`, 1–64 chars after strip; `main` reserved |
  | listing | every `*.yaml` except `main`; `{name, saved_at: "YYYY-MM-DD HH:MM"}` from mtime, newest first |
  | load: merge | deep-merge the preset over **main.yaml as on disk** |
  | load: sections | applied in this order, each validated alone (a bad one is logged and skipped): bands → filter_order → tau → autoscale → fft.n_bins → fft.window (always reconfigures the FFT, resetting its state) → peak_smear → tilt → onset per band |
  | load: failure | error if no section applied |

## WebSocket (`server/io/ws_server.py`, `server/control/dispatcher.py`)

- **Ports:** `ws://<ws.host>:<ws.port>`, default 8765. HTTP is a **separate** port, `ws.http_port` (default 8766).
- **Greeting** on connect, in order: `meta`, `devices`, `presets` on the control lane, then `server_status`.
- **Per-client outbound lanes**, drained in this priority:
  1. **control** — an unbounded FIFO. More than 1024 queued closes the client with code 1008.
  2. **status** — latest-only.
  3. **stream** — snapshots + binary FFT; drop-oldest, max 4.
- **Broadcast loop** at `snapshot_hz`, with deadline scheduling (resync if more than one period late):
  - **meta:** at most one coalesced meta per tick, to every client, when dirty.
  - **snapshot:** only when the FeatureStore `seq` changed. Fields `low/mid/high` = scaled × master_gain; `*_raw` = smoother values (no gain); `*_onset` = 1 if that band's monotonic onset counter changed since the last snapshot; `bpm`; `t` = server ms since WS start.
  - **binary FFT:** only when the FFTStore seq changed and FFT is enabled. It carries processed × master_gain, or raw dB untouched if `send_raw_db`.
- **Status loop:** `server_status` at 5 Hz to streaming clients only.
- **Idle:** with no client streaming, the broadcast loop sleeps until a client resumes or meta goes dirty. On resume the onset counters are re-baselined, so stale onsets don't flash.
- **`set_stream {enabled}`** is handled per connection in `WSServer._on_message`, not by the dispatcher. Disable drops the client's queued stream and status. Enable re-sends meta.
- **Binary FFT frame:** `[1:u8][0:u8][n_bins:u16 LE][f32 LE × n_bins]`.
- **Errors** are `{"type":"error","reason":…}`. Malformed JSON, a missing `type`, and an unknown `type` get **only** the error. A handler `ValueError` (validation, failed device switch) gets the error **plus an immediate targeted meta**, so the UI snaps back, and meta is marked dirty.

### Inbound messages (`Dispatcher._handlers` + `set_stream`)

Every state change marks meta dirty, so a coalesced broadcast goes out next tick. "persist" = `persister.request(commit=…)`; where a handler reads `commit` from the message it defaults to `true`.

| type | fields | effect | reply / persist |
|---|---|---|---|
| `set_fft` | enabled | enable: post-proc `reset()` + worker `resync()` if it was off; sets `fft.enabled` **and** `osc.send_fft` | commit |
| `set_band` | band, lo_hz, hi_hz, commit | validate → clamp to sr → cfg → 50 ms-debounced filter retune + immediate `apply_bands` (AutoScaler geometry, post-proc tau anchors, OSC meta dirty) | commit? |
| `set_smoothing` | tau{}, tau_attack{} (partial per band), commit | merge into cfg → ExpSmoother + post-proc per-bin taus | commit? |
| `set_autoscale` | any of tau_attack_s, tau_release_s, noise_floor, strength, master_gain; commit | `App.apply_autoscale` (master_gain is cfg-only) | commit? |
| `set_n_fft_bins` | n | FFT worker + post-proc reconfigure (state reset), AutoScaler log-bin geometry, OSC meta dirty | commit |
| `set_fft_send_raw_db` | send_raw_db | flips the stream both wires send | commit |
| `set_fft_peak_smear` | peak_smear_oct, commit | post-proc smear σ | commit? |
| `set_fft_tilt` | tilt_db_per_oct, commit | AutoScaler tilt + post-proc tilt | commit? |
| `set_filter_order` | order (1..8), commit | redesign all three bandpasses (state reset) | commit? |
| `set_onset` | band + any of sensitivity (>1), refractory_s, slow_tau_s, abs_floor; commit | `OnsetTracker.set_params` | commit? |
| `set_ws_snapshot_hz` | hz, commit | broadcast rate | commit? |
| `set_peak_decay` / `set_show_onsets` / `set_fft_view` / `set_fft3d_style` / `set_palette` / `set_ui_layout` | UI-only values | cfg only | commit (`set_peak_decay` / `set_ui_layout` take `commit`) |
| `list_devices` | probe | re-init audio + reopen the input (~150 ms gap), list inputs; probe opens each for 0.2 s | targeted `devices` |
| `set_device` | index | transactional hot-switch (below); cfg device = {index, name} | commit; **broadcast** `devices` |
| `list_presets` | — | | targeted `presets` |
| `save_preset` | name | write full snapshot | targeted `presets` |
| `load_preset` | name | section-by-section apply (above) | commit |
| `set_stream` | enabled | per-connection pause/resume | none (resume → meta) |

### Outbound messages

Shapes: `testdata/golden/wire/ws_*.json`. Highlights:

- **`meta`:** 26 keys, including `sr` as an **int**, a `bands` dict of `{lo_hz, hi_hz}`, the nested `autoscale` / `onset`, the UI fields, and `device` `{index, name}`.
- **`devices.items[]`:** `{index, name, hostapi, default_samplerate, max_input_channels}`, plus `probed_signal` / `probed_at` after a probe.
- **`server_status`:**
  - top level: `audio_stalled`, `cb_overruns`, `dsp_drops`, `fft_drops` (all counted from the current device session);
  - `perf`: `block_period_ms`, `hop_period_ms`, `ws_period_ms`;
  - per stage: `{avg_ms, p95_ms, load_pct}` for `cb`, `dsp`, `fft`, `ws`, `lmh_e2e`, `fft_e2e`, with an extra `enabled` on `fft` and `fft_e2e`.
- **Perf rings:** ns rings, sized cb 128 / dsp 128 / fft 64 / ws 32 / lmh_e2e 128 / fft_e2e 64. Stats skip zero slots; p95 is the element at `int(0.95·(n−1))` after a partial sort; `load_pct = avg / period · 100`.
- The UI Performance panel reads all of this; keep the schema.

## OSC (`server/io/osc_sender.py`, `server/io/osc_publisher.py`)

- **Transport:** UDP to every `osc.destinations` entry, from one non-blocking socket. Send errors are swallowed and logged at DEBUG. Byte layouts: `testdata/golden/wire/osc_*.bin`.

  | address | args | when |
  |---|---|---|
  | `/audio/lmh` | `,fff` scaled × master_gain | every DSP block (32 B) |
  | `/audio/bpm` | `,f` | every DSP block (20 B) |
  | `/audio/onset/{low,mid,high}` | `,i 1` | only on that band's onset blocks (28 B, constant) |
  | `/audio/fft` | `,f×n` processed × gain, or raw dB with sentinels and sub-floor values raised to `db_floor` | every hop while `osc.send_fft` and FFT is enabled. One datagram; `n ≤ 1024` keeps it under macOS's 9216-byte limit |
  | `/audio/meta` | `,iiiffffff` = sr, blocksize, n_bins, low_lo, low_hi, mid_lo, mid_hi, high_lo, high_hi | at startup, then every 1 s, ≥ 50 ms after a change to sr / bands / n_bins, and right after any stream change |

- **Silence:** `/audio/lmh 0 0 0` + `/audio/bpm 0` (the zero templates), sent once when the audio watchdog detects a stall.
- LMH, BPM and onsets go out on the DSP worker thread; FFT goes out on the FFT worker thread; meta goes out on the control loop.

## Static HTTP (`server/io/http_server.py`)

- Serves `<repo>/ui/` on `ws.host:ws.http_port` from its own thread.
- Sends `Cache-Control: no-cache` on every response.
- Forces MIME types for `.js` / `.mjs` (`text/javascript`), `.css`, `.json`, `.wasm`, `.svg` and `.html`.
- No access log.
- If `ui/` is missing, a warning is logged and HTTP is skipped.

## Devices (`server/audio/devices.py`, `App.*` in `server/main.py`)

- **Initial device:** `--device` → `cfg.audio.device.name` (looked up **by name**) → `cfg.audio.device.index` → the system default.
- **Stream:** opened with the device's default sr. `channels` is clamped to the device's maximum. Stereo is mixed to mono as `(L+R)/2`. A host latency over 80 ms logs a warning.
- **Hot-switch** (`hot_switch_device`):
  - under the device-switch lock;
  - the index must be a current input;
  - tear down → reset the ring, read pointers and perf → open the new device → rebuild the DSP for the new sr (clamp bands) → FFT reconfigure + post-proc `update_bands` + `reset`;
  - on failure: restore the bands, reopen the old device, raise → error + snap-back meta.
  - The workers stay alive throughout.
- **Watchdog** (`_audio_watchdog_loop`):
  - ticks every 0.25 s;
  - if no new ring block arrives for 1 s while no switch holds the lock: one warning, OSC silence, `audio_stalled = true`;
  - then reopens every 3 s: re-init audio → same device **by name** → else the system default.
  - Reopening the same device at the same sr keeps the DSP state (`keep_state=True`).
  - Auto-recovery never rewrites `cfg.audio.device`.
- **Drop counters** are re-baselined after every stream change, and again 250 ms later.

## Threading, backlog and timing policies

- **Callback:** mono-mix → slot-ring write (timestamped) → set both worker events.
- **Ring:** `SlotRing` holds `max(32, pow2 ≥ 2·⌈16384/blocksize⌉)` slots.
- **DSP worker:**
  - drains every available block on each wake;
  - if more than 4 blocks behind, it skips to `wi−1` and counts drops;
  - calls `OnsetTracker.advance(skipped)` so the BPM stays on audio time;
  - a failed read also drops and advances;
  - self-heals when the ring is reset.
- **FFT worker:**
  - drains every complete hop;
  - if more than `window + 2 hops` behind, it skips whole hops to stay aligned;
  - `resync()` after a re-enable skips the idle gap without counting drops.
- **Worker loops never die:** each iteration is wrapped and errors are logged at most once a second.
- **Priority:** macOS `QOS_CLASS_USER_INITIATED` for the workers and the main thread; Linux absolute nice −5 (best effort). The Python-only `sys.setswitchinterval(5e-4)` has no Rust equivalent and is not needed.
- **Filter retunes** from `set_band` are debounced 50 ms server-side.
