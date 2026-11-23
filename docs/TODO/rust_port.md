# Rust engine: what is still open

**Status:** the Rust engine (`engine/`) is at parity with the Python engine (`server/`) and is the default (`make start`). **Both engines are permanent** (decided by Xander, 2026-11-23, reversing the earlier "Python frozen, deleted at cutover" plan): the rule for keeping them in agreement is in [`reference/dual_engine.md`](../reference/dual_engine.md). Open work: **Pi validation** (below), then **distribution**, which is blocked on open question 1.

## Pi validation (open)

- Cross-compile for `aarch64-unknown-linux-gnu` (the target's std is installed on the M2; `priority.rs` type-checks for it; add `targets = ["aarch64-unknown-linux-gnu"]` to `engine/rust-toolchain.toml`). Check on the Pi that the workers get `SCHED_FIFO` or fall back to nice −5 (startup log `priority:` lines). **Expect neither as a plain user:** `SCHED_FIFO` needs `RLIMIT_RTPRIO` > 0 and nice −5 needs `RLIMIT_NICE` (or `CAP_SYS_NICE`), both 0 by default, and the M2 soaks showed un-boosted workers drop blocks. So ship the Pi setup with it: a systemd unit with `LimitRTPRIO=20` + `LimitNICE=-10` (or an `@audio` rule in `/etc/security/limits.d/` and the user in `audio`), and soak with and without it. (Cross type-check without the Pi's ALSA libs: a stub `alsa.pc` on `PKG_CONFIG_PATH` plus `PKG_CONFIG_ALLOW_CROSS=1`.) Run Python and Rust on a Pi 4 (and a Pi 5 if one is available) with the same config, and record `server_status.perf` plus `top` CPU for both in [`reference/perf_baseline.md`](../reference/perf_baseline.md).
- **Targets** (Pi 4, default config, UI open): total CPU < 10% of one core, DSP p95 < 0.5 ms, zero drops in a 10-minute soak.
- Before pointing Rust at the Pi's configs: load and re-save `configs/*.yaml` once with the Python engine (Rust has no legacy migrations). Then `grep -l "^beat:" configs/*.yaml` must return nothing.
- Device constraints from the port: device indices are positions in cpal's device list (= PortAudio's on CoreAudio; unverified on ALSA, see open question 3); `BufferSize::Fixed(blocksize)` is requested only when the device's range allows it, and ALSA re-blocking is untested on hardware.
- Still to do by hand (Xander, any machine): unplug/replug a USB mic while running (watchdog → reopen by name → recovery).

## Distribution (open, blocked on open question 1)

What stays valid whichever way question 1 goes, for shipping the Rust binary to people without a toolchain:

- **Self-contained binary:** embed `ui/` and `configs/main.example.yaml` into the binary (`rust-embed` or `include_dir`). An installed binary has no repo next to it.
- **Config location:** a repo checkout keeps `<repo>/configs/main.yaml` (as `server/main.py:_resolve_config_path` does). An installed binary uses the platform config dir (`directories` crate: `~/Library/Application Support/realtime-audio-server/`, `~/.config/realtime-audio-server/`), seeded from the embedded example on first run. `--config` overrides both. Detect "repo checkout" by a `configs/` dir beside the workspace root, resolved at build time via `CARGO_MANIFEST_DIR`, falling back to runtime lookup.
- **CI release matrix** (GitHub Actions): macOS arm64 + x86_64, Linux x86_64 + aarch64 (covers the Pi on 64-bit Raspberry Pi OS), Windows x86_64; raw binaries (tar/zip) to a GitHub Release. `cargo-dist` may be simpler than hand-rolled steps.
- **Trap:** cpal on Linux links `libasound.so.2` dynamically. In a manylinux wheel it isn't on the allowlist, so tell maturin/auditwheel to leave it external rather than vendor it (every Linux desktop and the Pi has ALSA). Document `sudo apt install libasound2` for minimal images.

## Open questions (for Xander)

1. **What does `pip install realtime-audio-server` ship, now that Python stays?** The old plan (written when Python was going to be deleted) rewrote `pyproject.toml` for maturin (`bindings = "bin"`) so the wheel carried the Rust binary as the `audio-server` command, with no Python engine. With both engines permanent that no longer fits as written. Options: (a) pip keeps shipping the Python engine, and the Rust engine is distributed as GitHub Release binaries only; (b) a second package (e.g. `realtime-audio-server-rs`) built with maturin; (c) one wheel carrying both, with the Rust binary as a second command. Undecided; nothing in the repo depends on it yet.
2. **Python library API?** Should a pip package also let people `import` the DSP from their own Python scripts (feed numpy blocks in, get L/M/H, onsets and FFT bins back)? The Python engine already allows this in-tree (`server/dsp/`); PyO3 bindings on the pure `dsp` crate would be the Rust route. *Recommendation:* only if someone asks.
3. **Device identity on Linux.** Saved configs reopen the input **by name**. cpal's ALSA device names will likely differ from PortAudio's (PortAudio embeds `(hw:2,0)`; unverified), so on the Pi a name saved by the Python engine may not match in Rust, and the server falls back to the system default once. cpal also offers a stable `id()` (CoreAudio UID / ALSA id). *Recommendation:* keep name matching; on a miss, fall back to default as today. Storing an id too would be a schema change, so both engines would need it. Not blocking the Pi validation.

## Later ideas (not scheduled)

Both would land in both engines (see the rule in [`reference/dual_engine.md`](../reference/dual_engine.md)); the first may not be reachable in Python, which is the open trade-off.
- Smaller `blocksize` (128/64) for lower latency, now that Rust's per-block overhead is ~µs.
- Multi-channel input / more bands.

## Done

Phases 0–3 (parity harness, `dsp` crate at parity, the `audio-server` binary with the full control plane, 10-minute zero-drop soak) — see [reference/rust_dsp.md](../reference/rust_dsp.md), [reference/rust_server.md](../reference/rust_server.md) and [reference/dual_engine.md](../reference/dual_engine.md).
