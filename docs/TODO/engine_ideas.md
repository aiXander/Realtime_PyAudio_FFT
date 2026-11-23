# Engine ideas (not scheduled)

Nothing here is planned. Each item waits for a real need. Anything with an outward effect lands in both engines (the rule: [`../reference/dual_engine.md`](../reference/dual_engine.md)).

## Prebuilt Rust binaries (only if someone asks)

Today the Rust engine is built from source (the distribution decision is in `dual_engine.md`). Shipping binaries would need:

- **A self-contained binary:** embed `ui/` and `configs/main.example.yaml` (`rust-embed` or `include_dir`). An installed binary has no repo next to it.
- **A config location:** a repo checkout keeps `<repo>/configs/main.yaml` (as `server/main.py:_resolve_config_path` does). An installed binary uses the platform config dir (`directories` crate), seeded from the embedded example on first run. `--config` overrides both.
- **A CI release matrix** (GitHub Actions): macOS arm64 + x86_64, Linux x86_64 + aarch64 (covers the Pi on 64-bit Raspberry Pi OS), Windows x86_64; raw binaries to a GitHub Release. `cargo-dist` may be simpler than hand-rolled steps.
- **Trap:** cpal on Linux links `libasound.so.2` dynamically; document `sudo apt install libasound2` for minimal images.

## Python library API (only if someone asks)

Let people `import` the DSP into their own scripts (feed numpy blocks in, get L/M/H, onsets and FFT bins back). The Python engine already allows this in-tree (`server/dsp/`); PyO3 bindings on the pure `dsp` crate would be the Rust route.

## Stable device identity

Saved configs reopen the input **by name**. On Linux, cpal's ALSA names may differ from PortAudio's (PortAudio embeds `(hw:2,0)`), so a name saved by one engine may miss in the other and the server falls back to the system default once. cpal offers a stable `id()` (CoreAudio UID / ALSA id). Storing it too is a config schema change, so both engines and every file in `configs/` would change together.

## Lower latency, more inputs

- Smaller `blocksize` (128/64) now that Rust's per-block overhead is ~µs. Python may not keep up, which is the open trade-off.
- Multi-channel input / more bands.
