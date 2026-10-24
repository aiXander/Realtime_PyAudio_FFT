# Parity goldens

What the Rust engine is tested against. Generated from the Python reference engine. Don't hand-edit anything here.

**Only this README and `wire/` are in git.** The DSP goldens (`manifest.json`, `signals/`, `cases/`, `filter_design_sos.npy`, ~20 MB) are gitignored. On a fresh clone, generate them with `uv run python tools/parity/make_goldens.py`. Rust parity tests skip when `manifest.json` is missing. Archive the generated set before the Python engine is deleted, because it can't be regenerated after that.

- `manifest.json`, `signals/`, `cases/`, `filter_design_sos.npy`: DSP goldens, written by `uv run python tools/parity/make_goldens.py`. `--check` regenerates into a temp dir and asserts the result is byte-identical. It is, run to run, on numpy 2.4 / scipy 1.17. Regenerating is only legitimate while the Python engine is still the reference, and only after a deliberate change that lands in both engines.
- `wire/`: one captured instance of every WS message, both binary FFT flavours, every OSC packet, and the YAML the persister and the preset writer produce. Written by `uv run tools/parity/record_wire.py` from a live server fed the BlackHole drum loop. The values are run-specific. The **shape** is the contract: JSON key sets and value types, the OSC address / type-tag / payload layout, the binary header. To check another engine, run `uv run tools/parity/record_wire.py --out <dir> --server-cmd <engine>` and then `--compare testdata/golden/wire <dir>`.

## File format

Every array is a `.npy` file: format version 1.0, C-order, little-endian. The dtypes are `float64`, `float32`, `int64` and `int8`. Read the files with `ndarray-npy`/`npyz`, or parse the 10-byte magic plus the header dict by hand. Parse `manifest.json` with serde_json's **`float_roundtrip`** feature. Python writes floats as shortest-round-trip `repr`, and without that feature serde_json can be 1 ULP off.

## Signals

`signals/<name>_<sr>.npy` holds float32 mono samples from `tools/signals.py`: drums (120 BPM), sweep, white, pink, silence, step (silence then drums) and clip. A case uses a **prefix** of its signal: `n_blocks × blocksize` samples. They're committed because numpy's RNG can't be reproduced in Rust.

## Cases (27)

`manifest.cases[]` fields:

| field | meaning |
|---|---|
| `id` | e.g. `default_drums`, `order8_drums`, `sr16000_drums`, `script_drums` |
| `config` | the **full validated config** (same shape as `configs/main.yaml`) *after* band edges were clamped for this `sr`. Build the DSP from this, not from the overrides. |
| `overrides` | what differs from `configs/main.example.yaml` (for humans) |
| `sr`, `blocksize`, `n_blocks`, `n_hops` | |
| `events` | scripted control changes (only `script_drums`): `{block, msg, calls}` |
| `files` | paths of the per-case arrays below |

Per-case arrays (`cases/<id>/`):

| file | dtype / shape | what |
|---|---|---|
| `raw.npy` | f64 (n_blocks, 3) | `ExpSmoother.values` after each block. This is the WS `*_raw` value. |
| `scaled.npy` | f64 (n_blocks, 3) | AutoScaler output (after the strength blend, **before master_gain**) = the onset detector's input |
| `onsets.npy` | i8 (n_blocks, 3) | per-block onset pulse per band |
| `bpm.npy` | f64 (n_blocks,) | `OnsetTracker.update` return value |
| `hops.npy` | i64 (n_hops, 2) | `[window_start_block, processed_after_block]` |
| `fft_offsets.npy` | i64 (n_hops+1,) | frame k of the two arrays below is `[off[k], off[k+1])`. Frames are variable length because `n_bins` changes mid-script. |
| `fft_raw_db.npy` | f32 (Σ n_bins,) | wire dB per hop, with `-1000` sentinels in empty log bins |
| `fft_processed.npy` | f32 (Σ n_bins,) | post-processor output per hop, in `[0, 1]` |
| `derived_*.npy` | | internal state right after construction: bin map, Hann, power correction, tilt, per-bin taus / alphas, smear kernel, AutoScaler noise budget / per-bin factor / tilt, smoother + onset alphas, the three SOS matrices. Compare these first when an output diverges. |

## Replay contract (what a Rust parity test must do)

For each block `b` in `0..n_blocks`:

1. Apply every event with `block == b`, in order. Each event's `calls` lists the DSP-level setter calls that the Python dispatcher made (`target`, `method`, `args`, `kwargs`, `depth`). Replay **depth-0 calls only**. Depth-1 calls happen *inside* a depth-0 call (e.g. `fft_worker.reconfigure` → `fft_postprocessor.reconfigure`) and are part of that call's semantics. `fft_worker.resync` / `reset` only move the read pointer; the `hops` table already encodes their effect.
2. Run the DSP block on samples `[b·bs, (b+1)·bs)` (f32 → f64 for the filters), then compare row `b` of raw / scaled / onsets / bpm.
3. For every `hops` row with `processed_after_block == b`, in order: build the window from blocks `[start, start + window_size/bs)` (f32 → f64), run the hop, and compare the frame.

The filter retunes (`set_band`) are applied immediately at the event. The live server debounces them by 50 ms, but that's a server timing detail, not DSP behaviour. `load_preset` deep-merges the preset (`preset_files`) over main.yaml as it was persisted at that moment, which is the current state.

## Tolerances

These live in `manifest.tolerances`: L/M/H raw and scaled, and BPM, ≤ 1e-9 abs. Onsets are exact; any exception must be documented as a threshold-edge case. FFT raw dB ≤ 1e-4 dB, with identical sentinel positions. FFT processed ≤ 1e-6 abs. Filter-design SOS ≤ 1e-12.

## Filter design table

`filter_design_sos.npy` is f64 `(rows, 8, 6)`, zero-padded past `n_sections`. `manifest.filter_design.rows[i]` = `{sr, order, lo_hz, hi_hz, lo_clamped, hi_clamped, n_sections, row}`. The grid covers sr 16/44.1/48/96 kHz, orders 1–8, and 9 band-edge pairs, including ones that need clamping (`[1, 30000]`, `[2944, 13952]` at 16 kHz). Each entry is `FilterBank._design_one` = `clamp_band_edges` + `scipy.signal.iirfilter(order, [lo/nyq, hi/nyq], btype="bandpass", ftype="butter", output="sos")`.
