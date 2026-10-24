"""Generate the DSP parity goldens the Rust `dsp` crate is tested against.

    uv run python tools/parity/make_goldens.py            # (re)write testdata/golden/
    uv run python tools/parity/make_goldens.py --check    # regenerate into a temp dir,
                                                          # assert byte-identical to the committed set

Drives the real Python engine offline (tools/parity/offline.py: production
App wiring, DSPWorker.process_block, FFTWorker.process_hop, the real
Dispatcher for scripted control changes). Everything the Rust side needs is
in testdata/golden/; see testdata/golden/README.md for the format and the
replay contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "parity"))

from offline import OfflineApp, OfflineEngine, load_cfg  # noqa: E402
from signals import generate  # noqa: E402

from server.config import config_to_dict, write_yaml_atomic  # noqa: E402
from server.dsp.filters import FilterBank  # noqa: E402

GOLDEN = REPO / "testdata" / "golden"
FORMAT_VERSION = 1

TOLERANCES = {
    "raw": {"max_abs": 1e-9},
    "scaled": {"max_abs": 1e-9},
    "bpm": {"max_abs": 1e-9},
    "onsets": {"exact": True,
               "note": "an exception must be a documented threshold-edge case: novelty within 1e-9 of the threshold"},
    "fft_raw_db": {"max_abs": 1e-4, "sentinel_positions": "identical"},
    "fft_processed": {"max_abs": 1e-6},
    "filter_design_sos": {"max_abs": 1e-12},
    "derived": {"max_rel": 1e-12},
}

# ---------------------------------------------------------------------------
# Matrix. Every case = base config (configs/main.example.yaml) + overrides,
# a signal, a sample rate, optional scripted control events.
# ---------------------------------------------------------------------------
SR = 48000
DUR = {"drums": 6.0, "sweep": 4.0, "white": 3.0, "pink": 3.0, "silence": 2.0, "step": 5.0, "clip": 3.0}

# Scripted control changes for the `script_drums` case, as real WS messages
# (the exact shape the UI sends). `block` = applied before that DSP block.
SCRIPT = [
    (200, {"type": "set_autoscale", "noise_floor": 1e-4, "commit": True}),
    (300, {"type": "set_band", "band": "mid", "lo_hz": 500.0, "hi_hz": 2500.0, "commit": False}),
    (380, {"type": "set_band", "band": "high", "lo_hz": 3000.0, "hi_hz": 30000.0, "commit": True}),
    (450, {"type": "set_smoothing", "tau": {"low": 0.05}, "tau_attack": {"high": 0.002}, "commit": True}),
    (550, {"type": "set_autoscale", "strength": 1.0, "tau_attack_s": 0.1, "tau_release_s": 5.0}),
    (650, {"type": "set_fft_tilt", "tilt_db_per_oct": 0.0}),
    (700, {"type": "set_fft_peak_smear", "peak_smear_oct": 0.8}),
    (800, {"type": "set_n_fft_bins", "n": 64}),
    (900, {"type": "set_filter_order", "order": 4}),
    (1000, {"type": "set_onset", "band": "low", "sensitivity": 1.4, "refractory_s": 0.15}),
    (1050, {"type": "set_onset", "band": "high", "slow_tau_s": 0.08, "abs_floor": 0.05}),
    (1100, {"type": "load_preset", "name": "golden-window"}),
    (1150, {"type": "set_autoscale", "strength": 0.0}),
    (1250, {"type": "set_fft", "enabled": False}),
    (1300, {"type": "set_fft", "enabled": True}),
    (1400, {"type": "set_autoscale", "strength": 0.5, "noise_floor": 0.0}),
    (1500, {"type": "set_n_fft_bins", "n": 512}),
    (1600, {"type": "set_fft_tilt", "tilt_db_per_oct": -3.0}),
    (1700, {"type": "set_autoscale", "master_gain": 2.0}),
]
# Partial preset loaded by the script: only these keys; everything else is
# deep-merged from main.yaml (= the current state, see offline._SyncPersister).
SCRIPT_PRESET = {
    "dsp": {"filter_order": 3},
    "fft": {"window_size": 2048, "hop": 512, "f_min": 40.0, "n_bins": 96},
    "onset": {"mid": {"sensitivity": 2.2}},
}


def _cases() -> list[dict]:
    cases = []
    for sig in ("drums", "sweep", "white", "pink", "silence", "step", "clip"):
        cases.append({"id": f"default_{sig}", "signal": sig, "sr": SR, "overrides": {}})
    variants = [
        ("strength0", {"autoscale": {"strength": 0.0}}),
        ("strength05", {"autoscale": {"strength": 0.5}}),
        ("strength1", {"autoscale": {"strength": 1.0}}),
        ("tilt0", {"fft": {"tilt_db_per_oct": 0.0}}),
        ("smear0", {"fft": {"peak_smear_oct": 0.0}}),
        ("bins16", {"fft": {"n_bins": 16}}),
        ("bins256", {"fft": {"n_bins": 256}}),
        ("bins1024", {"fft": {"n_bins": 1024}}),
        ("order1", {"dsp": {"filter_order": 1}}),
        ("order4", {"dsp": {"filter_order": 4}}),
        ("order8", {"dsp": {"filter_order": 8}}),
        ("win512", {"fft": {"window_size": 512, "hop": 256}}),
        ("win4096", {"fft": {"window_size": 4096, "hop": 1024}}),
        ("block128", {"audio": {"blocksize": 128}}),
        ("block512", {"audio": {"blocksize": 512}}),
    ]
    # Variants run on a 4 s prefix of the drum loop (8 beats: enough for BPM
    # to lock); default_drums keeps the full 6 s.
    for name, ov in variants:
        cases.append({"id": f"{name}_drums", "signal": "drums", "sr": SR, "overrides": ov,
                      "duration": 4.0})
    cases.append({"id": "sr16000_drums", "signal": "drums", "sr": 16000, "overrides": {}, "duration": 4.0})
    cases.append({"id": "sr44100_drums", "signal": "drums", "sr": 44100, "overrides": {}, "duration": 4.0})
    cases.append({"id": "nf0_white", "signal": "white", "sr": SR,
                  "overrides": {"autoscale": {"noise_floor": 0.0}}})
    cases.append({"id": "loudnf_pink", "signal": "pink", "sr": SR,
                  "overrides": {"autoscale": {"noise_floor": 0.01}}})
    cases.append({"id": "script_drums", "signal": "drums", "sr": SR, "overrides": {},
                  "duration": 10.0, "script": SCRIPT})
    # bins1024 is the heaviest fixture; keep it short.
    for c in cases:
        if c["id"] == "bins1024_drums":
            c["duration"] = 2.0
    return cases


# ---------------------------------------------------------------------------
# Call tracing: record the DSP-level setter calls each control event causes,
# in order, so the Rust dsp-crate replay doesn't depend on the dispatcher.
# ---------------------------------------------------------------------------
TRACED = {
    "filter_bank": ("retune", "set_order", "reset_state"),
    "smoother": ("set_tau", "reset"),
    "auto_scaler": ("set_taus", "set_noise_floor", "set_tilt", "set_bands", "set_n_fft_window",
                    "set_fft_geometry", "set_strength", "reset"),
    "onset_tracker": ("set_params", "reconfigure", "reset", "advance"),
    "fft_worker": ("reconfigure", "reset", "resync"),
    "fft_postprocessor": ("reconfigure", "update_smoothing", "update_bands", "update_smear",
                          "update_tilt", "update_autoscale", "reset"),
}


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("non-finite value in trace")
    return v


class Tracer:
    def __init__(self, app: OfflineApp):
        self.calls: list[dict] | None = None
        self.depth = 0
        for target, methods in TRACED.items():
            obj = getattr(app, target)
            for m in methods:
                setattr(obj, m, self._wrap(target, m, getattr(obj, m)))

    def _wrap(self, target, method, fn):
        def wrapper(*args, **kwargs):
            rec = None
            if self.calls is not None:
                rec = {"target": target, "method": method, "args": _jsonable(list(args)),
                       "kwargs": _jsonable(kwargs), "depth": self.depth}
                self.calls.append(rec)
            self.depth += 1
            try:
                return fn(*args, **kwargs)
            finally:
                self.depth -= 1
        return wrapper


# ---------------------------------------------------------------------------
def _save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr), allow_pickle=False)


def _derived(app: OfflineApp) -> dict[str, np.ndarray]:
    """Internal state right after construction — localises a divergence to
    one precomputation instead of 'the output differs'."""
    fw, pp, a = app.fft_worker, app.fft_postprocessor, app.auto_scaler
    fb = app.filter_bank
    return {
        "fft_bin_assign": fw.bin_assign.astype(np.int64),
        "fft_bin_counts": fw.bin_counts.astype(np.float64),
        "fft_hann": fw.hann.astype(np.float64),
        "fft_win_power_corr": np.array([fw._win_power_corr]),
        "post_tilt_db": pp.tilt_db.astype(np.float64),
        "post_tau_per_bin": pp.tau_per_bin.astype(np.float64),
        "post_tau_attack_per_bin": pp.tau_attack_per_bin.astype(np.float64),
        "post_alpha_smooth_release": pp.alpha_smooth_release.copy(),
        "post_alpha_smooth_attack": pp.alpha_smooth_attack.copy(),
        "post_peak_alphas": np.array([pp.alpha_attack, pp.alpha_release]),
        "post_smear_kernel": (pp._smear_kernel.copy() if pp._smear_kernel is not None
                              else np.zeros(0)),
        "as_tilt_lin": a._tilt_lin.copy(),
        "as_noise_pwr_per_band": a._noise_pwr_per_band.copy(),
        "as_per_bin_factor": a._per_bin_factor.copy(),
        "as_log_bins_per_band": a._log_bins_per_band.copy(),
        "as_peak_alphas": np.array([a._a_atk, a._a_rel]),
        "smoother_alphas": np.stack([app.smoother._alpha_attack, app.smoother._alpha_release]),
        "onset_alphas": np.array([app.onset_tracker.alpha_fast, app.onset_tracker.alpha_thr,
                                  *app.onset_tracker.alpha_slow]),
        "sos_low": fb.sos_low.copy(),
        "sos_mid": fb.sos_mid.copy(),
        "sos_high": fb.sos_high.copy(),
    }


def run_case(case: dict, sig: np.ndarray, out: Path, work: Path) -> dict:
    cfg = load_cfg(case["overrides"])
    cdir_work = work / case["id"]
    cdir_work.mkdir(parents=True)
    config_path = cdir_work / "main.yaml"
    app = OfflineApp(cfg, case["sr"], config_path=config_path)
    if case.get("script"):
        write_yaml_atomic(cdir_work / "golden-window.yaml", SCRIPT_PRESET)
    initial_cfg = config_to_dict(app.cfg)  # after band clamping for this sr
    derived = _derived(app)
    tracer = Tracer(app)
    eng = OfflineEngine(app)

    bs = app.cfg.audio.blocksize
    n_blocks = sig.size // bs
    raw = np.zeros((n_blocks, 3), np.float64)
    scaled = np.zeros((n_blocks, 3), np.float64)
    onsets = np.zeros((n_blocks, 3), np.int8)
    bpm = np.zeros(n_blocks, np.float64)
    hop_rows, raw_db_chunks, proc_chunks, offsets = [], [], [], [0]

    events_by_block: dict[int, list[dict]] = {}
    for b, msg in case.get("script", []):
        events_by_block.setdefault(b, []).append(msg)
    events_out = []

    for b in range(n_blocks):
        for msg in events_by_block.get(b, []):
            tracer.calls = []
            eng.dispatch(copy.deepcopy(msg))
            events_out.append({"block": b, "msg": msg, "calls": tracer.calls})
            tracer.calls = None
        block = sig[b * bs:(b + 1) * bs]
        bpm[b] = eng.step_dsp(block)
        raw[b] = app.smoother.values
        scaled[b] = app.dsp_worker.scaled_buf
        onsets[b] = app.dsp_worker.onsets_buf
        for start, bins_f32, processed in eng.fft_hops():
            hop_rows.append((start, b))
            raw_db_chunks.append(bins_f32.copy())
            proc_chunks.append(processed.copy())
            offsets.append(offsets[-1] + bins_f32.size)
    eng.close()

    cdir = out / "cases" / case["id"]
    files = {}

    def put(name, arr):
        _save(cdir / f"{name}.npy", arr)
        files[name] = f"cases/{case['id']}/{name}.npy"

    put("raw", raw)
    put("scaled", scaled)
    put("onsets", onsets)
    put("bpm", bpm)
    put("hops", np.array(hop_rows, np.int64).reshape(-1, 2))
    put("fft_offsets", np.array(offsets, np.int64))
    put("fft_raw_db", np.concatenate(raw_db_chunks).astype(np.float32) if raw_db_chunks
        else np.zeros(0, np.float32))
    put("fft_processed", np.concatenate(proc_chunks).astype(np.float32) if proc_chunks
        else np.zeros(0, np.float32))
    for k, v in derived.items():
        put(f"derived_{k}", v)

    return {
        "id": case["id"],
        "signal": f"{case['signal']}_{case['sr']}",
        "sr": case["sr"],
        "blocksize": bs,
        "n_blocks": n_blocks,
        "n_hops": len(hop_rows),
        "overrides": case["overrides"],
        "config": initial_cfg,
        "events": events_out,
        "preset_files": ({"golden-window": SCRIPT_PRESET} if case.get("script") else {}),
        "files": files,
        "summary": {
            "onsets_per_band": [int(x) for x in onsets.sum(axis=0)],
            "final_bpm": float(bpm[-1]),
            "scaled_mean": [float(x) for x in scaled.mean(axis=0)],
        },
    }


def filter_design_table(out: Path) -> dict:
    """Standalone SOS goldens: FilterBank._design_one over a grid of
    (sr, order, band) incl. edges that need clamping."""
    srs = (16000.0, 44100.0, 48000.0, 96000.0)
    bands = ((88.0, 184.0), (304.0, 1760.0), (2944.0, 13952.0), (72.0, 176.0), (432.0, 1760.0),
             (3008.0, 7200.0), (20.0, 60.0), (1.0, 30000.0), (5000.0, 5100.0))
    rows, mats = [], []
    for sr in srs:
        for order in range(1, 9):
            fb = FilterBank(sr=sr, bands={"low": bands[0], "mid": bands[1], "high": bands[2]},
                            blocksize=256, order=order)
            for lo, hi in bands:
                sos, (lo_c, hi_c) = fb._design_one(lo, hi)
                rows.append({"sr": sr, "order": order, "lo_hz": lo, "hi_hz": hi,
                             "lo_clamped": lo_c, "hi_clamped": hi_c,
                             "n_sections": int(sos.shape[0]), "row": len(mats)})
                pad = np.zeros((8, 6))
                pad[: sos.shape[0]] = sos
                mats.append(pad)
    _save(out / "filter_design_sos.npy", np.stack(mats))
    return {"file": "filter_design_sos.npy", "shape": "(n_rows, 8, 6) f64, zero-padded past n_sections",
            "rows": rows}


# What this script owns inside the golden dir. Everything else there
# (README.md, wire/ from record_wire.py) is left alone.
OWNED = ("manifest.json", "filter_design_sos.npy", "signals", "cases")


def generate_all(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in OWNED:
        p = out / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    logging.getLogger("server").setLevel(logging.ERROR)
    cases = _cases()
    signals: dict[str, dict] = {}
    sig_data: dict[str, np.ndarray] = {}
    for c in cases:
        dur = c.get("duration", DUR[c["signal"]])
        key = f"{c['signal']}_{c['sr']}"
        # One file per (signal, sr); the longest duration requested wins, and
        # shorter cases use a prefix.
        if key not in sig_data or sig_data[key].size < int(round(c["sr"] * dur)):
            sig_data[key] = generate(c["signal"], c["sr"], dur)
            signals[key] = {"file": f"signals/{key}.npy", "name": c["signal"], "sr": c["sr"],
                            "duration_s": dur, "dtype": "float32"}
    for key, arr in sig_data.items():
        _save(out / "signals" / f"{key}.npy", arr)

    manifest_cases = []
    with tempfile.TemporaryDirectory() as work:
        for c in cases:
            dur = c.get("duration", DUR[c["signal"]])
            sig = sig_data[f"{c['signal']}_{c['sr']}"][: int(round(c["sr"] * dur))]
            print(f"  {c['id']:<16} {dur:4.1f} s @ {c['sr']} Hz", flush=True)
            mc = run_case(c, sig, out, Path(work))
            mc["duration_s"] = dur
            manifest_cases.append(mc)

    manifest = {
        "format": FORMAT_VERSION,
        "generator": "tools/parity/make_goldens.py",
        "reference": {"numpy": np.__version__, "scipy": scipy.__version__,
                      "python": sys.version.split()[0]},
        "replay_order": [
            "for each block b: apply events with block == b (their `calls`, depth 0 only, in order)",
            "write block b (float32) into the slot ring",
            "DSP: filter -> rms -> smoother -> autoscaler -> onset; record raw/scaled/onsets/bpm",
            "FFT: for each row of hops with processed_after_block == b, window = blocks "
            "[start, start + window_blocks) cast to f64 -> process_hop; record raw_db, processed",
        ],
        "notes": [
            "raw = ExpSmoother.values (WS *_raw); scaled = AutoScaler output before master_gain "
            "(master_gain only multiplies wire outputs, never DSP state).",
            "fft_raw_db / fft_processed are per-hop frames concatenated; frame k is "
            "[fft_offsets[k], fft_offsets[k+1]) (n_bins changes mid-script).",
            "hops rows are [window_start_block, processed_after_block].",
            "events[].calls are the DSP-level setter calls the dispatcher made, depth 0 = called by "
            "App/Dispatcher, depth 1 = nested inside another traced call (e.g. FFTWorker.reconfigure "
            "-> FFTPostProcessor.reconfigure); replay depth 0 only.",
            "filter retunes are applied immediately at the event (the live server debounces 50 ms; "
            "that's server timing, not DSP semantics).",
            "load_preset deep-merges the preset over main.yaml as persisted at that moment "
            "(= current state); preset_files holds the preset body.",
            "parse this manifest with serde_json's `float_roundtrip` feature: floats are written "
            "as Python repr (shortest round-trip) and must parse to the exact same f64.",
        ],
        "tolerances": TOLERANCES,
        "signals": signals,
        "filter_design": filter_design_table(out),
        "cases": manifest_cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")


def _tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.relative_to(root).parts[0] in OWNED}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(GOLDEN))
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temp dir and compare with --out byte for byte")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp) / "golden"
            generate_all(fresh)
            a, b = _tree_digest(out), _tree_digest(fresh)
            if a != b:
                diff = sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b) if a[k] != b[k])
                print(f"MISMATCH in {len(diff)} files, e.g. {diff[:10]}")
                return 1
            print(f"OK: {len(a)} files byte-identical")
            return 0
    generate_all(out)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({total / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
