"""Per-stage CPU cost of the Python reference engine (offline, single thread).

    uv run python tools/bench/stage_bench.py                 # default config matrix
    uv run python tools/bench/stage_bench.py --quick         # default config only
    uv run python tools/bench/stage_bench.py --out tools/bench/results/stages_python_m2.json

Times every stage of the per-block DSP path and the per-hop FFT path with the
production classes (built through tools/parity/offline.py), fed the
deterministic drum loop, plus the "math floor" for each path — the bare
numerical kernels (SOS biquads, rfft) with no Python glue — to show how much of
the cost is interpreter / small-array overhead the Rust port removes. The Rust
`criterion` benches should report the same stage names so the two tables line
up (see docs/reference/perf_baseline.md).

Not engine-agnostic (imports `server`): deleted with the Python engine.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "parity"))

from offline import OfflineApp, load_cfg  # noqa: E402
from signals import generate  # noqa: E402

from server.dsp.features import block_rms  # noqa: E402
from server.dsp.filters import _sosfilt_kernel  # noqa: E402
from server.io.osc_publisher import OscPublisher  # noqa: E402
from server.io.osc_sender import OscSender  # noqa: E402
from server.config import OscDest  # noqa: E402
from server.io.ws_server import WSServer, encode_meta  # noqa: E402

N_BLOCKS = 3000
N_HOPS = 1500


def _stats(ns: list[int]) -> dict:
    a = np.array(ns, dtype=np.float64) / 1000.0  # µs
    return {"mean_us": float(a.mean()), "p50_us": float(np.percentile(a, 50)),
            "p95_us": float(np.percentile(a, 95)), "p99_us": float(np.percentile(a, 99)),
            "max_us": float(a.max())}


def bench_config(overrides: dict, sr: float = 48000.0) -> dict:
    cfg = load_cfg(overrides)
    app = OfflineApp(cfg, sr)
    bs = cfg.audio.blocksize
    sig = generate("drums", sr, 8.0)
    n_avail = sig.size // bs
    blocks = [sig[(i % n_avail) * bs:((i % n_avail) + 1) * bs] for i in range(N_BLOCKS)]
    pc = time.perf_counter_ns
    t = {k: [] for k in ("filter_bank", "rms_x3", "smoother", "autoscaler", "onset", "dsp_block_total",
                         "feature_store_publish", "osc_publish_lmh")}
    fb, sm, asc, ot, dw = app.filter_bank, app.smoother, app.auto_scaler, app.onset_tracker, app.dsp_worker
    sender = OscSender([OscDest("127.0.0.1", 9)])  # discard port; sendto still runs
    pub = OscPublisher(sender, get_master_gain=lambda: 1.0, get_db_floor=lambda: cfg.fft.db_floor,
                       get_send_fft=lambda: True, get_send_raw_db=lambda: False, get_fft_enabled=lambda: True)
    for x in blocks:
        t0 = pc(); lo, md, hi = fb.process(x); t1 = pc()
        r = (block_rms(lo), block_rms(md), block_rms(hi)); t2 = pc()
        sm.update(*r); t3 = pc()
        asc.update(sm.values, dw.scaled_buf); t4 = pc()
        bpm = ot.update(dw.scaled_buf, dw.onsets_buf); t5 = pc()
        app.features_store.publish(sm.values, dw.scaled_buf, dw.onsets_buf, ot.onset_count, bpm, 0); t6 = pc()
        pub.publish_lmh(dw.scaled_buf, dw.onsets_buf, bpm, 0); t7 = pc()
        for k, a, b in (("filter_bank", t0, t1), ("rms_x3", t1, t2), ("smoother", t2, t3),
                        ("autoscaler", t3, t4), ("onset", t4, t5), ("feature_store_publish", t5, t6),
                        ("osc_publish_lmh", t6, t7)):
            t[k].append(b - a)
    for x in blocks:  # the production entry point, end to end
        t0 = pc(); dw.process_block(x); t["dsp_block_total"].append(pc() - t0)

    # Math floor, DSP: the private Cython SOS kernel on 3 preallocated f64
    # buffers (= what a Rust biquad loop does) + dot-product RMS.
    floor_dsp = []
    bufs = [np.zeros((1, bs)) for _ in range(3)]
    sos = (fb.sos_low, fb.sos_mid, fb.sos_high)
    zis = [np.zeros((1, s.shape[0], 2)) for s in sos]
    if _sosfilt_kernel is not None:
        for x in blocks:
            t0 = pc()
            for buf, s, z in zip(bufs, sos, zis):
                buf[0] = x
                _sosfilt_kernel(s, buf, z)
                np.dot(buf[0], buf[0])
            floor_dsp.append(pc() - t0)

    # FFT path.
    fw, pp = app.fft_worker, app.fft_postprocessor
    ws = fw.window_size
    windows = [sig[(i * fw.hop) % (sig.size - ws):(i * fw.hop) % (sig.size - ws) + ws].astype(np.float64)
               for i in range(N_HOPS)]
    tf = {k: [] for k in ("window_rfft", "power_logbin_db", "postprocess", "fft_hop_total",
                          "fft_store_publish", "osc_publish_fft")}
    for w in windows:
        np.copyto(fw.window_buf, w)
        t0 = pc()
        np.multiply(fw.window_buf, fw.hann, out=fw.window_buf)
        np.fft.rfft(fw.window_buf, out=fw.spectrum)
        t1 = pc()
        tf["window_rfft"].append(t1 - t0)
    for w in windows:
        np.copyto(fw.window_buf, w)
        t0 = pc(); bins_f32, processed = fw.process_hop(); t1 = pc()
        app.fft_store.publish(bins_f32, processed, 0); t2 = pc()
        pub.publish_fft(bins_f32, processed, 0); t3 = pc()
        tf["fft_hop_total"].append(t1 - t0)
        tf["fft_store_publish"].append(t2 - t1)
        tf["osc_publish_fft"].append(t3 - t2)
    db_in = bins_f32.copy()
    for _ in range(N_HOPS):
        t0 = pc(); pp.process(db_in); tf["postprocess"].append(pc() - t0)
    tf["power_logbin_db"] = list(np.maximum(0, np.array(tf["fft_hop_total"]) - np.array(tf["window_rfft"])
                                            - np.array(tf["postprocess"])))
    floor_fft = []
    buf = np.zeros(ws)
    spec = np.zeros(ws // 2 + 1, np.complex128)
    for w in windows:
        t0 = pc(); np.multiply(w, fw.hann, out=buf); np.fft.rfft(buf, out=spec); floor_fft.append(pc() - t0)

    # WS encode costs (asyncio loop side, per broadcast tick).
    wss = WSServer.__new__(WSServer)
    wss._fft_scratch = None
    tw = {"snapshot_json": [], "fft_binary_encode": [], "meta_json": []}
    meta = app.snapshot_meta()
    for _ in range(2000):
        t0 = pc()
        json.dumps({"type": "snapshot", "seq": 1, "low": float(dw.scaled_buf[0]), "mid": 0.5, "high": 0.25,
                    "low_raw": 0.001, "mid_raw": 0.002, "high_raw": 0.003, "low_onset": 0,
                    "mid_onset": 1, "high_onset": 0, "bpm": 120.0, "t": 123456.7})
        t1 = pc()
        wss._encode_fft_binary(processed, 1.0)
        t2 = pc()
        encode_meta(meta)
        t3 = pc()
        tw["snapshot_json"].append(t1 - t0)
        tw["fft_binary_encode"].append(t2 - t1)
        tw["meta_json"].append(t3 - t2)

    block_period_us = bs / sr * 1e6
    hop_period_us = fw.hop / sr * 1e6
    out = {
        "config": {"sr": sr, "blocksize": bs, "filter_order": cfg.dsp.filter_order,
                   "n_bins": cfg.fft.n_bins, "window_size": ws, "hop": fw.hop,
                   "strength": cfg.autoscale.strength, "smear": cfg.fft.peak_smear_oct},
        "block_period_us": block_period_us, "hop_period_us": hop_period_us,
        "dsp": {k: _stats(v) for k, v in t.items()},
        "fft": {k: _stats(v) for k, v in tf.items()},
        "ws": {k: _stats(v) for k, v in tw.items()},
        "math_floor": {"dsp_sos3_rms3": _stats(floor_dsp) if floor_dsp else None,
                       "fft_window_rfft": _stats(floor_fft)},
    }
    # CPU share of one core the two worker paths would use at this config.
    out["load_pct"] = {
        "dsp_worker": out["dsp"]["dsp_block_total"]["mean_us"] / block_period_us * 100,
        "fft_worker": out["fft"]["fft_hop_total"]["mean_us"] / hop_period_us * 100,
    }
    return out


MATRIX = [
    ("default", {}),
    ("bins256", {"fft": {"n_bins": 256}}),
    ("bins1024", {"fft": {"n_bins": 1024}}),
    ("order1", {"dsp": {"filter_order": 1}}),
    ("order4", {"dsp": {"filter_order": 4}}),
    ("order8", {"dsp": {"filter_order": 8}}),
    ("win4096", {"fft": {"window_size": 4096, "hop": 1024}}),
    ("strength1", {"autoscale": {"strength": 1.0}}),
    ("block128", {"audio": {"blocksize": 128}}),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    import logging
    logging.getLogger("server").setLevel(logging.ERROR)
    results = {"host": {"machine": platform.machine(), "processor": platform.processor(),
                        "python": platform.python_version(), "numpy": np.__version__},
               "configs": {}}
    for name, ov in (MATRIX[:1] if args.quick else MATRIX):
        r = bench_config(ov)
        results["configs"][name] = r
        c = r["config"]
        print(f"\n== {name}: sr {c['sr']:.0f} block {c['blocksize']} order {c['filter_order']} "
              f"bins {c['n_bins']} window {c['window_size']}/{c['hop']} strength {c['strength']}")
        print(f"   {'stage':<24}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   (µs)")
        for grp in ("dsp", "fft", "ws"):
            for k, s in r[grp].items():
                print(f"   {grp + '.' + k:<24}{s['mean_us']:9.1f}{s['p50_us']:9.1f}{s['p95_us']:9.1f}"
                      f"{s['p99_us']:9.1f}{s['max_us']:9.1f}")
        for k, s in r["math_floor"].items():
            if s:
                print(f"   {'floor.' + k:<24}{s['mean_us']:9.1f}{s['p50_us']:9.1f}{s['p95_us']:9.1f}"
                      f"{s['p99_us']:9.1f}{s['max_us']:9.1f}")
        print(f"   load: dsp worker {r['load_pct']['dsp_worker']:.2f}% of one core "
              f"(block {r['block_period_us']:.0f} µs), fft worker {r['load_pct']['fft_worker']:.2f}% "
              f"(hop {r['hop_period_us']:.0f} µs)")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nJSON: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
