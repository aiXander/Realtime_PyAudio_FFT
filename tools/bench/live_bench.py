# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0", "sounddevice>=0.4.6", "websockets>=14,<17", "psutil>=5.9", "pyyaml>=6.0"]
# ///
"""Black-box live benchmark of an audio-server engine (Python today, Rust later).

Engine-agnostic on purpose: it only talks to the server through its CLI
(`--config`), its OSC output and its WebSocket API, so the exact same run can
be repeated against the Rust binary and compared number for number:

    uv run tools/bench/live_bench.py --label python
    uv run tools/bench/live_bench.py --label rust \
        --server-cmd "engine/target/release/audio-server"

What it does:
  1. Writes a throwaway config (base: configs/main.example.yaml) on private
     ports (WS 18765, HTTP 18766, OSC -> 127.0.0.1:19000) and, with
     `--input blackhole`, points the server at the "BlackHole 2ch" loopback
     input while this script plays a deterministic drum loop into the
     BlackHole output, so every run sees identical audio.
  2. Starts the server, then measures two phases:
       idle  no WS client (UI closed): only the OSC path runs.
       ui    one WS client streaming snapshots + binary FFT + server_status
             (what an open UI tab costs).
  3. Per phase: process CPU (% of one core, from CPU-time deltas), per-thread
     CPU, RSS; OSC packet rates and inter-arrival jitter per address; WS
     message rates; server_status perf (stage avg/p95, end-to-end latency,
     drops); and "kick -> /audio/onset/low" latency, i.e. from the moment the
     kick's first sample hits the BlackHole DAC clock to the OSC onset
     packet's arrival here (input buffering + DSP + detector delay).

Results print as a table and are written as JSON (`--out`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import sounddevice as sd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from signals import generate  # noqa: E402

WS_PORT = 18765
HTTP_PORT = 18766
OSC_PORT = 19000
BLACKHOLE = "BlackHole 2ch"
KICK_GAIN_THRESHOLD = 0.3  # a drum-loop sample above this marks a kick start


# --------------------------------------------------------------------------- config
def _set_dotted(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def write_config(base: Path, out_dir: Path, device_name: str | None, overrides: list[str]) -> Path:
    cfg = yaml.safe_load(base.read_text())
    cfg["audio"]["device"] = {"name": device_name, "index": None}
    cfg["ws"].update({"enabled": True, "host": "127.0.0.1", "port": WS_PORT, "http_port": HTTP_PORT})
    cfg["osc"]["destinations"] = [{"host": "127.0.0.1", "port": OSC_PORT}]
    cfg["osc"]["send_fft"] = True
    cfg["fft"]["enabled"] = True
    for ov in overrides:
        k, _, v = ov.partition("=")
        _set_dotted(cfg, k, yaml.safe_load(v))
    path = out_dir / "main.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# --------------------------------------------------------------------------- playback
class LoopPlayer:
    """Plays a signal in a loop into an output device and keeps a
    (first-sample-index, DAC time on the perf_counter clock) map so kick
    onsets can be timestamped. PortAudio's stream clock and perf_counter
    may differ, so the DAC time is re-based: perf_now + (dac - stream_now)."""

    def __init__(self, device: str, signal_name: str):
        info = sd.query_devices(device, "output")
        self.sr = float(info["default_samplerate"])
        self.sig = generate(signal_name, self.sr, 8.0)  # 8 s = 16 beats, loops seamlessly at 120 BPM
        self.n = self.sig.size
        self.pos = 0
        self.anchors: list[tuple[int, float]] = []  # (absolute sample index, dac perf time)
        self._lock = threading.Lock()
        x = np.abs(self.sig)
        above = x > KICK_GAIN_THRESHOLD
        starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
        # Keep only the first crossing of each beat (kicks are 0.5 s apart).
        kicks = []
        for s in starts:
            if not kicks or s - kicks[-1] > 0.25 * self.sr:
                kicks.append(int(s))
        self.kick_offsets = np.array(kicks, dtype=np.int64)
        self.stream = sd.OutputStream(device=device, samplerate=self.sr, channels=2,
                                      dtype="float32", blocksize=256, latency="low",
                                      callback=self._cb)

    def _cb(self, outdata, frames, time_info, status):
        perf_now = time.perf_counter()
        dac = time_info.outputBufferDacTime
        cur = time_info.currentTime
        dac_perf = perf_now + (dac - cur) if dac and cur else perf_now
        idx = np.arange(self.pos, self.pos + frames) % self.n
        chunk = self.sig[idx]
        outdata[:, 0] = chunk
        outdata[:, 1] = chunk
        with self._lock:
            self.anchors.append((self.pos, dac_perf))
        self.pos += frames

    def kick_times(self) -> np.ndarray:
        """DAC-clock perf_counter times of every kick played so far."""
        with self._lock:
            anchors = list(self.anchors)
        if len(anchors) < 2:
            return np.zeros(0)
        a_idx = np.array([a[0] for a in anchors], dtype=np.float64)
        a_t = np.array([a[1] for a in anchors], dtype=np.float64)
        total = int(a_idx[-1])
        loops = total // self.n + 1
        abs_kicks = (self.kick_offsets[None, :] + self.n * np.arange(loops)[:, None]).ravel()
        abs_kicks = abs_kicks[(abs_kicks >= a_idx[0]) & (abs_kicks <= a_idx[-1])]
        # Per-sample DAC time: nearest preceding anchor + offset / sr.
        j = np.searchsorted(a_idx, abs_kicks, side="right") - 1
        return a_t[j] + (abs_kicks - a_idx[j]) / self.sr


# --------------------------------------------------------------------------- OSC listener
class OscListener(threading.Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True, name="osc-listener")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.2)
        self.stop = False
        self.events: dict[str, list[float]] = {}
        self.sizes: dict[str, int] = {}
        self._lock = threading.Lock()

    def run(self):
        while not self.stop:
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            t = time.perf_counter()
            addr = data[: data.find(b"\x00")].decode("ascii", "replace")
            with self._lock:
                self.events.setdefault(addr, []).append(t)
                self.sizes[addr] = len(data)

    def window(self, t0: float, t1: float) -> dict[str, np.ndarray]:
        with self._lock:
            ev = {k: np.array(v) for k, v in self.events.items()}
        return {k: v[(v >= t0) & (v < t1)] for k, v in ev.items()}


# --------------------------------------------------------------------------- WS client
class WsClient(threading.Thread):
    """Streams like the UI does; records message counts and server_status."""

    def __init__(self, url: str):
        super().__init__(daemon=True, name="ws-client")
        self.url = url
        self.counts: dict[str, int] = {}
        self.fft_frames = 0
        self.statuses: list[tuple[float, dict]] = []
        self.meta: dict | None = None
        self.samples: dict[str, str] = {}
        self._quit = threading.Event()
        self.connected = threading.Event()

    def run(self):
        asyncio.run(self._main())

    async def _main(self):
        import websockets
        async with websockets.connect(self.url, max_size=None) as ws:
            self.connected.set()
            while not self._quit.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), 0.2)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, bytes):
                    self.fft_frames += 1
                    continue
                d = json.loads(msg)
                t = d.get("type", "?")
                self.counts[t] = self.counts.get(t, 0) + 1
                self.samples.setdefault(t, msg)
                if t == "server_status":
                    self.statuses.append((time.perf_counter(), d))
                elif t == "meta":
                    self.meta = d

    def close(self):
        self._quit.set()
        self.join(timeout=3)


# --------------------------------------------------------------------------- stats
def _dist(x: np.ndarray, scale: float = 1.0) -> dict:
    if x.size == 0:
        return {"n": 0}
    x = x * scale
    return {"n": int(x.size), "mean": float(np.mean(x)), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x))}


def _ps_time(s: str) -> float:
    # ps -M times: "m:ss.cc" or "h:mm:ss.cc"
    parts = [float(x) for x in s.split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def thread_times(pid: int) -> dict[str, float]:
    """Per-thread CPU seconds via `ps -M` (psutil's per-thread API needs
    task_for_pid, which macOS denies). Keyed "<ordinal>:<sched priority>" —
    thread order is stable for a process whose thread set doesn't change, and
    the priority column tells threads apart (e.g. 97R = CoreAudio realtime IO)."""
    if sys.platform != "darwin":
        try:
            return {str(t.id): t.user_time + t.system_time for t in psutil.Process(pid).threads()}
        except psutil.Error:
            return {}
    out = subprocess.run(["ps", "-M", "-p", str(pid)], capture_output=True, text=True).stdout
    res = {}
    for i, line in enumerate(out.splitlines()[1:]):
        cols = line.split()
        # first row: USER PID TT %CPU STAT PRI STIME UTIME CMD...;
        # other rows: PID %CPU STAT PRI STIME UTIME
        if len(cols) >= 8 and cols[1] == str(pid):
            cols = cols[3:]
        elif cols and cols[0] == str(pid):
            cols = cols[1:]
        if len(cols) < 5:
            continue
        pri, stime, utime = cols[2], cols[3], cols[4]
        res[f"{i}:{pri}"] = _ps_time(stime) + _ps_time(utime)
    return res


def cpu_snapshot(proc: psutil.Process) -> tuple[float, float, dict[str, float]]:
    ct = proc.cpu_times()
    return time.perf_counter(), ct.user + ct.system, thread_times(proc.pid)


def summarize_status(statuses: list[dict]) -> dict:
    if not statuses:
        return {}
    out = {"n": len(statuses)}
    for key in ("cb", "dsp", "fft", "ws", "lmh_e2e", "fft_e2e"):
        avgs = [s["perf"][key]["avg_ms"] for s in statuses if key in s.get("perf", {})]
        p95s = [s["perf"][key]["p95_ms"] for s in statuses if key in s.get("perf", {})]
        loads = [s["perf"][key]["load_pct"] for s in statuses if key in s.get("perf", {})]
        if avgs:
            out[key] = {"avg_ms_median": float(np.median(avgs)), "p95_ms_median": float(np.median(p95s)),
                        "p95_ms_max": float(np.max(p95s)), "load_pct_median": float(np.median(loads))}
    first, last = statuses[0], statuses[-1]
    for k in ("cb_overruns", "dsp_drops", "fft_drops"):
        out[k] = int(last.get(k, 0)) - int(first.get(k, 0))
    # Distinct drop events: status ticks (5 Hz) in which either counter rose.
    out["drop_events"] = sum(
        1 for a, b in zip(statuses, statuses[1:])
        if int(b.get("dsp_drops", 0)) > int(a.get("dsp_drops", 0))
        or int(b.get("fft_drops", 0)) > int(a.get("fft_drops", 0)))
    out["audio_stalled_any"] = any(s.get("audio_stalled") for s in statuses)
    out["block_period_ms"] = last["perf"]["block_period_ms"]
    out["hop_period_ms"] = last["perf"]["hop_period_ms"]
    return out


def measure_phase(name, proc, osc, player, duration, ws: WsClient | None):
    load0 = os.getloadavg()
    t0, c0, th0 = cpu_snapshot(proc)
    st0 = len(ws.statuses) if ws else 0
    counts0 = dict(ws.counts) if ws else {}
    fft0 = ws.fft_frames if ws else 0
    time.sleep(duration)
    t1, c1, th1 = cpu_snapshot(proc)
    wall = t1 - t0
    res = {"phase": name, "wall_s": wall,
           "cpu_pct_one_core": (c1 - c0) / wall * 100.0,
           "rss_mb": proc.memory_info().rss / 2**20,
           "num_threads": proc.num_threads(),
           "loadavg_1m_start_end": [round(load0[0], 2), round(os.getloadavg()[0], 2)]}
    per_thread = sorted(((tid, (th1.get(tid, 0.0) - th0.get(tid, 0.0)) / wall * 100.0) for tid in th1),
                        key=lambda x: -x[1])
    res["threads_cpu_pct"] = {tid: round(p, 2) for tid, p in per_thread if p > 0.05}
    win = osc.window(t0, t1)
    res["osc"] = {}
    for addr, ts in sorted(win.items()):
        d = {"rate_hz": ts.size / wall, "bytes": osc.sizes.get(addr)}
        if ts.size > 2 and addr in ("/audio/lmh", "/audio/fft", "/audio/bpm"):
            d["interarrival_ms"] = _dist(np.diff(ts), 1000.0)
        res["osc"][addr] = d
    if player is not None:
        kicks = player.kick_times()
        kicks = kicks[(kicks >= t0 - 0.5) & (kicks < t1)]
        onsets = win.get("/audio/onset/low", np.zeros(0))
        lats = []
        for k in kicks:
            later = onsets[(onsets > k) & (onsets < k + 0.25)]
            if later.size:
                lats.append(later[0] - k)
        res["kick_to_onset_low_ms"] = _dist(np.array(lats), 1000.0)
        res["kick_to_onset_low_ms"]["kicks"] = int(kicks.size)
    if ws is not None:
        res["ws_rates_hz"] = {k: (v - counts0.get(k, 0)) / wall for k, v in ws.counts.items()}
        res["ws_rates_hz"]["fft_binary"] = (ws.fft_frames - fft0) / wall
        res["server_status"] = summarize_status([s for _, s in ws.statuses[st0:]])
    return res


def probe_status(url: str) -> dict:
    """Connect briefly to read the greeting server_status (perf rings hold the
    last ~0.7 s of DSP / ~1.5 s of FFT samples, so they reflect the idle phase)."""
    c = WsClient(url)
    c.start()
    deadline = time.time() + 5
    while not c.statuses and time.time() < deadline:
        time.sleep(0.05)
    c.close()
    return summarize_status([s for _, s in c.statuses[:1]])


# --------------------------------------------------------------------------- main
def wait_port(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-cmd", default=f"{REPO}/.venv/bin/python -m server.main",
                   help="engine command; --config <tmp>/main.yaml is appended")
    p.add_argument("--label", default="engine")
    p.add_argument("--input", choices=("blackhole", "default"), default="blackhole")
    p.add_argument("--signal", default="drums")
    p.add_argument("--base-config", default=str(REPO / "configs" / "main.example.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, e.g. --set fft.n_bins=256 (repeatable)")
    p.add_argument("--warmup", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=30.0, help="seconds per phase")
    p.add_argument("--phases", default="idle,ui")
    p.add_argument("--out", default=None, help="JSON output path")
    args = p.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="live_bench_"))
    device = BLACKHOLE if args.input == "blackhole" else None
    cfg_path = write_config(Path(args.base_config), tmp, device, args.set)
    url = f"ws://127.0.0.1:{WS_PORT}"

    player = None
    if args.input == "blackhole":
        player = LoopPlayer(BLACKHOLE, args.signal)
        player.stream.start()

    osc = OscListener(OSC_PORT)
    osc.start()
    log_path = tmp / "server.log"
    cmd = shlex.split(args.server_cmd) + ["--config", str(cfg_path)]
    with open(log_path, "w") as log_fh:
        server = subprocess.Popen(cmd, cwd=REPO, stdout=log_fh, stderr=subprocess.STDOUT)
    results = {"label": args.label, "cmd": cmd, "config_overrides": args.set, "input": args.input,
               "signal": args.signal, "host": {"machine": os.uname().machine, "cpu_count": os.cpu_count()},
               "phases": []}
    try:
        if not wait_port(WS_PORT, 20):
            print(f"server did not open WS port; log: {log_path}", file=sys.stderr)
            print(log_path.read_text()[-3000:], file=sys.stderr)
            return 1
        proc = psutil.Process(server.pid)
        # Grab meta (sr etc.) once.
        c = WsClient(url)
        c.start()
        c.connected.wait(5)
        time.sleep(0.5)
        c.close()
        results["meta"] = {k: c.meta.get(k) for k in ("sr", "blocksize", "n_fft_bins", "fft_window_size",
                                                      "filter_order", "ws_snapshot_hz")} if c.meta else {}
        results["meta"]["hop"] = yaml.safe_load(cfg_path.read_text())["fft"]["hop"]
        time.sleep(args.warmup)
        for phase in args.phases.split(","):
            if phase == "idle":
                r = measure_phase("idle", proc, osc, player, args.duration, None)
                r["server_status"] = probe_status(url)
            elif phase == "ui":
                ws = WsClient(url)
                ws.start()
                ws.connected.wait(5)
                time.sleep(2.0)  # let the greeting burst pass
                r = measure_phase("ui", proc, osc, player, args.duration, ws)
                ws.close()
            else:
                raise SystemExit(f"unknown phase {phase}")
            results["phases"].append(r)
            time.sleep(1.0)
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        osc.stop = True
        if player is not None:
            player.stream.stop()
            player.stream.close()

    print_summary(results)
    out = Path(args.out) if args.out else tmp / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nJSON: {out}\nserver log: {log_path}")
    return 0


def print_summary(r: dict) -> None:
    m = r.get("meta", {})
    print(f"\n== {r['label']}  sr={m.get('sr')} block={m.get('blocksize')} "
          f"fft={m.get('fft_window_size')}/{m.get('hop')} bins={m.get('n_fft_bins')} "
          f"order={m.get('filter_order')} input={r['input']}:{r['signal']}")
    for ph in r["phases"]:
        print(f"\n-- phase {ph['phase']}: CPU {ph['cpu_pct_one_core']:.1f}% of one core, "
              f"RSS {ph['rss_mb']:.0f} MB, {ph['num_threads']} threads; "
              f"busiest threads % {dict(list(ph['threads_cpu_pct'].items())[:6])}")
        for addr, d in ph["osc"].items():
            ia = d.get("interarrival_ms")
            extra = (f"  inter-arrival ms p50 {ia['p50']:.2f} p95 {ia['p95']:.2f} "
                     f"p99 {ia['p99']:.2f} max {ia['max']:.2f}") if ia else ""
            print(f"   osc {addr:<18} {d['rate_hz']:7.1f} Hz {d['bytes']:>5} B{extra}")
        k = ph.get("kick_to_onset_low_ms")
        if k and k.get("n"):
            print(f"   kick -> /audio/onset/low ms: p50 {k['p50']:.1f} p95 {k['p95']:.1f} "
                  f"max {k['max']:.1f} (hit {k['n']}/{k['kicks']} kicks)")
        if "ws_rates_hz" in ph:
            print("   ws rates Hz: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(ph["ws_rates_hz"].items())))
        st = ph.get("server_status") or {}
        for key in ("cb", "dsp", "fft", "ws", "lmh_e2e", "fft_e2e"):
            if key in st:
                s = st[key]
                print(f"   status {key:<8} avg {s['avg_ms_median']*1000:8.1f} us  p95 {s['p95_ms_median']*1000:8.1f} us "
                      f"(worst p95 {s['p95_ms_max']*1000:8.1f} us)  load {s['load_pct_median']:5.2f}%")
        if st:
            print(f"   drops: cb_overruns {st.get('cb_overruns')} dsp {st.get('dsp_drops')} "
                  f"fft {st.get('fft_drops')} stalled={st.get('audio_stalled_any')}")


if __name__ == "__main__":
    sys.exit(main())
