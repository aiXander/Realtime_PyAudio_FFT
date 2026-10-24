# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0", "sounddevice>=0.4.6", "websockets>=14,<17", "psutil>=5.9", "pyyaml>=6.0"]
# ///
"""Record wire goldens from a live server: one of every WS message type, both
binary FFT flavours, one OSC packet per address, and the YAML the persister /
preset writer produce.

    uv run tools/parity/record_wire.py                      # Python engine -> testdata/golden/wire/
    uv run tools/parity/record_wire.py --out /tmp/rust_wire \\
        --server-cmd engine/target/release/audio-server     # same capture from the Rust engine

Values (levels, device names, timings) differ run to run; what the Rust
engine must match is the *shape*: JSON key sets + value types per message,
OSC address / type-tag / payload layout, binary FFT header. Compare two
captures with `--compare A B`. Feeds the server the BlackHole drum loop
(tools/bench/live_bench.py) so onsets fire.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
from live_bench import BLACKHOLE, OSC_PORT, WS_PORT, LoopPlayer, wait_port, write_config  # noqa: E402

OUT = REPO / "testdata" / "golden" / "wire"


class OscCapture(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", OSC_PORT))
        self.sock.settimeout(0.2)
        self.packets: dict[str, bytes] = {}
        self.fft_raw: bytes | None = None
        self.want_raw = False
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            addr = data[: data.find(b"\x00")].decode()
            if addr == "/audio/fft" and self.want_raw:
                self.fft_raw = data
            else:
                self.packets[addr] = data  # keep the latest (the first lmh/bpm are pre-input zeros)


async def ws_session(url: str, out: Path, osc: OscCapture) -> list[dict]:
    import websockets
    log: list[dict] = []
    seen: dict[str, str] = {}

    async with websockets.connect(url, max_size=None) as ws:
        async def recv_until(pred, timeout=5.0):
            deadline = time.time() + timeout
            got = []
            while time.time() < deadline:
                try:
                    m = await asyncio.wait_for(ws.recv(), 0.5)
                except asyncio.TimeoutError:
                    continue
                got.append(m)
                if pred(m):
                    return m, got
            raise TimeoutError("expected message not received")

        def is_type(t):
            return lambda m: isinstance(m, str) and json.loads(m).get("type") == t

        def save(name: str, msg, request=None):
            p = out / name
            if isinstance(msg, bytes):
                p.write_bytes(msg)
            else:
                p.write_text(msg)
            log.append({"file": name, "request": request})

        # Greeting: meta, devices, presets, server_status (+ stream).
        greet = []
        for t in ("meta", "devices", "presets", "server_status"):
            m, got = await recv_until(is_type(t))
            greet += got
            seen.setdefault(t, m)
        save("ws_meta.json", seen["meta"], "greeting")
        save("ws_devices.json", seen["devices"], "greeting")
        save("ws_presets.json", seen["presets"], "greeting")
        save("ws_server_status.json", seen["server_status"], "greeting / 5 Hz")
        snap, _ = await recv_until(lambda m: isinstance(m, str) and json.loads(m).get("type") == "snapshot"
                                   and json.loads(m).get("bpm", 0) > 0, timeout=15)
        save("ws_snapshot.json", snap, "stream (60 Hz)")
        onset_snap, _ = await recv_until(lambda m: isinstance(m, str) and json.loads(m).get("type") == "snapshot"
                                         and json.loads(m).get("low_onset") == 1, timeout=10)
        save("ws_snapshot_onset.json", onset_snap, "stream, a tick where low_onset fired")
        fft, _ = await recv_until(lambda m: isinstance(m, bytes))
        save("ws_fft_processed.bin", fft, "stream, meta.fft_send_raw_db=false")

        # Errors: unknown type, invalid value, structurally bad JSON.
        await ws.send(json.dumps({"type": "no_such_message"}))
        e1, got = await recv_until(is_type("error"))
        # (an unknown type gets the error only — no snap-back meta)
        save("ws_error_unknown_type.json", e1, {"type": "no_such_message"})
        bad = {"type": "set_band", "band": "low", "lo_hz": 500.0, "hi_hz": 100.0}
        await ws.send(json.dumps(bad))
        e2, _ = await recv_until(is_type("error"))
        save("ws_error_invalid_value.json", e2, bad)
        m, _ = await recv_until(is_type("meta"))
        save("ws_meta_after_error.json", m, "targeted snap-back meta right after a validation error")
        await ws.send("{not json")
        e3, _ = await recv_until(is_type("error"))
        save("ws_error_invalid_json.json", e3, "{not json")
        await ws.send(json.dumps({"no_type": 1}))
        e4, _ = await recv_until(is_type("error"))
        save("ws_error_missing_type.json", e4, {"no_type": 1})

        # Raw-dB mode: WS binary + OSC flip together.
        await ws.send(json.dumps({"type": "set_fft_send_raw_db", "send_raw_db": True}))
        m, _ = await recv_until(lambda m: isinstance(m, str) and json.loads(m).get("type") == "meta"
                                and json.loads(m).get("fft_send_raw_db") is True)
        save("ws_meta_raw_db.json", m, "coalesced meta after set_fft_send_raw_db true")
        osc.want_raw = True
        await asyncio.sleep(0.3)
        rawfft, _ = await recv_until(lambda m: isinstance(m, bytes))
        save("ws_fft_raw_db.bin", rawfft, "stream, meta.fft_send_raw_db=true")
        osc.want_raw = False
        await ws.send(json.dumps({"type": "set_fft_send_raw_db", "send_raw_db": False}))
        await recv_until(lambda m: isinstance(m, str) and json.loads(m).get("type") == "meta"
                         and json.loads(m).get("fft_send_raw_db") is False)

        # Presets + devices replies.
        await ws.send(json.dumps({"type": "save_preset", "name": "wire golden"}))
        m, _ = await recv_until(is_type("presets"))
        save("ws_presets_after_save.json", m, {"type": "save_preset", "name": "wire golden"})
        await ws.send(json.dumps({"type": "list_devices", "probe": False}))
        m, _ = await recv_until(is_type("devices"), timeout=10)
        save("ws_devices_reply.json", m, {"type": "list_devices", "probe": False})

        # set_stream pause/resume: resume re-sends meta.
        await ws.send(json.dumps({"type": "set_stream", "enabled": False}))
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "set_stream", "enabled": True}))
        m, _ = await recv_until(is_type("meta"))
        save("ws_meta_on_resume.json", m, {"type": "set_stream", "enabled": True})

        # A slider drag (commit=false) → persister writes main.yaml within ~1 s.
        await ws.send(json.dumps({"type": "set_autoscale", "strength": 0.65, "commit": False}))
        await recv_until(lambda m: isinstance(m, str) and json.loads(m).get("type") == "meta"
                         and json.loads(m)["autoscale"]["strength"] == 0.65)
    return log


def record(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file():
            p.unlink()
    tmp = Path(tempfile.mkdtemp(prefix="record_wire_"))
    cfg_path = write_config(REPO / "configs" / "main.example.yaml", tmp, BLACKHOLE, [])
    player = LoopPlayer(BLACKHOLE, "drums")
    player.stream.start()
    osc = OscCapture()
    osc.start()
    cmd = shlex.split(args.server_cmd) + ["--config", str(cfg_path)]
    log_fh = open(tmp / "server.log", "w")
    server = subprocess.Popen(cmd, cwd=REPO, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        if not wait_port(WS_PORT, 20):
            print((tmp / "server.log").read_text()[-3000:])
            return 1
        log = asyncio.run(ws_session(f"ws://127.0.0.1:{WS_PORT}", out, osc))
        time.sleep(1.5)  # persister drag debounce (1 s, capped 250 ms after first dirty)
    finally:
        server.send_signal(signal.SIGINT)
        server.wait(timeout=10)
        osc.stop = True
        player.stream.stop()
    for addr, data in sorted(osc.packets.items()):
        name = "osc" + addr.replace("/", "_") + ".bin"
        (out / name).write_bytes(data)
        log.append({"file": name, "request": "OSC capture (last packet before shutdown)"})
    if osc.fft_raw is not None:
        (out / "osc_audio_fft_raw_db.bin").write_bytes(osc.fft_raw)
        log.append({"file": "osc_audio_fft_raw_db.bin", "request": "OSC /audio/fft with send_raw_db=true"})
    (out / "persisted_main.yaml").write_text((tmp / "main.yaml").read_text())
    log.append({"file": "persisted_main.yaml", "request": "main.yaml as rewritten by the persister"})
    preset = tmp / "wire golden.yaml"
    if preset.exists():
        (out / "preset_saved.yaml").write_text(preset.read_text())
        log.append({"file": "preset_saved.yaml", "request": "save_preset 'wire golden'"})
    (out / "index.json").write_text(json.dumps({"server_cmd": cmd, "files": log}, indent=1) + "\n")
    missing = {"/audio/lmh", "/audio/bpm", "/audio/fft", "/audio/meta", "/audio/onset/low",
               "/audio/onset/mid", "/audio/onset/high"} - set(osc.packets)
    print(f"wrote {len(log)} files to {out}" + (f"; MISSING OSC {sorted(missing)}" if missing else ""))
    return 1 if missing or osc.fft_raw is None else 0


# --------------------------------------------------------------------------- compare
def json_shape(v):
    """Key sets + JSON value types (numbers collapse to 'number'; ints are
    kept apart where the Python side guarantees an int, e.g. meta.sr)."""
    if isinstance(v, dict):
        return {k: json_shape(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [json_shape(v[0])] if v else []
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "number"
    if v is None:
        return "null"
    return type(v).__name__


def osc_layout(data: bytes) -> dict:
    end = data.find(b"\x00")
    addr = data[:end].decode()
    i = (end // 4 + 1) * 4
    tend = data.find(b"\x00", i)
    tags = data[i:tend].decode()
    j = (tend // 4 + 1) * 4
    return {"address": addr, "address_field_bytes": i, "typetags": tags[:12] + ("…" if len(tags) > 12 else ""),
            "n_args": len(tags) - 1, "payload_offset": j, "total_bytes": len(data)}


def compare(a: Path, b: Path) -> int:
    bad = 0
    for pa in sorted(a.iterdir()):
        pb = b / pa.name
        if pa.name == "index.json" or not pb.exists():
            if not pb.exists() and pa.name != "index.json":
                print(f"MISSING in {b}: {pa.name}")
                bad += 1
            continue
        if pa.suffix == ".json":
            ok = json_shape(json.loads(pa.read_text())) == json_shape(json.loads(pb.read_text()))
        elif pa.name.startswith("osc_"):
            ok = osc_layout(pa.read_bytes()) == osc_layout(pb.read_bytes())
        elif pa.name.startswith("ws_fft"):
            ha, hb = pa.read_bytes()[:4], pb.read_bytes()[:4]
            ok = ha == hb and len(pa.read_bytes()) == len(pb.read_bytes())
        else:
            import yaml
            ok = json_shape(yaml.safe_load(pa.read_text())) == json_shape(yaml.safe_load(pb.read_text()))
        print(f"{'ok  ' if ok else 'DIFF'} {pa.name}")
        bad += not ok
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cmd", default=f"{REPO}/.venv/bin/python -m server.main")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args(argv)
    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))
    return record(args)


if __name__ == "__main__":
    sys.exit(main())
