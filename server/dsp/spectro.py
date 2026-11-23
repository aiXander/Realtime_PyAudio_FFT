"""Multi-resolution spectrogram for *looking at* sound (the UI's 3D FFT view).

Python twin of `engine/dsp/src/spectro.rs` + `engine/server/src/spectro.rs`:
same analysis, same `set_spectro` control message, same binary wire. Nothing
downstream (OSC, L/M/H, LEDs) reads it. `Spectro.process` is the source of
the spectro parity goldens (tools/parity/make_goldens.py) that the Rust twin
is tested against.

Every hop runs a ladder of Hann windows (×2 sizes, 512 … 8192 samples at
48 kHz) that all peak at one instant, ~21 ms behind the newest sample; the
long ones are lopsided so they can (`ladder_window`). Each log bin reads the
window whose
resolution suits its frequency (wanted window = sr / (f · rel_res), constant
Q, clamped to the ladder), crossfading in dB between neighbouring windows.
Amplitude-normalised (a sine reads its dBFS level), zero-padded 2×; a log bin
narrower than an rfft bin interpolates, a wider one takes the max of the bins
it covers. No smoothing, no autoscale.

`SpectroService` is the worker: a plain (non-realtime) thread that polls the
SlotRing every few ms while at least one streaming WS client subscribed, and
queues u8-quantised frames for the WS broadcast tick (`take_message`).

Wire (binary, little-endian), one message per broadcast tick:
  [0] u8 type=2  [1] u8 version=1  [2] u16 n_bins  [4] u16 n_frames  [6] u16 0
  [8] u32 first_seq  [12] f32 hop_s  [16] f32 f_min  [20] f32 f_max
  [24] f32 db0  [28] f32 db_step   (dB = db0 + q·db_step)
  [32] u8 × n_bins × n_frames, frame-major
"""

from __future__ import annotations

import logging
import struct
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

DB_FLOOR = -140.0
MIN_BINS, MAX_BINS, DEFAULT_BINS = 64, 2048, 1024
F_MIN, F_MAX = 20.0, 20_000.0
DB0, DB_STEP = -127.5, 0.5
REL_RES = {"time": 0.08, "balanced": 0.045, "pitch": 0.02}
_QUEUE_MAX_FRAMES = 1024
_MAX_BACKLOG_HOPS = 64
_POLL_S, _IDLE_S = 0.004, 0.1
_HEADER = struct.Struct("<BBHHHIfffff")


def _min_window(sr: float) -> int:
    p = min(14.0, max(6.0, round(np.log2(sr / 93.75))))
    return 1 << int(p)


def window_sizes(sr: float, max_window: int) -> list[int]:
    w_min = min(_min_window(sr), max_window)
    w_max = min(w_min * 16, max(max_window, w_min))
    out, w = [], w_min
    while w <= w_max:
        out.append(w)
        w *= 2
    return out


def ladder_window(w: int, sizes: list[int]) -> tuple[np.ndarray, int]:
    """Window `w` of the ladder: (weights, first sample within the newest
    `sizes[-1]` samples).

    The picture has to keep up with the sound, so every window peaks at the
    same instant, `delay` = half of the third window (1024 samples, ~21 ms at
    48 kHz) behind the newest sample. A window up to twice that long is a
    plain Hann centred there. A longer one cannot be centred that late, so it
    is lopsided: a long half-Hann rise out of the past and a short half-Hann
    fall of `delay` samples, ending at the newest sample. Next to a centred
    Hann of the same length, a kick shows in the 8192 window ~30 ms sooner
    (−12 dB after 32 ms instead of 63) and the peak is as narrow at −6 dB;
    the price is wider skirts (−20 dB at ±2.8 rfft bins instead of ±1.7).
    """
    delay = min(sizes[-1], sizes[0] * 4) // 2
    fall = min(w // 2, delay)
    rise = w - fall
    i = np.arange(w)
    phase = np.where(i < rise, (i + 0.5) / (2 * rise), 0.5 + (i - rise + 0.5) / (2 * fall))
    return 0.5 - 0.5 * np.cos(2 * np.pi * phase), sizes[-1] - delay - rise


def hop_blocks(sr: float, blocksize: int) -> int:
    return max(1, int(round(sr * 0.005 / blocksize)))


class Spectro:
    def __init__(self, sr: float, n_bins: int, f_min: float, f_max: float,
                 rel_res: float, max_window: int):
        sr = max(1000.0, float(sr))
        n_bins = max(2, int(n_bins))
        f_max = min(f_max, sr / 2)
        f_min = min(max(f_min, 1.0), f_max / 2)
        self.sr, self.n_bins, self.f_min, self.f_max = sr, n_bins, f_min, f_max
        self.sizes = window_sizes(sr, max_window)
        self.span = self.sizes[-1]
        self.starts = []
        self.hann = []
        self.norm2 = []
        self.offsets = []          # each window's start in the flat mag² array
        off = 0
        for w in self.sizes:
            h, start = ladder_window(w, self.sizes)
            self.hann.append(h)
            self.starts.append(start)
            self.norm2.append((2.0 / h.sum()) ** 2)
            self.offsets.append(off)
            off += w + 1           # rfft of 2w → w + 1 bins
        k_max = len(self.sizes) - 1
        edges = f_min * 2.0 ** (np.arange(n_bins + 1) / n_bins * np.log2(f_max / f_min))
        f_lo, f_hi = edges[:-1], edges[1:]
        f_c = np.sqrt(f_lo * f_hi)
        k = np.clip(np.log2(sr / (f_c * max(rel_res, 1e-4)) / self.sizes[0]), 0, k_max)
        a = np.minimum(np.floor(k).astype(int), max(0, k_max - 1))
        b = np.minimum(a + 1, k_max)
        self.t = np.where(a == b, 0.0, k - a)
        self.taps = [self._taps(a, f_lo, f_c, f_hi), self._taps(b, f_lo, f_c, f_hi)]
        self._flat = np.zeros(off)

    def _taps(self, win, f_lo, f_c, f_hi):
        """Per bin: either max over flat[idx[j, :]] or lerp flat[lo] → flat[lo+1] by frac."""
        sizes = np.array(self.sizes)[win]
        bin_hz = self.sr / (2 * sizes)
        last = sizes.astype(float)                  # last rfft index of a 2w transform
        lo = np.ceil(f_lo / bin_hz)
        hi = np.minimum(np.floor(f_hi / bin_hz), last)
        interp = hi < lo
        x = np.clip(f_c / bin_hz, 0, last - 1)
        ilo = np.floor(x)
        frac = x - ilo
        base = np.array(self.offsets)[win]
        width = int(np.max(np.where(interp, 1, hi - lo + 1)))
        steps = np.arange(width)
        rng = np.where(interp, 0, hi - lo)[:, None]
        idx = base[:, None] + np.where(interp[:, None], ilo[:, None], lo[:, None] + np.minimum(steps, rng))
        return {
            "idx": idx.astype(np.int64),
            "interp": interp,
            "ilo": (base + ilo).astype(np.int64),
            "frac": frac,
        }

    def process(self, samples: np.ndarray) -> np.ndarray:
        """One frame from the newest `span` samples → n_bins dB values (amplitude dBFS)."""
        x = samples[-self.span:]
        flat = self._flat
        for w, h, n2, off, o in zip(self.sizes, self.hann, self.norm2, self.offsets, self.starts):
            spec = np.fft.rfft(x[o:o + w] * h, n=2 * w)
            flat[off:off + w + 1] = (spec.real ** 2 + spec.imag ** 2) * n2
        out = []
        for tp in self.taps:
            vmax = flat[tp["idx"]].max(axis=1)
            vint = flat[tp["ilo"]] * (1 - tp["frac"]) + flat[tp["ilo"] + 1] * tp["frac"]
            p = np.where(tp["interp"], vint, vmax)
            out.append(np.maximum(10.0 * np.log10(p + 1e-20), DB_FLOOR))
        return out[0] + (out[1] - out[0]) * self.t


class SpectroService:
    """Spectro worker thread + the control-loop side (requests, queue, perf)."""

    def __init__(self, ring, sr: float, blocksize: int, max_window: int):
        self.ring = ring
        self.blocksize = blocksize
        self.max_window = max_window
        self._sr = float(sr)
        self._n_bins, self._res = DEFAULT_BINS, "balanced"
        self._epoch = 1
        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._q = None             # dict: epoch, n_bins, hop_s, f_min, f_max, first_seq, frames(list[bytes])
        self._perf = np.zeros(64, dtype=np.int64)
        self._perf_idx = 0
        self._thread = threading.Thread(target=self._run, name="spectro-worker", daemon=True)

    # ---------------- control side (asyncio loop) ----------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._enabled.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def set_enabled(self, on: bool) -> None:
        (self._enabled.set if on else self._enabled.clear)()

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    def set_request(self, n_bins: int, res: str) -> None:
        with self._lock:
            if (n_bins, res) != (self._n_bins, self._res):
                self._n_bins, self._res = n_bins, res
                self._epoch += 1

    def set_sr(self, sr: float) -> None:
        with self._lock:
            if float(sr) != self._sr:
                self._sr = float(sr)
                self._epoch += 1

    def request(self) -> tuple[int, str]:
        return self._n_bins, self._res

    def hop_s(self) -> float:
        return hop_blocks(self._sr, self.blocksize) * self.blocksize / self._sr

    def _params(self):
        with self._lock:
            sr, n, res, ep = self._sr, self._n_bins, self._res, self._epoch
        return ep, dict(sr=sr, n_bins=n, f_min=F_MIN, f_max=min(F_MAX, 0.49 * sr),
                        rel_res=REL_RES[res], max_window=self.max_window)

    def info(self, enabled: bool) -> dict:
        _, p = self._params()
        return {"type": "spectro", "enabled": enabled, "n_bins": self._n_bins, "res": self._res,
                "hop_s": self.hop_s(), "f_min": p["f_min"], "f_max": p["f_max"],
                "windows": window_sizes(p["sr"], self.max_window), "sr": p["sr"]}

    def perf_message(self) -> dict:
        n = min(self._perf_idx, self._perf.size)
        v = np.sort(self._perf[:n][self._perf[:n] > 0])
        avg = float(v.mean()) / 1e6 if v.size else 0.0
        p95 = float(v[int(0.95 * (v.size - 1))]) / 1e6 if v.size else 0.0
        return {"type": "spectro_perf", "avg_ms": avg, "p95_ms": p95,
                "load_pct": avg / (self.hop_s() * 1000.0) * 100.0}

    def take_message(self) -> bytes | None:
        with self._lock:
            q = self._q
            if not q or not q["frames"]:
                return None
            frames = q["frames"]
            head = _HEADER.pack(2, 1, q["n_bins"], len(frames), 0, q["first_seq"] & 0xFFFFFFFF,
                                q["hop_s"], q["f_min"], q["f_max"], DB0, DB_STEP)
            q["first_seq"] += len(frames)
            q["frames"] = []
        return head + b"".join(frames)

    # ---------------- worker thread ----------------
    def _push(self, epoch: int, seq: int, sp: Spectro, hop_s: float, db: np.ndarray) -> None:
        q8 = np.clip(np.round((db - DB0) / DB_STEP), 0, 255).astype(np.uint8).tobytes()
        with self._lock:
            q = self._q
            if (q is None or q["epoch"] != epoch or not q["frames"]
                    or seq != q["first_seq"] + len(q["frames"])
                    or len(q["frames"]) >= _QUEUE_MAX_FRAMES):
                q = self._q = {"epoch": epoch, "n_bins": sp.n_bins, "hop_s": hop_s, "f_min": sp.f_min,
                               "f_max": sp.f_max, "first_seq": seq, "frames": []}
            q["frames"].append(q8)

    def _run(self) -> None:
        bs = self.blocksize
        sp = None
        epoch_built = 0
        active = False
        read_idx = seq = 0
        span_blocks = hb = 1
        hop_s = 0.0
        buf = None
        while not self._stop.is_set():
            if not self._enabled.is_set():
                active = False
                self._enabled.wait(_IDLE_S)
                continue
            try:
                epoch, p = self._params()
                if sp is None or epoch != epoch_built:
                    sp = Spectro(**p)
                    epoch_built = epoch
                    span_blocks = -(-sp.span // bs)
                    hb = hop_blocks(p["sr"], bs)
                    hop_s = hb * bs / p["sr"]
                    buf = np.zeros(span_blocks * bs, dtype=np.float64)
                    active = False
                wi = self.ring.write_idx
                if not active or read_idx > wi:
                    read_idx = max(0, wi - span_blocks)
                    active = True
                if wi - read_idx > span_blocks + _MAX_BACKLOG_HOPS * hb:
                    k = (wi - span_blocks - read_idx) // hb
                    read_idx += k * hb
                    seq += k
                while wi - read_idx >= span_blocks and not self._stop.is_set():
                    t0 = time.perf_counter_ns()
                    if self.ring.try_read_window(read_idx, span_blocks, buf):
                        self._push(epoch_built, seq, sp, hop_s, sp.process(buf))
                        self._perf[self._perf_idx % self._perf.size] = time.perf_counter_ns() - t0
                        self._perf_idx += 1
                    read_idx += hb
                    seq += 1
            except Exception:
                log.exception("spectro worker iteration failed; continuing")
                active = False
            time.sleep(_POLL_S)
