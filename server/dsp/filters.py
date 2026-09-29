"""Three independent Butterworth bandpass filters: LOW / MID / HIGH, SOS form."""
from __future__ import annotations

import logging
import threading

import numpy as np
from scipy.signal import iirfilter, sosfilt

log = logging.getLogger(__name__)

BAND_NAMES = ("low", "mid", "high")

# Defensive band-edge limits applied at design time (see clamp_band_edges).
_HI_MAX_FRAC_OF_SR = 0.45   # hi edge ≤ 0.45·sr  (= 0.9·Nyquist)
_LO_MIN_HZ = 1.0            # lo edge ≥ 1 Hz (Wn must be > 0)
_LO_MAX_FRAC_OF_HI = 0.9    # lo edge ≤ 0.9·hi (bandpass needs lo < hi)


def clamp_band_edges(lo: float, hi: float, sr: float) -> tuple[float, float]:
    """Clamp a bandpass [lo, hi] (Hz) into a range scipy can always design
    at sample rate `sr`: 1 Hz ≤ lo ≤ 0.9·hi, hi ≤ 0.45·sr.

    Pure; no side effects. A 16000 Hz high edge on a 16/24 kHz device would
    otherwise make iirfilter raise "Wn must be 0 < Wn < 1".
    """
    sr = float(sr)
    hi = max(min(float(hi), _HI_MAX_FRAC_OF_SR * sr), 2.0 * _LO_MIN_HZ)
    lo = max(float(lo), _LO_MIN_HZ)
    lo = min(lo, _LO_MAX_FRAC_OF_HI * hi)
    return lo, hi


# ---------------------------------------------------------------------------
# Fast in-place SOS kernel.
#
# Public scipy.signal.sosfilt costs ~26 µs per band on a 256-sample block,
# almost all of it Python-side validation / dtype promotion / moveaxis /
# copies. The Cython kernel it wraps, scipy.signal._sosfilt._sosfilt(sos, x,
# zi), filters x in place and updates zi in place in ~4–5 µs with
# bit-identical output. It's private API, so: guarded import + a runtime
# self-check against the public function on a tiny input; fall back to the
# public sosfilt if either fails.
#
# Kernel contract (scipy 1.17 _signaltools.sosfilt): all three arrays share
# one dtype (float64 here), C-contiguous, shapes
#   sos: (n_sections, 6)   x: (n_signals, n_samples)   zi: (n_signals, n_sections, 2)
# ---------------------------------------------------------------------------
def _probe_private_sosfilt():
    try:
        from scipy.signal._sosfilt import _sosfilt as kernel
    except Exception:  # noqa: BLE001 — any import problem → fallback
        return None
    try:
        sos = np.ascontiguousarray(
            iirfilter(2, [0.1, 0.3], btype="bandpass", ftype="butter", output="sos"),
            dtype=np.float64,
        )
        rng = np.random.default_rng(1)
        x = rng.standard_normal(64).astype(np.float32)
        zi0 = rng.standard_normal((sos.shape[0], 2))
        ref_y, ref_zf = sosfilt(sos, x, zi=zi0)
        xb = np.empty((1, x.size), dtype=np.float64)
        xb[0] = x
        zb = np.ascontiguousarray(zi0.reshape(1, sos.shape[0], 2))
        kernel(sos, xb, zb)
        if np.array_equal(xb[0], ref_y) and np.array_equal(zb[0], ref_zf):
            return kernel
        log.warning("filters: private sosfilt kernel mismatch; using public sosfilt")
    except Exception as e:  # noqa: BLE001
        log.warning("filters: private sosfilt kernel unusable (%s); using public sosfilt", e)
    return None


_sosfilt_kernel = _probe_private_sosfilt()


class FilterBank:
    """Three independent bandpass filters, owned by the DSP worker.

    Each band has its own [lo_hz, hi_hz] edges. Bands may overlap or leave
    gaps — that's intentional; it lets you carve out specific energy regions
    (e.g. drop sub-rumble from "low" by setting low.lo_hz=30, or scoop a
    midrange hole). Edges are clamped at design time via `clamp_band_edges`
    so an edge above the device's Nyquist can never crash the design;
    `self.bands` keeps the requested values, `self.effective_bands` the
    clamped ones actually in use.

    `process(x)` is allocation-free on the fast path: the float32 block is
    cast into three preallocated float64 (1, blocksize) buffers which the
    private scipy kernel filters in place, with zi (float64, IIR numerical
    stability) also updated in place. The returned arrays are those
    preallocated buffers — valid until the next `process()` call.
    """

    def __init__(self, sr: float, bands: dict, blocksize: int, order: int = 4):
        # bands: {"low": (lo_hz, hi_hz), "mid": (...), "high": (...)}
        self.sr = float(sr)
        self.order = int(order)
        self.blocksize = int(blocksize)
        self._lock = threading.Lock()
        self._alloc_io(self.blocksize)
        self._design(bands)
        self._init_state()
        self.bands = {k: (float(v[0]), float(v[1])) for k, v in bands.items()}

    def _alloc_io(self, n: int) -> None:
        self._n = int(n)
        self._buf_low = np.zeros((1, n), dtype=np.float64)
        self._buf_mid = np.zeros((1, n), dtype=np.float64)
        self._buf_high = np.zeros((1, n), dtype=np.float64)
        # 1-D views, created once so the hot path doesn't build view objects.
        self._out_low = self._buf_low[0]
        self._out_mid = self._buf_mid[0]
        self._out_high = self._buf_high[0]

    def _design_one(self, lo: float, hi: float) -> tuple[np.ndarray, tuple[float, float]]:
        nyq = self.sr / 2.0
        lo_c, hi_c = clamp_band_edges(lo, hi, self.sr)
        sos = iirfilter(
            self.order, [lo_c / nyq, hi_c / nyq],
            btype="bandpass", ftype="butter", output="sos",
        )
        return np.ascontiguousarray(sos, dtype=np.float64), (lo_c, hi_c)

    def _design(self, bands: dict):
        self.sos_low, eff_low = self._design_one(*bands["low"])
        self.sos_mid, eff_mid = self._design_one(*bands["mid"])
        self.sos_high, eff_high = self._design_one(*bands["high"])
        self.effective_bands = {"low": eff_low, "mid": eff_mid, "high": eff_high}

    def _init_state(self):
        # (n_signals=1, n_sections, 2), C-contiguous — the private kernel's
        # zi layout. zi[0] is the public-sosfilt (n_sections, 2) layout.
        self.zi_low = np.zeros((1, self.sos_low.shape[0], 2), dtype=np.float64)
        self.zi_mid = np.zeros((1, self.sos_mid.shape[0], 2), dtype=np.float64)
        self.zi_high = np.zeros((1, self.sos_high.shape[0], 2), dtype=np.float64)

    def retune(self, bands: dict) -> None:
        """Called from the asyncio loop. Brief click acceptable on retune."""
        with self._lock:
            self._design(bands)
            self._init_state()
            self.bands = {k: (float(v[0]), float(v[1])) for k, v in bands.items()}

    def set_order(self, order: int) -> None:
        """Change the Butterworth order for all three bands. Higher order →
        steeper skirts, more biquads, more CPU. Brief click on change."""
        with self._lock:
            self.order = int(order)
            self._design(self.bands)
            self._init_state()

    def reset_state(self) -> None:
        with self._lock:
            self._init_state()

    def process(self, x: np.ndarray):
        with self._lock:
            if x.shape[0] != self._n:
                self._alloc_io(x.shape[0])  # off the steady-state path
            out_lo = self._out_low
            out_md = self._out_mid
            out_hi = self._out_high
            np.copyto(out_lo, x)
            np.copyto(out_md, x)
            np.copyto(out_hi, x)
            kernel = _sosfilt_kernel
            if kernel is not None:
                kernel(self.sos_low, self._buf_low, self.zi_low)
                kernel(self.sos_mid, self._buf_mid, self.zi_mid)
                kernel(self.sos_high, self._buf_high, self.zi_high)
            else:
                self._process_public(self.sos_low, out_lo, self.zi_low)
                self._process_public(self.sos_mid, out_md, self.zi_mid)
                self._process_public(self.sos_high, out_hi, self.zi_high)
        return out_lo, out_md, out_hi

    @staticmethod
    def _process_public(sos: np.ndarray, buf: np.ndarray, zi: np.ndarray) -> None:
        """Fallback: public sosfilt (allocates), results copied back in place."""
        y, zf = sosfilt(sos, buf, zi=zi[0])
        np.copyto(buf, y)
        np.copyto(zi[0], zf)
