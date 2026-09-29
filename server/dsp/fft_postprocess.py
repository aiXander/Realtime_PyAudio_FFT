"""Per-bin port of features.AutoScaler for the FFT log bins.

Mirrors the L/M/H pipeline (pre-tilt → smoother → asymmetric peak follower
→ peak smear → soft gate → tanh → strength blend) so a single set of
controls tunes both pipelines in lockstep.

Output is in [0, 1]; the WS broadcaster and OSC sender pick between this
"processed" stream and the raw dB stream based on `cfg.fft.send_raw_db`.

Threading: configured from the asyncio loop, but `process()` runs on the
FFT worker thread. Cross-thread mutations go through `update_*` methods
that take `_lock`; `process()` also takes `_lock`. The lock is held for
hundreds of µs per hop (n_bins ~128, all-vector ops) and reconfigure
events are rare.

Output is double-buffered: process() writes into one of two preallocated
f32 arrays and returns the just-written one, toggling for the next call.
The returned ref stays valid until two further process() calls; downstream
consumers (OSC, WS encoder) read the array synchronously and drop the ref
before then, so no copy is needed on the publish path.
"""
from __future__ import annotations

import math
import threading

import numpy as np
from scipy.ndimage import convolve1d

from ._math import (
    EPS,
    LN10_OVER_20,
    log_bin_centers,
    tau_to_alpha,
    tau_to_alpha_into,
    tilt_db_curve,
)


class FFTPostProcessor:
    def __init__(self, n_bins: int, f_min: float, sr: float,
                 bands: dict, tau: dict, tau_attack: dict,
                 tau_release_s: float, noise_floor: float, strength: float,
                 db_floor: float = -60.0, db_ceiling: float = 0.0,
                 hop_period_s: float = 512 / 48000.0,
                 tau_attack_s: float = 0.05,
                 peak_smear_oct: float = 0.3,
                 tilt_db_per_oct: float = 3.0):
        self._lock = threading.Lock()
        self.n_bins = int(n_bins)
        self.f_min = float(f_min)
        self.sr = float(sr)
        self.bands = bands
        self.tau = tau
        self.tau_attack = tau_attack
        self.tau_release_s = float(tau_release_s)
        self.noise_floor = max(0.0, float(noise_floor))
        self.strength = max(0.0, min(1.0, float(strength)))
        self.db_floor = float(db_floor)
        self.db_ceiling = float(db_ceiling)
        self.hop_period_s = float(hop_period_s)
        self.tau_attack_s = float(tau_attack_s)
        self.peak_smear_oct = max(0.0, float(peak_smear_oct))
        self.tilt_db_per_oct = float(tilt_db_per_oct)
        self._allocate()

    # ----------------- allocation -----------------
    def _allocate(self) -> None:
        n = self.n_bins
        self.smooth_lin = np.zeros(n, dtype=np.float64)
        self.peak_lin = np.full(n, max(self.noise_floor, EPS), dtype=np.float64)
        self.peak_smoothed = np.full(n, max(self.noise_floor, EPS), dtype=np.float64)
        self.interp_db = np.zeros(n, dtype=np.float64)
        # Double-buffered f32 output. process() writes into one buffer, returns
        # the ref, and toggles. Caller may read the returned ref until the
        # next-next process() call (two hops later) without copying. Single
        # producer + asyncio readers that drop the ref synchronously after
        # encoding makes this race-free.
        self._processed_buffers = (
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )
        self._proc_idx = 0
        # scratch (float64), repurposed across pipeline stages
        self._lin = np.zeros(n, dtype=np.float64)
        self._diff = np.zeros(n, dtype=np.float64)
        self._alpha = np.zeros(n, dtype=np.float64)
        self._scratch = np.zeros(n, dtype=np.float64)
        self._mask = np.zeros(n, dtype=bool)
        # Sentinel-fill LUT, built lazily on the first hop (post-processor
        # doesn't see the bin layout until it gets a real frame). Bin layout
        # is stable until reconfigure(), so the LUT is rebuilt once per
        # reconfigure and the hot path is alloc-free thereafter.
        self._empty_idx: np.ndarray | None = None
        self._left_idx: np.ndarray | None = None
        self._right_idx: np.ndarray | None = None
        self._w_left: np.ndarray | None = None
        self._w_right: np.ndarray | None = None
        self._left_vals: np.ndarray | None = None
        self._right_vals: np.ndarray | None = None
        self.tau_per_bin = self._build_per_bin_tau_for(self.tau)
        self.tau_attack_per_bin = self._build_per_bin_tau_for(self.tau_attack)
        # Per-bin alpha for the ASYMMETRIC signal smoother. alpha_smooth_release
        # is selected on falling samples (slow → averages sustained content);
        # alpha_smooth_attack is selected on rising samples (fast → onsets pass).
        self.alpha_smooth_release = np.zeros(n, dtype=np.float64)
        self.alpha_smooth_attack = np.zeros(n, dtype=np.float64)
        self.tilt_db = self._build_tilt_db()
        self._recompute_alphas()
        self._recompute_smear()
        self._warmed = False

    def _build_tilt_db(self) -> np.ndarray:
        f_center = log_bin_centers(self.n_bins, self.f_min, self.sr / 2.0)
        return tilt_db_curve(f_center, self.tilt_db_per_oct).astype(np.float64)

    def _build_sentinel_interp_lut(self, db_in: np.ndarray) -> None:
        """Detect empty-bin positions (sentinel = -1000 dB) and precompute a
        linear-interpolation LUT (left/right valid-neighbor source indices
        and per-empty-bin weights) so the hot path can fill sentinels with
        a fancy-take + multiply-add + fancy-scatter, no `np.interp` call
        and no per-hop temporaries.

        Edge handling: empty bins below the lowest valid bin take the
        rightmost valid neighbor's value (w_left=0); empty bins above the
        highest valid bin take the leftmost valid neighbor's value
        (w_right=0). If all bins are valid, the LUT is empty and the hot
        path skips the fill entirely.
        """
        valid_mask = db_in >= -500.0   # sentinel is -1000; real values >= db_floor
        valid_idx = np.flatnonzero(valid_mask)
        empty_idx = np.flatnonzero(~valid_mask)

        if empty_idx.size == 0 or valid_idx.size == 0:
            self._empty_idx = np.zeros(0, dtype=np.intp)
            self._left_idx = np.zeros(0, dtype=np.intp)
            self._right_idx = np.zeros(0, dtype=np.intp)
            self._w_left = np.zeros(0, dtype=np.float64)
            self._w_right = np.zeros(0, dtype=np.float64)
            self._left_vals = np.zeros(0, dtype=np.float64)
            self._right_vals = np.zeros(0, dtype=np.float64)
            return

        # right_pos = leftmost index in valid_idx whose value is >= empty_idx[k].
        # left_pos = right_pos - 1. May be out of range at the spectrum edges.
        right_pos = np.searchsorted(valid_idx, empty_idx, side="left")
        left_pos = right_pos - 1
        has_left = left_pos >= 0
        has_right = right_pos < valid_idx.size
        left_pos_safe = np.where(has_left, left_pos, 0)
        right_pos_safe = np.where(has_right, right_pos, valid_idx.size - 1)
        L = valid_idx[left_pos_safe]
        R = valid_idx[right_pos_safe]

        span = (R - L).astype(np.float64)
        safe_span = np.where(span > 0, span, 1.0)
        w_left_default = (R - empty_idx) / safe_span
        # has_left & has_right → linear blend; only-left → w_left=1; only-right → w_left=0.
        w_left = np.where(
            has_left & has_right, w_left_default,
            np.where(has_left, 1.0, 0.0),
        )

        self._empty_idx = empty_idx.astype(np.intp)
        self._left_idx = L.astype(np.intp)
        self._right_idx = R.astype(np.intp)
        self._w_left = w_left.astype(np.float64)
        self._w_right = (1.0 - w_left).astype(np.float64)
        self._left_vals = np.zeros(empty_idx.size, dtype=np.float64)
        self._right_vals = np.zeros(empty_idx.size, dtype=np.float64)

    def _recompute_smear(self) -> None:
        """Convert peak_smear_oct → Gaussian sigma in *bin index* units AND
        precompute the Gaussian kernel.

        The smear in process() uses mode='reflect', which is self-normalising
        at the edges (mirror-padded neighbours have the same kernel mass as
        the interior), so no per-bin normalisation weights are needed.

        Precomputing the kernel avoids ~20-50µs of per-hop overhead on Pi 4
        that scipy.ndimage.gaussian_filter1d spends rebuilding the kernel
        each call. We use convolve1d with these weights instead, which is
        otherwise identical (mode='reflect', truncate=4σ).
        """
        if self.peak_smear_oct <= 0.0:
            self._smear_sigma_bins = 0.0
            self._smear_kernel = None
            return
        f_max = self.sr / 2.0
        if f_max <= self.f_min:
            self._smear_sigma_bins = 0.0
            self._smear_kernel = None
            return
        bins_per_octave = self.n_bins / max(1e-6, math.log2(f_max / self.f_min))
        sigma = float(self.peak_smear_oct * bins_per_octave)
        self._smear_sigma_bins = sigma
        if sigma <= 0.0:
            self._smear_kernel = None
            return
        # Match scipy.ndimage.gaussian_filter1d defaults: truncate=4.0,
        # radius = int(truncate*sigma + 0.5). The kernel is symmetric and
        # normalised so the smear preserves total energy under interior bins.
        radius = max(1, int(4.0 * sigma + 0.5))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        self._smear_kernel = k

    def _build_per_bin_tau_for(self, tau_dict: dict) -> np.ndarray:
        anchors = []
        for name in ("low", "mid", "high"):
            b = self.bands[name]
            cen = math.sqrt(float(b["lo_hz"]) * float(b["hi_hz"]))
            anchors.append((math.log10(max(cen, 1e-3)), float(tau_dict[name])))
        anchors.sort(key=lambda x: x[0])
        log_f = np.log10(log_bin_centers(self.n_bins, self.f_min, self.sr / 2.0))
        xp = np.array([a[0] for a in anchors], dtype=np.float64)
        fp = np.array([a[1] for a in anchors], dtype=np.float64)
        # np.interp clamps to fp[0]/fp[-1] outside [xp[0], xp[-1]] — exactly the
        # nearest-band behaviour we want.
        return np.interp(log_f, xp, fp)

    def _recompute_alphas(self) -> None:
        h = self.hop_period_s
        # Per-bin signal smoother: separate attack / release alphas, selected
        # by sign(diff) per bin in the hot path.
        tau_to_alpha_into(h, self.tau_per_bin, self.alpha_smooth_release)
        tau_to_alpha_into(h, self.tau_attack_per_bin, self.alpha_smooth_attack)
        # Peak-follower alphas (scalar — single tau_attack_s/tau_release_s
        # shared with the L/M/H AutoScaler).
        self.alpha_attack = tau_to_alpha(h, self.tau_attack_s)
        self.alpha_release = tau_to_alpha(h, self.tau_release_s)

    # ----------------- live retune (asyncio thread) -----------------
    def reset(self) -> None:
        with self._lock:
            self.smooth_lin.fill(0.0)
            self.peak_lin.fill(max(self.noise_floor, EPS))
            self._warmed = False

    def reconfigure(self, *, n_bins=None, f_min=None, sr=None,
                    bands=None, tau=None, tau_attack=None, hop_period_s=None,
                    db_floor=None, db_ceiling=None) -> None:
        with self._lock:
            if n_bins is not None: self.n_bins = int(n_bins)
            if f_min is not None: self.f_min = float(f_min)
            if sr is not None: self.sr = float(sr)
            if bands is not None: self.bands = bands
            if tau is not None: self.tau = tau
            if tau_attack is not None: self.tau_attack = tau_attack
            if hop_period_s is not None: self.hop_period_s = float(hop_period_s)
            if db_floor is not None: self.db_floor = float(db_floor)
            if db_ceiling is not None: self.db_ceiling = float(db_ceiling)
            self._allocate()

    def update_smoothing(self, tau: dict, tau_attack: dict) -> None:
        with self._lock:
            self.tau = tau
            self.tau_attack = tau_attack
            self.tau_per_bin = self._build_per_bin_tau_for(self.tau)
            self.tau_attack_per_bin = self._build_per_bin_tau_for(self.tau_attack)
            self._recompute_alphas()

    def update_bands(self, bands: dict) -> None:
        with self._lock:
            self.bands = bands
            self.tau_per_bin = self._build_per_bin_tau_for(self.tau)
            self.tau_attack_per_bin = self._build_per_bin_tau_for(self.tau_attack)
            self._recompute_alphas()

    def update_smear(self, peak_smear_oct: float) -> None:
        with self._lock:
            self.peak_smear_oct = max(0.0, float(peak_smear_oct))
            self._recompute_smear()

    def update_tilt(self, tilt_db_per_oct: float) -> None:
        with self._lock:
            self.tilt_db_per_oct = float(tilt_db_per_oct)
            self.tilt_db = self._build_tilt_db()

    def update_autoscale(self, *, tau_attack_s=None, tau_release_s=None,
                         noise_floor=None, strength=None) -> None:
        with self._lock:
            if tau_attack_s is not None:
                self.tau_attack_s = float(tau_attack_s)
            if tau_release_s is not None:
                self.tau_release_s = float(tau_release_s)
            if noise_floor is not None:
                self.noise_floor = max(0.0, float(noise_floor))
                np.maximum(self.peak_lin, max(self.noise_floor, EPS), out=self.peak_lin)
            if strength is not None:
                self.strength = max(0.0, min(1.0, float(strength)))
            self._recompute_alphas()

    # ----------------- hot path (FFT worker thread) -----------------
    def process(self, db_in: np.ndarray) -> np.ndarray:
        """Run the pipeline on `db_in` (float32, length n_bins, raw wire dB
        with -1000 sentinels for empty log bins). Returns a float32 view of
        the just-written output buffer (in [0, 1]).

        The output is one of two preallocated buffers — the returned ref is
        valid until two subsequent process() calls. Don't keep it longer.
        """
        with self._lock:
            # Pick the back buffer (the one not last published). Toggle at end.
            out = self._processed_buffers[self._proc_idx]

            # 1. Sentinel interpolation. Empty bins arrive as -1000 dB and
            #    must be filled with values consistent with their valid
            #    neighbors — clamping to db_floor leaves visible gaps in the
            #    spectrum at the low end (where consecutive log bins are
            #    empty because rfft Δf > log-bin width). Use a precomputed
            #    LUT (built once per bin layout) for an alloc-free linear
            #    blend of the left/right valid neighbors.
            self.interp_db[:] = db_in   # f32 → f64
            if self._empty_idx is None:
                self._build_sentinel_interp_lut(db_in)
            if self._empty_idx.size > 0:
                # mode="clip": indices are in range by construction, and the
                # default mode="raise" always buffers `out` (allocates).
                np.take(self.interp_db, self._left_idx, out=self._left_vals, mode="clip")
                np.take(self.interp_db, self._right_idx, out=self._right_vals, mode="clip")
                np.multiply(self._left_vals, self._w_left, out=self._left_vals)
                np.multiply(self._right_vals, self._w_right, out=self._right_vals)
                np.add(self._left_vals, self._right_vals, out=self._left_vals)
                self.interp_db[self._empty_idx] = self._left_vals

            # 2. Pre-tilt (in dB) + dB→linear into _lin.
            #    interp_db is the UNTILTED clamped spectrum and is reused below
            #    for the strength<1 baseline; do not mutate it.
            #    10^(x/20) ≡ exp(x · ln10/20); np.exp is several× faster than
            #    np.power(10.0, ...) on ARM and x86.
            np.add(self.interp_db, self.tilt_db, out=self._lin)
            np.multiply(self._lin, LN10_OVER_20, out=self._lin)
            np.exp(self._lin, out=self._lin)

            # 3. Per-bin ASYMMETRIC EMA smoothing — fast attack, slow release,
            #    selected per bin by sign(diff). Same shape as the L/M/H
            #    ExpSmoother and as the peak follower below.
            if not self._warmed:
                np.copyto(self.smooth_lin, self._lin)
                # Don't saturate the peak follower to the first sample —
                # initialise to noise_floor and let attack chase up over
                # tau_attack_s. Avoids one transient locking in tau_release_s
                # of suppression on every other bin.
                self.peak_lin.fill(max(self.noise_floor, EPS))
                self._warmed = True
            else:
                np.subtract(self._lin, self.smooth_lin, out=self._diff)
                np.greater(self._diff, 0.0, out=self._mask)
                np.copyto(self._alpha, self.alpha_smooth_release)
                np.copyto(self._alpha, self.alpha_smooth_attack, where=self._mask)
                np.multiply(self._diff, self._alpha, out=self._diff)
                np.add(self.smooth_lin, self._diff, out=self.smooth_lin)

            # 4. Asymmetric peak follower (in-place, preallocated buffers).
            #    fill+masked-copy is alloc-free because _mask is preallocated;
            #    np.where(...) would otherwise allocate a fresh result each
            #    call (no out= support).
            np.subtract(self.smooth_lin, self.peak_lin, out=self._diff)
            np.greater(self._diff, 0.0, out=self._mask)
            self._alpha.fill(self.alpha_release)
            np.copyto(self._alpha, self.alpha_attack, where=self._mask)
            np.multiply(self._diff, self._alpha, out=self._diff)
            np.add(self.peak_lin, self._diff, out=self.peak_lin)

            # 4b. Spatial Gaussian smear of the peak across log-frequency bins.
            #     mode='reflect' is self-normalising at the edges (mirrored
            #     neighbours carry the same kernel mass), so no normalisation
            #     divide is needed. Kernel is precomputed in _recompute_smear
            #     so the hot path is a single convolve1d call.
            if self._smear_kernel is not None:
                convolve1d(self.peak_lin, self._smear_kernel,
                           output=self.peak_smoothed, mode="reflect")
            else:
                np.copyto(self.peak_smoothed, self.peak_lin)

            # 5. AutoScaler core: tanh( max(0, smooth - nf) / max(peak, nf) ).
            #    In-place via _diff (gated/scaled) and _alpha (denom).
            nf = max(self.noise_floor, EPS)
            np.subtract(self.smooth_lin, nf, out=self._diff)
            np.maximum(self._diff, 0.0, out=self._diff)
            np.maximum(self.peak_smoothed, nf, out=self._alpha)
            np.divide(self._diff, self._alpha, out=self._diff)
            np.tanh(self._diff, out=self._diff)

            # 6. Strength blend.
            s = self.strength
            if s >= 1.0:
                out[:] = self._diff   # f64 → f32
                self._proc_idx ^= 1
                return out

            # Raw dB-mapped baseline from UNTILTED interp_db, [0, 1]; gate-zero
            # bins whose untilted value is below the noise gate.
            disp_floor = self.db_floor
            disp_span = max(1.0, self.db_ceiling - disp_floor)
            np.subtract(self.interp_db, disp_floor, out=self._scratch)
            np.multiply(self._scratch, 1.0 / disp_span, out=self._scratch)
            np.clip(self._scratch, 0.0, 1.0, out=self._scratch)
            noise_floor_db = 20.0 * math.log10(nf)
            np.less(self.interp_db, noise_floor_db, out=self._mask)
            self._scratch[self._mask] = 0.0

            # blended = s·scaled + (1-s)·baseline
            np.multiply(self._diff, s, out=self._diff)
            np.multiply(self._scratch, 1.0 - s, out=self._scratch)
            np.add(self._diff, self._scratch, out=self._diff)
            out[:] = self._diff
            self._proc_idx ^= 1
            return out
