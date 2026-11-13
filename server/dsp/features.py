"""Per-band exponential smoother + rolling auto-scaler.

Mirrors `dsp/fft_postprocess.py`: tilt is applied as a per-band LINEAR
pre-multiplier on the smoothed RMS, the noise floor is a single global
linear value, and the strength<1 baseline is a dB-mapped [0,1] curve
(with the same gate-zeroing). One `noise_floor` slider therefore gates
both pipelines in lockstep.
"""
from __future__ import annotations

import math

import numpy as np

from ._math import EPS, LN10_OVER_20, tau_to_alpha, tilt_db_curve


def block_rms(x: np.ndarray) -> float:
    """RMS of a 1-D buffer; returns Python float.

    np.dot(x, x) is a single BLAS sdot/ddot call with no per-block temporary,
    vs np.square(...) which allocates a length-blocksize array each call.
    """
    return math.sqrt(float(np.dot(x, x)) / x.size)


# `/audio/level` scale. Constants, not config: consumers write their own
# thresholds in this scale, so it is part of the wire contract. Mirrored in
# engine/dsp/src/features.rs.
LEVEL_TAU_S = 0.1
LEVEL_FLOOR_DB = -60.0
_LEVEL_FLOOR_POWER = 10.0 ** (LEVEL_FLOOR_DB / 10.0)  # 1e-6


class LevelMeter:
    """Broadband loudness on a fixed scale, never auto-gained (`/audio/level`).

    An RMS meter over the mono block BEFORE the filter bank: one-pole smooth
    the mean square (0.1 s), then map dBFS linearly so -60 dB reads 0 and
    0 dB reads 1. Smoothing power (not dB) keeps the reading independent of
    the block size. Not scaled by master_gain, not noise-gated.
    """

    def __init__(self, sr: float, blocksize: int):
        self._scratch = np.zeros(int(blocksize), dtype=np.float64)
        self.reconfigure(sr, blocksize)

    def reconfigure(self, sr: float, blocksize: int) -> None:
        """New sr / blocksize (device hot-switch): re-derive alpha, zero p."""
        if self._scratch.size != int(blocksize):
            self._scratch = np.zeros(int(blocksize), dtype=np.float64)
        self._alpha = tau_to_alpha(int(blocksize) / float(sr), LEVEL_TAU_S)
        self._p = 0.0

    def update(self, x: np.ndarray) -> float:
        """One block of mono samples -> level in [0, 1]. No allocation."""
        s = self._scratch
        # float32 -> float64 first: np.dot on float32 sums in float32, which
        # is outside the 1e-9 parity tolerance against the Rust engine.
        np.copyto(s, x)
        ms = float(np.dot(s, s)) / s.size
        if not math.isfinite(ms):
            ms = 0.0
        p = self._p + self._alpha * (ms - self._p)
        self._p = p
        if p <= _LEVEL_FLOOR_POWER:  # also guards log10(0)
            return 0.0
        return min(1.0, max(0.0, 1.0 + 10.0 * math.log10(p) / -LEVEL_FLOOR_DB))


class ExpSmoother:
    """Per-band ASYMMETRIC one-pole envelope follower.

    Two per-band time constants — fast ATTACK on rising values, slow RELEASE
    on falling values:
        diff = rms - v
        alpha = alpha_attack[band]   if diff > 0
                alpha_release[band]  otherwise
        v += alpha * diff

    Why asymmetric: a symmetric smoother forces a single τ to cover both
    "catch the kick onset" (wants ~ms) and "don't bounce on harmonic flutter
    in a sustained note" (wants ~100 ms). No single value wins. Decoupling
    them lets onsets pass through with attack-shaped latency (a few ms) while
    sustained material averages out over the release τ. This is the standard
    envelope-follower / VU-meter shape, and it's what every audio compressor
    sidechain uses for the same perceptual reason.

    alpha = 1 - exp(-blocksize / (sr * tau)).
    """

    def __init__(self, sr: float, blocksize: int, tau: dict, tau_attack: dict):
        self.sr = float(sr)
        self.blocksize = int(blocksize)
        self._values = np.zeros(3, dtype=np.float64)
        self.set_tau(tau, tau_attack)

    def set_tau(self, tau: dict, tau_attack: dict) -> None:
        dt = self.blocksize / self.sr
        self._alpha_release = np.array(
            [tau_to_alpha(dt, tau[k]) for k in ("low", "mid", "high")],
            dtype=np.float64,
        )
        self._alpha_attack = np.array(
            [tau_to_alpha(dt, tau_attack[k]) for k in ("low", "mid", "high")],
            dtype=np.float64,
        )

    def update(self, rms_lo: float, rms_md: float, rms_hi: float) -> tuple[float, float, float]:
        # Unrolled 3-band asymmetric one-pole. Python scalars + per-element
        # branching — for n=3 this is trivially cheap and avoids the overhead
        # of allocating 3-element numpy temporaries on every audio block.
        v = self._values
        aa = self._alpha_attack
        ar = self._alpha_release
        d = rms_lo - v[0]; v[0] += (aa[0] if d > 0.0 else ar[0]) * d
        d = rms_md - v[1]; v[1] += (aa[1] if d > 0.0 else ar[1]) * d
        d = rms_hi - v[2]; v[2] += (aa[2] if d > 0.0 else ar[2]) * d
        return float(v[0]), float(v[1]), float(v[2])

    @property
    def values(self) -> np.ndarray:
        return self._values

    def reset(self) -> None:
        self._values.fill(0.0)


_DEFAULT_BANDS = {"low": (40.0, 250.0), "mid": (250.0, 2000.0), "high": (2000.0, 16000.0)}


class AutoScaler:
    """Rolling per-band normalizer.

    Pipeline (pre-tilted smoothed RMS in `v`):
        peak  ← asymmetric one-pole follower of v (fast attack, slow release)
        gated ← max(0, v - noise_floor)
        sc    ← tanh( gated / max(peak, noise_floor) )          # in [0, 1]
        raw   ← clip( (20·log10(v) - db_floor) / (db_ceil-db_floor), 0, 1 )
                with raw=0 where v < noise_floor                # in [0, 1]
        out   ← strength·sc + (1-strength)·raw                  # in [0, 1]

    Tilt is applied as a per-band linear pre-multiplier on `values_in`
    (= 10^(tilt_db/20), where tilt_db = tilt_db_per_oct · log2(f_c / 1 kHz)).
    Same shape as the FFT post-processor, where tilt is added in dB before
    the dB→linear conversion — so a single global noise_floor gates both
    pipelines identically (for tones).

    For broadband noise, the integrated band power scales with band width,
    so the same `noise_floor` value (which gates the FFT viz per-bin) does
    NOT gate the L/M/H aggregate equivalently — wider bands integrate more
    sub-floor noise and read higher. To keep the single knob coherent, we
    subtract a per-band noise budget before everything else:
        clean_rms² = max(0, rms² − noise_floor² · n_bins_eff)
    where `n_bins_eff = max(1, K_lin / N_log)` is the average count of
    linear rfft bins per FFT-viz log bin in this band (`K_avg_log`), with
    a 1-bin floor for narrow bands where log bins are smaller than rfft
    bins (low/mid). Semantically: this is "one log bin's worth of
    floor-level noise", which is exactly the threshold the FFT viz's
    per-bin gate uses (a log bin reads visible iff its linear amplitude
    exceeds `nf`, i.e. its summed-over-K_avg-linear-bins power exceeds
    `K_avg · nf²`). Anything visible on a single log bin in the FFT
    contributes ≥ this much to integrated band power and survives.
    Subtracting the full `K_lin · nf²` (the broadband-at-floor budget)
    over-penalizes narrowband content in wide bands by ~`N_log` —
    e.g. for the high band, K_lin ≈ 274 vs K_avg ≈ 9 — and would
    wipe out snare hits that show clearly on the FFT spectrum. The
    downstream `v − nf` gate + slow peak follower take care of any
    residual broadband bleed.
    """

    def __init__(self, sr: float, blocksize: int,
                 tau_attack_s: float = 0.05, tau_release_s: float = 60.0,
                 noise_floor: float = 1e-3, strength: float = 1.0,
                 tilt_db_per_oct: float = 0.0,
                 bands: dict | None = None,
                 n_fft_window: int = 1024,
                 fft_n_bins: int = 192,
                 fft_f_min: float = 30.0,
                 db_floor: float = -60.0, db_ceiling: float = 0.0):
        self.sr = float(sr)
        self.blocksize = int(blocksize)
        self.noise_floor = max(0.0, float(noise_floor))
        self.strength = max(0.0, min(1.0, float(strength)))
        self.tilt_db_per_oct = float(tilt_db_per_oct)
        self._bands = self._normalize_bands(bands if bands is not None else _DEFAULT_BANDS)
        self._n_fft_window = max(1, int(n_fft_window))
        self._fft_n_bins = max(1, int(fft_n_bins))
        self._fft_f_min = max(1e-3, float(fft_f_min))
        self._band_centers = np.zeros(3, dtype=np.float64)
        self._band_widths = np.zeros(3, dtype=np.float64)
        self._log_bins_per_band = np.zeros(3, dtype=np.float64)
        # Per-band 1/sqrt(K_lin), used to express integrated band RMS as
        # "average per-rfft-bin amplitude" in the strength<1 baseline so it
        # reads the same as an FFT log bin in the same frequency range.
        self._per_bin_factor = np.ones(3, dtype=np.float64)
        self._recompute_band_geom()
        self._recompute_log_bin_geom()
        self._recompute_per_bin_factor()
        self.db_floor = float(db_floor)
        self.db_ceiling = float(db_ceiling)
        self._tilt_lin = np.ones(3, dtype=np.float64)
        self._noise_pwr_per_band = np.zeros(3, dtype=np.float64)  # nf² · bins_in_band
        self._v = np.zeros(3, dtype=np.float64)         # tilted, noise-subtracted input
        self._v2 = np.zeros(3, dtype=np.float64)        # scratch for v² / clean_rms
        self._diff = np.zeros(3, dtype=np.float64)      # peak follower delta
        self._alpha = np.zeros(3, dtype=np.float64)     # per-band attack/release
        self._mask = np.zeros(3, dtype=bool)            # rising-edge mask (preallocated)
        self._scaled = np.zeros(3, dtype=np.float64)    # tanh output
        self._raw = np.zeros(3, dtype=np.float64)       # dB-mapped baseline
        self._peak = np.full(3, max(self.noise_floor, EPS), dtype=np.float64)
        self._warmed = False
        self.tau_attack_s = float(tau_attack_s)
        self.tau_release_s = float(tau_release_s)
        self._recompute_tilt_lin()
        self._recompute_noise_pwr()
        self.set_taus(tau_attack_s=self.tau_attack_s, tau_release_s=self.tau_release_s)

    @staticmethod
    def _normalize_bands(bands: dict) -> dict:
        return {k: (float(bands[k][0]), float(bands[k][1])) for k in ("low", "mid", "high")}

    def _recompute_band_geom(self) -> None:
        for i, name in enumerate(("low", "mid", "high")):
            lo, hi = self._bands[name]
            self._band_centers[i] = math.sqrt(max(lo, 1e-3) * max(hi, 1e-3))
            self._band_widths[i] = max(hi - lo, 0.0)

    def _recompute_tilt_lin(self) -> None:
        if self.tilt_db_per_oct == 0.0:
            self._tilt_lin.fill(1.0)
            return
        # 10^(tilt_db/20) ≡ exp(tilt_db · LN10_OVER_20); avoids np.power's
        # internal log+mul+exp.
        tilt_db = tilt_db_curve(self._band_centers, self.tilt_db_per_oct)
        np.exp(tilt_db * LN10_OVER_20, out=self._tilt_lin)

    def _recompute_per_bin_factor(self) -> None:
        # K_lin = BW · n_fft_window / sr — count of linear rfft bins in the
        # band. _per_bin_factor = 1/sqrt(max(K_lin, 1)). For broadband-uniform
        # input at amplitude `a` per linear bin, rms_band = a · sqrt(K_lin),
        # so rms_band · _per_bin_factor = a — i.e. what an FFT log bin in
        # this range would read. Floored at 1 so very narrow bands (K_lin < 1)
        # don't artificially boost the reading.
        scale = self._n_fft_window / max(self.sr, 1e-3)
        for i in range(3):
            k_lin = max(self._band_widths[i] * scale, 1.0)
            self._per_bin_factor[i] = 1.0 / math.sqrt(k_lin)

    def _recompute_log_bin_geom(self) -> None:
        # FFT-viz log-bin count inside each band. log_bins_per_octave is
        # constant across the spectrum; multiply by the band's octave width
        # to get its log-bin count. Used to cap the linear-bin noise budget.
        nyq = max(self.sr / 2.0, 1e-3)
        log_oct_total = math.log2(max(nyq / self._fft_f_min, 1.0 + 1e-6))
        log_bins_per_oct = self._fft_n_bins / max(log_oct_total, 1e-3)
        for i, name in enumerate(("low", "mid", "high")):
            lo, hi = self._bands[name]
            oct_band = math.log2(max(hi, 1e-3) / max(lo, 1e-3))
            self._log_bins_per_band[i] = log_bins_per_oct * max(oct_band, 0.0)

    def _recompute_noise_pwr(self) -> None:
        # Per-band noise budget: nf² · n_bins_eff, where
        #   n_bins_eff = max(1, K_lin / N_log)  — average linear bins per
        #                                          FFT-viz log bin in the
        #                                          band (K_avg_log), floored
        #                                          at 1 for narrow bands
        #                                          where log bins are smaller
        #                                          than rfft bins.
        # This is "one log bin's worth of floor noise" — exactly the
        # threshold the FFT viz's per-bin gate uses. Any single log bin
        # visible on the FFT contributes ≥ this much to integrated band
        # power and survives. Subtracting K_lin · nf² (full broadband
        # budget) instead would over-penalize narrowband content by ~N_log
        # and kill snare hits that show clearly on the FFT.
        nf2 = float(self.noise_floor) ** 2
        scale = self._n_fft_window / max(self.sr, 1e-3)
        # k_lin = BW · n_fft_window / sr; n_bins_eff = max(1, k_lin / N_log)
        np.multiply(self._band_widths, scale, out=self._noise_pwr_per_band)
        # Divide by N_log; guard against zero-width log bands.
        np.divide(self._noise_pwr_per_band,
                  np.maximum(self._log_bins_per_band, 1e-3),
                  out=self._noise_pwr_per_band)
        np.maximum(self._noise_pwr_per_band, 1.0, out=self._noise_pwr_per_band)
        np.multiply(self._noise_pwr_per_band, nf2, out=self._noise_pwr_per_band)

    def set_taus(self, tau_attack_s: float, tau_release_s: float) -> None:
        dt = self.blocksize / self.sr
        self._a_atk = tau_to_alpha(dt, tau_attack_s)
        self._a_rel = tau_to_alpha(dt, tau_release_s)
        self.tau_attack_s = float(tau_attack_s)
        self.tau_release_s = float(tau_release_s)

    def set_noise_floor(self, floor: float) -> None:
        self.noise_floor = max(float(floor), 0.0)
        np.maximum(self._peak, max(self.noise_floor, EPS), out=self._peak)
        self._recompute_noise_pwr()

    def set_tilt(self, tilt_db_per_oct: float) -> None:
        self.tilt_db_per_oct = float(tilt_db_per_oct)
        self._recompute_tilt_lin()

    def set_bands(self, bands: dict) -> None:
        self._bands = self._normalize_bands(bands)
        self._recompute_band_geom()
        self._recompute_tilt_lin()
        self._recompute_log_bin_geom()
        self._recompute_per_bin_factor()
        self._recompute_noise_pwr()

    def set_n_fft_window(self, n: int) -> None:
        self._n_fft_window = max(1, int(n))
        self._recompute_per_bin_factor()
        self._recompute_noise_pwr()

    def set_fft_geometry(self, n_bins: int | None = None,
                         f_min: float | None = None) -> None:
        """Update FFT-viz log-bin geometry (n_bins, f_min). Recomputes the
        per-band log-bin count → noise budget cap. sr changes go through
        the construction path (hot-switch rebuilds AutoScaler), so they
        don't need a runtime setter here."""
        if n_bins is not None:
            self._fft_n_bins = max(1, int(n_bins))
        if f_min is not None:
            self._fft_f_min = max(1e-3, float(f_min))
        self._recompute_log_bin_geom()
        self._recompute_noise_pwr()

    def set_strength(self, strength: float) -> None:
        self.strength = max(0.0, min(1.0, float(strength)))

    def reset(self) -> None:
        self._peak.fill(max(self.noise_floor, EPS))
        self._warmed = False

    def update(self, values_in: np.ndarray, out: np.ndarray) -> np.ndarray:
        # Hot path: scalar Python unroll for n=3. Numpy ufunc dispatch costs
        # ~1-3µs per call regardless of array length — for length-3 arrays
        # that overhead dominates the actual arithmetic by ~20×. Calling the
        # ~16 ufuncs the vectorised version used was burning ~6-15ms/sec on
        # Pi 4 just on Python-side bookkeeping. The unrolled version below
        # is a few µs per block. Mirrors ExpSmoother.update for the same reason.
        nf2 = self.noise_floor                    # gate threshold (may be 0)
        nf = nf2 if nf2 > EPS else EPS            # divisor floor

        peak = self._peak                         # length-3 f64 (persistent)
        tilt = self._tilt_lin
        npwr = self._noise_pwr_per_band
        a_atk = self._a_atk
        a_rel = self._a_rel

        # Read input scalars once (numpy scalar attribute lookups aren't free).
        r0 = float(values_in[0]); r1 = float(values_in[1]); r2 = float(values_in[2])

        # Bandwidth-aware noise subtraction in power, then sqrt + tilt.
        c0 = r0 * r0 - npwr[0]; c0 = 0.0 if c0 < 0.0 else c0
        c1 = r1 * r1 - npwr[1]; c1 = 0.0 if c1 < 0.0 else c1
        c2 = r2 * r2 - npwr[2]; c2 = 0.0 if c2 < 0.0 else c2
        v0 = math.sqrt(c0) * tilt[0]
        v1 = math.sqrt(c1) * tilt[1]
        v2 = math.sqrt(c2) * tilt[2]

        # Asymmetric peak follower (warm to nf — don't saturate to first sample).
        if not self._warmed:
            peak[0] = nf; peak[1] = nf; peak[2] = nf
            self._warmed = True
        p0 = peak[0]; p1 = peak[1]; p2 = peak[2]
        d0 = v0 - p0; p0 += (a_atk if d0 > 0.0 else a_rel) * d0
        d1 = v1 - p1; p1 += (a_atk if d1 > 0.0 else a_rel) * d1
        d2 = v2 - p2; p2 += (a_atk if d2 > 0.0 else a_rel) * d2
        peak[0] = p0; peak[1] = p1; peak[2] = p2

        # tanh( max(0, v - nf2) / max(peak, nf) )
        g0 = v0 - nf2; g0 = 0.0 if g0 < 0.0 else g0
        g1 = v1 - nf2; g1 = 0.0 if g1 < 0.0 else g1
        g2 = v2 - nf2; g2 = 0.0 if g2 < 0.0 else g2
        sc0 = math.tanh(g0 / (p0 if p0 > nf else nf))
        sc1 = math.tanh(g1 / (p1 if p1 > nf else nf))
        sc2 = math.tanh(g2 / (p2 if p2 > nf else nf))

        s = self.strength
        if s >= 1.0:
            out[0] = sc0; out[1] = sc1; out[2] = sc2
            return out

        # Raw dB-mapped baseline (strength<1 path). Mirrors the FFT
        # post-processor's strength<1 path: UNTILTED, per-rfft-bin equivalent
        # (raw · _per_bin_factor), gate at noise_floor (matches FFT viz).
        span = self.db_ceiling - self.db_floor
        if span < 1.0: span = 1.0
        inv_span = 1.0 / span
        df = self.db_floor
        pbf = self._per_bin_factor

        def _baseline(raw, factor):
            pb = raw * factor
            if pb < nf2:
                return 0.0
            if pb < EPS:
                pb = EPS
            v = (20.0 * math.log10(pb) - df) * inv_span
            if v < 0.0: return 0.0
            if v > 1.0: return 1.0
            return v

        b0 = _baseline(r0, pbf[0])
        b1 = _baseline(r1, pbf[1])
        b2 = _baseline(r2, pbf[2])

        if s <= 0.0:
            out[0] = b0; out[1] = b1; out[2] = b2
        else:
            inv_s = 1.0 - s
            out[0] = s * sc0 + inv_s * b0
            out[1] = s * sc1 + inv_s * b1
            out[2] = s * sc2 + inv_s * b2
        return out
