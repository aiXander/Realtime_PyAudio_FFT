"""Per-band onset detector + low-band BPM estimator.

Runs once per audio block inside DSPWorker on the FULLY POST-PROCESSED
L/M/H signals (after smoother → auto-scaler → strength blend — i.e. the
same values sent on `/audio/lmh` and rendered on the bars/lines viz).
Every UI knob that shapes the visual `low`/`mid`/`high` tracks (bandpass
edges, smoothing, autoscale strength, noise gate) also shapes what the
detector sees, so the user dials each band until it pulses cleanly on
the relevant transients (kicks, snares, hats) and there is no second
pipeline to tune.

Each band has its own three detection parameters (`sensitivity`,
`refractory_s`, `slow_tau_s`) since kicks / snares / hats have very
different transient shapes and density. BPM is computed only from the
low-band onset stream (it's the only band whose onset rate is musically
tempo-shaped — kick / sub events).

Algorithm (industry-standard real-time onset detection — per band):

1. Two one-pole envelopes of the band's input:
     fast (~12 ms, shared, fixed) — removes per-block jitter
     slow (`slow_tau_s` per band) — local DC reference / "what 'normal' sounds like"
2. Novelty function: half-wave rectified flux, `nov = max(0, fast - slow)`.
   Equivalent to a band-pass on the envelope: positive only when the signal
   is rising faster than the slow envelope can track. Standard real-time
   onset shape.
3. Adaptive threshold: a slow EMA (~1 s) of `nov` itself, multiplied by
   `sensitivity` for the trigger floor and by `K_LOW` for the Schmitt
   release. Plus an absolute floor so silence-noise can never trigger.
4. Schmitt trigger with refractory period: fire (and arm) when
   `nov > thr_high` AND time-since-last-fire > `refractory_s`; drop back
   to idle when `nov < thr_low`. Exactly one `1` per onset.
5. BPM (low band only): median of recent inter-onset intervals (IOIs),
   with outlier rejection (drop IOIs outside 0.5×–2× the running median),
   octave-folded into [BPM_MIN, BPM_MAX], smoothed via a per-onset EMA.
6. Silence handling (low band only): if no onset for >
   `BPM_DECAY_AFTER_S`, zero out the BPM and clear the IOI ring.

Hot path: pure scalar Python unrolled for n=3 (same justification as
ExpSmoother / AutoScaler.update — numpy ufunc overhead dominates length-3
ops by ~20×). Allocation only inside the low-band `_on_low_fire`, bounded
< 4 Hz by the refractory period.
"""
from __future__ import annotations

import math

import numpy as np


BAND_NAMES = ("low", "mid", "high")


class OnsetTracker:
    # ---------- Per-band tunable params (mutable; see set_params) ----------

    # Schmitt high threshold multiplier on the slow EMA of novelty itself.
    # Higher → stricter (fewer triggers, only the strongest onsets fire).
    # Lower → more sensitive.
    DEFAULT_K_HIGH = 1.8

    # Refractory: shortest allowed inter-onset interval. 0.25 s ≈ 240 BPM
    # upper detection limit.
    DEFAULT_MIN_IOI_S = 0.25

    # Slow envelope time constant. What counts as a "transient":
    # smaller → only very fast rises trigger; larger → slower swells qualify.
    DEFAULT_SLOW_TAU_S = 0.30

    # Default absolute floor on the Schmitt trigger threshold. The high
    # threshold is `max(K_HIGH × thr_avg, abs_floor[band])` — when adaptive
    # novelty is small (room/electronic noise after the autoscaler has
    # ramped it up but nothing musically loud is happening), the adaptive
    # part collapses and this is what actually gates the trigger. Since the
    # post-processed band values are nominally [0, 1] and real-onset
    # novelty typically reaches 0.2–0.5+, a floor of 0.10 (= 10% of full
    # scale) rejects noise-floor micro-flux while preserving headroom for
    # soft hits. Exposed per-band so noisy bands can be raised independently.
    DEFAULT_ABS_FLOOR = 0.10

    # ---------- Shared (non-exposed) constants ----------

    # Schmitt release multiplier; armed → idle when nov < K_LOW × thr_avg.
    # Held constant — exposing both endpoints is over-tuning.
    K_LOW = 0.6

    # Fast envelope time constant — fixed; just removes per-block jitter.
    FAST_TAU_S = 0.012

    # Adaptive-threshold EMA tau — how quickly the trigger floor follows
    # changing dynamic content. Held constant.
    THR_TAU_S = 1.000

    # ---------- BPM (low band only) ----------

    BPM_MIN = 60.0
    BPM_MAX = 180.0
    BPM_RING = 12
    BPM_DECAY_AFTER_S = 5.0
    BPM_SMOOTH_ALPHA = 0.3

    # Which band index feeds the BPM estimator (0=low, 1=mid, 2=high).
    BPM_BAND_IDX = 0

    def __init__(self, sr: float, blocksize: int,
                 params: dict | None = None):
        """`params` is `{band: {sensitivity, refractory_s, slow_tau_s}}` for
        each of low/mid/high. Missing entries use class defaults."""
        params = params or {}
        self.k_high = np.empty(3, dtype=np.float64)
        self.min_ioi_s = np.empty(3, dtype=np.float64)
        self.slow_tau_s = np.empty(3, dtype=np.float64)
        self.abs_floor = np.empty(3, dtype=np.float64)
        for i, name in enumerate(BAND_NAMES):
            p = params.get(name) or {}
            self.k_high[i] = float(p.get("sensitivity", self.DEFAULT_K_HIGH))
            self.min_ioi_s[i] = float(p.get("refractory_s", self.DEFAULT_MIN_IOI_S))
            self.slow_tau_s[i] = float(p.get("slow_tau_s", self.DEFAULT_SLOW_TAU_S))
            self.abs_floor[i] = float(p.get("abs_floor", self.DEFAULT_ABS_FLOOR))
        self.alpha_slow = np.empty(3, dtype=np.float64)
        # Monotonic per-band counters. The WS broadcaster reads these and
        # emits an onset flag whenever the counter advances between snapshots,
        # so onsets that fall between WS snapshot ticks (audio block rate
        # ≈ 187 Hz at 48k/256 vs ws_snapshot_hz default 60) are never lost.
        # Allocated once and NEVER reset (not even by reset()): the
        # broadcaster compares with `!=`, so zeroing them on a device switch
        # would emit a spurious onset on every band.
        self.onset_count = np.zeros(3, dtype=np.int64)
        self._configure(sr, blocksize)
        self.reset()

    def _configure(self, sr: float, blocksize: int) -> None:
        self.dt = float(blocksize) / max(float(sr), 1.0)
        self.alpha_fast = 1.0 - math.exp(-self.dt / max(self.FAST_TAU_S, 1e-3))
        self.alpha_thr = 1.0 - math.exp(-self.dt / max(self.THR_TAU_S, 1e-3))
        for i in range(3):
            self.alpha_slow[i] = 1.0 - math.exp(-self.dt / max(float(self.slow_tau_s[i]), 1e-3))

    def reconfigure(self, sr: float, blocksize: int) -> None:
        """Called from the asyncio loop on device hot-switch (sr changes).
        Worker is briefly stalled by the orchestrator before this runs."""
        self._configure(sr, blocksize)

    def set_params(self, band: str, *, sensitivity: float | None = None,
                   refractory_s: float | None = None,
                   slow_tau_s: float | None = None,
                   abs_floor: float | None = None) -> None:
        """Atomic single-attribute swaps from the asyncio loop. The DSP
        worker re-reads on the next iteration; no lock needed. `band` is
        one of "low" / "mid" / "high"."""
        i = BAND_NAMES.index(band)
        if sensitivity is not None:
            self.k_high[i] = float(sensitivity)
        if refractory_s is not None:
            self.min_ioi_s[i] = float(refractory_s)
        if slow_tau_s is not None:
            self.slow_tau_s[i] = float(slow_tau_s)
            self.alpha_slow[i] = 1.0 - math.exp(-self.dt / max(float(self.slow_tau_s[i]), 1e-3))
        if abs_floor is not None:
            self.abs_floor[i] = float(abs_floor)

    def reset(self) -> None:
        self.fast = np.zeros(3, dtype=np.float64)
        self.slow = np.zeros(3, dtype=np.float64)
        self.thr_avg = np.zeros(3, dtype=np.float64)
        self.armed = [False, False, False]
        self.t_now = 0.0
        self.last_fire_t = np.full(3, -1e9, dtype=np.float64)
        # onset_count is deliberately NOT reset — see __init__.
        # BPM state (low band only)
        self.onset_times: list[float] = []
        self.bpm_smoothed = 0.0

    def advance(self, n_blocks: int) -> None:
        """Advance the detector clock by `n_blocks` blocks that were
        dropped without being processed (DSPWorker backlog skip / failed
        ring read). Keeps IOIs — and so BPM — on real audio time across
        stalls. Envelope state is left as-is. Called on the DSP worker."""
        if n_blocks > 0:
            self.t_now += n_blocks * self.dt

    def update(self, lmh: np.ndarray, out_onsets: np.ndarray) -> float:
        """One block. Writes per-band onset pulses (0/1) into `out_onsets`
        (length-3 array). Returns the smoothed BPM (0 during silence or
        before enough onsets accumulate).

        `lmh` is a length-3 float64 array of the post-processed band values.
        """
        self.t_now += self.dt
        t = self.t_now

        # Locals — attribute lookups aren't free on the hot path.
        fast = self.fast
        slow = self.slow
        thr_avg = self.thr_avg
        a_fast = self.alpha_fast
        a_slow = self.alpha_slow
        a_thr = self.alpha_thr
        k_high = self.k_high
        min_ioi = self.min_ioi_s
        last_fire = self.last_fire_t
        onset_count = self.onset_count
        armed = self.armed
        k_low = self.K_LOW
        abs_floor = self.abs_floor   # per-band length-3 array

        x0 = float(lmh[0]); x1 = float(lmh[1]); x2 = float(lmh[2])

        # Fast envelopes
        f0 = fast[0] + a_fast * (x0 - fast[0]); fast[0] = f0
        f1 = fast[1] + a_fast * (x1 - fast[1]); fast[1] = f1
        f2 = fast[2] + a_fast * (x2 - fast[2]); fast[2] = f2

        # Slow envelopes (per-band tau)
        s0 = slow[0] + a_slow[0] * (x0 - slow[0]); slow[0] = s0
        s1 = slow[1] + a_slow[1] * (x1 - slow[1]); slow[1] = s1
        s2 = slow[2] + a_slow[2] * (x2 - slow[2]); slow[2] = s2

        # Half-wave-rectified novelty
        n0 = f0 - s0
        if n0 < 0.0: n0 = 0.0
        n1 = f1 - s1
        if n1 < 0.0: n1 = 0.0
        n2 = f2 - s2
        if n2 < 0.0: n2 = 0.0

        # Adaptive threshold EMA of novelty
        ta0 = thr_avg[0] + a_thr * (n0 - thr_avg[0]); thr_avg[0] = ta0
        ta1 = thr_avg[1] + a_thr * (n1 - thr_avg[1]); thr_avg[1] = ta1
        ta2 = thr_avg[2] + a_thr * (n2 - thr_avg[2]); thr_avg[2] = ta2

        # ----- Band 0 (low) — also drives BPM -----
        af0 = abs_floor[0]
        th = ta0 * k_high[0]
        if th < af0: th = af0
        tl = ta0 * k_low
        af_lo = af0 * 0.5
        if tl < af_lo: tl = af_lo
        o0 = 0
        if not armed[0]:
            if n0 > th and (t - last_fire[0]) > min_ioi[0]:
                o0 = 1
                armed[0] = True
                last_fire[0] = t
                onset_count[0] += 1
                self._on_low_fire(t)
        else:
            if n0 < tl:
                armed[0] = False

        # ----- Band 1 (mid) -----
        af1 = abs_floor[1]
        th = ta1 * k_high[1]
        if th < af1: th = af1
        tl = ta1 * k_low
        af_lo = af1 * 0.5
        if tl < af_lo: tl = af_lo
        o1 = 0
        if not armed[1]:
            if n1 > th and (t - last_fire[1]) > min_ioi[1]:
                o1 = 1
                armed[1] = True
                last_fire[1] = t
                onset_count[1] += 1
        else:
            if n1 < tl:
                armed[1] = False

        # ----- Band 2 (high) -----
        af2 = abs_floor[2]
        th = ta2 * k_high[2]
        if th < af2: th = af2
        tl = ta2 * k_low
        af_lo = af2 * 0.5
        if tl < af_lo: tl = af_lo
        o2 = 0
        if not armed[2]:
            if n2 > th and (t - last_fire[2]) > min_ioi[2]:
                o2 = 1
                armed[2] = True
                last_fire[2] = t
                onset_count[2] += 1
        else:
            if n2 < tl:
                armed[2] = False

        out_onsets[0] = o0
        out_onsets[1] = o1
        out_onsets[2] = o2

        # BPM silence handling — wipe stale tempo after long quiet on the
        # primary (low) band.
        if (t - last_fire[self.BPM_BAND_IDX]) > self.BPM_DECAY_AFTER_S:
            if self.bpm_smoothed != 0.0 or self.onset_times:
                self.bpm_smoothed = 0.0
                self.onset_times.clear()

        return self.bpm_smoothed

    def _on_low_fire(self, t: float) -> None:
        ot = self.onset_times
        ot.append(t)
        if len(ot) > self.BPM_RING:
            ot.pop(0)
        if len(ot) < 4:
            return  # need a few onsets before estimating

        n = len(ot)
        iois = sorted(ot[i + 1] - ot[i] for i in range(n - 1))
        if not iois:
            return
        m = iois[len(iois) // 2]
        if m <= 0.0:
            return

        # Outlier rejection: keep IOIs within 0.5×–2× the running median.
        kept = [v for v in iois if 0.5 * m < v < 2.0 * m]
        if len(kept) < 2:
            return
        med = kept[len(kept) // 2]
        if med <= 0.0:
            return

        bpm = 60.0 / med
        while bpm < self.BPM_MIN:
            bpm *= 2.0
        while bpm > self.BPM_MAX:
            bpm *= 0.5

        if self.bpm_smoothed <= 0.0:
            self.bpm_smoothed = bpm
        else:
            self.bpm_smoothed += self.BPM_SMOOTH_ALPHA * (bpm - self.bpm_smoothed)
