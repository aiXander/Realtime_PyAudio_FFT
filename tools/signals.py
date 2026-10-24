"""Deterministic test signals shared by the parity goldens and the live bench.

numpy only (no scipy) so the engine-agnostic live bench keeps working after
the Python engine is deleted. Every generator is seeded: the same (name, sr,
duration) always yields the same float32 samples on the same numpy version.
The parity goldens commit the generated samples as .npy anyway, so a numpy
RNG change can never silently move them.

Signals:
  drums    120 BPM loop: decaying-sine kick on every beat, noise+tone snare on
           beats 2 and 4, high-passed noise hats on every 8th note.
  sweep    log sine sweep 20 Hz -> 0.45*sr.
  white    white noise.
  pink     pink (1/f power) noise, FFT-shaped.
  silence  digital zeros.
  step     silence for the first 40%, then the drum loop at full level
           (warm-up / peak-follower attack / first-onset behaviour).
  clip     sine + drums driven past full scale, hard-clipped to [-1, 1].
"""
from __future__ import annotations

import numpy as np

SIGNAL_NAMES = ("drums", "sweep", "white", "pink", "silence", "step", "clip")

_SEEDS = {"drums": 1, "sweep": 2, "white": 3, "pink": 4, "silence": 5, "step": 6, "clip": 7}


def _highpass_noise(rng: np.random.Generator, n: int, passes: int = 3) -> np.ndarray:
    """White noise pushed toward the top octaves by repeated first differences
    (each pass is a +6 dB/oct tilt). Normalised to unit peak."""
    x = rng.standard_normal(n + passes)
    for _ in range(passes):
        x = np.diff(x)
    return x / max(np.max(np.abs(x)), 1e-12)


def _drums(sr: float, n: int, rng: np.random.Generator, bpm: float = 120.0) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm

    # Kick: 150 ms pitch-dropping sine (110 -> 45 Hz), exponential decay.
    kl = int(0.15 * sr)
    tk = np.arange(kl) / sr
    f_k = 45.0 + 65.0 * np.exp(-tk / 0.03)
    kick = np.sin(2 * np.pi * np.cumsum(f_k) / sr) * np.exp(-tk / 0.05)

    # Snare: 120 ms noise burst + 190 Hz body.
    sl = int(0.12 * sr)
    ts = np.arange(sl) / sr
    snare = (0.6 * rng.standard_normal(sl) + 0.5 * np.sin(2 * np.pi * 190.0 * ts)) * np.exp(-ts / 0.035)

    # Hat: 30 ms high-passed noise.
    hl = int(0.03 * sr)
    th = np.arange(hl) / sr
    hat_src = _highpass_noise(rng, hl)
    hat = hat_src * np.exp(-th / 0.008)

    def place(sample: np.ndarray, t: float, gain: float) -> None:
        i = int(round(t * sr))
        if i >= n:
            return
        m = min(sample.size, n - i)
        out[i:i + m] += gain * sample[:m]

    t = 0.0
    k = 0
    while t < n / sr:
        place(kick, t, 0.8)
        if k % 2 == 1:
            place(snare, t, 0.45)
        place(hat, t, 0.25)
        place(hat, t + beat / 2.0, 0.18)
        t += beat
        k += 1
    return out


def generate(name: str, sr: float, duration_s: float) -> np.ndarray:
    """Return `name` as float32 mono samples, length int(sr * duration_s)."""
    if name not in _SEEDS:
        raise ValueError(f"unknown signal {name!r}; expected one of {SIGNAL_NAMES}")
    sr = float(sr)
    n = int(round(sr * duration_s))
    rng = np.random.default_rng(_SEEDS[name])
    t = np.arange(n, dtype=np.float64) / sr

    if name == "drums":
        x = _drums(sr, n, rng)
    elif name == "sweep":
        f0, f1 = 20.0, 0.45 * sr
        k = np.log(f1 / f0)
        phase = 2 * np.pi * f0 * duration_s / k * (np.exp(t / duration_s * k) - 1.0)
        x = 0.3 * np.sin(phase)
    elif name == "white":
        x = 0.1 * rng.standard_normal(n)
    elif name == "pink":
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1.0 / sr)
        f[0] = f[1] if n > 1 else 1.0
        spec /= np.sqrt(f)
        x = np.fft.irfft(spec, n)
        x *= 0.1 / max(np.std(x), 1e-12)
    elif name == "silence":
        x = np.zeros(n)
    elif name == "step":
        x = np.zeros(n)
        start = int(0.4 * n)
        x[start:] = _drums(sr, n - start, rng)
    else:  # clip
        x = 1.4 * np.sin(2 * np.pi * 220.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)) + 2.0 * _drums(sr, n, rng)
        x = np.clip(x, -1.0, 1.0)
    return x.astype(np.float32)
