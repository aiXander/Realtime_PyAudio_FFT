"""Pure validators for control messages and persisted config values.

Philosophy: the UI is the gatekeeper for "sane" ranges (slider min/max etc.).
This module only enforces invariants that would otherwise crash the server,
silently corrupt state (NaN/Inf poisoning a smoother forever), or break a
security boundary (preset names hit the filesystem). Type-correct, finite
values outside "tasteful" ranges are passed through — let people send a
30-second smoothing tau or a 0.5 noise floor if they want to.
"""
from __future__ import annotations

import math
import re

BAND_NAMES = ("low", "mid", "high")


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite_float(x, name, *, gt: float | None = None, ge: float | None = None):
    """Coerce x to float and assert finite. Optional `gt` / `ge` bounds raise
    a precise message when violated."""
    if not _is_number(x):
        raise ValueError(f"{name} must be numeric")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if gt is not None and v <= gt:
        raise ValueError(f"{name} must be > {gt:g}")
    if ge is not None and v < ge:
        raise ValueError(f"{name} must be >= {ge:g}")
    return v


def validate_band(name, lo_hz, hi_hz, sr):
    """Validate a single bandpass.

    Hard invariants only: lo > 0, hi > lo, hi < sr/2 (Nyquist). Without these,
    `scipy.signal.iirfilter` either errors or produces an unstable filter.
    """
    lo_hz = _finite_float(lo_hz, f"{name}.lo_hz")
    hi_hz = _finite_float(hi_hz, f"{name}.hi_hz")
    sr = _finite_float(sr, "sr")
    if lo_hz <= 0.0:
        raise ValueError(f"{name}: lo_hz must be > 0")
    if hi_hz <= lo_hz:
        raise ValueError(f"{name}: hi_hz must be > lo_hz")
    nyq = 0.5 * sr
    if hi_hz >= nyq:
        raise ValueError(f"{name}: hi_hz must be < sr/2 ({nyq:.0f} Hz)")
    return lo_hz, hi_hz


def validate_bands(bands_dict, sr):
    """Validate {low: {lo_hz, hi_hz}, mid: {...}, high: {...}}. Returns {name: (lo, hi)}."""
    if not isinstance(bands_dict, dict):
        raise ValueError("bands must be a dict")
    out = {}
    for name in BAND_NAMES:
        b = bands_dict.get(name)
        if not isinstance(b, dict):
            raise ValueError(f"missing band {name!r}")
        out[name] = validate_band(name, b.get("lo_hz"), b.get("hi_hz"), sr)
    return out


def validate_tau(tau_dict):
    if not isinstance(tau_dict, dict):
        raise ValueError("tau must be a dict")
    out = {}
    for k, v in tau_dict.items():
        if k not in BAND_NAMES:
            raise ValueError(f"unknown band {k!r}")
        out[k] = _finite_float(v, f"tau[{k}]", gt=0.0)
    return out


def validate_filter_order(n):
    """Per-band Butterworth bandpass order. 1..8 covers the useful range
    (1 → ~6 dB/oct skirt, 8 → ~48 dB/oct). CPU cost scales linearly with
    order (sosfilt biquad count), so this is the main "fast vs. precise"
    knob for the L/M/H pipeline."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("filter_order must be an int")
    if not (1 <= n <= 8):
        raise ValueError("filter_order must be in 1..8")
    return n


def validate_n_fft_bins(n):
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("n_fft_bins must be an int")
    if n < 1:
        raise ValueError("n_fft_bins must be >= 1")
    return n


def validate_fft_window(window_size=None, hop=None, f_min=None, *, blocksize):
    """FFT window geometry. Hard invariants only:

    - window_size / hop must be positive multiples of the audio blocksize.
      FFTWorker._allocate derives its block counts by integer division; a
      sub-blocksize or non-multiple value leaves the worker with a
      zero-block window/hop that spins without advancing the read pointer.
    - f_min must be a positive finite frequency (log-spaced bin edges take
      log10(f_min)).

    Returns a dict of only the keys that were provided (same shape as
    validate_autoscale).
    """
    out = {}
    if window_size is not None:
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise ValueError("window_size must be an int")
        if window_size < blocksize or window_size % blocksize:
            raise ValueError(f"window_size must be a positive multiple of blocksize ({blocksize})")
        out["window_size"] = window_size
    if hop is not None:
        if not isinstance(hop, int) or isinstance(hop, bool):
            raise ValueError("hop must be an int")
        if hop < blocksize or hop % blocksize:
            raise ValueError(f"hop must be a positive multiple of blocksize ({blocksize})")
        out["hop"] = hop
    if f_min is not None:
        out["f_min"] = _finite_float(f_min, "f_min", gt=0.0)
    return out


def validate_autoscale(tau_attack_s=None, tau_release_s=None, noise_floor=None, strength=None,
                       master_gain=None):
    out = {}
    if tau_attack_s is not None:
        out["tau_attack_s"] = _finite_float(tau_attack_s, "tau_attack_s", gt=0.0)
    if tau_release_s is not None:
        out["tau_release_s"] = _finite_float(tau_release_s, "tau_release_s", gt=0.0)
    if noise_floor is not None:
        out["noise_floor"] = _finite_float(noise_floor, "noise_floor")
    if strength is not None:
        out["strength"] = _finite_float(strength, "strength")
    if master_gain is not None:
        out["master_gain"] = _finite_float(master_gain, "master_gain")
    return out


def validate_peak_smear_oct(v):
    return _finite_float(v, "peak_smear_oct")


def validate_fft_tilt_db_per_oct(v):
    return _finite_float(v, "tilt_db_per_oct")


def validate_peak_decay_per_s(v):
    return _finite_float(v, "peak_decay_per_s")


def validate_onset(sensitivity=None, refractory_s=None, slow_tau_s=None,
                   abs_floor=None):
    """Per-band onset-detector tunables. Hard invariants only:
      - sensitivity > 1.0 (Schmitt high must exceed the slow EMA of novelty
        itself, otherwise the trigger would always be tripped).
      - refractory_s > 0 (zero refractory → a single onset retriggers each block).
      - slow_tau_s > 0 (used as denominator in tau→alpha).
      - abs_floor >= 0 (finite; 0 means the absolute floor never gates).
    UI sliders enforce taste-level ranges; here we only block crashes.
    """
    out = {}
    if sensitivity is not None:
        out["sensitivity"] = _finite_float(sensitivity, "sensitivity", gt=1.0)
    if refractory_s is not None:
        out["refractory_s"] = _finite_float(refractory_s, "refractory_s", gt=0.0)
    if slow_tau_s is not None:
        out["slow_tau_s"] = _finite_float(slow_tau_s, "slow_tau_s", gt=0.0)
    if abs_floor is not None:
        out["abs_floor"] = _finite_float(abs_floor, "abs_floor", ge=0.0)
    return out


def validate_onset_band(band):
    if band not in BAND_NAMES:
        raise ValueError(f"band must be one of {BAND_NAMES}")
    return band


def validate_ws_snapshot_hz(hz):
    return int(round(_finite_float(hz, "hz", gt=0.0)))


_PRESET_NAME_RE = re.compile(r"^[a-zA-Z0-9_\- ]+$")


def validate_preset_name(name):
    """Preset names hit the filesystem (preset-<name>.yaml) — keep strict."""
    if not isinstance(name, str):
        raise ValueError("preset name must be a string")
    n = name.strip()
    if not (1 <= len(n) <= 64):
        raise ValueError("preset name must be 1-64 characters")
    if n.lower() == "main":
        raise ValueError("'main' is reserved for the active config file")
    if not _PRESET_NAME_RE.match(n):
        raise ValueError(
            "preset name may only contain letters, digits, spaces, hyphens, underscores"
        )
    return n


CARD_IDS = ("bars", "lines", "scene", "fft")


def validate_ui_layout(layout):
    """Validate a tiling 2x2 layout: {split_x, split_y, quadrants: [TL,TR,BL,BR]}.

    quadrants must be a permutation of CARD_IDS — that's structural, not taste:
    duplicate or missing IDs would break the UI's mount logic.
    """
    if not isinstance(layout, dict):
        raise ValueError("layout must be a dict")
    sx = _finite_float(layout.get("split_x"), "split_x")
    sy = _finite_float(layout.get("split_y"), "split_y")
    quads = layout.get("quadrants")
    if not (isinstance(quads, list) and len(quads) == 4):
        raise ValueError("quadrants must be a 4-item list (TL, TR, BL, BR)")
    if sorted(quads) != sorted(CARD_IDS):
        raise ValueError(f"quadrants must be a permutation of {list(CARD_IDS)}")
    return {"split_x": sx, "split_y": sy, "quadrants": list(quads)}


def validate_device_index(idx):
    if not isinstance(idx, int) or isinstance(idx, bool):
        raise ValueError("device index must be an int")
    if idx < 0:
        raise ValueError("device index must be >= 0")
    return idx
