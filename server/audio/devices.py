"""Input device discovery and signal-active probing."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


def list_input_devices() -> list[dict[str, Any]]:
    """Return all devices with at least one input channel."""
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    out = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) >= 1:
            out.append(
                {
                    "index": i,
                    "name": d["name"],
                    "hostapi": hostapis[d["hostapi"]]["name"] if d.get("hostapi") is not None else "",
                    "default_samplerate": float(d.get("default_samplerate") or 0.0),
                    "max_input_channels": int(d.get("max_input_channels", 0)),
                }
            )
    return out


def reinitialize_portaudio() -> None:
    """Re-run Pa_Terminate + Pa_Initialize so PortAudio re-enumerates devices.

    PortAudio snapshots the device list at initialisation: without this a
    device plugged in after startup never shows up, and indices can go stale
    after an unplug. Pa_Terminate closes every open stream, so the caller
    must have torn its own stream down first (hold App.device_switch_lock).
    ~2 ms on macOS.
    """
    sd._terminate()
    sd._initialize()


def find_input_by_name(name: str | None) -> int | None:
    """Current index of the input device called `name`, or None."""
    if not name:
        return None
    for d in list_input_devices():
        if d["name"] == name:
            return d["index"]
    return None


def default_input() -> int | None:
    # sd.default.device may be a list, tuple, plain int, or sounddevice's
    # _InputOutputPair (subscriptable but not a list/tuple). Duck-type the
    # first-element access so all platforms / sounddevice versions resolve.
    try:
        d = sd.default.device
        if isinstance(d, int) and d >= 0:
            return int(d)
        try:
            in_idx = d[0]
        except (TypeError, IndexError, KeyError):
            return None
        if in_idx is None or in_idx == -1:
            return None
        return int(in_idx)
    except Exception:
        return None


def signal_active_probe(devices: list[dict], duration: float = 0.2, threshold: float = 1e-4) -> dict[int, bool]:
    """Open each device briefly; return {index: saw_signal_during_probe}.

    A pure heuristic — proves the device produced *some* audio in the probe
    window, nothing more. Failures are logged and treated as 'no signal'.
    """
    results: dict[int, bool] = {}
    for d in devices:
        idx = d["index"]
        try:
            sr = int(d.get("default_samplerate") or 48000)
            ch = min(2, d.get("max_input_channels", 1))
            n = max(int(sr * duration), 256)
            buf = sd.rec(n, samplerate=sr, channels=ch, device=idx, dtype="float32", blocking=True)
            rms = float(np.sqrt(np.mean(np.square(buf))))
            results[idx] = rms > threshold
        except Exception as e:
            log.debug("probe failed for device %d: %s", idx, e)
            results[idx] = False
    return results


def resolve_initial_device(cfg_device, cli_device) -> int | None:
    """CLI > config.name > config.index > system default."""
    if cli_device is not None:
        return cli_device
    if cfg_device.name:
        idx = find_input_by_name(cfg_device.name)
        if idx is not None:
            return idx
        log.warning("configured device %r not found; falling back", cfg_device.name)
    if cfg_device.index is not None:
        for d in list_input_devices():
            if d["index"] == cfg_device.index:
                return cfg_device.index
        log.warning("configured device index %d not found; falling back", cfg_device.index)
    return default_input()


def device_info(idx: int) -> dict | None:
    try:
        d = sd.query_devices(idx)
        return dict(d)
    except Exception:
        return None
