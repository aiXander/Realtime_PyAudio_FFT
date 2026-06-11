"""Inbound WS message dispatcher: validate + apply + persist + reply.

Returns (targeted_replies, broadcasts) — never raises. The WS layer turns
exceptions into {"type":"error",...}; the dispatcher prefers to do that
itself with a precise reason.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager

import yaml

from . import validate as V
from ..config import write_yaml_atomic

log = logging.getLogger(__name__)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge: overlay wins; nested dicts merge key-by-key."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class _SkipSection(Exception):
    """Raised inside a preset section to silently skip without logging
    (used when the section has no relevant keys to apply)."""


@contextmanager
def _preset_section(label: str, applied: list[str]):
    """Run a preset-section block. On success, append `label` to `applied`.
    `_SkipSection` skips silently; any other Exception is logged and
    swallowed so one bad section doesn't fail the whole preset."""
    try:
        yield
    except _SkipSection:
        return
    except Exception as e:
        log.warning("preset %s invalid: %s", label, e)
        return
    applied.append(label)


class Dispatcher:
    def __init__(self, app):
        # `app` is the orchestrator (server.main.App); we touch its attributes directly.
        self.app = app

    async def __call__(self, msg: dict):
        t = msg.get("type")
        h = self._handlers.get(t)
        if h is None:
            return [{"type": "error", "reason": f"unknown type {t!r}"}], []
        try:
            return await h(self, msg)
        except ValueError as e:
            return [{"type": "error", "reason": str(e)}], []
        except Exception as e:
            log.exception("dispatcher handler %s failed", t)
            return [{"type": "error", "reason": f"internal: {e}"}], []

    # ---------- handlers ----------

    async def _set_fft(self, msg):
        enabled = bool(msg.get("enabled", False))
        if enabled:
            if not self.app.fft_enabled.is_set():
                # Reset post-processor state so we don't reuse stale smoother /
                # peak values from before the FFT was disabled, and snap the
                # worker's read pointer past the disabled gap so elapsed idle
                # time isn't counted as fft_drops.
                if self.app.fft_postprocessor is not None:
                    self.app.fft_postprocessor.reset()
                if self.app.fft_worker is not None:
                    self.app.fft_worker.resync()
            self.app.fft_enabled.set()
        else:
            self.app.fft_enabled.clear()
        self.app.cfg.fft.enabled = enabled
        # The ENABLE toggle is the single user-facing FFT switch — keep OSC
        # transmission in lockstep so consumers downstream get the FFT stream
        # whenever the UI says FFT is on.
        self.app.cfg.osc.send_fft = enabled
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_band(self, msg):
        band = msg.get("band")
        if band not in ("low", "mid", "high"):
            raise ValueError("band must be 'low', 'mid', or 'high'")
        commit = bool(msg.get("commit", True))
        sr = self.app.current_sr()
        lo, hi = V.validate_band(band, msg.get("lo_hz"), msg.get("hi_hz"), sr)
        getattr(self.app.cfg.dsp, band).lo_hz = lo
        getattr(self.app.cfg.dsp, band).hi_hz = hi
        # IIR retune is debounced (50ms); pipeline knobs that depend on the
        # band edges (per-band linear tilt, bandwidth-aware noise subtraction,
        # FFT-postprocessor's per-bin smoothing tau anchors) update synchronously.
        self.app.schedule_filter_retune()
        self.app.apply_bands()
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_smoothing(self, msg):
        commit = bool(msg.get("commit", True))
        tau = V.validate_tau(msg.get("tau") or {})
        tau_atk = V.validate_tau(msg.get("tau_attack") or {})
        if tau:
            self.app.cfg.dsp.tau = {**self.app.cfg.dsp.tau, **tau}
        if tau_atk:
            self.app.cfg.dsp.tau_attack = {**self.app.cfg.dsp.tau_attack, **tau_atk}
        self.app.apply_smoothing()
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_autoscale(self, msg):
        commit = bool(msg.get("commit", True))
        ok = V.validate_autoscale(
            tau_attack_s=msg.get("tau_attack_s"),
            tau_release_s=msg.get("tau_release_s"),
            noise_floor=msg.get("noise_floor"),
            strength=msg.get("strength"),
            master_gain=msg.get("master_gain"),
        )
        self.app.apply_autoscale(ok)
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _list_devices(self, msg):
        probe = bool(msg.get("probe", False))
        items = await asyncio.to_thread(self.app.list_devices_with_probe, probe)
        return [{"type": "devices", "items": items}], []

    async def _set_device(self, msg):
        idx = V.validate_device_index(msg.get("index"))
        await self.app.hot_switch_device(idx)
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()},
                    {"type": "devices", "items": self.app.list_devices_with_probe(False)}]

    async def _set_n_fft_bins(self, msg):
        n = V.validate_n_fft_bins(msg.get("n"))
        self.app.apply_fft_n_bins(n)
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_ws_snapshot_hz(self, msg):
        commit = bool(msg.get("commit", True))
        hz = V.validate_ws_snapshot_hz(msg.get("hz"))
        self.app.ws.set_snapshot_hz(hz)
        self.app.cfg.ws.snapshot_hz = hz
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    # ---------- presets ----------

    async def _list_presets(self, msg):
        items = self.app.list_presets()
        return [{"type": "presets", "items": items}], []

    async def _save_preset(self, msg):
        name = V.validate_preset_name(msg.get("name", ""))
        body = self.app.preset_body(name)
        path = self.app.preset_path(name)
        await asyncio.to_thread(write_yaml_atomic, path, body)
        items = self.app.list_presets()
        return [{"type": "presets", "items": items}], []

    async def _load_preset(self, msg):
        name = V.validate_preset_name(msg.get("name", ""))
        path = self.app.preset_path(name)
        if not path.exists():
            raise ValueError(f"preset {name!r} not found")
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as e:
            raise ValueError(f"preset {name!r} parse failed: {e}")
        # Deep-merge over main.yaml so any missing keys fall back to the active config.
        try:
            main_data = yaml.safe_load(self.app.config_path.read_text()) or {}
            if isinstance(main_data, dict) and isinstance(data, dict):
                data = _deep_merge(main_data, data)
        except Exception as e:
            log.warning("preset %r: main.yaml fallback unavailable (%s)", name, e)

        sr = self.app.current_sr()
        applied: list[str] = []
        d = data.get("dsp", {}) or {}
        f = data.get("fft", {}) or {}
        a = data.get("autoscale", {}) or {}
        onset = data.get("onset")
        if not isinstance(onset, dict):
            # Legacy single-band `beat:` block — port to low-band onset; leave
            # mid/high untouched so the active config's per-band tuning isn't
            # overwritten by a preset that only knew about the old shape.
            legacy = data.get("beat", {}) or {}
            onset = {"low": legacy} if legacy else {}

        with _preset_section("dsp.bands", applied):
            bands_raw = {k: d.get(k) for k in ("low", "mid", "high") if isinstance(d.get(k), dict)}
            if len(bands_raw) != 3:
                raise _SkipSection
            ok = V.validate_bands(bands_raw, sr)
            for nm, (lo, hi) in ok.items():
                cfg_band = getattr(self.app.cfg.dsp, nm)
                cfg_band.lo_hz, cfg_band.hi_hz = lo, hi
            self.app.schedule_filter_retune()
            self.app.apply_bands()

        with _preset_section("dsp.tau", applied):
            if "tau" not in d and "tau_attack" not in d:
                raise _SkipSection
            if "tau" in d:
                tau = V.validate_tau(d["tau"])
                self.app.cfg.dsp.tau = {**self.app.cfg.dsp.tau, **tau}
            if "tau_attack" in d:
                tau_atk = V.validate_tau(d["tau_attack"])
                self.app.cfg.dsp.tau_attack = {**self.app.cfg.dsp.tau_attack, **tau_atk}
            self.app.apply_smoothing()

        with _preset_section("autoscale", applied):
            ok = V.validate_autoscale(
                tau_attack_s=a.get("tau_attack_s"),
                tau_release_s=a.get("tau_release_s"),
                noise_floor=a.get("noise_floor"),
                strength=a.get("strength"),
                master_gain=a.get("master_gain"),
            )
            if not ok:
                raise _SkipSection
            self.app.apply_autoscale(ok)

        with _preset_section("fft.n_bins", applied):
            if "n_bins" not in f:
                raise _SkipSection
            self.app.apply_fft_n_bins(V.validate_n_fft_bins(f["n_bins"]))

        with _preset_section("fft.window", applied):
            ws = f.get("window_size") if isinstance(f.get("window_size"), int) else None
            hop = f.get("hop") if isinstance(f.get("hop"), int) else None
            fmin = f.get("f_min") if isinstance(f.get("f_min"), (int, float)) else None
            if not (ws or hop or fmin):
                raise _SkipSection
            self.app.apply_fft_window(window_size=ws, hop=hop,
                                      f_min=float(fmin) if fmin is not None else None)

        with _preset_section("fft.peak_smear_oct", applied):
            if "peak_smear_oct" not in f:
                raise _SkipSection
            self.app.apply_fft_peak_smear(V.validate_peak_smear_oct(f["peak_smear_oct"]))

        with _preset_section("fft.tilt_db_per_oct", applied):
            if "tilt_db_per_oct" not in f:
                raise _SkipSection
            self.app.apply_fft_tilt(V.validate_fft_tilt_db_per_oct(f["tilt_db_per_oct"]))

        with _preset_section("onset", applied):
            any_applied = False
            for band in ("low", "mid", "high"):
                b = onset.get(band)
                if not isinstance(b, dict):
                    continue
                ok = V.validate_onset(
                    sensitivity=b.get("sensitivity"),
                    refractory_s=b.get("refractory_s"),
                    slow_tau_s=b.get("slow_tau_s"),
                    abs_floor=b.get("abs_floor"),
                )
                if ok:
                    self.app.apply_onset(band, ok)
                    any_applied = True
            if not any_applied:
                raise _SkipSection

        if not applied:
            raise ValueError(f"preset {name!r} produced no valid fields")
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_fft_send_raw_db(self, msg):
        send_raw = bool(msg.get("send_raw_db", False))
        self.app.cfg.fft.send_raw_db = send_raw
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_fft_peak_smear(self, msg):
        commit = bool(msg.get("commit", True))
        v = V.validate_peak_smear_oct(msg.get("peak_smear_oct"))
        self.app.apply_fft_peak_smear(v)
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_ui_layout(self, msg):
        commit = bool(msg.get("commit", True))
        layout = V.validate_ui_layout(msg.get("layout") or {})
        self.app.cfg.ui.layout = layout
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_peak_decay(self, msg):
        commit = bool(msg.get("commit", True))
        v = V.validate_peak_decay_per_s(msg.get("peak_decay_per_s"))
        self.app.cfg.ui.peak_decay_per_s = v
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_show_onsets(self, msg):
        self.app.cfg.ui.show_onsets = bool(msg.get("show_onsets", False))
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_fft_tilt(self, msg):
        commit = bool(msg.get("commit", True))
        v = V.validate_fft_tilt_db_per_oct(msg.get("tilt_db_per_oct"))
        self.app.apply_fft_tilt(v)
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_filter_order(self, msg):
        commit = bool(msg.get("commit", True))
        n = V.validate_filter_order(msg.get("order"))
        self.app.apply_filter_order(n)
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_onset(self, msg):
        commit = bool(msg.get("commit", True))
        band = V.validate_onset_band(msg.get("band"))
        ok = V.validate_onset(
            sensitivity=msg.get("sensitivity"),
            refractory_s=msg.get("refractory_s"),
            slow_tau_s=msg.get("slow_tau_s"),
            abs_floor=msg.get("abs_floor"),
        )
        if not ok:
            raise ValueError("set_onset needs at least one of sensitivity / refractory_s / slow_tau_s / abs_floor")
        self.app.apply_onset(band, ok)
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    _handlers = {
        "set_fft": _set_fft,
        "set_band": _set_band,
        "set_smoothing": _set_smoothing,
        "set_autoscale": _set_autoscale,
        "list_devices": _list_devices,
        "set_device": _set_device,
        "set_n_fft_bins": _set_n_fft_bins,
        "set_ws_snapshot_hz": _set_ws_snapshot_hz,
        "set_fft_send_raw_db": _set_fft_send_raw_db,
        "set_fft_peak_smear": _set_fft_peak_smear,
        "set_fft_tilt": _set_fft_tilt,
        "set_onset": _set_onset,
        "set_filter_order": _set_filter_order,
        "set_peak_decay": _set_peak_decay,
        "set_show_onsets": _set_show_onsets,
        "set_ui_layout": _set_ui_layout,
        "list_presets": _list_presets,
        "save_preset": _save_preset,
        "load_preset": _load_preset,
    }
