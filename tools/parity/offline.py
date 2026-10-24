"""Offline (no audio device, no threads, no sockets) driver for the real
Python engine. Shared by make_goldens.py and tools/bench/stage_bench.py.

`OfflineApp` is the production `server.main.App` with the device / network
edges stubbed out: it builds the DSP + FFT pipeline through the same code
paths as `App.start()` (`_build_pipeline_for_sr`, `_build_fft_pipeline`), so
the Dispatcher handlers and `App.apply_*` fan-out run unmodified on it. The
only behavioural differences, all deliberate and documented in the golden
manifest:

  * `schedule_filter_retune` retunes immediately (production debounces 50 ms;
    the debounce is a server-crate concern, not DSP semantics).
  * the persister is a no-op.
  * no OSC / WS (the publishers are None; FeatureStore still runs).

`OfflineEngine` steps the pipeline one block at a time in a fixed, documented
order (events -> ring write -> DSP block -> every FFT hop that is now
complete), mirroring the read-pointer logic of `FFTWorker.run` so hop
alignment after a reconfigure matches the live server.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from pathlib import Path

import numpy as np

from server.config import Config, config_to_dict, load_config, write_yaml_atomic
from server.control.dispatcher import Dispatcher
from server.dsp.fft import MAX_BACKLOG_HOPS
from server.dsp.worker import DSPWorker
from server.main import App

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO / "configs" / "main.example.yaml"


class _NullPersister:
    def request(self, commit: bool = True) -> None:
        pass

    def flush_now_sync(self) -> None:
        pass


class _SyncPersister:
    """Writes the config on every request. Production debounces (<= 250 ms),
    so by the time a later `load_preset` deep-merges over main.yaml the file
    holds the current state; writing synchronously models exactly that."""

    def __init__(self, path: Path, get_state):
        self.path = path
        self._get_state = get_state

    def request(self, commit: bool = True) -> None:
        write_yaml_atomic(self.path, self._get_state())

    def flush_now_sync(self) -> None:
        self.request()


class OfflineApp(App):
    def __init__(self, cfg: Config, sr: float, config_path: Path | None = None):
        """`config_path` given: main.yaml (and presets) live in its directory
        and every persist request rewrites it. None: nothing touches disk."""
        args = argparse.Namespace(no_ws=True, open=False, device=None, config=None,
                                  log_level="WARNING")
        super().__init__(args, cfg, config_path or (REPO / "configs" / "main.yaml"))
        self._sr = float(sr)
        if config_path is None:
            self.persister = _NullPersister()
        else:
            self.persister = _SyncPersister(config_path, lambda: config_to_dict(self.cfg))
            self.persister.request()
        self._build_pipeline_for_sr(self._sr)
        self.dsp_worker = DSPWorker(
            ring=self.ring,
            dsp_event=self.dsp_event,
            stop_flag=self.stop_flag,
            filter_bank=self.filter_bank,
            smoother=self.smoother,
            auto_scaler=self.auto_scaler,
            onset_tracker=self.onset_tracker,
            features_store=self.features_store,
            osc_publisher=None,
            blocksize=cfg.audio.blocksize,
            perf_ring=self.perf_dsp,
        )
        self._build_fft_pipeline(self._sr)

    def current_sr(self) -> float:
        return self._sr

    def schedule_filter_retune(self) -> None:
        self.filter_bank.retune(self._bands_tuple_dict())


def load_cfg(overrides: dict | None = None, base: Path = EXAMPLE_CONFIG) -> Config:
    """Example config with nested-dict overrides applied, routed through the
    production loader (so the same validation/clamping applies)."""
    import tempfile

    import yaml

    raw = yaml.safe_load(base.read_text())

    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)

    merge(raw, overrides or {})
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
        tmp = Path(fh.name)
    try:
        return load_config(tmp)
    finally:
        tmp.unlink()


class OfflineEngine:
    """Block-stepped driver. Per block b:
        1. apply every scripted event scheduled at block b (dispatcher messages)
        2. ring.write_block(signal block b)
        3. DSPWorker.process_block -> record raw / scaled / onsets / bpm
        4. while a full FFT window is available: FFTWorker.process_hop
           (window read through SlotRing.try_read_window) -> record
    """

    def __init__(self, app: OfflineApp):
        self.app = app
        self.dispatcher = Dispatcher(app)
        self._loop = asyncio.new_event_loop()

    def dispatch(self, msg: dict):
        targeted, _ = self._loop.run_until_complete(self.dispatcher(msg))
        errors = [r for r in targeted if r.get("type") == "error"]
        if errors:
            raise RuntimeError(f"dispatcher rejected {msg}: {errors[0]['reason']}")

    def step_dsp(self, block: np.ndarray) -> float:
        app = self.app
        app.ring.write_block(block, 0)
        return app.dsp_worker.process_block(block)

    def fft_hops(self):
        """Yield (start_block, bins_f32, processed) for every hop that is now
        complete. Same pointer arithmetic as FFTWorker.run (self-heal after a
        reset, backlog skip, drain-all)."""
        app = self.app
        fw = app.fft_worker
        if not app.fft_enabled.is_set():
            return
        wi = app.ring.write_idx
        if fw.read_block_idx > wi:
            fw.read_block_idx = 0
        if wi - fw.read_block_idx > fw.n_blocks_per_window + MAX_BACKLOG_HOPS * fw.hop_blocks:
            target_block = wi - fw.n_blocks_per_window
            skipped = (target_block - fw.read_block_idx) // fw.hop_blocks
            if skipped > 0:
                fw.read_block_idx += int(skipped) * fw.hop_blocks
        while wi - fw.read_block_idx >= fw.n_blocks_per_window:
            start = fw.read_block_idx
            ok = app.ring.try_read_window(start, fw.n_blocks_per_window, fw.window_buf)
            if not ok:
                raise RuntimeError("offline ring read failed (ring too small?)")
            bins_f32, processed = fw.process_hop()
            yield start, bins_f32, processed
            fw.read_block_idx += fw.hop_blocks

    def close(self):
        self._loop.close()
