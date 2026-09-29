"""Server entry point: orchestrates startup/shutdown, owns all state.

Threads:
  - PortAudio C thread runs AudioCallback (memcpy + signal).
  - DSP worker thread (filter + RMS + smoother + autoscaler -> FeatureStore
    + direct OSC dispatch via OscPublisher).
  - FFT worker thread (optional; window + rfft + log-bin + post-process
    -> FFTStore + direct OSC dispatch).
  - Main asyncio loop: WS server, broadcaster, status, dispatcher, persister.
"""
from __future__ import annotations

import argparse
import asyncio
import errno
import logging
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
import sounddevice as sd

from .audio import devices as devmod
from .audio.callback import AudioCallback
from .audio.ringbuffer import SlotRing
from .audio.stream import StreamHandle, open_input_stream
from .config import (
    CANONICAL_CONFIG_PATH, Config, Persister, config_to_dict, load_config_or_fallback,
)
from .control import validate as V
from .control.dispatcher import Dispatcher
from .dsp.features import AutoScaler, ExpSmoother
from .dsp.fft import FFTWorker
from .dsp.fft_postprocess import FFTPostProcessor
from .dsp.filters import FilterBank, clamp_band_edges
from .dsp.onset import OnsetTracker
from .dsp.worker import DSPWorker
from .io.http_server import StaticHTTPServer
from .io.osc_publisher import OscPublisher
from .io.osc_sender import OscSender
from .io.stores import FFTStore, FeatureStore
from .io.ws_server import WSServer
from .priority import boost_current_thread

log = logging.getLogger(__name__)

_LOOP_ERR_LOG_INTERVAL_S = 5.0
# OSC /audio/meta: resent at this period so late-joining OSC consumers pick
# it up, plus (debounced to one per tick) whenever sr / bands / n_bins change.
_OSC_META_PERIOD_S = 1.0
_OSC_META_TICK_S = 0.05


class StartupError(Exception):
    """Expected startup failure (port in use, no audio device, bad config):
    main() prints `message` + `hint` as plain lines and exits non-zero
    instead of dumping a traceback."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


def ring_slots_for(blocksize: int) -> int:
    """SlotRing size (power of two, >= 32) that holds at least 2× the largest
    accepted FFT window (validate.MAX_FFT_WINDOW) at this blocksize, so any
    window the validators accept always fits with a full window of headroom
    for the producer to keep writing while the FFT worker reads. blocksize
    is fixed for the process lifetime (hot-switch keeps it), so this is
    computed once."""
    need = 2 * -(-V.MAX_FFT_WINDOW // int(blocksize))
    n = 32
    while n < need:
        n *= 2
    return n


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="audio-server", description="Realtime audio feature server")
    p.add_argument("--config", default=None,
                   help="path to main config file (default: <repo>/configs/main.yaml)")
    p.add_argument("--no-ws", action="store_true", help="disable the WebSocket server (headless OSC)")
    p.add_argument("--open", action="store_true", help="open the bundled UI in the default browser")
    p.add_argument("--device", type=int, default=None, help="override input device index")
    p.add_argument("--log-level", default="INFO", help="logging level")
    return p.parse_args(argv)


class App:
    """Long-lived state shared across threads. Mutations happen on the asyncio loop."""

    def __init__(self, args, cfg: Config, config_path: Path):
        self.args = args
        self.cfg = cfg
        self.config_path = config_path
        self.config_dir = config_path.parent.resolve() if config_path.parent != Path("") else Path.cwd()

        # ----- Threading primitives -----
        self.dsp_event = threading.Event()
        self.fft_event = threading.Event()
        self.fft_enabled = threading.Event()
        if cfg.fft.enabled:
            self.fft_enabled.set()
        self.stop_flag = threading.Event()

        # ----- Stores -----
        self.features_store = FeatureStore()
        self.fft_store = FFTStore()

        # ----- Ring buffer -----
        self.ring = SlotRing(n_slots_pow2=ring_slots_for(cfg.audio.blocksize),
                             blocksize=cfg.audio.blocksize)

        # ----- Perf rings (int64 ns) -----
        self.perf_cb  = np.zeros(128, dtype=np.int64)
        self.perf_dsp = np.zeros(128, dtype=np.int64)
        self.perf_fft = np.zeros(64,  dtype=np.int64)
        self.perf_ws  = np.zeros(32,  dtype=np.int64)
        # End-to-end latency rings (audio-block receive -> OSC dispatch).
        # Written by the OSC sender task on each successful send.
        self.perf_lmh_e2e = np.zeros(128, dtype=np.int64)
        self.perf_fft_e2e = np.zeros(64,  dtype=np.int64)
        self.perf_e2e_idx = {"lmh": 0, "fft": 0}

        # ----- Async / loop bits -----
        self.loop: asyncio.AbstractEventLoop | None = None

        # ----- Initialised by start() -----
        self.callback: AudioCallback | None = None
        self.stream: StreamHandle | None = None
        self.filter_bank: FilterBank | None = None
        self.smoother: ExpSmoother | None = None
        self.auto_scaler: AutoScaler | None = None
        self.onset_tracker: OnsetTracker | None = None
        self.dsp_worker: DSPWorker | None = None
        self.fft_worker: FFTWorker | None = None
        self.fft_postprocessor: FFTPostProcessor | None = None
        self.osc_sender: OscSender | None = None
        self.osc_publisher: OscPublisher | None = None
        self.ws: WSServer | None = None
        self.http: StaticHTTPServer | None = None
        self.persister: Persister | None = None

        self._filter_retune_handle: asyncio.TimerHandle | None = None
        self._device_switch_lock: asyncio.Lock | None = None
        # OSC /audio/meta: dirty flag drained by _osc_meta_loop (<= 1 send
        # per tick, so band drags don't flood) + a periodic resend.
        self._osc_meta_dirty = False
        self._osc_meta_task: asyncio.Task | None = None
        # Counter baselines so the status panel's overruns/drops count from
        # the current device session: a hot-switch resets the ring and the
        # workers' read pointers, which can register a few spurious drops.
        self._drops_base = {"cb": 0, "dsp": 0, "fft": 0}
        self._last_err_log: dict[str, float] = {}

    @property
    def device_switch_lock(self) -> asyncio.Lock:
        """Serialises every PortAudio stream open/close (hot-switch, probe)."""
        return self._device_switch_lock

    # ----------------- meta fan-out -----------------
    def mark_meta_dirty(self) -> None:
        """Ask the WS broadcaster for one coalesced meta on its next tick."""
        if self.ws is not None:
            self.ws.mark_meta_dirty()

    def mark_osc_meta_dirty(self) -> None:
        """Ask _osc_meta_loop to resend /audio/meta on its next tick."""
        self._osc_meta_dirty = True

    def send_osc_meta(self) -> None:
        if self.osc_sender is None or self.stream is None:
            return
        self.osc_sender.send_meta(
            sr=int(self.stream.samplerate),
            blocksize=self.cfg.audio.blocksize,
            n_fft_bins=self.cfg.fft.n_bins,
            bands=self._bands_tuple_dict(),
        )

    async def _osc_meta_loop(self) -> None:
        loop = asyncio.get_running_loop()
        last_sent = loop.time()  # start() already sent one
        while True:
            await asyncio.sleep(_OSC_META_TICK_S)
            try:
                now = loop.time()
                if self._osc_meta_dirty or now - last_sent >= _OSC_META_PERIOD_S:
                    self._osc_meta_dirty = False
                    last_sent = now
                    self.send_osc_meta()
            except Exception:
                self._log_loop_error("osc meta loop")

    def _log_loop_error(self, where: str) -> None:
        now = time.monotonic()
        if now - self._last_err_log.get(where, -1e9) >= _LOOP_ERR_LOG_INTERVAL_S:
            self._last_err_log[where] = now
            log.exception("%s iteration failed (continuing; repeats suppressed for %.0fs)",
                          where, _LOOP_ERR_LOG_INTERVAL_S)

    def start_background_tasks(self) -> None:
        self._osc_meta_task = asyncio.get_running_loop().create_task(
            self._osc_meta_loop(), name="osc-meta")

    # ----------------- starting up -----------------
    def _bands_tuple_dict(self) -> dict:
        d = self.cfg.dsp
        return {
            "low":  (d.low.lo_hz,  d.low.hi_hz),
            "mid":  (d.mid.lo_hz,  d.mid.hi_hz),
            "high": (d.high.lo_hz, d.high.hi_hz),
        }

    def _bands_meta_dict(self) -> dict:
        """{lo_hz, hi_hz}-shape used by FFTPostProcessor + snapshot_meta."""
        d = self.cfg.dsp
        return {
            "low":  {"lo_hz": d.low.lo_hz,  "hi_hz": d.low.hi_hz},
            "mid":  {"lo_hz": d.mid.lo_hz,  "hi_hz": d.mid.hi_hz},
            "high": {"lo_hz": d.high.lo_hz, "hi_hz": d.high.hi_hz},
        }

    # ----------------- pipeline-bus: fan settings out to both pipelines -----
    # Callers mutate cfg first (they own the validated values), then call
    # one apply_* to push the change into AutoScaler + FFTPostProcessor in
    # lockstep. Hides the "is fft_postprocessor None?" guards and the dual
    # set/update API so both handlers and the preset loader stay terse.

    def apply_bands(self) -> None:
        """Push current cfg.dsp band edges into both pipelines."""
        self.auto_scaler.set_bands(self._bands_tuple_dict())
        if self.fft_postprocessor is not None:
            self.fft_postprocessor.update_bands(self._bands_meta_dict())
        self.mark_osc_meta_dirty()

    def apply_smoothing(self) -> None:
        """Push current cfg.dsp.tau / tau_attack into both pipelines."""
        self.smoother.set_tau(self.cfg.dsp.tau, self.cfg.dsp.tau_attack)
        if self.fft_postprocessor is not None:
            self.fft_postprocessor.update_smoothing(
                self.cfg.dsp.tau, self.cfg.dsp.tau_attack,
            )

    def apply_autoscale(self, ok: dict) -> None:
        """Apply a validated autoscale-update dict to cfg and both pipelines.

        `ok` may contain any subset of {tau_attack_s, tau_release_s,
        noise_floor, strength, master_gain}. Missing keys are unchanged.
        master_gain is config-only (read live by the OSC sender).
        """
        a = self.cfg.autoscale
        for k in ("tau_attack_s", "tau_release_s", "noise_floor", "strength", "master_gain"):
            if k in ok:
                setattr(a, k, ok[k])
        if "tau_attack_s" in ok or "tau_release_s" in ok:
            self.auto_scaler.set_taus(a.tau_attack_s, a.tau_release_s)
        if "noise_floor" in ok:
            self.auto_scaler.set_noise_floor(a.noise_floor)
        if "strength" in ok:
            self.auto_scaler.set_strength(a.strength)
        if self.fft_postprocessor is not None and any(
            k in ok for k in ("tau_attack_s", "tau_release_s", "noise_floor", "strength")
        ):
            self.fft_postprocessor.update_autoscale(
                tau_attack_s=ok.get("tau_attack_s"),
                tau_release_s=ok.get("tau_release_s"),
                noise_floor=ok.get("noise_floor"),
                strength=ok.get("strength"),
            )

    def apply_onset(self, band: str, ok: dict) -> None:
        """Apply a validated per-band onset-detector update dict to cfg +
        tracker.

        `band` is one of "low" / "mid" / "high". `ok` may contain any subset
        of {sensitivity, refractory_s, slow_tau_s}. Missing keys are
        unchanged. Tracker setters are atomic (single-attribute writes the
        worker re-reads on next iteration); no lock needed.
        """
        cfg_band = getattr(self.cfg.onset, band)
        for k in ("sensitivity", "refractory_s", "slow_tau_s", "abs_floor"):
            if k in ok:
                setattr(cfg_band, k, ok[k])
        if self.onset_tracker is not None and ok:
            self.onset_tracker.set_params(
                band,
                sensitivity=ok.get("sensitivity"),
                refractory_s=ok.get("refractory_s"),
                slow_tau_s=ok.get("slow_tau_s"),
                abs_floor=ok.get("abs_floor"),
            )

    def apply_fft_n_bins(self, n: int) -> None:
        self.cfg.fft.n_bins = n
        self.fft_worker.reconfigure(n_bins=n)
        # n_bins controls FFT-viz log-bin density; AutoScaler caps its
        # noise budget at the per-band log-bin count, so keep it in sync.
        self.auto_scaler.set_fft_geometry(n_bins=n)
        self.mark_osc_meta_dirty()

    def apply_fft_window(self, *, window_size: int | None = None,
                         hop: int | None = None, f_min: float | None = None) -> None:
        """Reconfigure FFT window/hop/f_min and mirror knobs that depend on
        the resulting Δf_lin / log-bins-per-octave into AutoScaler."""
        # Validate BEFORE touching the worker: a non-blocksize-aligned
        # window/hop would leave FFTWorker with a zero-block geometry that
        # spins without advancing. Raises ValueError (preset loader logs &
        # skips the section).
        ok = V.validate_fft_window(window_size=window_size, hop=hop, f_min=f_min,
                                   blocksize=self.cfg.audio.blocksize)
        window_size = ok.get("window_size")
        hop = ok.get("hop")
        f_min = ok.get("f_min")
        self.fft_worker.reconfigure(window_size=window_size, hop=hop, f_min=f_min)
        if window_size is not None:
            self.cfg.fft.window_size = window_size
            # Δf_lin = sr/n_fft scales the K_lin term in noise budget.
            self.auto_scaler.set_n_fft_window(window_size)
        if hop is not None:
            self.cfg.fft.hop = hop
        if f_min is not None:
            self.cfg.fft.f_min = float(f_min)
            # f_min changes log_bins_per_octave -> N_log cap.
            self.auto_scaler.set_fft_geometry(f_min=float(f_min))

    def apply_fft_tilt(self, v: float) -> None:
        self.cfg.fft.tilt_db_per_oct = v
        # AutoScaler always exists; FFTPostProcessor may not.
        self.auto_scaler.set_tilt(v)
        if self.fft_postprocessor is not None:
            self.fft_postprocessor.update_tilt(v)

    def apply_filter_order(self, n: int) -> None:
        """Change Butterworth order of all three L/M/H bandpasses live.
        Brief click on change (filter state resets); the user is consciously
        making a fast/precise tradeoff."""
        self.cfg.dsp.filter_order = int(n)
        if self.filter_bank is not None:
            self.filter_bank.set_order(n)

    def apply_fft_peak_smear(self, v: float) -> None:
        self.cfg.fft.peak_smear_oct = v
        if self.fft_postprocessor is not None:
            self.fft_postprocessor.update_smear(v)

    def _clamp_cfg_bands_for_sr(self, sr: float) -> bool:
        """Clamp cfg band edges into the Nyquist-safe range for `sr`
        (hi ≤ 0.45·sr, 1 Hz ≤ lo ≤ 0.9·hi — `clamp_band_edges`) and write
        them back to cfg, so cfg, meta, OSC meta and the filters all agree.
        A 16/22.05/24 kHz device would otherwise get a 13952 Hz high band it
        can't represent. Returns True if anything changed (then meta is
        marked dirty and, once the persister exists, a save is requested)."""
        changed = False
        for name in ("low", "mid", "high"):
            b = getattr(self.cfg.dsp, name)
            lo, hi = clamp_band_edges(b.lo_hz, b.hi_hz, sr)
            if lo != b.lo_hz or hi != b.hi_hz:
                log.warning("band %s [%.1f, %.1f] Hz not representable at sr=%.0f; "
                            "clamped to [%.1f, %.1f] Hz",
                            name, b.lo_hz, b.hi_hz, sr, lo, hi)
                b.lo_hz, b.hi_hz = lo, hi
                changed = True
        if changed:
            self.mark_meta_dirty()
            self.mark_osc_meta_dirty()
            if self.persister is not None:
                self.persister.request(commit=True)
        return changed

    def _build_pipeline_for_sr(self, sr: float) -> bool:
        """(Re)build every sr-dependent DSP object into self.* (NOT into the
        workers — the caller swaps those in). Clamps cfg bands for `sr`
        first; returns whether they were clamped."""
        cfg = self.cfg
        clamped = self._clamp_cfg_bands_for_sr(sr)
        self.filter_bank = FilterBank(
            sr=sr,
            bands=self._bands_tuple_dict(),
            blocksize=cfg.audio.blocksize,
            order=cfg.dsp.filter_order,
        )
        self.smoother = ExpSmoother(sr=sr, blocksize=cfg.audio.blocksize,
                                    tau=cfg.dsp.tau, tau_attack=cfg.dsp.tau_attack)
        self.auto_scaler = AutoScaler(
            sr=sr,
            blocksize=cfg.audio.blocksize,
            tau_attack_s=cfg.autoscale.tau_attack_s,
            tau_release_s=cfg.autoscale.tau_release_s,
            noise_floor=cfg.autoscale.noise_floor,
            strength=cfg.autoscale.strength,
            tilt_db_per_oct=cfg.fft.tilt_db_per_oct,
            bands=self._bands_tuple_dict(),
            n_fft_window=cfg.fft.window_size,
            fft_n_bins=cfg.fft.n_bins,
            fft_f_min=cfg.fft.f_min,
            db_floor=cfg.fft.db_floor,
            db_ceiling=cfg.fft.db_ceiling,
        )
        if self.onset_tracker is not None:
            # Hot-switch: alphas depend on dt = blocksize / sr. User-tuned
            # per-band params are persisted on the instance and survive
            # reconfigure().
            self.onset_tracker.reconfigure(sr, cfg.audio.blocksize)
            self.onset_tracker.reset()
            return clamped
        self.onset_tracker = OnsetTracker(
            sr=sr,
            blocksize=cfg.audio.blocksize,
            params={
                "low":  {"sensitivity": cfg.onset.low.sensitivity,
                         "refractory_s": cfg.onset.low.refractory_s,
                         "slow_tau_s": cfg.onset.low.slow_tau_s,
                         "abs_floor": cfg.onset.low.abs_floor},
                "mid":  {"sensitivity": cfg.onset.mid.sensitivity,
                         "refractory_s": cfg.onset.mid.refractory_s,
                         "slow_tau_s": cfg.onset.mid.slow_tau_s,
                         "abs_floor": cfg.onset.mid.abs_floor},
                "high": {"sensitivity": cfg.onset.high.sensitivity,
                         "refractory_s": cfg.onset.high.refractory_s,
                         "slow_tau_s": cfg.onset.high.slow_tau_s,
                         "abs_floor": cfg.onset.high.abs_floor},
            },
        )
        return clamped

    def start(self) -> None:
        cfg = self.cfg
        self.loop = asyncio.get_running_loop()
        self._device_switch_lock = asyncio.Lock()

        # -------- Resolve & open audio device --------
        device_idx = devmod.resolve_initial_device(cfg.audio.device, self.args.device)
        log.info("initial device index: %s", device_idx)

        # Build callback (perf ring is shared)
        # We need the stream's sample rate before building DSP.
        # Open a placeholder callback first, then swap. Cleanest: build callback,
        # open stream (which calls back at start()), build DSP from sr, START stream.
        self.callback = AudioCallback(
            ring=self.ring,
            dsp_event=self.dsp_event,
            fft_event=self.fft_event,
            channels=cfg.audio.channels,
            blocksize=cfg.audio.blocksize,
            perf_ring=self.perf_cb,
        )
        try:
            self.stream = open_input_stream(
                device=device_idx,
                blocksize=cfg.audio.blocksize,
                channels=cfg.audio.channels,
                callback=self.callback,
            )
        except Exception as e:
            which = "the system default input" if device_idx is None else f"input device {device_idx}"
            raise StartupError(
                f"could not open {which}: {e}",
                "plug in / enable a microphone, or pick another input with --device N "
                "(list them with `uv run python -m sounddevice`)",
            ) from e

        bands_clamped = self._build_pipeline_for_sr(self.stream.samplerate)

        # -------- OSC: wire sender + publisher BEFORE workers so they can
        # dispatch directly on their own threads (no asyncio hop). --------
        self.osc_sender = OscSender(cfg.osc.destinations)
        self.osc_publisher = OscPublisher(
            self.osc_sender,
            get_master_gain=lambda: self.cfg.autoscale.master_gain,
            get_db_floor=lambda: self.cfg.fft.db_floor,
            get_send_fft=lambda: self.cfg.osc.send_fft,
            get_send_raw_db=lambda: self.cfg.fft.send_raw_db,
            get_fft_enabled=lambda: self.fft_enabled.is_set(),
            perf_lmh_e2e=self.perf_lmh_e2e,
            perf_fft_e2e=self.perf_fft_e2e,
            perf_idx_state=self.perf_e2e_idx,
        )

        # -------- Workers --------
        self.dsp_worker = DSPWorker(
            ring=self.ring,
            dsp_event=self.dsp_event,
            stop_flag=self.stop_flag,
            filter_bank=self.filter_bank,
            smoother=self.smoother,
            auto_scaler=self.auto_scaler,
            onset_tracker=self.onset_tracker,
            features_store=self.features_store,
            osc_publisher=self.osc_publisher,
            blocksize=cfg.audio.blocksize,
            perf_ring=self.perf_dsp,
        )
        self.fft_postprocessor = FFTPostProcessor(
            n_bins=cfg.fft.n_bins,
            f_min=cfg.fft.f_min,
            sr=self.stream.samplerate,
            bands=self._bands_meta_dict(),
            tau=dict(cfg.dsp.tau),
            tau_attack=dict(cfg.dsp.tau_attack),
            tau_release_s=cfg.autoscale.tau_release_s,
            noise_floor=cfg.autoscale.noise_floor,
            strength=cfg.autoscale.strength,
            db_floor=cfg.fft.db_floor,
            db_ceiling=cfg.fft.db_ceiling,
            hop_period_s=cfg.fft.hop / self.stream.samplerate,
            tau_attack_s=cfg.autoscale.tau_attack_s,
            peak_smear_oct=cfg.fft.peak_smear_oct,
            tilt_db_per_oct=cfg.fft.tilt_db_per_oct,
        )
        self.fft_worker = FFTWorker(
            ring=self.ring,
            fft_event=self.fft_event,
            fft_enabled=self.fft_enabled,
            stop_flag=self.stop_flag,
            fft_store=self.fft_store,
            osc_publisher=self.osc_publisher,
            blocksize=cfg.audio.blocksize,
            sr=self.stream.samplerate,
            window_size=cfg.fft.window_size,
            hop=cfg.fft.hop,
            n_bins=cfg.fft.n_bins,
            f_min=cfg.fft.f_min,
            perf_ring=self.perf_fft,
            db_floor=cfg.fft.db_floor,
            post_processor=self.fft_postprocessor,
        )
        self.dsp_worker.start()
        self.fft_worker.start()

        # Send the OSC meta packet now that the sender is wired (resent at
        # 1 Hz + on change by _osc_meta_loop).
        self.send_osc_meta()

        # -------- Persister --------
        self.persister = Persister(self.config_path, get_state=lambda: config_to_dict(self.cfg))
        self.persister.attach(self.loop)
        if bands_clamped:
            self.persister.request(commit=True)

        # -------- WebSocket server (optional) --------
        if cfg.ws.enabled and not self.args.no_ws:
            dispatcher = Dispatcher(self)
            self.ws = WSServer(
                host=cfg.ws.host,
                port=cfg.ws.port,
                snapshot_hz=cfg.ws.snapshot_hz,
                features_store=self.features_store,
                fft_store=self.fft_store,
                get_meta=self.snapshot_meta,
                get_devices=lambda: self.list_devices_with_probe(False),
                get_presets=self.list_presets,
                get_server_status=self.snapshot_server_status,
                get_fft_enabled=lambda: self.fft_enabled.is_set(),
                get_fft_send_raw_db=lambda: self.cfg.fft.send_raw_db,
                get_master_gain=lambda: self.cfg.autoscale.master_gain,
                dispatcher_handle=dispatcher,
                perf_ring=self.perf_ws,
            )

            # Static HTTP server for the UI (ES modules need http://, not file://).
            ui_root = (Path(__file__).resolve().parent.parent / "ui")
            if ui_root.exists():
                http = StaticHTTPServer(host=cfg.ws.host, port=cfg.ws.http_port, root=ui_root)
                try:
                    http.start()
                except OSError as e:
                    raise _bind_error("UI HTTP", cfg.ws.host, cfg.ws.http_port,
                                      "ws.http_port", e) from e
                self.http = http
            else:
                log.warning("ui directory not found at %s; static http server skipped", ui_root)
        else:
            self.ws = None

        # -------- Start stream LAST, after workers are up --------
        try:
            self.stream.start()
        except Exception as e:
            raise StartupError(
                f"could not start audio input device {self.stream.device}: {e}",
                "the device may be busy or unplugged; pick another with --device N",
            ) from e

        # -------- Optional UI launch --------
        if self.args.open:
            if self.ws is not None and self.http is not None:
                url = f"http://{cfg.ws.host}:{cfg.ws.http_port}/index.html"
                try:
                    webbrowser.open(url)
                except Exception as e:
                    log.warning("failed to open browser: %s", e)
            else:
                log.info("--open ignored: WS or HTTP server disabled")

    # ----------------- snapshots / readers exposed to dispatcher + ws -----------------
    def current_sr(self) -> float:
        return float(self.stream.samplerate) if self.stream else 48000.0

    def snapshot_meta(self) -> dict:
        cfg = self.cfg
        device = {"index": None, "name": None}
        if self.stream is not None:
            try:
                info = devmod.device_info(int(self.stream.device)) if isinstance(self.stream.device, int) else None
                if info:
                    device = {"index": int(self.stream.device), "name": info.get("name")}
            except Exception:
                pass
        return {
            "sr": int(self.current_sr()),
            "blocksize": cfg.audio.blocksize,
            "n_fft_bins": cfg.fft.n_bins,
            "bands": self._bands_meta_dict(),
            "fft_enabled": self.fft_enabled.is_set(),
            "fft_db_floor": cfg.fft.db_floor,
            "fft_db_ceiling": cfg.fft.db_ceiling,
            "fft_f_min": cfg.fft.f_min,
            "fft_send_raw_db": cfg.fft.send_raw_db,
            "fft_peak_smear_oct": cfg.fft.peak_smear_oct,
            "fft_tilt_db_per_oct": cfg.fft.tilt_db_per_oct,
            "tau": dict(cfg.dsp.tau),
            "tau_attack": dict(cfg.dsp.tau_attack),
            "filter_order": cfg.dsp.filter_order,
            "autoscale": {
                "tau_attack_s": cfg.autoscale.tau_attack_s,
                "tau_release_s": cfg.autoscale.tau_release_s,
                "noise_floor": cfg.autoscale.noise_floor,
                "strength": cfg.autoscale.strength,
                "master_gain": cfg.autoscale.master_gain,
            },
            "onset": {
                "low": {
                    "sensitivity": cfg.onset.low.sensitivity,
                    "refractory_s": cfg.onset.low.refractory_s,
                    "slow_tau_s": cfg.onset.low.slow_tau_s,
                    "abs_floor": cfg.onset.low.abs_floor,
                },
                "mid": {
                    "sensitivity": cfg.onset.mid.sensitivity,
                    "refractory_s": cfg.onset.mid.refractory_s,
                    "slow_tau_s": cfg.onset.mid.slow_tau_s,
                    "abs_floor": cfg.onset.mid.abs_floor,
                },
                "high": {
                    "sensitivity": cfg.onset.high.sensitivity,
                    "refractory_s": cfg.onset.high.refractory_s,
                    "slow_tau_s": cfg.onset.high.slow_tau_s,
                    "abs_floor": cfg.onset.high.abs_floor,
                },
            },
            "ws_snapshot_hz": self.ws.snapshot_hz if self.ws else cfg.ws.snapshot_hz,
            "ui_peak_decay_per_s": cfg.ui.peak_decay_per_s,
            "ui_show_onsets": cfg.ui.show_onsets,
            "ui_layout": cfg.ui.layout,
            "device": device,
        }

    def list_devices_with_probe(self, probe: bool) -> list:
        items = devmod.list_input_devices()
        if probe:
            results = devmod.signal_active_probe(items, duration=0.2)
            for it in items:
                it["probed_signal"] = bool(results.get(it["index"], False))
                it["probed_at"] = time.strftime("%H:%M:%S")
        return items

    def list_presets(self) -> list[dict]:
        items = []
        try:
            self.config_dir.mkdir(exist_ok=True)
            for p in sorted(self.config_dir.glob("*.yaml")):
                if p.stem.lower() == "main":
                    continue
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                saved_at = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
                items.append({"name": p.stem, "saved_at": saved_at})
        except Exception:
            pass
        items.sort(key=lambda x: x["saved_at"], reverse=True)
        return items

    def preset_path(self, name: str) -> Path:
        # Caller has already passed `name` through validate_preset_name, so it's
        # a stripped, anchored, character-class-restricted string.
        self.config_dir.mkdir(exist_ok=True)
        return self.config_dir / f"{name}.yaml"

    def preset_body(self, name: str) -> dict:
        # Presets are full snapshots — same shape as main.yaml — so loading is
        # a deep-merge over main as fallback for any missing keys.
        return config_to_dict(self.cfg)

    # ----------------- perf summary -----------------
    @staticmethod
    def _ring_stats(ring: np.ndarray, count: int) -> tuple[float, float]:
        if count == 0:
            return 0.0, 0.0
        n = min(count, ring.shape[0])
        valid = ring[:n] if count <= ring.shape[0] else ring
        # filter zeros — slot may be unwritten
        nz = valid[valid > 0]
        sz = nz.size
        if sz == 0:
            return 0.0, 0.0
        avg_ns = float(np.mean(nz))
        # p95 via np.partition — O(n) quickselect, vs np.percentile's O(n log n)
        # full sort + linear interpolation. Status loop runs 6 of these at 5 Hz,
        # so on Pi 4 the partition swap saves a few ms/sec. Exact integer index
        # (no interpolation between samples) is fine for monitoring.
        k = int(0.95 * (sz - 1))
        p95_ns = float(np.partition(nz, k)[k])
        return avg_ns / 1e6, p95_ns / 1e6  # ms

    def snapshot_server_status(self) -> dict:
        cfg = self.cfg
        sr = self.current_sr()
        block_period_ms = cfg.audio.blocksize / sr * 1000.0
        hop_period_ms = (cfg.fft.hop / sr) * 1000.0
        ws_hz = self.ws.snapshot_hz if self.ws else cfg.ws.snapshot_hz
        ws_period_ms = 1000.0 / max(ws_hz, 1)

        cb_avg, cb_p95 = self._ring_stats(self.perf_cb,  self.callback.perf_idx if self.callback else 0)
        ds_avg, ds_p95 = self._ring_stats(self.perf_dsp, self.dsp_worker.perf_idx if self.dsp_worker else 0)
        ff_avg, ff_p95 = self._ring_stats(self.perf_fft, self.fft_worker.perf_idx if self.fft_worker else 0)
        ws_avg, ws_p95 = self._ring_stats(self.perf_ws,  self.ws.perf_idx if self.ws else 0)
        lmh_e2e_avg, lmh_e2e_p95 = self._ring_stats(self.perf_lmh_e2e, self.perf_e2e_idx["lmh"])
        fft_e2e_avg, fft_e2e_p95 = self._ring_stats(self.perf_fft_e2e, self.perf_e2e_idx["fft"])

        def load(avg_ms, period_ms):
            if period_ms <= 0:
                return 0.0
            return float(avg_ms / period_ms * 100.0)

        return {
            "type": "server_status",
            # Counted from the current device session (see _drops_base).
            "cb_overruns": max(0, int(self.callback.cb_overruns) - self._drops_base["cb"])
                           if self.callback else 0,
            "dsp_drops": max(0, int(self.dsp_worker.dsp_drops) - self._drops_base["dsp"])
                         if self.dsp_worker else 0,
            "fft_drops": max(0, int(self.fft_worker.fft_drops) - self._drops_base["fft"])
                         if self.fft_worker else 0,
            "perf": {
                "block_period_ms": block_period_ms,
                "hop_period_ms": hop_period_ms,
                "ws_period_ms": ws_period_ms,
                "lmh_e2e": {"avg_ms": lmh_e2e_avg, "p95_ms": lmh_e2e_p95,
                            "load_pct": load(lmh_e2e_avg, block_period_ms)},
                "fft_e2e": {"avg_ms": fft_e2e_avg, "p95_ms": fft_e2e_p95,
                            "load_pct": load(fft_e2e_avg, hop_period_ms),
                            "enabled": self.fft_enabled.is_set() and bool(cfg.osc.send_fft)},
                "cb":  {"avg_ms": cb_avg, "p95_ms": cb_p95, "load_pct": load(cb_avg, block_period_ms)},
                "dsp": {"avg_ms": ds_avg, "p95_ms": ds_p95, "load_pct": load(ds_avg, block_period_ms)},
                "fft": {"avg_ms": ff_avg, "p95_ms": ff_p95, "load_pct": load(ff_avg, hop_period_ms),
                        "enabled": self.fft_enabled.is_set()},
                "ws":  {"avg_ms": ws_avg, "p95_ms": ws_p95, "load_pct": load(ws_avg, ws_period_ms)},
            },
        }

    # ----------------- mutators called from dispatcher -----------------
    def schedule_filter_retune(self) -> None:
        """Server-side 50ms debounce of cutoff retunes.

        Reads cfg.dsp at fire time so multiple per-band updates within the
        debounce window collapse into a single retune of the latest state.
        """
        loop = self.loop
        if loop is None:
            return
        if self._filter_retune_handle is not None:
            self._filter_retune_handle.cancel()
        self._filter_retune_handle = loop.call_later(
            0.05, lambda: self.filter_bank.retune(self._bands_tuple_dict())
        )

    def _capture_drop_baselines(self) -> None:
        self._drops_base = {
            "cb": int(self.callback.cb_overruns) if self.callback else 0,
            "dsp": int(self.dsp_worker.dsp_drops) if self.dsp_worker else 0,
            "fft": int(self.fft_worker.fft_drops) if self.fft_worker else 0,
        }

    def _teardown_stream(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _reset_stream_state(self) -> None:
        """Rewind ring + worker read pointers + perf rings for a fresh stream.
        Only called while no stream is running (callback fires nothing)."""
        self.ring.reset()
        self.dsp_worker.read_block_idx = 0
        self.fft_worker.reset()
        # zero perf rings so new sr's load is visible cleanly
        for r in (self.perf_cb, self.perf_dsp, self.perf_fft, self.perf_ws,
                  self.perf_lmh_e2e, self.perf_fft_e2e):
            r.fill(0)
        self.perf_e2e_idx["lmh"] = 0
        self.perf_e2e_idx["fft"] = 0
        self.callback.perf_idx = 0
        self.dsp_worker.perf_idx = 0
        self.fft_worker.perf_idx = 0
        if self.ws is not None:
            self.ws.reset_perf()

    def _install_stream(self, idx: int | None) -> None:
        """Open `idx`, rebuild the sr-dependent pipeline, swap it into the
        workers and start the stream. On failure closes whatever it opened
        (self.stream stays None) and re-raises; the caller rolls back."""
        stream = open_input_stream(
            device=idx,
            blocksize=self.cfg.audio.blocksize,
            channels=self.cfg.audio.channels,
            callback=self.callback,
        )
        try:
            new_sr = stream.samplerate
            # Rebuild filter + autoscaler + smoother + onset alphas for the
            # new sr (clamps cfg bands to the new Nyquist first).
            self._build_pipeline_for_sr(new_sr)
            # Swap into worker (idle: no stream is running).
            self.dsp_worker.filter_bank = self.filter_bank
            self.dsp_worker.smoother = self.smoother
            self.dsp_worker.auto_scaler = self.auto_scaler
            # FFT worker rebuild for sr (this also reconfigures the
            # post-processor's hop_period_s / sr — see FFTWorker.reconfigure).
            self.fft_worker.reconfigure(sr=new_sr)
            self.fft_postprocessor.update_bands(self._bands_meta_dict())
            self.fft_postprocessor.reset()
            self.stream = stream
            stream.start()
        except Exception:
            self.stream = None
            stream.stop()
            stream.close()
            raise

    async def hot_switch_device(self, new_idx: int) -> None:
        """Tear down stream, rebuild for new device, start fresh.

        Transactional: if anything fails (index vanished, device unplugged
        or busy, filter design error), the previous device, sample rate and
        pipeline are restored and the old stream restarted, then ValueError
        is raised so the dispatcher replies with an error + snap-back meta.

        The DSP/FFT workers stay alive — they idle on dsp_event/fft_event
        timeouts while the stream is down (callback fires nothing).
        """
        async with self._device_switch_lock:
            # Upper-bound / existence check BEFORE touching the running
            # stream: the index must be a current input device.
            available = {d["index"] for d in devmod.list_input_devices()}
            if new_idx not in available:
                raise ValueError(f"device {new_idx} is not an available input device")

            old_idx = self.stream.device if self.stream is not None else None
            if not isinstance(old_idx, int):
                old_idx = self.cfg.audio.device.index
            d = self.cfg.dsp
            old_bands = {n: (getattr(d, n).lo_hz, getattr(d, n).hi_hz)
                         for n in ("low", "mid", "high")}

            self._teardown_stream()
            self._reset_stream_state()
            try:
                self._install_stream(new_idx)
            except Exception as e:
                log.warning("switch to device %s failed (%s); rolling back to device %s",
                            new_idx, e, old_idx)
                # Undo any band clamp done for the failed device's sr.
                for n, (lo, hi) in old_bands.items():
                    getattr(d, n).lo_hz, getattr(d, n).hi_hz = lo, hi
                self._reset_stream_state()
                try:
                    self._install_stream(old_idx)
                except Exception as e2:
                    log.error("rollback to device %s also failed (%s); audio input is "
                              "stopped — pick a device in the UI", old_idx, e2)
                self._after_stream_change()
                raise ValueError(f"could not switch to device {new_idx}: {e}") from e

            # Update cfg with the device choice
            info = devmod.device_info(new_idx) or {}
            self.cfg.audio.device.index = new_idx
            self.cfg.audio.device.name = info.get("name")
            self._after_stream_change()

    def _after_stream_change(self) -> None:
        # Push fresh meta over OSC now (sr may have changed) + WS meta on the
        # next broadcast tick.
        self.send_osc_meta()
        self.mark_meta_dirty()
        # Count drops/overruns from this device session. Re-baseline once
        # more shortly after, to exclude the few spurious drops the workers
        # can register while self-healing their read pointers onto the reset
        # ring on the new stream's first blocks.
        self._capture_drop_baselines()
        if self.loop is not None:
            self.loop.call_later(0.25, self._capture_drop_baselines)

    # ----------------- shutdown -----------------
    async def shutdown(self) -> None:
        # Also used to clean up after a failed start(), so every step
        # tolerates partially-initialised state.
        if self._osc_meta_task is not None:
            self._osc_meta_task.cancel()
            try:
                await self._osc_meta_task
            except (asyncio.CancelledError, Exception):
                pass
        log.info("shutdown: stopping audio stream")
        self._teardown_stream()
        log.info("shutdown: stopping workers")
        self.stop_flag.set()
        self.dsp_event.set()
        self.fft_event.set()
        for w in (self.dsp_worker, self.fft_worker):
            if w is not None and w.is_alive():
                w.join(timeout=1.0)
        if self.ws is not None:
            await self.ws.stop()
        if self.http is not None:
            self.http.stop()
        if self.persister is not None:
            self.persister.flush_now_sync()


def _bind_error(what: str, host: str, port: int, cfg_key: str, e: OSError) -> StartupError:
    if e.errno == errno.EADDRINUSE:
        return StartupError(
            f"{what} port {port} on {host} is already in use",
            f"another audio-server is probably running (check `lsof -i :{port}`); "
            f"stop it, or change {cfg_key} in the config",
        )
    return StartupError(f"{what} server could not bind {host}:{port}: {e}",
                        f"check ws.host / {cfg_key} in the config")


async def _run(args, cfg, config_path):
    app = App(args, cfg, config_path)
    try:
        app.start()
        if app.ws is not None:
            try:
                await app.ws.start()
            except OSError as e:
                raise _bind_error("WebSocket", cfg.ws.host, cfg.ws.port, "ws.port", e) from e
        app.start_background_tasks()
    except BaseException:
        await app.shutdown()
        raise

    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows fallback

    try:
        await stop_event.wait()
    finally:
        await app.shutdown()


def _migrate_legacy_config_layout(config_path: Path) -> None:
    """One-shot migration: fold legacy `config.yaml` + `presets/*.yaml` into `configs/`."""
    configs_dir = config_path.parent
    try:
        configs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    cwd = Path.cwd().resolve()
    # Legacy main config at cwd/config.yaml -> configs/main.yaml
    legacy_main = cwd / "config.yaml"
    if legacy_main.exists() and not config_path.exists() and legacy_main.resolve() != config_path:
        try:
            legacy_main.rename(config_path)
            log.info("migrated %s -> %s", legacy_main, config_path)
        except OSError as e:
            log.warning("could not migrate legacy config.yaml: %s", e)
    # Legacy presets/*.yaml or presets/preset-*.yaml -> configs/<name>.yaml
    legacy_presets = cwd / "presets"
    if legacy_presets.is_dir():
        for p in legacy_presets.glob("*.yaml"):
            stem = p.stem
            if stem.startswith("preset-"):
                stem = stem[len("preset-"):]
            if stem.lower() == "main":
                continue
            target = configs_dir / f"{stem}.yaml"
            if target.exists():
                continue
            try:
                p.rename(target)
            except OSError:
                pass
        try:
            legacy_presets.rmdir()
        except OSError:
            pass
    # Legacy preset-*.yaml at cwd
    for p in cwd.glob("preset-*.yaml"):
        stem = p.stem[len("preset-"):]
        if not stem or stem.lower() == "main":
            continue
        target = configs_dir / f"{stem}.yaml"
        if target.exists():
            continue
        try:
            p.rename(target)
        except OSError:
            pass


def _resolve_config_path(arg_value: str | None) -> Path:
    """Find the config file deterministically, regardless of CWD.

    - --config omitted: use the canonical <repo>/configs/main.yaml.
    - --config absolute: use as-is.
    - --config relative: try CWD first, then repo-root.
    """
    if arg_value is None:
        return CANONICAL_CONFIG_PATH.resolve()
    p = Path(arg_value)
    if p.is_absolute():
        return p.resolve()
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = (CANONICAL_CONFIG_PATH.parent.parent / p).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return cwd_candidate  # surface a sensible path in the error message


def _fatal(message: str, hint: str = "") -> int:
    print(f"audio-server: error: {message}", file=sys.stderr)
    if hint:
        print(f"audio-server: hint: {hint}", file=sys.stderr)
    return 1


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Python's default GIL switch interval is 5 ms: a thread that wants the
    # GIL (the audio callback's Python body, the DSP/FFT workers after an
    # event wake) waits up to that long whenever another thread (typically
    # the asyncio loop JSON-encoding WS frames) holds it without blocking —
    # a few consecutive hand-offs add up to ~10+ ms of wake-up latency, i.e.
    # multiple audio blocks. 0.5 ms keeps the hand-off well under one block
    # period (5.3 ms at 48k/256) for a small context-switch overhead.
    sys.setswitchinterval(5e-4)
    # Lift the main thread (which becomes the asyncio loop thread, running
    # the OSC sender / WS broadcaster / dispatcher). Best-effort; no-ops on
    # platforms / permissions where it can't apply.
    boost_current_thread("main")
    config_path = _resolve_config_path(args.config)
    _migrate_legacy_config_layout(config_path)
    if not config_path.exists() and args.config is None:
        # Fresh clone: seed configs/main.yaml from the checked-in example so
        # the persister has a writable target. Example has device=null so
        # resolve_initial_device falls through to the system default.
        example = config_path.parent / "main.example.yaml"
        if example.exists():
            import shutil
            shutil.copyfile(example, config_path)
            log.info("seeded %s from %s", config_path, example)
    if not config_path.exists():
        return _fatal(f"config file not found: {config_path}",
                      "omit --config to use configs/main.yaml (seeded from main.example.yaml)")
    log.info("loading config from %s", config_path)
    example = config_path.parent / "main.example.yaml"
    if not example.exists():
        example = CANONICAL_CONFIG_PATH.parent / "main.example.yaml"
    try:
        # A torn / empty / unparseable file is backed up as
        # <name>.corrupt-<timestamp> and replaced by the example defaults.
        cfg = load_config_or_fallback(config_path, example)
    except (ValueError, OSError) as e:
        return _fatal(f"invalid config: {e}",
                      f"fix the value in {config_path.name}, or delete the file to "
                      "regenerate it from configs/main.example.yaml")
    if args.no_ws:
        cfg.ws.enabled = False
    try:
        asyncio.run(_run(args, cfg, config_path))
    except KeyboardInterrupt:
        pass
    except StartupError as e:
        return _fatal(e.message, e.hint)
    except sd.PortAudioError as e:
        return _fatal(f"audio device error: {e}",
                      "plug in a mic and restart, or pick another input with --device N")
    return 0


if __name__ == "__main__":
    sys.exit(main())
