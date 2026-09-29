"""Config dataclass, YAML load/save, debounced atomic Persister.

There are NO hardcoded defaults in this file. Every value is loaded from
the YAML config file (default: ``<repo>/configs/main.yaml``). The dataclasses
below are bare typed schemas; ``load_config`` constructs them from a
fully-populated dict read off disk. The committed ``configs/main.yaml`` is
the canonical default state — the runtime persister keeps it complete.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

from .control import validate as V

log = logging.getLogger(__name__)

CANONICAL_CONFIG_PATH = (Path(__file__).resolve().parent.parent / "configs" / "main.yaml")


@dataclass
class DeviceCfg:
    name: str | None
    index: int | None


@dataclass
class AudioCfg:
    device: DeviceCfg
    blocksize: int
    channels: int


@dataclass
class BandCfg:
    lo_hz: float
    hi_hz: float


@dataclass
class DspCfg:
    low: BandCfg
    mid: BandCfg
    high: BandCfg
    tau: dict
    tau_attack: dict
    filter_order: int


@dataclass
class AutoscaleCfg:
    tau_attack_s: float
    tau_release_s: float
    noise_floor: float
    strength: float
    master_gain: float


@dataclass
class FftCfg:
    enabled: bool
    n_bins: int
    window_size: int
    hop: int
    f_min: float
    db_floor: float
    db_ceiling: float
    peak_smear_oct: float
    tilt_db_per_oct: float
    send_raw_db: bool


@dataclass
class OnsetBandCfg:
    sensitivity: float    # K_HIGH multiplier on adaptive threshold
    refractory_s: float   # min inter-onset interval (seconds)
    slow_tau_s: float     # slow envelope tau (seconds)
    abs_floor: float      # absolute floor on the Schmitt high threshold


@dataclass
class OnsetCfg:
    low: OnsetBandCfg
    mid: OnsetBandCfg
    high: OnsetBandCfg


@dataclass
class OscDest:
    host: str
    port: int


@dataclass
class OscCfg:
    destinations: list
    send_fft: bool


@dataclass
class UiCfg:
    peak_decay_per_s: float
    layout: dict
    show_onsets: bool


@dataclass
class WsCfg:
    enabled: bool
    host: str
    port: int
    snapshot_hz: int
    http_port: int


@dataclass
class Config:
    audio: AudioCfg
    dsp: DspCfg
    autoscale: AutoscaleCfg
    fft: FftCfg
    onset: OnsetCfg
    osc: OscCfg
    ws: WsCfg
    ui: UiCfg


def _build_config(d: dict) -> Config:
    """Build a Config from a fully-populated dict.

    Missing keys raise KeyError — config files are expected to be
    complete (the runtime persister always writes the full state).
    Values are routed through validators where the live control path has
    them, so a corrupt-but-parseable YAML still surfaces a clear error.
    """
    a = d["audio"]
    dev = a.get("device") or {}
    audio = AudioCfg(
        device=DeviceCfg(name=dev.get("name"), index=dev.get("index")),
        blocksize=V.validate_blocksize(a["blocksize"]),
        channels=int(a["channels"]),
    )

    ds = d["dsp"]
    bands_raw = {k: ds[k] for k in ("low", "mid", "high")}
    # The device sample rate isn't known at load time, so only the
    # structural invariants (finite, lo > 0, hi > lo) are checked here.
    # App clamps the edges against the ACTUAL sr (hi ≤ 0.45·sr) as soon as
    # the stream is open, writes them back to cfg and persists
    # (App._clamp_cfg_bands_for_sr). Validating against a guessed 48k here
    # used to reject a config saved on a 96k device.
    ok = V.validate_bands(bands_raw, None)
    dsp = DspCfg(
        low=BandCfg(*ok["low"]),
        mid=BandCfg(*ok["mid"]),
        high=BandCfg(*ok["high"]),
        tau=V.validate_tau(ds["tau"]),
        tau_attack=V.validate_tau(ds["tau_attack"]),
        # filter_order added after the initial schema; default to 2 (cheap,
        # 12 dB/oct skirt — plenty for envelope/beat use).
        filter_order=V.validate_filter_order(ds.get("filter_order", 2)),
    )

    asd = d["autoscale"]
    ok_as = V.validate_autoscale(
        tau_attack_s=asd["tau_attack_s"],
        tau_release_s=asd["tau_release_s"],
        noise_floor=asd["noise_floor"],
        strength=asd["strength"],
        master_gain=asd["master_gain"],
    )
    autoscale = AutoscaleCfg(**ok_as)

    fd = d["fft"]
    # Same validator the preset loader / App.apply_fft_window use: window and
    # hop must be positive multiples of blocksize (else FFTWorker spins on a
    # zero-block geometry) and fit the SlotRing; f_min > 0.
    ok_win = V.validate_fft_window(
        window_size=fd["window_size"], hop=fd["hop"], f_min=fd["f_min"],
        blocksize=audio.blocksize,
    )
    fft = FftCfg(
        enabled=bool(fd["enabled"]),
        n_bins=V.validate_n_fft_bins(fd["n_bins"]),
        window_size=ok_win["window_size"],
        hop=ok_win["hop"],
        f_min=ok_win["f_min"],
        db_floor=float(fd["db_floor"]),
        db_ceiling=float(fd["db_ceiling"]),
        peak_smear_oct=V.validate_peak_smear_oct(fd["peak_smear_oct"]),
        tilt_db_per_oct=V.validate_fft_tilt_db_per_oct(fd["tilt_db_per_oct"]),
        send_raw_db=bool(fd["send_raw_db"]),
    )

    # Per-band onset detector. The legacy single-band `beat:` block (low-band
    # only) is migrated transparently here: if `onset:` is missing but `beat:`
    # is present, use it for the low band and seed mid/high with snappier
    # defaults (shorter refractory + slow tau — snares and hats are sharper
    # and denser than kicks). On the next config write the new `onset:` shape
    # is persisted and the `beat:` block disappears.
    onset_raw = d.get("onset")
    if not isinstance(onset_raw, dict):
        legacy = d.get("beat") or {}
        onset_raw = {
            "low": {
                "sensitivity": legacy.get("sensitivity", 1.8),
                "refractory_s": legacy.get("refractory_s", 0.25),
                "slow_tau_s": legacy.get("slow_tau_s", 0.30),
            },
            "mid":  {"sensitivity": 1.8, "refractory_s": 0.18, "slow_tau_s": 0.20},
            "high": {"sensitivity": 1.8, "refractory_s": 0.12, "slow_tau_s": 0.15},
        }
    onset_defaults = {
        "low":  {"sensitivity": 1.8, "refractory_s": 0.25, "slow_tau_s": 0.30, "abs_floor": 0.10},
        "mid":  {"sensitivity": 1.8, "refractory_s": 0.18, "slow_tau_s": 0.20, "abs_floor": 0.10},
        "high": {"sensitivity": 1.8, "refractory_s": 0.12, "slow_tau_s": 0.15, "abs_floor": 0.10},
    }
    onset_bands = {}
    for name in ("low", "mid", "high"):
        b = onset_raw.get(name) or {}
        dflt = onset_defaults[name]
        ok = V.validate_onset(
            sensitivity=b.get("sensitivity", dflt["sensitivity"]),
            refractory_s=b.get("refractory_s", dflt["refractory_s"]),
            slow_tau_s=b.get("slow_tau_s", dflt["slow_tau_s"]),
            abs_floor=b.get("abs_floor", dflt["abs_floor"]),
        )
        onset_bands[name] = OnsetBandCfg(**ok)
    onset = OnsetCfg(**onset_bands)

    od = d["osc"]
    dests = [OscDest(host=str(x["host"]), port=int(x["port"])) for x in od["destinations"]]
    osc = OscCfg(destinations=dests, send_fft=bool(od["send_fft"]))

    wd = d["ws"]
    ws = WsCfg(
        enabled=bool(wd["enabled"]),
        host=str(wd["host"]),
        port=int(wd["port"]),
        snapshot_hz=V.validate_ws_snapshot_hz(wd["snapshot_hz"]),
        http_port=int(wd["http_port"]),
    )

    ud = d["ui"]
    ui = UiCfg(
        peak_decay_per_s=V.validate_peak_decay_per_s(ud["peak_decay_per_s"]),
        layout=V.validate_ui_layout(ud["layout"]),
        # Added after the initial schema; legacy configs without the key
        # default to OFF (onset squares hidden — matches the user's first-run
        # expectation of a minimal bars+peak-hold view).
        show_onsets=bool(ud.get("show_onsets", False)),
    )

    return Config(audio=audio, dsp=dsp, autoscale=autoscale, fft=fft, onset=onset, osc=osc, ws=ws, ui=ui)


class ConfigCorruptError(ValueError):
    """The config file is unreadable as a config: YAML parse error, empty
    file, non-mapping root, or a missing required section/key. Distinct from
    a plain ValueError (parseable file with an invalid value), which the
    caller reports instead of silently replacing the file."""


def load_config(path: Path | str) -> Config:
    """Load and validate a config file. No fallbacks, no merging.

    Raises ``FileNotFoundError`` if missing, ``ConfigCorruptError`` if the
    file doesn't parse / is empty / is missing required keys, and
    ``ValueError`` if a value is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        raise ConfigCorruptError(f"config {path} failed to parse: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigCorruptError(f"config root must be a YAML mapping (empty file?): {path}")
    try:
        return _build_config(raw)
    except (KeyError, TypeError) as e:
        # KeyError: missing section/key. TypeError: a section that's not a
        # mapping (e.g. `dsp: null`) being subscripted.
        raise ConfigCorruptError(f"config {path} missing/malformed required key: {e}") from e


def load_config_or_fallback(path: Path | str, example_path: Path | str) -> Config:
    """Load `path`; if it is corrupt (see ConfigCorruptError), move it aside
    as ``<name>.corrupt-<timestamp>``, re-seed it from `example_path` and
    load that instead, so a torn / emptied main.yaml never bricks startup.
    Invalid-but-parseable values still raise ValueError (the user's file is
    likely a hand edit worth fixing, not garbage)."""
    path = Path(path)
    try:
        return load_config(path)
    except ConfigCorruptError as e:
        example_path = Path(example_path)
        if not example_path.exists():
            raise
        stamp = time.strftime('%Y%m%d-%H%M%S')
        backup = path.with_name(f"{path.name}.corrupt-{stamp}")
        n = 1
        while backup.exists():  # never overwrite an earlier backup
            backup = path.with_name(f"{path.name}.corrupt-{stamp}-{n}")
            n += 1
        log.warning("%s — backing it up as %s and falling back to %s defaults",
                    e, backup.name, example_path.name)
        os.replace(path, backup)
        shutil.copyfile(example_path, path)
        return load_config(path)


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def write_yaml_atomic(path: Path | str, data: dict) -> None:
    """Atomic, durable write: tmp file → flush + fsync → os.replace → fsync
    the directory. Without the fsync, a power loss shortly after the rename
    can leave a zero-length main.yaml (the rename is journaled before the
    data blocks hit disk)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        # Best-effort: persist the rename itself (POSIX; no-op elsewhere).
        try:
            dfd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            dfd = None
        if dfd is not None:
            try:
                os.fsync(dfd)
            except OSError:
                pass
            finally:
                os.close(dfd)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class Persister:
    """Drag-aware debounced persister for config.yaml.

    Schedules atomic writes on the asyncio loop. Two debounce levels:
      - commit=False (drag): 1s delay, but a 250ms cap from the *first*
        unsaved change ensures we never lag too far behind.
      - commit=True / discrete events: 50ms delay, also capped at 250ms.
    """

    DRAG_DELAY = 1.0
    COMMIT_DELAY = 0.05
    MAX_DELAY = 0.25

    def __init__(self, path: Path | str, get_state):
        self.path = Path(path)
        self._get_state = get_state
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handle: asyncio.TimerHandle | None = None
        self._earliest_deadline: float | None = None
        self._dirty = False

    def attach(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def request(self, commit: bool):
        if self._loop is None:
            return
        self._dirty = True
        now = self._loop.time()
        delay = self.COMMIT_DELAY if commit else self.DRAG_DELAY
        target = now + delay
        if self._earliest_deadline is None:
            self._earliest_deadline = now + self.MAX_DELAY
        target = min(target, self._earliest_deadline)
        if self._handle is not None:
            self._handle.cancel()
        self._handle = self._loop.call_at(target, self._flush)

    def _flush(self):
        self._handle = None
        self._earliest_deadline = None
        if not self._dirty:
            return
        self._dirty = False
        try:
            data = self._get_state()
            write_yaml_atomic(self.path, data)
        except Exception as e:
            log.warning("persist failed: %s", e)

    def flush_now_sync(self):
        """Synchronous flush for shutdown. Cancels pending timer."""
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        if not self._dirty:
            return
        self._dirty = False
        try:
            data = self._get_state()
            write_yaml_atomic(self.path, data)
        except Exception as e:
            log.warning("final persist failed: %s", e)
