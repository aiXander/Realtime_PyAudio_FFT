"""sounddevice.InputStream lifecycle wrapper."""
from __future__ import annotations

import logging

import sounddevice as sd

from . import devices, system_tap

log = logging.getLogger(__name__)


class StreamHandle:
    def __init__(self, stream: sd.InputStream, tap: system_tap.SystemTap | None = None):
        self.stream = stream
        # Set for "System audio": the tap this stream reads, closed with it.
        self.tap = tap
        self.samplerate = float(stream.samplerate)
        # InputStream.latency is normally a scalar, but some sounddevice
        # builds / duplex paths report a (input, output) tuple — pick the
        # input side so the warning math below stays meaningful.
        lat_raw = getattr(stream, "latency", 0.0) or 0.0
        if isinstance(lat_raw, (list, tuple)):
            lat_raw = lat_raw[0] if lat_raw else 0.0
        self.latency = float(lat_raw)
        self.blocksize = int(stream.blocksize) if stream.blocksize else 0
        self.channels = int(stream.channels)
        # The device-list entry this stream is (for System audio: its entry,
        # not the hidden tap device PortAudio actually opened).
        if tap is not None:
            self.device = devices.system_audio_index()
            self.device_name = system_tap.SYSTEM_AUDIO_NAME
        else:
            self.device = stream.device
            try:
                self.device_name = sd.query_devices(stream.device)["name"]
            except Exception:
                self.device_name = None

    def output_moved(self) -> bool:
        """True when this is System audio and the default output changed
        since the tap was made (it keeps recording the old device)."""
        return self.tap is not None and system_tap.default_output_uid() not in (None, self.tap.output_uid)

    def start(self):
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop(ignore_errors=True)
        except Exception:
            pass

    def abort(self):
        """Stop without draining pending buffers — for a stream whose device
        has gone away, where a draining stop() can block."""
        try:
            self.stream.abort(ignore_errors=True)
        except Exception:
            pass

    def close(self):
        try:
            self.stream.close(ignore_errors=True)
        except Exception:
            pass
        if self.tap is not None:
            self.tap.close()


def open_input_stream(device: int | None, blocksize: int, channels: int, callback) -> StreamHandle:
    """Open the input stream and return a handle. Sample rate is device-driven.
    `device == devices.system_audio_index()` opens System audio (a tap on the
    default output); the caller must have closed its previous stream, since
    making PortAudio see the new tap device needs a re-init."""
    tap = None
    if device is not None and device == devices.system_audio_index() and system_tap.available():
        uid = system_tap.default_output_uid()
        if uid is None:
            raise RuntimeError("System audio: no default output device")
        tap = system_tap.SystemTap(uid)
        devices.reinitialize_portaudio()
        device = next((i for i, d in enumerate(sd.query_devices()) if d["name"] == tap.name), None)
        if device is None:
            _close_tap(tap)
            raise RuntimeError("System audio: the tap device did not appear")
    try:
        return _open(device, blocksize, channels, callback, tap)
    except Exception:
        if tap is not None:
            _close_tap(tap)
        raise


def _close_tap(tap: system_tap.SystemTap) -> None:
    """Destroy a tap whose stream failed to open, and re-init PortAudio so
    device indices are back to what the caller resolved them against (its
    rollback reopens the previous device by index)."""
    tap.close()
    devices.reinitialize_portaudio()


def _open(device: int | None, blocksize: int, channels: int, callback, tap) -> StreamHandle:
    # Clamp `channels` to what the device actually exposes — a user with channels=2
    # in config but a mono mic would otherwise hit a PortAudio error at boot.
    if device is not None:
        try:
            info = sd.query_devices(device)
            max_in = int(info.get("max_input_channels", 0) or 0)
            if max_in >= 1 and channels > max_in:
                log.warning("requested channels=%d but device exposes %d; clamping",
                            channels, max_in)
                channels = max_in
        except Exception as e:
            log.debug("device probe for channel clamp failed: %s", e)
    stream = sd.InputStream(
        device=device,
        samplerate=None,
        blocksize=blocksize,
        channels=channels,
        dtype="float32",
        callback=callback,
        latency="low",
    )
    h = StreamHandle(stream, tap)
    log.info(
        "stream open: device=%s (%s) sr=%.0f blocksize=%d channels=%d latency=%.1fms",
        h.device, h.device_name, h.samplerate, blocksize, channels, h.latency * 1000.0
    )
    # Only warn on truly excessive host latency. On macOS CoreAudio the reported
    # device latency includes a hardware safety offset (~30–40ms is normal for
    # the built-in mic at blocksize=256/48k), so the old "2x blocksize/sr"
    # threshold misfired on every clean run. Warn only above an absolute floor
    # that actually correlates with audible problems.
    if h.latency > 0.080:
        log.warning(
            "device latency %.1fms is high; check Audio MIDI Setup buffer",
            h.latency * 1000.0,
        )
    return h
