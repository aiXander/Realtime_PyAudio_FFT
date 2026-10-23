"""sounddevice.InputStream lifecycle wrapper."""
from __future__ import annotations

import logging

import sounddevice as sd

log = logging.getLogger(__name__)


class StreamHandle:
    def __init__(self, stream: sd.InputStream):
        self.stream = stream
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
        self.device = stream.device

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


def open_input_stream(device: int | None, blocksize: int, channels: int, callback) -> StreamHandle:
    """Open the input stream and return a handle. Sample rate is device-driven."""
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
    h = StreamHandle(stream)
    log.info(
        "stream open: device=%s sr=%.0f blocksize=%d channels=%d latency=%.1fms",
        device, h.samplerate, blocksize, channels, h.latency * 1000.0
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
