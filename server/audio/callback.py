"""Hot-path audio callback: memcpy + ring write + signal. No allocation."""
from __future__ import annotations

import threading
import time

import numpy as np


class AudioCallback:
    """Stateful callable. PortAudio invokes __call__ from its C thread.

    Per call:
      1. Sticky-check status.input_overflow -> bump cb_overruns.
      2. Mono-mix (if stereo) into preallocated mono_buf via two in-place ufuncs.
      3. ring.write_block(mono_buf) — single np.copyto + two int stores.
      4. dsp_event.set(); fft_event.set() (cheap; no alloc).
    """

    def __init__(self, ring, dsp_event: threading.Event, fft_event: threading.Event,
                 channels: int, blocksize: int, perf_ring: np.ndarray):
        self.ring = ring
        self.dsp_event = dsp_event
        self.fft_event = fft_event
        self.channels = channels
        self.blocksize = blocksize
        self.mono_buf = np.zeros(blocksize, dtype=np.float32)
        self.cb_overruns = 0
        self.perf_ring = perf_ring  # int64 ns-deltas; len must be power-of-two friendly
        self.perf_len = perf_ring.shape[0]
        self.perf_idx = 0

    def __call__(self, in_data, frames, time_info, status):
        t0 = time.perf_counter_ns()
        if status and getattr(status, "input_overflow", False):
            self.cb_overruns += 1
        # Drive the mono-mix off in_data's actual width, not self.channels.
        # PortAudio / sounddevice may clamp the opened stream to fewer channels
        # than requested (e.g. config asks for stereo on a mono USB mic on
        # Linux/Windows) — indexing in_data[:, 1] under that condition would
        # raise IndexError from the audio thread.
        ch = in_data.shape[1] if in_data.ndim == 2 else 1
        if ch == 1:
            np.copyto(self.mono_buf, in_data[:, 0])
        else:
            np.add(in_data[:, 0], in_data[:, 1], out=self.mono_buf)
            np.multiply(self.mono_buf, 0.5, out=self.mono_buf)
        t1 = time.perf_counter_ns()
        # Stamp the slot with the moment the block became available. Done
        # before signalling so consumers reading the slot always see a valid
        # timestamp paired with the data.
        self.ring.write_block(self.mono_buf, t1)
        self.dsp_event.set()
        self.fft_event.set()
        # Perf ring measures the whole body (mix + ring write + event sets),
        # not just the mix; t1 stays the block's availability timestamp.
        i = self.perf_idx
        self.perf_ring[i % self.perf_len] = time.perf_counter_ns() - t0
        self.perf_idx = i + 1
