"""Worker-thread-callable OSC publisher.

Composes the low-level `OscSender` (wire format + sendto) with live config
getters and per-block latency accounting. Workers call `publish_lmh` /
`publish_fft` directly after they publish to their respective stores — no
asyncio hop, no event wake-up, no scheduling jitter from the loop.

Why this exists (vs. the old `osc_sender_task`):
  * `call_soon_threadsafe(sender_event.set)` + loop scheduling cost ~30–80 µs
    typical and ~500 µs–1 ms when the loop is busy with WS broadcasts or
    HTTP requests. Direct dispatch from the worker thread eliminates both.
  * The send path stays a synchronous mutate-buffer + sendto call (~10–30 µs
    to localhost), so the DSP/FFT worker thread is not meaningfully slowed.
  * The separation between "wire layer" (OscSender) and "what gets sent
    under which config" (this class) makes it easy to later swap one for
    e.g. a dedicated sender thread with its own queue.

Thread-safety:
  * `publish_lmh` is intended to be called from a single thread (the DSP
    worker); `publish_fft` from a single thread (the FFT worker). Inside
    OscSender the two methods touch disjoint packet buffers. The shared
    UDP socket is thread-safe at the kernel level.
  * Config getters are plain attribute reads on the App/Config objects;
    those reads are atomic under the GIL. Stale values for one block are
    acceptable (same semantic as the old async task).
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np

from .osc_sender import OscSender

log = logging.getLogger(__name__)


class OscPublisher:
    def __init__(
        self,
        sender: OscSender,
        *,
        get_master_gain: Callable[[], float],
        get_db_floor: Callable[[], float],
        get_send_fft: Callable[[], bool],
        get_send_raw_db: Callable[[], bool],
        get_fft_enabled: Callable[[], bool],
        perf_lmh_e2e: Optional[np.ndarray] = None,
        perf_fft_e2e: Optional[np.ndarray] = None,
        perf_idx_state: Optional[dict] = None,
    ):
        self._sender = sender
        self._get_master_gain = get_master_gain
        self._get_db_floor = get_db_floor
        self._get_send_fft = get_send_fft
        self._get_send_raw_db = get_send_raw_db
        self._get_fft_enabled = get_fft_enabled
        self._perf_lmh = perf_lmh_e2e
        self._perf_fft = perf_fft_e2e
        self._perf_idx = perf_idx_state if perf_idx_state is not None else {"lmh": 0, "fft": 0}
        self._lmh_len = perf_lmh_e2e.shape[0] if perf_lmh_e2e is not None else 0
        self._fft_len = perf_fft_e2e.shape[0] if perf_fft_e2e is not None else 0

    # ---------------- per-block L/M/H + onset + BPM ----------------
    def publish_lmh(self, scaled: np.ndarray, onsets: np.ndarray,
                    bpm: float, t_recv_ns: int = 0) -> None:
        """Called by the DSP worker right after `features_store.publish`.

        `scaled` is the length-3 post-autoscale f64 array; `onsets` the
        length-3 int8 per-band onset pulse (1 on the firing block, 0 else);
        `bpm` the smoothed tempo (0.0 when not locked).
        """
        g = self._get_master_gain()
        s = self._sender
        try:
            s.send_lmh(scaled[0] * g, scaled[1] * g, scaled[2] * g)
            s.send_bpm(bpm)
            if onsets[0]:
                s.send_onset(0)
            if onsets[1]:
                s.send_onset(1)
            if onsets[2]:
                s.send_onset(2)
        except Exception as e:
            log.debug("publish_lmh failed: %s", e)
            return
        self._record_latency("lmh", t_recv_ns, self._perf_lmh, self._lmh_len)

    # ---------------- per-hop FFT ----------------
    def publish_fft(self, raw_db: np.ndarray,
                    processed: Optional[np.ndarray],
                    t_recv_ns: int = 0) -> None:
        """Called by the FFT worker right after `fft_store.publish`.

        Picks raw-dB vs. processed based on `cfg.fft.send_raw_db` (read via
        the getter, so the live UI toggle is honored without restart).
        """
        if not self._get_send_fft() or not self._get_fft_enabled():
            return
        s = self._sender
        try:
            if self._get_send_raw_db():
                # Master gain does not apply to the raw-dB monitor stream.
                s.send_fft(raw_db, self._get_db_floor())
            elif processed is not None:
                s.send_fft_processed(processed, self._get_master_gain())
            else:
                return
        except Exception as e:
            log.debug("publish_fft failed: %s", e)
            return
        self._record_latency("fft", t_recv_ns, self._perf_fft, self._fft_len)

    # ---------------- perf bookkeeping ----------------
    def _record_latency(self, key: str, t_recv_ns: int,
                        ring: Optional[np.ndarray], ring_len: int) -> None:
        if ring is None or ring_len == 0 or not t_recv_ns:
            return
        latency = time.perf_counter_ns() - t_recv_ns
        if latency <= 0:
            return
        i = self._perf_idx[key]
        ring[i % ring_len] = latency
        self._perf_idx[key] = i + 1
