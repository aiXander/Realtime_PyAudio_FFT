"""Rate-limited exception logging for the DSP / FFT worker loops.

Workers catch per-block / per-hop exceptions so one bad iteration can't kill
the thread silently. A persistent fault would otherwise log at block rate
(~190 Hz), so this collapses repeats: at most one traceback per `interval_s`,
with a count of the ones suppressed in between. Called only from worker
threads (never the PortAudio callback), so logging is allowed here.
"""
from __future__ import annotations

import logging
import time


class RateLimitedErrorLog:
    def __init__(self, logger: logging.Logger, label: str, interval_s: float = 1.0):
        self._log = logger
        self._label = label
        self._interval = float(interval_s)
        self._last_t = -1e18
        self._suppressed = 0
        self.total = 0

    def exception(self, exc: BaseException) -> None:
        self.total += 1
        now = time.monotonic()
        if now - self._last_t < self._interval:
            self._suppressed += 1
            return
        self._last_t = now
        extra = f" ({self._suppressed} similar suppressed)" if self._suppressed else ""
        self._suppressed = 0
        self._log.error("%s: iteration failed, continuing%s", self._label, extra,
                        exc_info=exc)
