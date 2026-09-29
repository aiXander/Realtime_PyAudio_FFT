"""Best-effort scheduling-priority elevation.

Goal: get the audio server "slightly above average" CPU priority without any
user setup (no sudo, no launchctl plist, no env vars). Every call is wrapped
to never raise — if the platform call fails or doesn't exist, we silently
fall back to default scheduling.

macOS: pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED) per-thread.
  No privileges needed. On Apple Silicon this also biases the thread onto
  P-cores under the default scheduler policy. We deliberately don't claim
  QOS_CLASS_USER_INTERACTIVE — that tier is for UI-critical work and would
  be impolite for a long-running background server.

Linux: absolute niceness -5 per-thread via setpriority(PRIO_PROCESS, 0, -5)
  (on Linux, who=0 targets the calling thread; niceness is per-thread). Must
  be absolute, not os.nice(-5): nice() is relative, and threads spawned after
  the main thread was boosted inherit its -5 and would compound to -10. Never
  lowers a thread that already runs at or below the target. Will silently
  no-op without CAP_SYS_NICE / appropriate rlimits.

Windows / other: no-op.

The PortAudio C callback thread already gets time-constraint scheduling from
CoreAudio on macOS / from PortAudio's host API on other platforms, which is
strictly higher than anything reachable from user-space here — we only need
to lift the DSP/FFT worker threads and the asyncio main thread.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys

log = logging.getLogger(__name__)

# <sys/qos.h>
_QOS_CLASS_USER_INITIATED = 0x19

# Linux absolute target niceness for boosted threads.
_LINUX_TARGET_NICE = -5

_set_qos = None
_macos_probed = False


def _macos_init() -> bool:
    global _set_qos, _macos_probed
    if _macos_probed:
        return _set_qos is not None
    _macos_probed = True
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib", use_errno=True)
        fn = libc.pthread_set_qos_class_self_np
        fn.argtypes = [ctypes.c_uint, ctypes.c_int]
        fn.restype = ctypes.c_int
        _set_qos = fn
    except (OSError, AttributeError):
        _set_qos = None
    return _set_qos is not None


def boost_current_thread(label: str = "thread") -> bool:
    """Bump the calling thread's scheduling priority. Returns True on success."""
    try:
        if sys.platform == "darwin":
            if _macos_init():
                rc = _set_qos(_QOS_CLASS_USER_INITIATED, 0)
                if rc == 0:
                    log.info("priority: %s -> QOS_CLASS_USER_INITIATED", label)
                    return True
                log.debug("priority: pthread_set_qos rc=%d for %s", rc, label)
        elif sys.platform.startswith("linux"):
            try:
                cur = os.getpriority(os.PRIO_PROCESS, 0)
                if cur <= _LINUX_TARGET_NICE:
                    return True  # already at/above target priority (inherited or preset)
                os.setpriority(os.PRIO_PROCESS, 0, _LINUX_TARGET_NICE)
                log.info("priority: %s -> nice %d", label, _LINUX_TARGET_NICE)
                return True
            except (PermissionError, OSError):
                pass
    except Exception as e:
        log.debug("priority boost failed for %s: %s", label, e)
    return False
