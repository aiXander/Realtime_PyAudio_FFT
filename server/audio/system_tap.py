"""System audio capture on macOS 14.2+ via a Core Audio process tap.

The "System audio" input records whatever plays on the current default
output device, so no BlackHole / Multi-Output Device is needed and the
volume keys keep working: the tap takes each app's audio before the output
device's volume, so the analysis level doesn't follow the volume keys.

What gets built (same in `engine/server/src/system_tap.rs`):
  - a process tap on the output device (all processes, unmuted: you still
    hear everything);
  - a private aggregate device (visible to this process only) holding the
    tap, with the output device itself as clock sub-device. Without that
    clock the aggregate only ticks while something plays, so silence would
    deliver no blocks at all and look like a stalled input to the watchdog.
    The tap is NOT auto-started (`tapautostart`): an auto-started tap made
    the aggregate wait for the first sound before delivering anything.
This process then opens the aggregate as an ordinary input device.

Plain ctypes against CoreAudio / CoreFoundation / the ObjC runtime (no
pyobjc). Only called when a stream opens/closes or the watchdog ticks,
never from the audio callback.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, byref, c_bool, c_char_p, c_int32, c_long, c_uint32, c_void_p

# Device-list entry name (and persisted cfg.audio.device.name), both engines.
SYSTEM_AUDIO_NAME = "System audio"
# Our private aggregate devices carry this name prefix; device lists hide them.
TAP_DEVICE_PREFIX = "audio-server system tap"
# ... and this UID prefix (Rust hides by UID too; PortAudio only has names).
TAP_UID_PREFIX = "com.audio-server.system-tap."


def _fourcc(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


class _Addr(ctypes.Structure):
    _fields_ = [("mSelector", c_uint32), ("mScope", c_uint32), ("mElement", c_uint32)]


_SYSTEM_OBJECT = 1
_GLOBAL = _fourcc("glob")
_DEFAULT_OUTPUT = _fourcc("dOut")
_DEVICE_UID = _fourcc("uid ")
_PROCESS_LIST = _fourcc("prs#")
_PROCESS_PID = _fourcc("ppid")
_PROCESS_RUNNING_OUTPUT = _fourcc("piro")
_UTF8 = 0x08000100

_lib = None


def _libs():
    """(CoreAudio, CoreFoundation, libobjc), or None without process taps."""
    global _lib
    if _lib is not None:
        return _lib or None
    _lib = False
    if sys.platform != "darwin":
        return None
    try:
        ca = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        ca.AudioHardwareCreateProcessTap  # noqa: B018 — macOS 14.2+ only
    except (OSError, AttributeError):
        return None
    objc.objc_getClass.restype = c_void_p
    objc.objc_getClass.argtypes = [c_char_p]
    objc.sel_registerName.restype = c_void_p
    objc.sel_registerName.argtypes = [c_char_p]
    objc.objc_autoreleasePoolPush.restype = c_void_p
    objc.objc_autoreleasePoolPop.argtypes = [c_void_p]
    if not objc.objc_getClass(b"CATapDescription"):
        return None
    ca.AudioObjectGetPropertyDataSize.restype = c_int32
    ca.AudioObjectGetPropertyDataSize.argtypes = [c_uint32, POINTER(_Addr), c_uint32, c_void_p, POINTER(c_uint32)]
    ca.AudioObjectGetPropertyData.restype = c_int32
    ca.AudioObjectGetPropertyData.argtypes = [c_uint32, POINTER(_Addr), c_uint32, c_void_p, POINTER(c_uint32), c_void_p]
    for fn in ("AudioHardwareCreateProcessTap", "AudioHardwareCreateAggregateDevice"):
        getattr(ca, fn).restype = c_int32
        getattr(ca, fn).argtypes = [c_void_p, POINTER(c_uint32)]
    for fn in ("AudioHardwareDestroyProcessTap", "AudioHardwareDestroyAggregateDevice"):
        getattr(ca, fn).restype = c_int32
        getattr(ca, fn).argtypes = [c_uint32]
    cf.CFStringCreateWithCString.restype = c_void_p
    cf.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
    cf.CFStringGetCString.restype = c_bool
    cf.CFStringGetCString.argtypes = [c_void_p, c_char_p, c_long, c_uint32]
    cf.CFDictionaryCreateMutable.restype = c_void_p
    cf.CFDictionaryCreateMutable.argtypes = [c_void_p, c_long, c_void_p, c_void_p]
    cf.CFDictionarySetValue.restype = None
    cf.CFDictionarySetValue.argtypes = [c_void_p, c_void_p, c_void_p]
    cf.CFArrayCreate.restype = c_void_p
    cf.CFArrayCreate.argtypes = [c_void_p, c_void_p, c_long, c_void_p]
    cf.CFRelease.restype = None
    cf.CFRelease.argtypes = [c_void_p]
    _lib = (ca, cf, objc)
    return _lib


def available() -> bool:
    """True on macOS 14.2+ (Core Audio process taps)."""
    return _libs() is not None


def _msg(restype, *argtypes):
    """A typed `objc_msgSend` (arm64 needs the exact non-variadic prototype)."""
    addr = ctypes.cast(_libs()[2].objc_msgSend, c_void_p).value
    return ctypes.CFUNCTYPE(restype, c_void_p, c_void_p, *argtypes)(addr)


def _sel(name: str) -> int:
    return _libs()[2].sel_registerName(name.encode())


def _cfstr(s: str) -> int:
    return _libs()[1].CFStringCreateWithCString(None, s.encode(), _UTF8)


def _prop(obj: int, selector: int, ctype):
    """A fixed-size property value, or None."""
    val = ctype()
    size = c_uint32(ctypes.sizeof(ctype))
    st = _libs()[0].AudioObjectGetPropertyData(obj, byref(_Addr(selector, _GLOBAL, 0)), 0, None, byref(size), byref(val))
    return val.value if st == 0 else None


def _prop_str(obj: int, selector: int) -> str | None:
    ref = _prop(obj, selector, c_void_p)
    if not ref:
        return None
    cf = _libs()[1]
    try:
        buf = ctypes.create_string_buffer(1024)
        return buf.value.decode() if cf.CFStringGetCString(ref, buf, len(buf), _UTF8) else None
    finally:
        cf.CFRelease(ref)


def default_output_uid() -> str | None:
    """UID of the current default output device (what a new tap records)."""
    if not available():
        return None
    dev = _prop(_SYSTEM_OBJECT, _DEFAULT_OUTPUT, c_uint32)
    return _prop_str(dev, _DEVICE_UID) if dev else None


def output_playing_elsewhere() -> bool:
    """Whether another process is playing audio right now: the device probe's
    answer for "System audio" (a real signal probe would need a second tap,
    and in Python a PortAudio re-init that cuts the live stream)."""
    if not available():
        return False
    ca = _libs()[0]
    addr = _Addr(_PROCESS_LIST, _GLOBAL, 0)
    size = c_uint32(0)
    if ca.AudioObjectGetPropertyDataSize(_SYSTEM_OBJECT, byref(addr), 0, None, byref(size)) != 0:
        return False
    procs = (c_uint32 * (size.value // 4))()
    if ca.AudioObjectGetPropertyData(_SYSTEM_OBJECT, byref(addr), 0, None, byref(size), procs) != 0:
        return False
    me = os.getpid()
    return any(_prop(p, _PROCESS_PID, c_int32) != me and _prop(p, _PROCESS_RUNNING_OUTPUT, c_uint32)
               for p in procs)


class SystemTap:
    """A process tap on output device `output_uid` + the private aggregate
    device wrapping it (named `self.name`; open it as an input from this
    process). `close()` destroys both."""

    _instances = 0

    def __init__(self, output_uid: str):
        libs = _libs()
        if libs is None:
            raise RuntimeError("system audio capture needs macOS 14.2+")
        ca, cf, objc = libs
        self.output_uid = output_uid
        SystemTap._instances += 1
        tag = f"{os.getpid()}.{SystemTap._instances}"
        self.name = f"{TAP_DEVICE_PREFIX} {tag}"
        self._tap_id = 0
        self._agg_id = 0
        owned: list[int] = []  # +1 references to release at the end

        def own(ref: int) -> int:
            owned.append(ref)
            return ref

        pool = objc.objc_autoreleasePoolPush()
        try:
            arr_cb = ctypes.cast(cf.kCFTypeArrayCallBacks, c_void_p)
            key_cb = ctypes.cast(cf.kCFTypeDictionaryKeyCallBacks, c_void_p)
            val_cb = ctypes.cast(cf.kCFTypeDictionaryValueCallBacks, c_void_p)
            true = c_void_p.in_dll(cf, "kCFBooleanTrue").value

            def cfdict(**items) -> int:
                d = own(cf.CFDictionaryCreateMutable(None, len(items), key_cb, val_cb))
                for k, v in items.items():
                    cf.CFDictionarySetValue(d, own(_cfstr(k)), v)
                return d

            def cfarray(*vals) -> int:
                return own(cf.CFArrayCreate(None, (c_void_p * len(vals))(*vals), len(vals), arr_cb))

            out_uid = own(_cfstr(output_uid))
            # [[CATapDescription alloc] initWithProcesses:@[] andDeviceUID:uid withStream:0];
            # an empty process list + exclusive = every process.
            desc = _msg(c_void_p)(objc.objc_getClass(b"CATapDescription"), _sel("alloc"))
            desc = _msg(c_void_p, c_void_p, c_void_p, c_long)(
                desc, _sel("initWithProcesses:andDeviceUID:withStream:"), cfarray(), out_uid, 0)
            if not desc:
                raise RuntimeError("CATapDescription init failed")
            own(desc)
            _msg(None, c_long)(desc, _sel("setMuteBehavior:"), 0)  # CATapUnmuted
            _msg(None, c_bool)(desc, _sel("setPrivate:"), True)
            _msg(None, c_bool)(desc, _sel("setExclusive:"), True)
            _msg(None, c_void_p)(desc, _sel("setName:"), own(_cfstr(self.name)))
            tap_id = c_uint32(0)
            st = ca.AudioHardwareCreateProcessTap(desc, byref(tap_id))
            if st != 0:
                raise RuntimeError(f"AudioHardwareCreateProcessTap failed (OSStatus {st})")
            self._tap_id = tap_id.value
            tap_uid = _msg(c_void_p)(_msg(c_void_p)(desc, _sel("UUID")), _sel("UUIDString"))

            props = cfdict(
                name=own(_cfstr(self.name)),
                uid=own(_cfstr(f"{TAP_UID_PREFIX}{tag}")),
                master=out_uid,
                subdevices=cfarray(cfdict(uid=out_uid)),
                taps=cfarray(cfdict(uid=tap_uid, drift=true)),
                private=true,
            )
            agg_id = c_uint32(0)
            st = ca.AudioHardwareCreateAggregateDevice(props, byref(agg_id))
            if st != 0:
                raise RuntimeError(f"AudioHardwareCreateAggregateDevice failed (OSStatus {st})")
            self._agg_id = agg_id.value
        except Exception:
            self.close()
            raise
        finally:
            for ref in owned:
                cf.CFRelease(ref)
            objc.objc_autoreleasePoolPop(pool)

    def close(self) -> None:
        ca = _libs()[0]
        if self._agg_id:
            ca.AudioHardwareDestroyAggregateDevice(self._agg_id)
            self._agg_id = 0
        if self._tap_id:
            ca.AudioHardwareDestroyProcessTap(self._tap_id)
            self._tap_id = 0
