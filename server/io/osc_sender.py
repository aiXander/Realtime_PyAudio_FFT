"""OSC UDP wire layer.

Pure synchronous packet builder + UDP sendto. No event loop, no threads.
Each address has a pre-encoded packet template; hot-path sends mutate the
float payload in place and call `sendto` — zero allocation per packet.

Thread-safety:
  * Each `send_*` method mutates its own packet bytearray. As long as a
    given method is only called from one thread, no locking is needed.
    Current usage: LMH/onset/BPM from the DSP worker; FFT from the FFT
    worker. The two methods touch disjoint buffers.
  * The shared UDP socket is thread-safe at the kernel level (`sendto`
    releases the GIL and is atomic at the OS layer for datagrams).

`send_meta` is the only path still using pythonosc (variable-shape payload,
called rarely from the asyncio loop).
"""
from __future__ import annotations

import logging
import socket
import struct

import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from ..config import OscDest

log = logging.getLogger(__name__)


# Pre-encoded OSC packet for "/audio/lmh ,fff <f><f><f>".
# Address "/audio/lmh" padded to 12 bytes, type tag ",fff" padded to 8 bytes,
# then 3 big-endian float32 args = 32 bytes total. Floats start at offset 20.
_LMH_TEMPLATE = b"/audio/lmh\x00\x00,fff\x00\x00\x00\x00" + b"\x00" * 12
_LMH_FLOAT_OFFSET = 20
_LMH_PACK = struct.Struct("!fff").pack_into

# Pre-encoded "/audio/onset/<band> ,i <int=1>" packets. Each onset is sent
# ONLY on onset blocks (no zero pulses), so the value is always 1 and the
# whole packet is constant per band — no per-send packing.
#
# OSC string padding: pad to a multiple of 4 bytes with at least one NUL.
#   - "/audio/onset/low"  = 16 chars  → 4 NULs → 20 bytes address
#   - "/audio/onset/mid"  = 16 chars  → 4 NULs → 20 bytes address
#   - "/audio/onset/high" = 17 chars  → 3 NULs → 20 bytes address
# Type tag ",i" = 2 chars → 2 NULs → 4 bytes.
# int32 BE = 4 bytes (value 1 = "\x00\x00\x00\x01"). Total: 28 bytes per packet.
_ONSET_PACKETS = (
    b"/audio/onset/low\x00\x00\x00\x00,i\x00\x00\x00\x00\x00\x01",   # 0 = low
    b"/audio/onset/mid\x00\x00\x00\x00,i\x00\x00\x00\x00\x00\x01",   # 1 = mid
    b"/audio/onset/high\x00\x00\x00,i\x00\x00\x00\x00\x00\x01",      # 2 = high
)

# Pre-encoded "/audio/bpm ,f <f>". Float arg is mutated in place.
# "/audio/bpm" (10 bytes) + NUL padding to 12 (2 NULs) = 12.
# ",f" (2 bytes) + NUL padding to 4 (2 NULs) = 4.
# float32 BE = 4 bytes. Float starts at offset 16. Total: 20.
_BPM_TEMPLATE = b"/audio/bpm\x00\x00,f\x00\x00\x00\x00\x00\x00"
_BPM_FLOAT_OFFSET = 16
_BPM_PACK = struct.Struct("!f").pack_into


def _build_fft_packet(n_bins: int) -> tuple[bytearray, np.ndarray, int]:
    """Build an "/audio/fft ,ff...f <f32 * n_bins>" packet template.

    Returns (buf, payload_view, payload_offset) where `payload_view` is a
    big-endian f32 numpy view writable into the packet's payload region.
    Writing native-endian floats into this view auto-byteswaps to BE on copy.
    """
    addr = b"/audio/fft"
    # OSC: pad to next multiple of 4 with at least one NUL.
    addr_padded_len = ((len(addr) // 4) + 1) * 4
    addr_field = addr.ljust(addr_padded_len, b"\x00")
    tag = b"," + b"f" * n_bins
    tag_padded_len = ((len(tag) // 4) + 1) * 4
    tag_field = tag.ljust(tag_padded_len, b"\x00")
    payload_offset = len(addr_field) + len(tag_field)
    buf = bytearray(payload_offset + n_bins * 4)
    buf[:payload_offset] = addr_field + tag_field
    # writable view because bytearray exposes a mutable buffer.
    payload_view = np.frombuffer(buf, dtype=">f4", count=n_bins, offset=payload_offset)
    return buf, payload_view, payload_offset


class OscSender:
    def __init__(self, destinations: list[OscDest]):
        self._dests = list(destinations)
        self._clients = [SimpleUDPClient(d.host, d.port) for d in self._dests]
        # Per-block hot path: bypass pythonosc (which builds a fresh
        # OscMessage + bytes object per call) by mutating a single
        # preallocated packet buffer and sending it via raw UDP. Zero
        # allocation per send.
        self._lmh_buf = bytearray(_LMH_TEMPLATE)
        self._bpm_buf = bytearray(_BPM_TEMPLATE)
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.setblocking(False)
        self._addrs = [(d.host, int(d.port)) for d in self._dests]
        # FFT packet template — lazy-built on first send and rebuilt only
        # when n_bins changes (rare, via UI). The native-endian scratch
        # holds the clamped / gain-multiplied frame before the byteswap
        # copy into the BE payload view.
        self._fft_n_bins: int = 0
        self._fft_buf: bytearray | None = None
        self._fft_payload_view: np.ndarray | None = None
        self._fft_scratch: np.ndarray | None = None

    def _ensure_fft_packet(self, n_bins: int) -> tuple[bytearray, np.ndarray, np.ndarray]:
        if self._fft_n_bins != n_bins or self._fft_buf is None:
            buf, view, _ = _build_fft_packet(n_bins)
            self._fft_buf = buf
            self._fft_payload_view = view
            self._fft_scratch = np.empty(n_bins, dtype=np.float32)
            self._fft_n_bins = n_bins
        return self._fft_buf, self._fft_payload_view, self._fft_scratch

    def send_meta(self, sr: int, blocksize: int, n_fft_bins: int, bands: dict) -> None:
        # Payload: [sr, blocksize, n_fft_bins, low_lo, low_hi, mid_lo, mid_hi, high_lo, high_hi]
        msg = [
            int(sr), int(blocksize), int(n_fft_bins),
            float(bands["low"][0]),  float(bands["low"][1]),
            float(bands["mid"][0]),  float(bands["mid"][1]),
            float(bands["high"][0]), float(bands["high"][1]),
        ]
        for c in self._clients:
            try:
                c.send_message("/audio/meta", msg)
            except Exception as e:
                log.debug("osc meta send failed: %s", e)

    def send_lmh(self, low: float, mid: float, high: float) -> None:
        # Zero-alloc hot path: pack into the preallocated packet buffer
        # and sendto each destination. struct.pack_into and socket.sendto
        # do not allocate Python objects.
        _LMH_PACK(self._lmh_buf, _LMH_FLOAT_OFFSET, low, mid, high)
        buf = self._lmh_buf
        sock = self._udp_sock
        for addr in self._addrs:
            try:
                sock.sendto(buf, addr)
            except Exception as e:
                log.debug("osc lmh send failed: %s", e)

    def send_onset(self, band_idx: int) -> None:
        """Sends `/audio/onset/{low,mid,high} 1`. Constant packet — no
        packing needed. Called only on onset blocks; absence of a packet on
        an address is silence on that band (no `0`s sent), giving downstream
        consumers a clean rising-edge trigger semantic."""
        pkt = _ONSET_PACKETS[band_idx]
        sock = self._udp_sock
        for addr in self._addrs:
            try:
                sock.sendto(pkt, addr)
            except Exception as e:
                log.debug("osc onset send failed: %s", e)

    def send_bpm(self, bpm: float) -> None:
        """Sends `/audio/bpm <f>`. Per-block; cheap (4-byte float pack +
        20-byte UDP packet). Downstream caches whatever was last received."""
        _BPM_PACK(self._bpm_buf, _BPM_FLOAT_OFFSET, float(bpm))
        buf = self._bpm_buf
        sock = self._udp_sock
        for addr in self._addrs:
            try:
                sock.sendto(buf, addr)
            except Exception as e:
                log.debug("osc bpm send failed: %s", e)

    def send_silence(self) -> None:
        """Send `/audio/lmh 0 0 0` + `/audio/bpm 0`. Safe from any thread:
        the unmutated templates (all-zero floats) are immutable bytes, so
        this never touches the buffers the DSP worker packs into. Sent when
        audio input stalls, so downstream consumers settle to silence
        instead of holding the last level."""
        sock = self._udp_sock
        for pkt in (_LMH_TEMPLATE, _BPM_TEMPLATE):
            for addr in self._addrs:
                try:
                    sock.sendto(pkt, addr)
                except Exception as e:
                    log.debug("osc silence send failed: %s", e)

    def send_fft(self, bins: np.ndarray, db_floor: float) -> None:
        """Send raw-dB FFT frame. Clamps sentinels (-1000) and any sub-floor
        values to `db_floor` in a single fused C pass into a native-endian
        scratch, then byteswap-copies into the BE payload view of the
        preallocated packet buffer. Zero allocation per call."""
        n = int(bins.shape[0])
        buf, payload_view, scratch = self._ensure_fft_packet(n)
        np.maximum(bins, np.float32(db_floor), out=scratch)
        np.copyto(payload_view, scratch)  # native f32 → BE f32 (byteswap)
        sock = self._udp_sock
        for addr in self._addrs:
            try:
                sock.sendto(buf, addr)
            except Exception as e:
                log.debug("osc fft send failed: %s", e)

    def send_fft_processed(self, bins: np.ndarray, gain: float = 1.0) -> None:
        """Send post-processed FFT frame, optionally scaled by master gain.
        Master gain >1 may push bins above 1.0 — by design; downstream must
        accept it. Zero allocation per call (gain==1.0 byteswaps directly
        from the source array; gain!=1.0 fuses the multiply into scratch)."""
        n = int(bins.shape[0])
        buf, payload_view, scratch = self._ensure_fft_packet(n)
        if gain != 1.0:
            np.multiply(bins, np.float32(gain), out=scratch)
            np.copyto(payload_view, scratch)
        else:
            np.copyto(payload_view, bins)
        sock = self._udp_sock
        for addr in self._addrs:
            try:
                sock.sendto(buf, addr)
            except Exception as e:
                log.debug("osc fft send failed: %s", e)
