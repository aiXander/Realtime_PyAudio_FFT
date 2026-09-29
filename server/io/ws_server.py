"""WebSocket server: broadcaster + per-client outbound + inbound dispatch."""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from collections import deque

import numpy as np
import websockets
from websockets.asyncio.server import serve

log = logging.getLogger(__name__)


# Stream frames (snapshot JSON + binary FFT) per client: latest-N, drop
# oldest. A slow client only ever loses stale visualisation frames.
STREAM_QUEUE_MAX = 4
# Control messages (meta, devices, presets, error, dispatcher replies) are
# never dropped. They are low-rate (meta is coalesced to <= 1 per broadcast
# tick), so a client that falls this far behind is dead or stalled; it is
# disconnected rather than silently losing state updates.
CONTROL_QUEUE_MAX = 1024
# Min seconds between repeated exception logs from the background loops.
_LOOP_ERR_LOG_INTERVAL_S = 5.0


class _Client:
    """Per-client outbound state, drained by one sender task.

    Three lanes, sent in priority order:
      1. `control` — FIFO, never dropped (meta / devices / presets / error /
         dispatcher replies).
      2. `status`  — latest-only slot for the 5 Hz server_status (a stale
         status is worthless, so it's overwritten rather than queued).
      3. `stream`  — deque(maxlen=STREAM_QUEUE_MAX): appending to a full
         deque drops the oldest frame (drop-oldest, never block).
    """
    __slots__ = ("ws", "control", "status", "stream", "wake", "sender_task", "closing")

    def __init__(self, ws):
        self.ws = ws
        self.control: deque = deque()
        self.status = None
        self.stream: deque = deque(maxlen=STREAM_QUEUE_MAX)
        self.wake = asyncio.Event()
        self.sender_task = None
        self.closing = False

    def put_control(self, msg) -> None:
        if self.closing:
            return
        if len(self.control) >= CONTROL_QUEUE_MAX:
            self.closing = True
            log.warning("ws client control queue overflow (%d msgs); disconnecting",
                        len(self.control))
            asyncio.ensure_future(self.ws.close(code=1008, reason="client too slow"))
            return
        self.control.append(msg)
        self.wake.set()

    def put_status(self, msg) -> None:
        self.status = msg
        self.wake.set()

    def put_stream(self, msg) -> None:
        self.stream.append(msg)
        self.wake.set()

    def next_msg(self):
        """Pop the highest-priority pending message, or None."""
        if self.control:
            return self.control.popleft()
        if self.status is not None:
            m, self.status = self.status, None
            return m
        if self.stream:
            return self.stream.popleft()
        return None


class WSServer:
    """Owns the websockets server, the broadcast loop, and inbound dispatch."""

    def __init__(self, host: str, port: int, snapshot_hz: int,
                 features_store, fft_store, get_meta, get_devices, get_presets,
                 get_server_status, get_fft_enabled, get_fft_send_raw_db,
                 get_master_gain,
                 dispatcher_handle,
                 perf_ring: np.ndarray | None = None):
        self.host = host
        self.port = port
        self._snapshot_hz = max(15, min(240, int(snapshot_hz)))
        self.features_store = features_store
        self.fft_store = fft_store
        self.get_meta = get_meta
        self.get_devices = get_devices
        self.get_presets = get_presets
        self.get_server_status = get_server_status
        self.get_fft_enabled = get_fft_enabled
        self.get_fft_send_raw_db = get_fft_send_raw_db
        self.get_master_gain = get_master_gain
        self.dispatcher_handle = dispatcher_handle
        self.clients: set[_Client] = set()
        self._server = None
        self._broadcast_task: asyncio.Task | None = None
        self._status_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._t0 = time.monotonic()
        self.perf_ring: np.ndarray | None = perf_ring
        self._perf_idx = 0
        # Preallocated f32 scratch for the gain-multiplied FFT frame on the
        # broadcast hot path. Resized lazily if n_bins changes.
        self._fft_scratch: np.ndarray | None = None
        # Coalesced meta broadcast: dispatcher handlers call mark_meta_dirty()
        # instead of broadcasting the full (~1.5 KB) meta per slider-drag
        # message; the broadcast loop emits at most one meta per tick.
        self._meta_dirty = False
        self._last_err_log: dict[str, float] = {}

    @property
    def snapshot_hz(self) -> int:
        return self._snapshot_hz

    def set_snapshot_hz(self, hz: int) -> None:
        self._snapshot_hz = max(15, min(240, int(hz)))

    def mark_meta_dirty(self) -> None:
        """Request a meta broadcast to all clients on the next broadcast tick."""
        self._meta_dirty = True

    def _log_loop_error(self, where: str) -> None:
        now = time.monotonic()
        if now - self._last_err_log.get(where, -1e9) >= _LOOP_ERR_LOG_INTERVAL_S:
            self._last_err_log[where] = now
            log.exception("%s iteration failed (continuing; repeats suppressed for %.0fs)",
                          where, _LOOP_ERR_LOG_INTERVAL_S)

    def _server_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    async def start(self) -> None:
        self._server = await serve(self._handle_client, self.host, self.port)
        log.info("ws server listening on ws://%s:%d", self.host, self.port)
        self._broadcast_task = asyncio.create_task(self._broadcast_loop(), name="ws-broadcast")
        self._status_task = asyncio.create_task(self._status_loop(), name="ws-status")

    async def stop(self) -> None:
        self._stop.set()
        for t in (self._broadcast_task, self._status_task):
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
        for c in list(self.clients):
            try:
                await c.ws.close()
            except Exception:
                pass

    # ---------------- per-client lifecycle ----------------
    async def _handle_client(self, ws):
        client = _Client(ws)
        self.clients.add(client)
        log.info("ws client connected (n=%d)", len(self.clients))
        client.sender_task = asyncio.create_task(self._client_sender(client))
        try:
            # Greet with meta/devices/presets/server_status
            await self._greet(client)
            async for raw in ws:
                await self._on_message(client, raw)
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            log.warning("ws client error: %s", e)
        finally:
            self.clients.discard(client)
            if client.sender_task:
                client.sender_task.cancel()
            log.info("ws client disconnected (n=%d)", len(self.clients))

    async def _client_sender(self, client: _Client):
        try:
            while True:
                msg = client.next_msg()
                if msg is None:
                    # No await between the emptiness check and clear(), so a
                    # put_* can't slip in between (single-threaded loop).
                    client.wake.clear()
                    await client.wake.wait()
                    continue
                try:
                    await client.ws.send(msg)
                except Exception:
                    return
        except asyncio.CancelledError:
            return

    async def _greet(self, client: _Client):
        client.put_control(encode_meta(self.get_meta()))
        client.put_control(json.dumps({"type": "devices", "items": self.get_devices()}))
        client.put_control(json.dumps({"type": "presets", "items": self.get_presets()}))
        client.put_status(json.dumps(self._make_server_status()))

    async def _on_message(self, client: _Client, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            await self._reply(client, {"type": "error", "reason": "invalid JSON"})
            return
        if not isinstance(msg, dict) or "type" not in msg:
            await self._reply(client, {"type": "error", "reason": "missing 'type'"})
            return
        try:
            replies = await self.dispatcher_handle(msg)
        except Exception as e:
            await self._reply(client, {"type": "error", "reason": str(e)})
            return
        # Dispatcher returns ('targeted_replies', 'broadcasts'): see dispatcher.
        if not replies:
            return
        targeted, broadcasts = replies
        for r in targeted:
            await self._reply(client, r)
        for b in broadcasts:
            await self._broadcast(b)

    async def _reply(self, client: _Client, msg: dict):
        client.put_control(json.dumps(msg))

    async def _broadcast(self, msg):
        """Broadcast a control message (never dropped) to every client."""
        text = msg if isinstance(msg, (bytes, str)) else json.dumps(msg)
        # Single-threaded asyncio: queue ops don't await, so iterating the
        # set is safe (no concurrent add/remove from _handle_client).
        for c in self.clients:
            c.put_control(text)

    # ---------------- broadcast loop (60 Hz default) ----------------
    async def _broadcast_loop(self):
        loop = asyncio.get_running_loop()
        # Deadline scheduling: sleeping a fixed period after each tick's work
        # makes the real rate drift low by the work time (+ loop jitter).
        # Instead aim at absolute deadlines; if we fall behind by more than a
        # full period (loop stalled), resync to now rather than bursting
        # catch-up ticks that would only send duplicate frames.
        next_t = loop.time()
        state = {"feat_seq": 0, "fft_seq": 0, "onset_counts": (0, 0, 0)}
        while not self._stop.is_set():
            period = 1.0 / max(self._snapshot_hz, 1)
            next_t += period
            delay = next_t - loop.time()
            if delay < -period:
                next_t = loop.time()
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))
            if self._stop.is_set():
                return
            try:
                self._broadcast_tick(state)
            except Exception:
                self._log_loop_error("ws broadcast loop")

    def _broadcast_tick(self, state: dict) -> None:
        t_start = time.perf_counter_ns()
        clients = self.clients
        # Coalesced meta: at most one per tick, however many control
        # messages marked it dirty since the last tick.
        if self._meta_dirty:
            self._meta_dirty = False
            if clients:
                text = encode_meta(self.get_meta())
                for c in clients:
                    c.put_control(text)
        if not clients:
            return
        # L/M/H snapshot
        seq, raw, scaled, _t, _onsets_block, onset_counts, bpm = self.features_store.read()
        if seq != state["feat_seq"]:
            state["feat_seq"] = seq
            g = self.get_master_gain()
            # Onset detection runs at audio-block rate (~187 Hz at 48k/256)
            # while WS broadcasts at snapshot_hz (default 60 Hz). Per-band
            # onset pulses can fall between two snapshots and be missed.
            # Compare the monotonic per-band counters instead so each onset
            # triggers exactly one snapshot with that band's flag set.
            last = state["onset_counts"]
            o_lo = 1 if onset_counts[0] != last[0] else 0
            o_md = 1 if onset_counts[1] != last[1] else 0
            o_hi = 1 if onset_counts[2] != last[2] else 0
            state["onset_counts"] = onset_counts
            msg = {
                "type": "snapshot",
                "seq": seq,
                "low": scaled[0] * g,
                "mid": scaled[1] * g,
                "high": scaled[2] * g,
                "low_raw": raw[0],
                "mid_raw": raw[1],
                "high_raw": raw[2],
                "low_onset": o_lo,
                "mid_onset": o_md,
                "high_onset": o_hi,
                "bpm": bpm,
                "t": self._server_ms(),
            }
            text = json.dumps(msg)
            for c in clients:
                c.put_stream(text)
        # FFT — pick raw vs processed stream based on the user-facing flag
        # so what the UI renders is byte-identical to what OSC sends.
        if self.get_fft_enabled():
            kind = "raw_db" if self.get_fft_send_raw_db() else "processed"
            fseq, frame, _ft = self.fft_store.read(kind)
            if fseq != state["fft_seq"] and frame is not None:
                state["fft_seq"] = fseq
                # Master gain only multiplies the processed feature
                # output; the raw-dB monitor stream is sent untouched.
                gain = 1.0 if kind == "raw_db" else self.get_master_gain()
                # Encoded synchronously (tobytes copies) so the
                # double-buffered frame ref is dropped before any await.
                payload = self._encode_fft_binary(frame, gain)
                for c in clients:
                    c.put_stream(payload)
        if self.perf_ring is not None:
            self._record_perf(time.perf_counter_ns() - t_start)

    def _encode_fft_binary(self, frame: np.ndarray, gain: float) -> bytes:
        """Wire layout: [type=1:u8][reserved:u8][n_bins:u16][float32 * n_bins] LE.

        Avoids the per-tick `arr * gain` allocation by writing into a
        preallocated f32 scratch when gain != 1.0. The final bytes object
        is unavoidable (we hand it to per-client queues; downstream send
        coroutines need a stable bytes-like).
        """
        n = int(frame.shape[0])
        if gain != 1.0:
            scratch = self._fft_scratch
            if scratch is None or scratch.size != n:
                scratch = np.empty(n, dtype=np.float32)
                self._fft_scratch = scratch
            np.multiply(frame, np.float32(gain), out=scratch)
            buf = scratch
        else:
            # frame is already float32; tobytes() is the only alloc.
            buf = frame
        header = struct.pack("<BBH", 1, 0, n)
        return header + buf.tobytes(order="C")

    def _record_perf(self, dt_ns: int):
        ring = self.perf_ring
        if ring is None:
            return
        i = self._perf_idx
        ring[i % ring.shape[0]] = dt_ns
        self._perf_idx = i + 1

    def reset_perf(self) -> None:
        self._perf_idx = 0

    @property
    def perf_idx(self) -> int:
        return self._perf_idx

    # ---------------- status loop (5 Hz) ----------------
    async def _status_loop(self):
        while not self._stop.is_set():
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                return
            if self._stop.is_set():
                return
            if not self.clients:
                continue
            try:
                text = json.dumps(self._make_server_status())
                for c in self.clients:
                    c.put_status(text)
            except Exception:
                self._log_loop_error("ws status loop")

    def _make_server_status(self) -> dict:
        return self.get_server_status()


def encode_meta(meta_dict: dict) -> str:
    return json.dumps({"type": "meta", **meta_dict})
