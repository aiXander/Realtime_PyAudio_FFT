"""FFT worker: hop-driven, log-spaced bins, dB output."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..io.osc_publisher import OscPublisher
from ..priority import boost_current_thread

log = logging.getLogger(__name__)

MAX_BACKLOG_HOPS = 2
EPS = 1e-12
# Sentinel for log bins with no rfft bin mapped (happens at the low end where
# rfft frequency resolution is coarser than the log spacing). Chosen far below
# any physically achievable dB value (rfft of EPS clamps at -240 dB) so the UI
# can distinguish "empty bin" from "very quiet real measurement". OSC sender
# replaces this with db_floor before transmitting to keep the wire contract.
EMPTY_BIN_SENTINEL = -1000.0


def build_log_bin_map(window_size: int, sr: float, n_bins: int, f_min: float):
    """Compute log-spaced bin assignment for an rfft of length window_size.

    Returns (bin_assign, bin_valid_mask, bin_idx_valid, bin_counts).
    bin_assign[k] is the log-bin index for rfft bin k, or -1 if outside [f_min, sr/2].
    bin_counts[j] is the number of rfft bins mapped to log bin j (0 for empty bins).
    """
    n_rfft = window_size // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_rfft, dtype=np.float64)
    f_max = sr / 2.0
    if f_max <= f_min:
        f_max = f_min * 2.0
    edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
    bin_assign = np.full(n_rfft, -1, dtype=np.int64)
    for k, f in enumerate(freqs):
        if f < f_min or f > f_max:
            continue
        idx = int(np.searchsorted(edges, f, side="right") - 1)
        if idx < 0:
            continue
        if idx >= n_bins:
            idx = n_bins - 1
        bin_assign[k] = idx
    bin_valid_mask = bin_assign >= 0
    bin_idx_valid = bin_assign[bin_valid_mask].astype(np.int64)
    bin_counts = np.bincount(bin_idx_valid, minlength=n_bins).astype(np.float64)
    if bin_counts.shape[0] > n_bins:
        bin_counts = bin_counts[:n_bins]
    return bin_assign, bin_valid_mask, bin_idx_valid, bin_counts


class FFTWorker(threading.Thread):
    def __init__(self, ring, fft_event: threading.Event, fft_enabled: threading.Event,
                 stop_flag: threading.Event, fft_store,
                 osc_publisher: OscPublisher | None,
                 blocksize: int, sr: float, window_size: int, hop: int,
                 n_bins: int, f_min: float, perf_ring: np.ndarray,
                 db_floor: float = -60.0, post_processor=None):
        super().__init__(name="fft-worker", daemon=True)
        self.ring = ring
        self.fft_event = fft_event
        self.fft_enabled = fft_enabled
        self.stop_flag = stop_flag
        self.fft_store = fft_store
        # Direct OSC dispatch on this thread — see DSPWorker for rationale.
        self.osc_publisher = osc_publisher
        self.blocksize = blocksize
        self.sr = float(sr)
        self.window_size = int(window_size)
        self.hop = int(hop)
        self.n_bins = int(n_bins)
        self.f_min = float(f_min)
        self.db_floor = float(db_floor)
        self.post_processor = post_processor  # FFTPostProcessor or None
        self.read_block_idx = 0
        self.fft_drops = 0
        self.perf_ring = perf_ring
        self.perf_len = perf_ring.shape[0]
        self.perf_idx = 0
        self._lock = threading.Lock()
        self._allocate()

    def _allocate(self) -> None:
        ws = self.window_size
        self.n_blocks_per_window = ws // self.blocksize
        self.hop_blocks = self.hop // self.blocksize
        if self.n_blocks_per_window < 1 or self.hop_blocks < 1:
            raise ValueError("window_size/hop must be multiples of blocksize")
        self.window_buf = np.zeros(ws, dtype=np.float32)
        self.hann = np.hanning(ws).astype(np.float32)
        # Window correction: a sine of amplitude A produces a peak rfft bin
        # magnitude of ≈A·sum(hann)/2 after windowing. We want each log bin's
        # mean-of-power to equal that band's RMS² (so a pure sine at amplitude
        # A reads as A²/2 = sine RMS², matching the time-domain RMS² used by
        # the L/M/H pipeline). That gives `power = mag² · 2 / sum(hann)²`.
        coh_gain = float(np.sum(self.hann.astype(np.float64)))
        self._win_power_corr = 2.0 / max(coh_gain * coh_gain, 1e-12)
        self.spectrum = np.zeros(ws // 2 + 1, dtype=np.complex64)
        self.mag_buf = np.zeros(ws // 2 + 1, dtype=np.float32)
        self.power_buf = np.zeros(ws // 2 + 1, dtype=np.float64)
        self.bin_assign, self.bin_valid_mask, self.bin_idx_valid, self.bin_counts = build_log_bin_map(
            ws, self.sr, self.n_bins, self.f_min
        )
        # Precompute the divisor (1/count where count>0; 0 elsewhere — empty bins
        # are overwritten with sentinel below) and the empty-bin index. The
        # cached integer index lets the hot path do a small scatter-assign
        # instead of full-array .any() + masked write.
        self._bin_count_inv = np.where(self.bin_counts > 0, 1.0 / np.maximum(self.bin_counts, 1.0), 0.0)
        self._bin_empty_mask = self.bin_counts == 0
        self._empty_bin_idx = np.flatnonzero(self._bin_empty_mask)
        self._has_empty_bins = bool(self._empty_bin_idx.size > 0)
        # Precomputed integer indices for valid rfft bins, plus a same-length
        # f64 scratch. Lets the hot path do `np.take(power_buf, idx, out=...)`
        # instead of `power_buf[bin_valid_mask]`, which allocates a fresh
        # array on every hop.
        self._valid_rfft_idx = np.flatnonzero(self.bin_valid_mask).astype(np.intp)
        self._valid_power = np.zeros(self._valid_rfft_idx.size, dtype=np.float64)
        # Replacement for the per-hop `np.bincount(...)` allocation.
        # bin_idx_valid is monotonic non-decreasing (rfft freqs and log-bin
        # edges are both monotonic, so consecutive valid rfft bins fall into
        # contiguous runs of the same log-bin index). That's the precondition
        # for np.add.reduceat — it sums consecutive ranges with no allocation.
        # We precompute:
        #   _reduceat_starts:  start position of each run within _valid_power
        #   _reduceat_targets: the log-bin index that each run scatters into
        #   _reduceat_out:     preallocated reduceat output (len = num runs)
        #   _bins_scratch:     n_bins-length scratch the result is scattered
        #                      into; replaces the bincount return array.
        # Empty log bins (no rfft bins map there) keep their stale value in
        # _bins_scratch but get zeroed by `* _bin_count_inv` (inv = 0 there)
        # and finally overwritten by EMPTY_BIN_SENTINEL in the f32 wire output.
        if self.bin_idx_valid.size > 0:
            unique_bins, start_positions = np.unique(self.bin_idx_valid, return_index=True)
            self._reduceat_starts = start_positions.astype(np.intp)
            self._reduceat_targets = unique_bins.astype(np.intp)
            self._reduceat_out = np.zeros(unique_bins.size, dtype=np.float64)
        else:
            self._reduceat_starts = np.zeros(0, dtype=np.intp)
            self._reduceat_targets = np.zeros(0, dtype=np.intp)
            self._reduceat_out = np.zeros(0, dtype=np.float64)
        self._bins_scratch = np.zeros(self.n_bins, dtype=np.float64)
        # Double-buffered f32 wire format. Each hop writes into the back buffer
        # and publishes that ref. Two-buffer alternation is safe because both
        # consumers (WS encoder via tobytes, OSC sender via tolist) drain the
        # array synchronously inside their await point and drop the ref before
        # we cycle back ~2 hops later.
        self._bins_f32_buffers = (
            np.zeros(self.n_bins, dtype=np.float32),
            np.zeros(self.n_bins, dtype=np.float32),
        )
        self._wire_idx = 0

    # ------------- live retune from asyncio thread -------------
    def reconfigure(self, n_bins: int | None = None, sr: float | None = None,
                    window_size: int | None = None, hop: int | None = None,
                    f_min: float | None = None) -> None:
        with self._lock:
            if n_bins is not None: self.n_bins = int(n_bins)
            if sr is not None: self.sr = float(sr)
            if window_size is not None: self.window_size = int(window_size)
            if hop is not None: self.hop = int(hop)
            if f_min is not None: self.f_min = float(f_min)
            self._allocate()
            self.read_block_idx = 0  # alignment changed
        # Mirror to post-processor (if any). Outside the worker lock to avoid
        # nested-lock surprises — the post-processor has its own _lock.
        if self.post_processor is not None:
            self.post_processor.reconfigure(
                n_bins=self.n_bins, sr=self.sr, f_min=self.f_min,
                hop_period_s=self.hop / self.sr,
            )

    def reset(self) -> None:
        with self._lock:
            self.read_block_idx = 0

    def resync(self) -> None:
        """Snap the read pointer to 'now' (one full window back) without
        counting the skipped backlog as drops. Called when the FFT is
        re-enabled after a disabled period — the gap is deliberate idle
        time, not an overrun, and shouldn't pollute the fft_drops counter."""
        with self._lock:
            self.read_block_idx = max(0, self.ring.write_idx - self.n_blocks_per_window)

    def run(self) -> None:
        boost_current_thread("fft-worker")
        while True:
            self.fft_event.wait(timeout=0.1)
            if self.stop_flag.is_set():
                return
            self.fft_event.clear()
            if not self.fft_enabled.is_set():
                continue

            with self._lock:
                wi = self.ring.write_idx

                # Self-heal after a producer reset (ring.reset() on device
                # hot-switch). See dsp/worker.py for the same pattern.
                if self.read_block_idx > wi:
                    self.read_block_idx = 0

                if wi - self.read_block_idx > self.n_blocks_per_window + MAX_BACKLOG_HOPS * self.hop_blocks:
                    target_block = wi - self.n_blocks_per_window
                    skipped_hops = (target_block - self.read_block_idx) // self.hop_blocks
                    if skipped_hops > 0:
                        self.fft_drops += int(skipped_hops)
                        self.read_block_idx += int(skipped_hops) * self.hop_blocks

                # Drain every hop currently available on each wake. fft_event is
                # coalesced the same way dsp_event is — without draining, the worker
                # advances at most one hop per callback and can't catch up after
                # falling behind. The post-processor's smoothers / peak followers
                # require every hop to be processed for correct state.
                while wi - self.read_block_idx >= self.n_blocks_per_window:
                    t0 = time.perf_counter_ns()
                    # The hop "becomes available" the instant the last block of the
                    # window lands; that's the latency clock's t=0 for FFT e2e.
                    last_block_idx = self.read_block_idx + self.n_blocks_per_window - 1
                    t_recv_ns = int(self.ring.block_t_ns[last_block_idx & self.ring.mask])
                    if not self.ring.try_read_window(self.read_block_idx, self.n_blocks_per_window, self.window_buf):
                        self.fft_drops += 1
                        self.read_block_idx = max(
                            self.read_block_idx + self.hop_blocks,
                            wi - self.n_blocks_per_window,
                        )
                        continue

                    np.multiply(self.window_buf, self.hann, out=self.window_buf)
                    np.fft.rfft(self.window_buf, out=self.spectrum)
                    np.abs(self.spectrum, out=self.mag_buf)
                    # Window-corrected RMS² power per rfft bin (float64 for log10).
                    self.power_buf[:] = self.mag_buf
                    np.multiply(self.power_buf, self.power_buf, out=self.power_buf)
                    self.power_buf *= self._win_power_corr
                    # Mean-of-power log-bin aggregation: sum(power) per log bin /
                    # rfft-count per log bin. Mean-of-dB underweights peaks; this
                    # is energy-conserving. Alloc-free hot path:
                    #   1. np.take valid rfft powers into preallocated scratch.
                    #   2. np.add.reduceat with precomputed run starts → per-run
                    #      sums into preallocated _reduceat_out.
                    #   3. scatter into _bins_scratch (skipping empty log bins).
                    #   4. multiply by _bin_count_inv (zeros empty bins).
                    # Replaces np.bincount, which allocated a fresh n_bins-length
                    # array every hop.
                    np.take(self.power_buf, self._valid_rfft_idx, out=self._valid_power)
                    bins = self._bins_scratch
                    if self._reduceat_starts.size > 0:
                        np.add.reduceat(
                            self._valid_power, self._reduceat_starts,
                            out=self._reduceat_out,
                        )
                        bins[self._reduceat_targets] = self._reduceat_out
                    np.multiply(bins, self._bin_count_inv, out=bins)
                    # → dB (10·log10(RMS²) ≡ 20·log10(RMS)). Floor at EPS to keep
                    # log10 finite; empty bins are overwritten with sentinel below.
                    np.maximum(bins, EPS, out=bins)
                    np.log10(bins, out=bins)
                    np.multiply(bins, 10.0, out=bins)
                    # Cast once into the back buffer of the double-buffered wire
                    # output, then patch sentinels via the cached index.
                    bins_f32 = self._bins_f32_buffers[self._wire_idx]
                    bins_f32[:] = bins
                    if self._has_empty_bins:
                        bins_f32[self._empty_bin_idx] = EMPTY_BIN_SENTINEL
                    # Run the post-processor (if present) and publish both streams.
                    # Producer-side double-buffering on both streams means the
                    # published refs stay stable across the next hop's writes; no
                    # copies needed on the publish path.
                    processed = None
                    if self.post_processor is not None:
                        processed = self.post_processor.process(bins_f32)
                    self.fft_store.publish(bins_f32, processed, t_recv_ns)
                    self._wire_idx ^= 1
                    # Direct OSC dispatch on this thread (skip if disabled
                    # via cfg.osc.send_fft / fft_enabled — the publisher
                    # checks both internally).
                    if self.osc_publisher is not None:
                        self.osc_publisher.publish_fft(
                            bins_f32, processed, t_recv_ns,
                        )
                    t1 = time.perf_counter_ns()
                    i = self.perf_idx
                    self.perf_ring[i % self.perf_len] = t1 - t0
                    self.perf_idx = i + 1
                    self.read_block_idx += self.hop_blocks
