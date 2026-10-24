//! State shared between the audio callback, the two workers and the control
//! loop: the slot ring, live flags the workers read every block/hop, drop
//! counters and the perf rings. Everything here is atomic; nothing locks.

use crate::osc::OscSender;
use crate::ring::SlotRing;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Monotonic nanoseconds (CLOCK_MONOTONIC): the one clock the callback
/// stamps blocks with and the workers measure e2e latency against.
#[inline]
pub fn now_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: valid pointer to a timespec; CLOCK_MONOTONIC always exists.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// An f64 stored as bits in an `AtomicU64`.
pub struct AtomicF64(AtomicU64);

impl AtomicF64 {
    pub fn new(v: f64) -> Self {
        AtomicF64(AtomicU64::new(v.to_bits()))
    }
    #[inline]
    pub fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
    pub fn store(&self, v: f64) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
}

/// Fixed-size ring of ns durations (the reference's int64 perf arrays plus
/// their write index). One writer; the control loop reads stats.
pub struct PerfRing {
    buf: Box<[AtomicU64]>,
    idx: AtomicU64,
}

impl PerfRing {
    pub fn new(len: usize) -> Self {
        PerfRing { buf: (0..len).map(|_| AtomicU64::new(0)).collect(), idx: AtomicU64::new(0) }
    }

    #[inline]
    pub fn record(&self, ns: u64) {
        let i = self.idx.load(Ordering::Relaxed);
        self.buf[(i % self.buf.len() as u64) as usize].store(ns, Ordering::Relaxed);
        self.idx.store(i + 1, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        for x in self.buf.iter() {
            x.store(0, Ordering::Relaxed);
        }
        self.idx.store(0, Ordering::Relaxed);
    }

    /// `App._ring_stats`: (avg_ms, p95_ms) over the written, non-zero slots;
    /// p95 is the element at `int(0.95·(n−1))` after a partial sort.
    pub fn stats(&self) -> (f64, f64) {
        let count = self.idx.load(Ordering::Relaxed) as usize;
        if count == 0 {
            return (0.0, 0.0);
        }
        let n = count.min(self.buf.len());
        let mut nz: Vec<u64> = self.buf[..n]
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .filter(|&x| x > 0)
            .collect();
        if nz.is_empty() {
            return (0.0, 0.0);
        }
        let avg = nz.iter().map(|&x| x as f64).sum::<f64>() / nz.len() as f64;
        let k = (0.95 * (nz.len() - 1) as f64) as usize;
        let (_, p95, _) = nz.select_nth_unstable(k);
        (avg / 1e6, *p95 as f64 / 1e6)
    }
}

pub struct Perf {
    pub cb: PerfRing,
    pub dsp: PerfRing,
    pub fft: PerfRing,
    pub ws: PerfRing,
    pub lmh_e2e: PerfRing,
    pub fft_e2e: PerfRing,
}

impl Perf {
    fn new() -> Self {
        Perf {
            cb: PerfRing::new(128),
            dsp: PerfRing::new(128),
            fft: PerfRing::new(64),
            ws: PerfRing::new(32),
            lmh_e2e: PerfRing::new(128),
            fft_e2e: PerfRing::new(64),
        }
    }

    pub fn reset_all(&self) {
        for r in [&self.cb, &self.dsp, &self.fft, &self.ws, &self.lmh_e2e, &self.fft_e2e] {
            r.reset();
        }
    }
}

pub struct Shared {
    pub ring: SlotRing,
    pub osc: OscSender,
    pub perf: Perf,
    // Live values the workers read per block / hop (the reference's config
    // getters on the OSC publisher).
    pub fft_enabled: AtomicBool,
    pub send_fft: AtomicBool,
    pub send_raw_db: AtomicBool,
    pub master_gain: AtomicF64,
    pub db_floor: AtomicF64,
    pub stop: AtomicBool,
    // Monotonic counters (status reports them minus per-session baselines).
    pub cb_overruns: AtomicU64,
    pub dsp_drops: AtomicU64,
    pub fft_drops: AtomicU64,
}

impl Shared {
    pub fn new(ring: SlotRing, osc: OscSender, cfg: &crate::config::Config) -> Self {
        Shared {
            ring,
            osc,
            perf: Perf::new(),
            fft_enabled: AtomicBool::new(cfg.fft.enabled),
            send_fft: AtomicBool::new(cfg.osc.send_fft),
            send_raw_db: AtomicBool::new(cfg.fft.send_raw_db),
            master_gain: AtomicF64::new(cfg.autoscale.master_gain),
            db_floor: AtomicF64::new(cfg.fft.db_floor),
            stop: AtomicBool::new(false),
            cb_overruns: AtomicU64::new(0),
            dsp_drops: AtomicU64::new(0),
            fft_drops: AtomicU64::new(0),
        }
    }
}

/// Worker-loop error logging, at most once per `interval` (the reference's
/// `RateLimitedErrorLog`): repeats are counted and reported with the next one.
pub struct RateLimitedLog {
    label: &'static str,
    interval_ns: u64,
    last_ns: u64,
    suppressed: u64,
}

impl RateLimitedLog {
    pub fn new(label: &'static str) -> Self {
        RateLimitedLog { label, interval_ns: 1_000_000_000, last_ns: 0, suppressed: 0 }
    }

    pub fn error(&mut self, what: &str) {
        let now = now_ns();
        if self.last_ns != 0 && now - self.last_ns < self.interval_ns {
            self.suppressed += 1;
            return;
        }
        self.last_ns = now;
        let s = std::mem::take(&mut self.suppressed);
        assert_no_alloc::permit_alloc(|| {
            if s > 0 {
                tracing::error!("{}: iteration failed, continuing ({s} similar suppressed): {what}", self.label);
            } else {
                tracing::error!("{}: iteration failed, continuing: {what}", self.label);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perf_stats_match_reference() {
        let r = PerfRing::new(8);
        assert_eq!(r.stats(), (0.0, 0.0));
        for v in [1_000_000u64, 2_000_000, 3_000_000, 4_000_000] {
            r.record(v);
        }
        let (avg, p95) = r.stats();
        assert!((avg - 2.5).abs() < 1e-12);
        // int(0.95·3) = 2 → third smallest.
        assert_eq!(p95, 3.0);
        for _ in 0..20 {
            r.record(5_000_000);
        }
        assert_eq!(r.stats(), (5.0, 5.0));
    }
}
