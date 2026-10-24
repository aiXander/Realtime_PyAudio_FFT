//! Block-aligned slot ring: one producer (the audio callback), two readers
//! (DSP + FFT workers). Port of `server/audio/ringbuffer.py`.
//!
//! Slots rather than a FIFO because the FFT reads overlapping windows. A
//! seqlock over plain memory is UB in Rust, so samples live in `AtomicU32`
//! (f32 bits, Relaxed) guarded by a per-slot sequence. Unlike the Python
//! protocol, the writer *invalidates* the slot's sequence before touching
//! the data, so a reader can never accept a block the writer lapped
//! mid-copy (Python could, in principle, accept a torn read). That's a
//! transport-robustness fix, not a DSP change.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};

pub struct SlotRing {
    mask: usize,
    blocksize: usize,
    samples: Box<[AtomicU32]>,
    /// `read_idx + 1` once block `read_idx` is fully written; 0 while a
    /// write is in progress (or the slot was never written).
    seq: Box<[AtomicU64]>,
    /// Monotonic ns at which the block became available (e2e latency clock).
    t_ns: Box<[AtomicU64]>,
    write_idx: AtomicU64,
}

/// Python `ring_slots_for`: a power of two ≥ 32 holding ≥ 2× the largest
/// accepted FFT window at this blocksize.
pub fn ring_slots_for(blocksize: usize) -> usize {
    let need = 2 * crate::validate::MAX_FFT_WINDOW.div_ceil(blocksize);
    let mut n = 32;
    while n < need {
        n *= 2;
    }
    n
}

impl SlotRing {
    pub fn new(n_slots: usize, blocksize: usize) -> Self {
        assert!(n_slots.is_power_of_two() && n_slots >= 8, "n_slots must be a power of two >= 8");
        SlotRing {
            mask: n_slots - 1,
            blocksize,
            samples: (0..n_slots * blocksize).map(|_| AtomicU32::new(0)).collect(),
            seq: (0..n_slots).map(|_| AtomicU64::new(0)).collect(),
            t_ns: (0..n_slots).map(|_| AtomicU64::new(0)).collect(),
            write_idx: AtomicU64::new(0),
        }
    }

    /// Monotonic count of blocks written since the last reset.
    #[inline]
    pub fn write_idx(&self) -> u64 {
        self.write_idx.load(Ordering::Acquire)
    }

    /// Producer only (one thread at a time). Allocation- and lock-free.
    #[inline]
    pub fn write_block(&self, src: &[f32], t_ns: u64) {
        let wi = self.write_idx.load(Ordering::Relaxed);
        let slot = (wi as usize) & self.mask;
        self.seq[slot].store(0, Ordering::Relaxed);
        fence(Ordering::Release);
        let base = slot * self.blocksize;
        for (dst, &x) in self.samples[base..base + self.blocksize].iter().zip(src) {
            dst.store(x.to_bits(), Ordering::Relaxed);
        }
        self.t_ns[slot].store(t_ns, Ordering::Relaxed);
        self.seq[slot].store(wi + 1, Ordering::Release);
        self.write_idx.store(wi + 1, Ordering::Release);
    }

    /// Copy block `read_idx` into `out`; false if it isn't there (yet / any
    /// more) or was overwritten mid-copy. Returns the block's timestamp.
    #[inline]
    pub fn try_read_block(&self, read_idx: u64, out: &mut [f32]) -> Option<u64> {
        let slot = (read_idx as usize) & self.mask;
        let s1 = self.seq[slot].load(Ordering::Acquire);
        if s1 != read_idx + 1 {
            return None;
        }
        let base = slot * self.blocksize;
        for (o, src) in out.iter_mut().zip(&self.samples[base..base + self.blocksize]) {
            *o = f32::from_bits(src.load(Ordering::Relaxed));
        }
        let t = self.t_ns[slot].load(Ordering::Relaxed);
        fence(Ordering::Acquire);
        (self.seq[slot].load(Ordering::Relaxed) == s1).then_some(t)
    }

    /// Copy `n_blocks` consecutive blocks starting at `start` into `out`
    /// (f32 → f64, the FFT window's precision). Returns the last block's
    /// timestamp.
    #[inline]
    pub fn try_read_window(&self, start: u64, n_blocks: usize, out: &mut [f64]) -> Option<u64> {
        let bs = self.blocksize;
        let mut t = 0;
        for k in 0..n_blocks {
            let idx = start + k as u64;
            let slot = (idx as usize) & self.mask;
            let s1 = self.seq[slot].load(Ordering::Acquire);
            if s1 != idx + 1 {
                return None;
            }
            let base = slot * bs;
            for (o, src) in out[k * bs..(k + 1) * bs].iter_mut().zip(&self.samples[base..base + bs]) {
                *o = f32::from_bits(src.load(Ordering::Relaxed)) as f64;
            }
            t = self.t_ns[slot].load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            if self.seq[slot].load(Ordering::Relaxed) != s1 {
                return None;
            }
        }
        Some(t)
    }

    /// Wipe state (device hot-switch; called while no stream runs).
    pub fn reset(&self) {
        self.write_idx.store(0, Ordering::Release);
        for s in self.seq.iter() {
            s.store(0, Ordering::Relaxed);
        }
        for t in self.t_ns.iter() {
            t.store(0, Ordering::Relaxed);
        }
        for x in self.samples.iter() {
            x.store(0, Ordering::Relaxed);
        }
        fence(Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn slot_count() {
        assert_eq!(ring_slots_for(256), 128);
        assert_eq!(ring_slots_for(16384), 32);
        assert_eq!(ring_slots_for(16), 2048);
    }

    #[test]
    fn read_back_and_lapping() {
        let r = SlotRing::new(8, 4);
        let mut out = [0f32; 4];
        assert!(r.try_read_block(0, &mut out).is_none());
        for i in 0..10u64 {
            r.write_block(&[i as f32; 4], 100 + i);
        }
        assert_eq!(r.write_idx(), 10);
        assert!(r.try_read_block(1, &mut out).is_none(), "lapped");
        assert_eq!(r.try_read_block(9, &mut out), Some(109));
        assert_eq!(out, [9.0; 4]);
        let mut w = [0f64; 8];
        assert_eq!(r.try_read_window(8, 2, &mut w), Some(109));
        assert_eq!(&w[..4], &[8.0; 4]);
        r.reset();
        assert!(r.try_read_block(9, &mut out).is_none());
    }

    /// Stress: a producer hammers a tiny ring while two readers race it.
    /// Every accepted block must be internally consistent (all samples carry
    /// the block's own index) — i.e. no torn reads ever get through.
    #[test]
    fn stress_no_torn_reads() {
        let bs = 64;
        let ring = Arc::new(SlotRing::new(8, bs));
        let stop = Arc::new(AtomicBool::new(false));
        let n_blocks = 200_000u64;
        let prod = {
            let ring = ring.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut buf = vec![0f32; bs];
                for i in 0..n_blocks {
                    buf.iter_mut().for_each(|x| *x = i as f32);
                    ring.write_block(&buf, i + 1);
                }
                stop.store(true, Ordering::Release);
            })
        };
        let readers: Vec<_> = (0..2)
            .map(|which| {
                let ring = ring.clone();
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let (mut ok, mut torn_checked) = (0u64, 0u64);
                    let mut blk = vec![0f32; bs];
                    let mut win = vec![0f64; 3 * bs];
                    while !stop.load(Ordering::Acquire) {
                        let wi = ring.write_idx();
                        if wi < 4 {
                            continue;
                        }
                        if which == 0 {
                            let idx = wi - 1;
                            if let Some(t) = ring.try_read_block(idx, &mut blk) {
                                assert!(blk.iter().all(|&x| x == idx as f32), "torn block read");
                                assert_eq!(t, idx + 1);
                                ok += 1;
                            }
                        } else {
                            let start = wi - 3;
                            if ring.try_read_window(start, 3, &mut win).is_some() {
                                for k in 0..3 {
                                    let want = (start + k as u64) as f32 as f64;
                                    assert!(win[k * bs..(k + 1) * bs].iter().all(|&x| x == want), "torn window");
                                }
                                ok += 1;
                            }
                        }
                        torn_checked += 1;
                    }
                    (ok, torn_checked)
                })
            })
            .collect();
        prod.join().unwrap();
        for r in readers {
            let (ok, n) = r.join().unwrap();
            assert!(n > 0 && ok > 0, "reader made no progress ({ok}/{n})");
        }
    }
}
