//! DSP + FFT worker threads (`server/dsp/worker.py`, `FFTWorker.run`).
//!
//! Each worker owns its DSP state outright. The control loop changes it by
//! pushing commands on an SPSC queue the worker drains before every block /
//! hop. Anything that allocates (filter designs are fixed-size, but a new
//! `FftChain`, a smear kernel, a hot-switch's rebuilt filter/smoother/scaler)
//! is built on the control thread and sent boxed; the replaced box comes back
//! over a return queue so it's dropped off the realtime threads.
//!
//! Backlog policies are the reference's: the DSP worker drains every block,
//! skips to `wi − 1` when more than 4 blocks behind (counting drops and
//! advancing the onset clock); the FFT worker drains every complete hop and
//! skips whole hops when more than `window + 2 hops` behind.

use crate::osc::{FftPacker, LmhPacker};
use crate::shared::{RateLimitedLog, Shared, now_ns};
use dsp::features::{AutoScaler, ExpSmoother};
use dsp::fft_post::{AutoscaleUpdate, Smear};
use dsp::filters::{FilterBank, FilterDesign};
use dsp::onset::OnsetParamsUpdate;
use dsp::{Bands, DspChain, FftChain};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::time::Duration;

pub const MAX_DSP_BACKLOG_BLOCKS: u64 = 4;
pub const MAX_FFT_BACKLOG_HOPS: u64 = 2;
const WAKE_TIMEOUT: Duration = Duration::from_millis(100);

/// The sr-dependent part of the DSP chain rebuilt on a device hot-switch
/// (the OnsetTracker is kept: reconfigured + reset).
pub struct DspCore {
    pub filter_bank: FilterBank,
    pub smoother: ExpSmoother,
    pub auto_scaler: AutoScaler,
}

// By value, not boxed: the queue slots are preallocated, so the worker
// never frees a box for a filter install.
#[allow(clippy::large_enum_variant)]
pub enum DspCmd {
    InstallFilter(FilterDesign),
    SetTau([f64; 3], [f64; 3]),
    SetAutoscaleTaus(f64, f64),
    SetNoiseFloor(f64),
    SetStrength(f64),
    SetTilt(f64),
    SetBands(Bands),
    SetNFftWindow(usize),
    SetFftGeometry(Option<usize>, Option<f64>),
    SetOnset(usize, OnsetParamsUpdate),
    /// Hot-switch: new filter/smoother/scaler for `sr`; onset reconfigure + reset.
    ReplaceCore(Box<DspCore>, f64),
    /// Rewind the read pointer (the ring was reset).
    ResetReadPtr,
}

pub enum FftCmd {
    /// Any FFT reconfigure: a whole new chain built from the current cfg;
    /// the read pointer goes back to 0 (alignment changed).
    ReplaceChain(Box<FftChain>),
    UpdateSmoothing([f64; 3], [f64; 3]),
    UpdateBands(Bands),
    UpdateAutoscale(AutoscaleUpdate),
    UpdateTilt(f64),
    /// `(peak_smear_oct, smear, sigma, built-for n_bins)`.
    InstallSmear(f64, Option<Box<Smear>>, f64, usize),
    /// Post-processor reset (smooth + peak + warm-up).
    ResetPost,
    /// Snap the read pointer to one window back without counting drops.
    Resync,
    ResetReadPtr,
}

/// Replaced state on its way back to the control thread to be dropped.
#[allow(dead_code)] // held only so the drop happens on the control thread
pub enum Garbage {
    DspCore(Box<DspCore>),
    FftChain(Box<FftChain>),
    Smear(Option<Box<Smear>>),
}

/// Latest L/M/H result (FeatureStore).
#[derive(Clone, Copy, Debug, Default)]
pub struct FeatureSnap {
    pub seq: u64,
    pub raw: [f64; 3],
    pub scaled: [f64; 3],
    pub onset_counts: [u64; 3],
    pub bpm: f64,
}

/// Latest FFT frame pair (FFTStore). Fixed capacity: n_bins changes never
/// allocate.
#[derive(Clone)]
pub struct FftFrame {
    pub seq: u64,
    pub n: usize,
    pub raw_db: Box<[f32; crate::validate::MAX_N_FFT_BINS]>,
    pub processed: Box<[f32; crate::validate::MAX_N_FFT_BINS]>,
    pub t_recv_ns: u64,
}

impl Default for FftFrame {
    fn default() -> Self {
        FftFrame {
            seq: 0,
            n: 0,
            raw_db: Box::new([0.0; crate::validate::MAX_N_FFT_BINS]),
            processed: Box::new([0.0; crate::validate::MAX_N_FFT_BINS]),
            t_recv_ns: 0,
        }
    }
}

/// Control-side handle to one worker's command queue.
pub struct CmdTx<T> {
    tx: rtrb::Producer<T>,
    thread: std::thread::Thread,
}

impl<T> CmdTx<T> {
    /// Push and wake the worker. The queue is deep and the worker drains it
    /// every ≤ 100 ms, so a full queue only waits briefly.
    pub fn send(&mut self, mut cmd: T) {
        for _ in 0..2000 {
            match self.tx.push(cmd) {
                Ok(()) => {
                    self.thread.unpark();
                    return;
                }
                Err(rtrb::PushError::Full(c)) => {
                    cmd = c;
                    self.thread.unpark();
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
        tracing::error!("worker command queue stuck full; command dropped");
    }
}

pub struct Workers {
    pub dsp_tx: CmdTx<DspCmd>,
    pub fft_tx: CmdTx<FftCmd>,
    pub garbage: [rtrb::Consumer<Garbage>; 2],
    pub feat_out: triple_buffer::Output<FeatureSnap>,
    pub fft_out: triple_buffer::Output<FftFrame>,
    pub dsp_thread: std::thread::Thread,
    pub fft_thread: std::thread::Thread,
    joins: Vec<JoinHandle<()>>,
}

impl Workers {
    pub fn spawn(shared: Arc<Shared>, dsp: DspChain, fft: FftChain, blocksize: usize) -> std::io::Result<Self> {
        let (dsp_ctx, dsp_crx) = rtrb::RingBuffer::<DspCmd>::new(256);
        let (fft_ctx, fft_crx) = rtrb::RingBuffer::<FftCmd>::new(256);
        let (dsp_gtx, dsp_grx) = rtrb::RingBuffer::<Garbage>::new(64);
        let (fft_gtx, fft_grx) = rtrb::RingBuffer::<Garbage>::new(64);
        let (feat_in, feat_out) = triple_buffer::triple_buffer(&FeatureSnap::default());
        let (fft_in, fft_out) = triple_buffer::triple_buffer(&FftFrame::default());

        let mut d = DspWorker {
            shared: shared.clone(),
            chain: dsp,
            cmds: dsp_crx,
            garbage: dsp_gtx,
            store: feat_in,
            packer: LmhPacker::default(),
            buf: vec![0.0; blocksize],
            read_idx: 0,
            seq: 0,
            err: RateLimitedLog::new("dsp-worker"),
        };
        let dsp_join = std::thread::Builder::new()
            .name("dsp-worker".into())
            .spawn(move || d.run())?;
        let mut f = FftWorker {
            shared: shared.clone(),
            chain: Box::new(fft),
            fresh: true,
            cmds: fft_crx,
            garbage: fft_gtx,
            store: fft_in,
            packer: FftPacker::default(),
            read_idx: 0,
            seq: 0,
            err: RateLimitedLog::new("fft-worker"),
        };
        let fft_join = std::thread::Builder::new()
            .name("fft-worker".into())
            .spawn(move || f.run())?;
        let dsp_thread = dsp_join.thread().clone();
        let fft_thread = fft_join.thread().clone();
        Ok(Workers {
            dsp_tx: CmdTx { tx: dsp_ctx, thread: dsp_thread.clone() },
            fft_tx: CmdTx { tx: fft_ctx, thread: fft_thread.clone() },
            garbage: [dsp_grx, fft_grx],
            feat_out,
            fft_out,
            dsp_thread,
            fft_thread,
            joins: vec![dsp_join, fft_join],
        })
    }

    /// Drop whatever the workers handed back.
    pub fn collect_garbage(&mut self) {
        for g in &mut self.garbage {
            while let Ok(x) = g.pop() {
                drop(x);
            }
        }
    }

    /// Stop + join (≤ 1 s each, like the reference).
    pub fn stop(&mut self, shared: &Shared) {
        shared.stop.store(true, Ordering::Release);
        self.dsp_thread.unpark();
        self.fft_thread.unpark();
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        for j in self.joins.drain(..) {
            while !j.is_finished() && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if j.is_finished() {
                let _ = j.join();
            }
        }
        self.collect_garbage();
    }
}

/// Runs `f` under `assert_no_alloc` in debug builds (a no-op in release).
#[inline(always)]
fn realtime<T>(f: impl FnOnce() -> T) -> T {
    assert_no_alloc::assert_no_alloc(f)
}

fn panic_msg(p: &(dyn std::any::Any + Send)) -> &str {
    p.downcast_ref::<&str>()
        .copied()
        .or_else(|| p.downcast_ref::<String>().map(|s| s.as_str()))
        .unwrap_or("panic")
}

struct DspWorker {
    shared: Arc<Shared>,
    chain: DspChain,
    cmds: rtrb::Consumer<DspCmd>,
    garbage: rtrb::Producer<Garbage>,
    store: triple_buffer::Input<FeatureSnap>,
    packer: LmhPacker,
    buf: Vec<f32>,
    read_idx: u64,
    seq: u64,
    err: RateLimitedLog,
}

impl DspWorker {
    fn run(&mut self) {
        crate::priority::boost_current_thread("dsp-worker");
        loop {
            std::thread::park_timeout(WAKE_TIMEOUT);
            if self.shared.stop.load(Ordering::Acquire) {
                return;
            }
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| realtime(|| self.wake()))) {
                self.err.error(panic_msg(&*p));
                self.read_idx += 1;
            }
        }
    }

    /// Returns true when a command rewound the read pointer.
    fn drain_cmds(&mut self) -> bool {
        let mut rewound = false;
        while let Ok(cmd) = self.cmds.pop() {
            let c = &mut self.chain;
            match cmd {
                DspCmd::InstallFilter(d) => c.filter_bank.install(d),
                DspCmd::SetTau(t, ta) => c.smoother.set_tau(t, ta),
                DspCmd::SetAutoscaleTaus(a, r) => c.auto_scaler.set_taus(a, r),
                DspCmd::SetNoiseFloor(nf) => c.auto_scaler.set_noise_floor(nf),
                DspCmd::SetStrength(s) => c.auto_scaler.set_strength(s),
                DspCmd::SetTilt(t) => c.auto_scaler.set_tilt(t),
                DspCmd::SetBands(b) => c.auto_scaler.set_bands(b),
                DspCmd::SetNFftWindow(n) => c.auto_scaler.set_n_fft_window(n),
                DspCmd::SetFftGeometry(n, f) => c.auto_scaler.set_fft_geometry(n, f),
                DspCmd::SetOnset(band, u) => c.onset_tracker.set_params(band, u),
                DspCmd::ReplaceCore(mut core, sr) => {
                    std::mem::swap(&mut c.filter_bank, &mut core.filter_bank);
                    std::mem::swap(&mut c.smoother, &mut core.smoother);
                    std::mem::swap(&mut c.auto_scaler, &mut core.auto_scaler);
                    c.onset_tracker.reconfigure(sr, self.buf.len());
                    c.onset_tracker.reset();
                    self.give_back(Garbage::DspCore(core));
                }
                DspCmd::ResetReadPtr => {
                    self.read_idx = 0;
                    rewound = true;
                }
            }
        }
        rewound
    }

    fn give_back(&mut self, g: Garbage) {
        if let Err(rtrb::PushError::Full(g)) = self.garbage.push(g) {
            // Control drains this every loop tick; a full queue means it's
            // gone. Leak rather than free on the realtime thread.
            std::mem::forget(g);
        }
    }

    fn wake(&mut self) {
        self.drain_cmds();
        let ring = &self.shared.ring;
        let wi = ring.write_idx();
        // Self-heal after a producer reset (ring.reset() rewinds write_idx).
        if self.read_idx > wi {
            self.read_idx = wi;
        }
        if wi - self.read_idx > MAX_DSP_BACKLOG_BLOCKS {
            let skipped = (wi - 1).saturating_sub(self.read_idx);
            if skipped > 0 {
                self.shared.dsp_drops.fetch_add(skipped, Ordering::Relaxed);
                // Skipped blocks are real elapsed time for the BPM clock.
                self.chain.onset_tracker.advance(skipped);
            }
            self.read_idx = wi - 1;
        }
        // Drain every block in the ring (wakes coalesce).
        while self.read_idx < wi {
            if self.drain_cmds() {
                return; // pointer rewound: re-snapshot on the next wake
            }
            let ring = &self.shared.ring;
            match ring.try_read_block(self.read_idx, &mut self.buf) {
                Some(t_recv) => {
                    let t0 = now_ns();
                    let out = self.chain.process_block(&self.buf);
                    self.seq += 1;
                    let snap = self.store.input_buffer_mut();
                    *snap = FeatureSnap {
                        seq: self.seq,
                        raw: out.raw,
                        scaled: out.scaled,
                        onset_counts: self.chain.onset_tracker.onset_count(),
                        bpm: out.bpm,
                    };
                    self.store.publish();
                    let sh = &*self.shared;
                    self.packer.publish(&sh.osc, out.scaled, sh.master_gain.load(), out.onsets, out.bpm);
                    record_latency(&sh.perf.lmh_e2e, t_recv);
                    sh.perf.dsp.record(now_ns() - t0);
                    self.read_idx += 1;
                }
                None => {
                    self.shared.dsp_drops.fetch_add(1, Ordering::Relaxed);
                    let nxt = (self.read_idx + 1).max(wi.saturating_sub(1));
                    self.chain.onset_tracker.advance(nxt - self.read_idx);
                    self.read_idx = nxt;
                }
            }
        }
    }
}

#[inline]
fn record_latency(ring: &crate::shared::PerfRing, t_recv: u64) {
    if t_recv == 0 {
        return;
    }
    let now = now_ns();
    if now > t_recv {
        ring.record(now - t_recv);
    }
}

struct FftWorker {
    shared: Arc<Shared>,
    chain: Box<FftChain>,
    /// The chain hasn't processed a hop yet: its first hop builds the
    /// sentinel LUT (the one accepted allocation, parity trap 6).
    fresh: bool,
    cmds: rtrb::Consumer<FftCmd>,
    garbage: rtrb::Producer<Garbage>,
    store: triple_buffer::Input<FftFrame>,
    packer: FftPacker,
    read_idx: u64,
    seq: u64,
    err: RateLimitedLog,
}

impl FftWorker {
    fn run(&mut self) {
        crate::priority::boost_current_thread("fft-worker");
        loop {
            std::thread::park_timeout(WAKE_TIMEOUT);
            if self.shared.stop.load(Ordering::Acquire) {
                return;
            }
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| realtime(|| self.wake()))) {
                self.err.error(panic_msg(&*p));
                self.read_idx += self.chain.analyzer.geometry().hop_blocks() as u64;
            }
        }
    }

    fn give_back(&mut self, g: Garbage) {
        if let Err(rtrb::PushError::Full(g)) = self.garbage.push(g) {
            std::mem::forget(g);
        }
    }

    fn geometry(&self) -> (u64, u64) {
        let g = self.chain.analyzer.geometry();
        (g.n_blocks_per_window() as u64, g.hop_blocks() as u64)
    }

    /// Returns true when the read pointer / geometry changed.
    fn drain_cmds(&mut self) -> bool {
        let mut moved = false;
        while let Ok(cmd) = self.cmds.pop() {
            match cmd {
                FftCmd::ReplaceChain(mut c) => {
                    std::mem::swap(&mut self.chain, &mut c);
                    self.fresh = true;
                    self.read_idx = 0;
                    moved = true;
                    self.give_back(Garbage::FftChain(c));
                }
                FftCmd::UpdateSmoothing(t, ta) => self.chain.post.update_smoothing(t, ta),
                FftCmd::UpdateBands(b) => self.chain.post.update_bands(b),
                FftCmd::UpdateAutoscale(u) => self.chain.post.update_autoscale(u),
                FftCmd::UpdateTilt(t) => self.chain.post.update_tilt(t),
                FftCmd::InstallSmear(oct, smear, sigma, n_bins) => {
                    if n_bins == self.chain.post.n_bins() {
                        let old = self.chain.post.install_smear(oct, smear, sigma);
                        self.give_back(Garbage::Smear(old));
                    } else {
                        // Built for a geometry that has since been replaced
                        // (the replacement chain already carries the cfg smear).
                        self.give_back(Garbage::Smear(smear));
                    }
                }
                FftCmd::ResetPost => self.chain.post.reset(),
                FftCmd::Resync => {
                    let (nbw, _) = self.geometry();
                    self.read_idx = self.shared.ring.write_idx().saturating_sub(nbw);
                    moved = true;
                }
                FftCmd::ResetReadPtr => {
                    self.read_idx = 0;
                    moved = true;
                }
            }
        }
        moved
    }

    fn wake(&mut self) {
        self.drain_cmds();
        if !self.shared.fft_enabled.load(Ordering::Acquire) {
            return;
        }
        // Commands pushed before the enable flag was set (reset + resync).
        self.drain_cmds();
        let (nbw, hb) = self.geometry();
        let wi = self.shared.ring.write_idx();
        if self.read_idx > wi {
            self.read_idx = 0;
        }
        if wi - self.read_idx > nbw + MAX_FFT_BACKLOG_HOPS * hb {
            let target = wi - nbw;
            let skipped = target.saturating_sub(self.read_idx) / hb;
            if skipped > 0 {
                self.shared.fft_drops.fetch_add(skipped, Ordering::Relaxed);
                self.read_idx += skipped * hb;
            }
        }
        while wi - self.read_idx >= nbw {
            if self.drain_cmds() {
                return;
            }
            let t0 = now_ns();
            let read_idx = self.read_idx;
            let Some(t_recv) =
                self.shared.ring.try_read_window(read_idx, nbw as usize, self.chain.window_mut())
            else {
                self.shared.fft_drops.fetch_add(1, Ordering::Relaxed);
                self.read_idx = (read_idx + hb).max(wi - nbw);
                continue;
            };
            if self.fresh {
                self.fresh = false;
                assert_no_alloc::permit_alloc(|| self.hop(t_recv));
            } else {
                self.hop(t_recv);
            }
            self.shared.perf.fft.record(now_ns() - t0);
            self.read_idx += hb;
        }
    }

    /// One hop on the filled window → FFTStore + `/audio/fft`.
    #[inline]
    fn hop(&mut self, t_recv: u64) {
        let (raw, processed) = self.chain.process_hop();
        self.seq += 1;
        let n = raw.len();
        let fr = self.store.input_buffer_mut();
        fr.seq = self.seq;
        fr.n = n;
        fr.raw_db[..n].copy_from_slice(raw);
        fr.processed[..n].copy_from_slice(processed);
        fr.t_recv_ns = t_recv;
        self.store.publish();
        let sh = &*self.shared;
        if sh.send_fft.load(Ordering::Relaxed) && sh.fft_enabled.load(Ordering::Relaxed) {
            if sh.send_raw_db.load(Ordering::Relaxed) {
                self.packer.send_raw_db(&sh.osc, raw, sh.db_floor.load());
            } else {
                self.packer.send_processed(&sh.osc, processed, sh.master_gain.load());
            }
            record_latency(&sh.perf.fft_e2e, t_recv);
        }
    }
}
