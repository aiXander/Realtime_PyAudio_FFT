//! The spectro worker: `dsp::spectro::Spectro` on its own thread, feeding the
//! UI's 3D view a frame every ~5 ms (Rust engine only, no Python reference).
//!
//! Opt-in per WS client (`set_spectro {enabled, n_bins?, res?}`, handled in
//! `App::on_message` like `set_stream`); the thread idles unless at least one
//! streaming client subscribed, so a closed 3D view costs nothing. It is not
//! realtime: it polls the slot ring every few ms (the ring is multi-reader,
//! and keeps ≥ 2 × 16384 samples, more than the longest window), so the audio
//! callback doesn't know it exists. Frames queue here until the WS broadcast
//! tick drains them into one binary message (`take_message`).
//!
//! Wire (binary, little-endian), one message per broadcast tick:
//! ```text
//! [0]  u8  type = 2          [1]  u8  version = 1
//! [2]  u16 n_bins            [4]  u16 n_frames        [6] u16 reserved
//! [8]  u32 first_seq         (frame k is seq first_seq + k; seq counts hops)
//! [12] f32 hop_s             [16] f32 f_min_hz        [20] f32 f_max_hz
//! [24] f32 db0               [28] f32 db_step         (dB = db0 + q · db_step)
//! [32] u8 × n_bins × n_frames, frame-major, bins log-spaced f_min → f_max
//! ```

use crate::shared::{AtomicF64, PerfRing, Shared};
use dsp::spectro::{Spectro, SpectroParams};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

pub const MIN_BINS: usize = 64;
pub const MAX_BINS: usize = 2048;
pub const DEFAULT_BINS: usize = 1024;
pub const F_MIN: f64 = 20.0;
pub const F_MAX: f64 = 20_000.0;
/// u8 quantisation: 0.5 dB steps from −127.5 dBFS (q = 0) to 0 dBFS (q = 255).
pub const DB0: f32 = -127.5;
pub const DB_STEP: f32 = 0.5;
/// Frames held for the WS tick before the queue restarts (nobody draining).
const QUEUE_MAX_FRAMES: usize = 1024;
/// More than this many hops behind → skip ahead (a gap the client sees).
const MAX_BACKLOG_HOPS: u64 = 64;
const POLL: Duration = Duration::from_millis(4);
const IDLE: Duration = Duration::from_millis(100);
const HEADER: usize = 32;

/// Time ↔ frequency trade-off: the wanted rfft bin spacing as a fraction of
/// the frequency (`SpectroParams::rel_res`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Res {
    Time,
    Balanced,
    Pitch,
}

impl Res {
    pub fn parse(s: &str) -> Option<Res> {
        match s {
            "time" => Some(Res::Time),
            "balanced" => Some(Res::Balanced),
            "pitch" => Some(Res::Pitch),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Res::Time => "time",
            Res::Balanced => "balanced",
            Res::Pitch => "pitch",
        }
    }
    fn rel_res(self) -> f64 {
        match self {
            Res::Time => 0.08,
            Res::Balanced => 0.045,
            Res::Pitch => 0.02,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request {
    pub n_bins: usize,
    pub res: Res,
}

struct Queue {
    epoch: u64,
    n_bins: usize,
    hop_s: f32,
    f_min: f32,
    f_max: f32,
    first_seq: u64,
    count: usize,
    data: Vec<u8>,
}

/// Shared between the control loop and the worker.
pub struct SpectroCtl {
    enabled: AtomicBool,
    req: Mutex<Request>,
    sr: AtomicF64,
    blocksize: usize,
    /// Bumped on any request / sample-rate change: the worker rebuilds.
    epoch: AtomicU64,
    queue: Mutex<Queue>,
    pub perf: PerfRing,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Ring blocks per hop: the whole number of blocks closest to 5 ms.
fn hop_blocks(sr: f64, blocksize: usize) -> usize {
    ((sr * 0.005 / blocksize as f64).round() as usize).max(1)
}

impl SpectroCtl {
    pub fn new(sr: f64, blocksize: usize) -> Self {
        SpectroCtl {
            enabled: AtomicBool::new(false),
            req: Mutex::new(Request { n_bins: DEFAULT_BINS, res: Res::Balanced }),
            sr: AtomicF64::new(sr),
            blocksize,
            epoch: AtomicU64::new(1),
            queue: Mutex::new(Queue {
                epoch: 0,
                n_bins: 0,
                hop_s: 0.0,
                f_min: 0.0,
                f_max: 0.0,
                first_seq: 0,
                count: 0,
                data: Vec::new(),
            }),
            perf: PerfRing::new(64),
        }
    }

    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Release);
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub fn request(&self) -> Request {
        *lock(&self.req)
    }

    pub fn set_request(&self, r: Request) {
        let mut cur = lock(&self.req);
        if *cur != r {
            *cur = r;
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub fn set_sr(&self, sr: f64) {
        if self.sr.load() != sr {
            self.sr.store(sr);
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub fn hop_s(&self) -> f64 {
        let sr = self.sr.load();
        (hop_blocks(sr, self.blocksize) * self.blocksize) as f64 / sr
    }

    fn params(&self) -> SpectroParams {
        let r = self.request();
        let sr = self.sr.load();
        SpectroParams {
            sr,
            n_bins: r.n_bins,
            f_min: F_MIN,
            f_max: F_MAX.min(0.49 * sr),
            rel_res: r.res.rel_res(),
            max_window: crate::validate::MAX_FFT_WINDOW,
        }
    }

    /// The reply to `set_spectro` (and what a client needs to label the view).
    pub fn info(&self, enabled: bool) -> Value {
        let r = self.request();
        let p = self.params();
        let windows: Vec<usize> = Spectro::new(p).window_sizes().collect();
        json!({
            "type": "spectro",
            "enabled": enabled,
            "n_bins": r.n_bins,
            "res": r.res.name(),
            "hop_s": self.hop_s(),
            "f_min": p.f_min,
            "f_max": p.f_max,
            "windows": windows,
            "sr": p.sr,
        })
    }

    pub fn perf_message(&self) -> Value {
        let (avg, p95) = self.perf.stats();
        let hop_ms = self.hop_s() * 1000.0;
        json!({"type": "spectro_perf", "avg_ms": avg, "p95_ms": p95, "load_pct": avg / hop_ms * 100.0})
    }

    /// Everything queued since the last call, as one binary message.
    pub fn take_message(&self) -> Option<Vec<u8>> {
        let mut q = lock(&self.queue);
        if q.count == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(HEADER + q.data.len());
        out.push(2);
        out.push(1);
        out.extend_from_slice(&(q.n_bins as u16).to_le_bytes());
        out.extend_from_slice(&(q.count as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(q.first_seq as u32).to_le_bytes());
        for v in [q.hop_s, q.f_min, q.f_max, DB0, DB_STEP] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&q.data);
        q.data.clear();
        q.first_seq += q.count as u64;
        q.count = 0;
        Some(out)
    }

    fn push(&self, epoch: u64, seq: u64, p: &SpectroParams, hop_s: f64, db: &[f32]) {
        let mut q = lock(&self.queue);
        let contiguous = q.count > 0 && q.epoch == epoch && seq == q.first_seq + q.count as u64;
        if !contiguous || q.count >= QUEUE_MAX_FRAMES {
            // New geometry, a gap, or nobody draining: restart the queue.
            *q = Queue {
                epoch,
                n_bins: db.len(),
                hop_s: hop_s as f32,
                f_min: p.f_min as f32,
                f_max: p.f_max as f32,
                first_seq: seq,
                count: 0,
                data: std::mem::take(&mut q.data),
            };
            q.data.clear();
        }
        let inv = 1.0 / DB_STEP;
        q.data.extend(db.iter().map(|&v| ((v - DB0) * inv).round().clamp(0.0, 255.0) as u8));
        q.count += 1;
    }
}

pub fn spawn(shared: Arc<Shared>, ctl: Arc<SpectroCtl>) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new().name("spectro-worker".into()).spawn(move || run(&shared, &ctl))
}

struct Plan {
    epoch: u64,
    spectro: Spectro,
    span_blocks: u64,
    hop_blocks: u64,
    hop_s: f64,
    buf: Vec<f64>,
}

fn run(shared: &Shared, ctl: &SpectroCtl) {
    let bs = ctl.blocksize;
    let mut plan: Option<Plan> = None;
    let mut active = false;
    let mut read_idx = 0u64;
    let mut seq = 0u64;
    loop {
        if shared.stop.load(Ordering::Acquire) {
            return;
        }
        if !ctl.enabled() {
            active = false;
            std::thread::park_timeout(IDLE);
            continue;
        }
        let epoch = ctl.epoch.load(Ordering::Acquire);
        if plan.as_ref().is_none_or(|p| p.epoch != epoch) {
            let p = ctl.params();
            let spectro = Spectro::new(p);
            let span_blocks = spectro.span().div_ceil(bs) as u64;
            let hb = hop_blocks(p.sr, bs) as u64;
            plan = Some(Plan {
                epoch,
                span_blocks,
                hop_blocks: hb,
                hop_s: (hb as usize * bs) as f64 / p.sr,
                buf: vec![0.0; span_blocks as usize * bs],
                spectro,
            });
            active = false;
        }
        let p = plan.as_mut().unwrap_or_else(|| unreachable!());
        let wi = shared.ring.write_idx();
        if !active || read_idx > wi {
            // (Re)start one window back; read_idx > wi = the ring was reset.
            read_idx = wi.saturating_sub(p.span_blocks);
            active = true;
        }
        if wi - read_idx > p.span_blocks + MAX_BACKLOG_HOPS * p.hop_blocks {
            let k = (wi - p.span_blocks - read_idx) / p.hop_blocks;
            read_idx += k * p.hop_blocks;
            seq += k;
        }
        while wi - read_idx >= p.span_blocks {
            let t0 = crate::shared::now_ns();
            if shared.ring.try_read_window(read_idx, p.span_blocks as usize, &mut p.buf).is_some() {
                let params = *p.spectro.params();
                let db = p.spectro.process(&p.buf);
                ctl.push(p.epoch, seq, &params, p.hop_s, db);
                ctl.perf.record(crate::shared::now_ns() - t0);
            }
            read_idx += p.hop_blocks;
            seq += 1;
        }
        std::thread::park_timeout(POLL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_layout() {
        let ctl = SpectroCtl::new(48_000.0, 256);
        let p = ctl.params();
        ctl.push(1, 7, &p, 0.005, &[-127.5, 0.0, -60.0]);
        ctl.push(1, 8, &p, 0.005, &[-200.0, 3.0, -60.2]);
        let m = ctl.take_message().unwrap();
        assert_eq!(&m[..8], &[2, 1, 3, 0, 2, 0, 0, 0]);
        assert_eq!(u32::from_le_bytes(m[8..12].try_into().unwrap()), 7);
        assert_eq!(&m[HEADER..], &[0, 255, 135, 0, 255, 135]);
        assert!(ctl.take_message().is_none());
        // Next batch continues the sequence; a gap restarts it.
        ctl.push(1, 9, &p, 0.005, &[0.0; 3]);
        ctl.push(1, 20, &p, 0.005, &[0.0; 3]);
        let m = ctl.take_message().unwrap();
        assert_eq!(u32::from_le_bytes(m[8..12].try_into().unwrap()), 20);
        assert_eq!(u16::from_le_bytes(m[4..6].try_into().unwrap()), 1);
    }
}
