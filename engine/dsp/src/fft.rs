//! Hop DSP of the FFT worker (port of `server/dsp/fft.py`: `build_log_bin_map`,
//! `FFTWorker._allocate` and `process_hop`). The worker loop, ring reads and
//! backlog policy live in the server crate.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::math::{EPS, linspace, logspace, pairwise_sum, reduce_segment};

/// Wire value for log bins with no rfft bin mapped (low end).
pub const EMPTY_BIN_SENTINEL: f32 = -1000.0;

/// Log-spaced bin assignment for an rfft of length `window_size` (parity
/// trap 4). `assign[k]` is the log bin of rfft bin k or -1;
/// `counts[j]` the number of rfft bins in log bin j.
pub struct LogBinMap {
    pub assign: Vec<i64>,
    pub counts: Vec<f64>,
}

pub fn build_log_bin_map(window_size: usize, sr: f64, n_bins: usize, f_min: f64) -> LogBinMap {
    let n_rfft = window_size / 2 + 1;
    let freqs = linspace(0.0, sr / 2.0, n_rfft);
    let mut f_max = sr / 2.0;
    if f_max <= f_min {
        f_max = f_min * 2.0;
    }
    let edges = logspace(f_min.log10(), f_max.log10(), n_bins + 1);
    let mut assign = vec![-1i64; n_rfft];
    let mut counts = vec![0.0f64; n_bins];
    for (k, &f) in freqs.iter().enumerate() {
        // searchsorted(side="right") - 1, clamped to n_bins - 1
        let mut idx = edges.partition_point(|&e| e <= f) as i64 - 1;
        idx = idx.min(n_bins as i64 - 1);
        if f >= f_min && f <= f_max && idx >= 0 {
            assign[k] = idx;
            counts[idx as usize] += 1.0;
        }
    }
    LogBinMap { assign, counts }
}

/// `np.hanning(m)`: the *symmetric* Hann (parity trap 1).
pub fn hanning(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let pi = std::f64::consts::PI;
    (0..m)
        .map(|i| {
            let n = (2 * i) as f64 - (m as f64 - 1.0); // arange(1-M, M, 2)
            0.5 + 0.5 * (pi * n / (m as f64 - 1.0)).cos()
        })
        .collect()
}

/// Geometry of the FFT stage. `hop` / `blocksize` only matter to the worker
/// (window/hop must be multiples of the block), kept here so a reconfigure
/// is one value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FftGeometry {
    pub sr: f64,
    pub blocksize: usize,
    pub window_size: usize,
    pub hop: usize,
    pub n_bins: usize,
    pub f_min: f64,
}

impl FftGeometry {
    pub fn n_blocks_per_window(&self) -> usize {
        self.window_size / self.blocksize
    }
    pub fn hop_blocks(&self) -> usize {
        self.hop / self.blocksize
    }
    pub fn hop_period_s(&self) -> f64 {
        self.hop as f64 / self.sr
    }
}

/// Window → Hann → rfft → window-corrected power → mean-of-power log bins
/// → dB → f32 wire frame with sentinels. Everything is preallocated in
/// `new`; `process_hop` doesn't allocate.
pub struct FftAnalyzer {
    geom: FftGeometry,
    window_buf: Vec<f64>,
    hann: Vec<f64>,
    win_power_corr: f64,
    plan: Arc<dyn RealToComplex<f64>>,
    fft_in: Vec<f64>,
    spectrum: Vec<Complex<f64>>,
    fft_scratch: Vec<Complex<f64>>,
    power: Vec<f64>,
    map: LogBinMap,
    /// First valid rfft bin; valid rfft bins are one contiguous range.
    valid_start: usize,
    /// Runs of consecutive rfft bins (within the valid range) per log bin:
    /// `(start, end, target)`; `start`/`end` index `power`.
    runs: Vec<(usize, usize, usize)>,
    bin_count_inv: Vec<f64>,
    /// Empty log bins hold the sentinel from construction on; `process_hop`
    /// only writes the bins that have rfft bins.
    wire: Vec<f32>,
}

impl FftAnalyzer {
    /// Errors when window / hop are not ≥ 1 block (Python raises ValueError).
    pub fn new(geom: FftGeometry) -> Result<Self, String> {
        if geom.n_blocks_per_window() < 1 || geom.hop_blocks() < 1 {
            return Err("window_size/hop must be multiples of blocksize".into());
        }
        let ws = geom.window_size;
        let hann = hanning(ws);
        let coh_gain = pairwise_sum(&hann);
        let win_power_corr = 2.0 / (coh_gain * coh_gain).max(1e-12);
        let mut planner = RealFftPlanner::<f64>::new();
        let plan = planner.plan_fft_forward(ws);
        let fft_in = plan.make_input_vec();
        let spectrum = plan.make_output_vec();
        let fft_scratch = plan.make_scratch_vec();
        let map = build_log_bin_map(ws, geom.sr, geom.n_bins, geom.f_min);

        let valid: Vec<usize> = (0..map.assign.len())
            .filter(|&k| map.assign[k] >= 0)
            .collect();
        let valid_start = valid.first().copied().unwrap_or(0);
        for w in valid.windows(2) {
            assert_eq!(w[1], w[0] + 1, "valid rfft bins must be contiguous");
        }
        let mut runs = Vec::new();
        let mut k = 0;
        while k < valid.len() {
            let target = map.assign[valid[k]] as usize;
            let start = k;
            while k < valid.len() && map.assign[valid[k]] as usize == target {
                k += 1;
            }
            runs.push((valid_start + start, valid_start + k, target));
        }
        let bin_count_inv = map
            .counts
            .iter()
            .map(|&c| if c > 0.0 { 1.0 / c.max(1.0) } else { 0.0 })
            .collect();
        let wire = map
            .counts
            .iter()
            .map(|&c| if c == 0.0 { EMPTY_BIN_SENTINEL } else { 0.0 })
            .collect();

        Ok(FftAnalyzer {
            geom,
            window_buf: vec![0.0; ws],
            hann,
            win_power_corr,
            plan,
            fft_in,
            spectrum,
            fft_scratch,
            power: vec![0.0; ws / 2 + 1],
            map,
            valid_start,
            runs,
            bin_count_inv,
            wire,
        })
    }

    pub fn geometry(&self) -> &FftGeometry {
        &self.geom
    }
    pub fn hann(&self) -> &[f64] {
        &self.hann
    }
    pub fn win_power_corr(&self) -> f64 {
        self.win_power_corr
    }
    pub fn bin_map(&self) -> &LogBinMap {
        &self.map
    }
    /// First valid rfft bin (diagnostics).
    pub fn valid_start(&self) -> usize {
        self.valid_start
    }

    /// The window the caller fills before `process_hop` (f32 ring samples
    /// cast to f64).
    pub fn window_mut(&mut self) -> &mut [f64] {
        &mut self.window_buf
    }

    /// One hop on the filled window. Returns the f32 wire frame (raw dB with
    /// -1000 sentinels; parity trap 5), valid until the next call.
    pub fn process_hop(&mut self) -> &[f32] {
        for ((o, &w), &h) in self.fft_in.iter_mut().zip(&self.window_buf).zip(&self.hann) {
            *o = w * h;
        }
        self.plan
            .process_with_scratch(&mut self.fft_in, &mut self.spectrum, &mut self.fft_scratch)
            .expect("rfft buffer sizes are fixed at construction");
        let corr = self.win_power_corr;
        for (p, c) in self.power.iter_mut().zip(&self.spectrum) {
            *p = (c.re * c.re + c.im * c.im) * corr;
        }
        // Each log bin's rfft bins are one run, so this visits every
        // non-empty log bin exactly once (and skips the log10 of empty ones).
        for &(start, end, target) in &self.runs {
            let v = (reduce_segment(&self.power[start..end]) * self.bin_count_inv[target]).max(EPS);
            self.wire[target] = (v.log10() * 10.0) as f32;
        }
        &self.wire
    }

    /// The last wire frame.
    pub fn wire(&self) -> &[f32] {
        &self.wire
    }
}
