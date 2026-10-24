//! Pure DSP for the realtime audio server: a 1:1 port of the Python
//! reference in `server/dsp/`. No IO, no threads, no allocation in the
//! per-block / per-hop paths after construction. Parity against the
//! Python goldens is tested in `tests/parity.rs`.

// Pure port: `x.min(1.0).max(0.0)` mirrors Python's `max(0, min(1, x))`
// (clamp differs on NaN), and the indexed n=3 loops mirror the unrolled
// Python scalar code line by line.
#![allow(clippy::manual_clamp, clippy::needless_range_loop)]

pub mod features;
pub mod fft;
pub mod fft_post;
pub mod filters;
pub mod math;
pub mod onset;

use features::{AutoScaler, AutoScalerParams, ExpSmoother, block_rms};
use fft::{FftAnalyzer, FftGeometry};
use fft_post::{FftPostParams, FftPostProcessor, PostReconfigure};
use filters::FilterBank;
use onset::{OnsetParams, OnsetTracker};

/// `[(lo_hz, hi_hz); 3]` for low / mid / high.
pub type Bands = [(f64, f64); 3];

pub const BAND_NAMES: [&str; 3] = ["low", "mid", "high"];

/// Everything the DSP side reads from the config (the `cfg.*` values the
/// Python `App._build_pipeline_for_sr` / `_build_fft_pipeline` pass).
/// Band edges are the already-clamped values.
#[derive(Clone, Copy, Debug)]
pub struct EngineParams {
    pub sr: f64,
    pub blocksize: usize,
    pub bands: Bands,
    pub filter_order: usize,
    pub tau: [f64; 3],
    pub tau_attack: [f64; 3],
    pub autoscale_tau_attack_s: f64,
    pub autoscale_tau_release_s: f64,
    pub noise_floor: f64,
    pub strength: f64,
    pub fft_n_bins: usize,
    pub fft_window_size: usize,
    pub fft_hop: usize,
    pub fft_f_min: f64,
    pub fft_db_floor: f64,
    pub fft_db_ceiling: f64,
    pub fft_peak_smear_oct: f64,
    pub fft_tilt_db_per_oct: f64,
    pub onset: [OnsetParams; 3],
}

/// Result of one DSP block.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlockOut {
    /// Smoothed RMS (WS `*_raw`).
    pub raw: [f64; 3],
    /// AutoScaler output after the strength blend, before master_gain.
    pub scaled: [f64; 3],
    pub onsets: [i8; 3],
    pub bpm: f64,
}

/// The DSP worker's state: filter → RMS → smoother → autoscaler → onset
/// (Python `DSPWorker.process_block`).
pub struct DspChain {
    pub filter_bank: FilterBank,
    pub smoother: ExpSmoother,
    pub auto_scaler: AutoScaler,
    pub onset_tracker: OnsetTracker,
}

impl DspChain {
    pub fn new(p: &EngineParams) -> Self {
        DspChain {
            filter_bank: FilterBank::new(p.sr, p.bands, p.blocksize, p.filter_order),
            smoother: ExpSmoother::new(p.sr, p.blocksize, p.tau, p.tau_attack),
            auto_scaler: AutoScaler::new(
                p.sr,
                p.blocksize,
                AutoScalerParams {
                    tau_attack_s: p.autoscale_tau_attack_s,
                    tau_release_s: p.autoscale_tau_release_s,
                    noise_floor: p.noise_floor,
                    strength: p.strength,
                    tilt_db_per_oct: p.fft_tilt_db_per_oct,
                    bands: p.bands,
                    n_fft_window: p.fft_window_size,
                    fft_n_bins: p.fft_n_bins,
                    fft_f_min: p.fft_f_min,
                    db_floor: p.fft_db_floor,
                    db_ceiling: p.fft_db_ceiling,
                },
            ),
            onset_tracker: OnsetTracker::new(p.sr, p.blocksize, p.onset),
        }
    }

    /// One audio block. Allocation-free.
    #[inline]
    pub fn process_block(&mut self, x: &[f32]) -> BlockOut {
        let [lo, md, hi] = self.filter_bank.process(x);
        let rms = [block_rms(lo), block_rms(md), block_rms(hi)];
        let raw = self.smoother.update(rms);
        let scaled = self.auto_scaler.update(raw);
        let (onsets, bpm) = self.onset_tracker.update(scaled);
        BlockOut {
            raw,
            scaled,
            onsets,
            bpm,
        }
    }
}

/// The FFT worker's state: analyzer + post-processor (Python
/// `FFTWorker.process_hop` with its `post_processor`).
pub struct FftChain {
    pub analyzer: FftAnalyzer,
    pub post: FftPostProcessor,
}

impl FftChain {
    pub fn new(p: &EngineParams) -> Result<Self, String> {
        let geom = FftGeometry {
            sr: p.sr,
            blocksize: p.blocksize,
            window_size: p.fft_window_size,
            hop: p.fft_hop,
            n_bins: p.fft_n_bins,
            f_min: p.fft_f_min,
        };
        Ok(FftChain {
            analyzer: FftAnalyzer::new(geom)?,
            post: FftPostProcessor::new(FftPostParams {
                n_bins: p.fft_n_bins,
                f_min: p.fft_f_min,
                sr: p.sr,
                bands: p.bands,
                tau: p.tau,
                tau_attack: p.tau_attack,
                tau_release_s: p.autoscale_tau_release_s,
                noise_floor: p.noise_floor,
                strength: p.strength,
                db_floor: p.fft_db_floor,
                db_ceiling: p.fft_db_ceiling,
                hop_period_s: p.fft_hop as f64 / p.sr,
                tau_attack_s: p.autoscale_tau_attack_s,
                peak_smear_oct: p.fft_peak_smear_oct,
                tilt_db_per_oct: p.fft_tilt_db_per_oct,
            }),
        })
    }

    /// Python `FFTWorker.reconfigure(...)`: rebuild the analyzer for the new
    /// geometry and mirror `(n_bins, sr, f_min, hop_period_s)` into the
    /// post-processor (which reallocates and re-warms).
    pub fn reconfigure(
        &mut self,
        n_bins: Option<usize>,
        sr: Option<f64>,
        window_size: Option<usize>,
        hop: Option<usize>,
        f_min: Option<f64>,
    ) -> Result<(), String> {
        let mut g = *self.analyzer.geometry();
        if let Some(v) = n_bins {
            g.n_bins = v;
        }
        if let Some(v) = sr {
            g.sr = v;
        }
        if let Some(v) = window_size {
            g.window_size = v;
        }
        if let Some(v) = hop {
            g.hop = v;
        }
        if let Some(v) = f_min {
            g.f_min = v;
        }
        self.analyzer = FftAnalyzer::new(g)?;
        self.post.reconfigure(PostReconfigure {
            n_bins: Some(g.n_bins),
            sr: Some(g.sr),
            f_min: Some(g.f_min),
            hop_period_s: Some(g.hop_period_s()),
            ..Default::default()
        });
        Ok(())
    }

    /// The window to fill (f64) before `process_hop`.
    pub fn window_mut(&mut self) -> &mut [f64] {
        self.analyzer.window_mut()
    }

    /// One hop: returns `(raw_db_wire, processed)`, both valid until the
    /// next call. Allocation-free (after a layout's first frame).
    pub fn process_hop(&mut self) -> (&[f32], &[f32]) {
        let wire = self.analyzer.process_hop();
        let processed = self.post.process(wire);
        (wire, processed)
    }
}
