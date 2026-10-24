//! Per-band RMS, asymmetric exponential smoother and rolling auto-scaler
//! (port of `server/dsp/features.py`). Scalar n=3 code, 1:1 with Python.

use crate::Bands;
use crate::math::{EPS, ln10_over_20, tau_to_alpha, tilt_db};

/// RMS of a block in f64: `sqrt(dot(x, x) / n)`.
#[inline]
pub fn block_rms(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        acc += v * v;
    }
    (acc / x.len() as f64).sqrt()
}

/// Per-band asymmetric one-pole envelope follower: attack alpha when
/// `rms - v > 0.0` (strict), release alpha otherwise.
pub struct ExpSmoother {
    sr: f64,
    blocksize: usize,
    values: [f64; 3],
    alpha_release: [f64; 3],
    alpha_attack: [f64; 3],
}

impl ExpSmoother {
    pub fn new(sr: f64, blocksize: usize, tau: [f64; 3], tau_attack: [f64; 3]) -> Self {
        let mut s = ExpSmoother {
            sr,
            blocksize,
            values: [0.0; 3],
            alpha_release: [0.0; 3],
            alpha_attack: [0.0; 3],
        };
        s.set_tau(tau, tau_attack);
        s
    }

    pub fn set_tau(&mut self, tau: [f64; 3], tau_attack: [f64; 3]) {
        let dt = self.blocksize as f64 / self.sr;
        for i in 0..3 {
            self.alpha_release[i] = tau_to_alpha(dt, tau[i]);
            self.alpha_attack[i] = tau_to_alpha(dt, tau_attack[i]);
        }
    }

    #[inline]
    pub fn update(&mut self, rms: [f64; 3]) -> [f64; 3] {
        for i in 0..3 {
            let d = rms[i] - self.values[i];
            let a = if d > 0.0 {
                self.alpha_attack[i]
            } else {
                self.alpha_release[i]
            };
            self.values[i] += a * d;
        }
        self.values
    }

    pub fn values(&self) -> [f64; 3] {
        self.values
    }

    /// `[alpha_attack, alpha_release]` (golden `derived_smoother_alphas`).
    pub fn alphas(&self) -> [[f64; 3]; 2] {
        [self.alpha_attack, self.alpha_release]
    }

    pub fn reset(&mut self) {
        self.values = [0.0; 3];
    }
}

/// Construction parameters for `AutoScaler` (Python keyword defaults).
#[derive(Clone, Copy, Debug)]
pub struct AutoScalerParams {
    pub tau_attack_s: f64,
    pub tau_release_s: f64,
    pub noise_floor: f64,
    pub strength: f64,
    pub tilt_db_per_oct: f64,
    pub bands: Bands,
    pub n_fft_window: usize,
    pub fft_n_bins: usize,
    pub fft_f_min: f64,
    pub db_floor: f64,
    pub db_ceiling: f64,
}

impl Default for AutoScalerParams {
    fn default() -> Self {
        AutoScalerParams {
            tau_attack_s: 0.05,
            tau_release_s: 60.0,
            noise_floor: 1e-3,
            strength: 1.0,
            tilt_db_per_oct: 0.0,
            bands: [(40.0, 250.0), (250.0, 2000.0), (2000.0, 16000.0)],
            n_fft_window: 1024,
            fft_n_bins: 192,
            fft_f_min: 30.0,
            db_floor: -60.0,
            db_ceiling: 0.0,
        }
    }
}

/// Rolling per-band normalizer; see the Python docstring and CLAUDE.md
/// ("L/M/H bandwidth-aware noise gate", "raw-dB baseline mirrors the FFT
/// viz") for the semantics. Two noise-floor variables on purpose (parity
/// trap 10): the gate uses raw `noise_floor` (may be 0), the divisor uses
/// `max(noise_floor, EPS)`.
pub struct AutoScaler {
    sr: f64,
    blocksize: usize,
    noise_floor: f64,
    strength: f64,
    tilt_db_per_oct: f64,
    bands: Bands,
    n_fft_window: usize,
    fft_n_bins: usize,
    fft_f_min: f64,
    band_centers: [f64; 3],
    band_widths: [f64; 3],
    log_bins_per_band: [f64; 3],
    per_bin_factor: [f64; 3],
    db_floor: f64,
    db_ceiling: f64,
    tilt_lin: [f64; 3],
    noise_pwr_per_band: [f64; 3],
    peak: [f64; 3],
    warmed: bool,
    tau_attack_s: f64,
    tau_release_s: f64,
    a_atk: f64,
    a_rel: f64,
}

impl AutoScaler {
    pub fn new(sr: f64, blocksize: usize, p: AutoScalerParams) -> Self {
        let noise_floor = p.noise_floor.max(0.0);
        let mut s = AutoScaler {
            sr,
            blocksize,
            noise_floor,
            strength: p.strength.min(1.0).max(0.0),
            tilt_db_per_oct: p.tilt_db_per_oct,
            bands: p.bands,
            n_fft_window: p.n_fft_window.max(1),
            fft_n_bins: p.fft_n_bins.max(1),
            fft_f_min: p.fft_f_min.max(1e-3),
            band_centers: [0.0; 3],
            band_widths: [0.0; 3],
            log_bins_per_band: [0.0; 3],
            per_bin_factor: [1.0; 3],
            db_floor: p.db_floor,
            db_ceiling: p.db_ceiling,
            tilt_lin: [1.0; 3],
            noise_pwr_per_band: [0.0; 3],
            peak: [noise_floor.max(EPS); 3],
            warmed: false,
            tau_attack_s: p.tau_attack_s,
            tau_release_s: p.tau_release_s,
            a_atk: 0.0,
            a_rel: 0.0,
        };
        s.recompute_band_geom();
        s.recompute_log_bin_geom();
        s.recompute_per_bin_factor();
        s.recompute_tilt_lin();
        s.recompute_noise_pwr();
        s.set_taus(p.tau_attack_s, p.tau_release_s);
        s
    }

    fn recompute_band_geom(&mut self) {
        for i in 0..3 {
            let (lo, hi) = self.bands[i];
            self.band_centers[i] = (lo.max(1e-3) * hi.max(1e-3)).sqrt();
            self.band_widths[i] = (hi - lo).max(0.0);
        }
    }

    fn recompute_tilt_lin(&mut self) {
        if self.tilt_db_per_oct == 0.0 {
            self.tilt_lin = [1.0; 3];
            return;
        }
        let k = ln10_over_20();
        for i in 0..3 {
            self.tilt_lin[i] = (tilt_db(self.band_centers[i], self.tilt_db_per_oct) * k).exp();
        }
    }

    fn recompute_per_bin_factor(&mut self) {
        let scale = self.n_fft_window as f64 / self.sr.max(1e-3);
        for i in 0..3 {
            let k_lin = (self.band_widths[i] * scale).max(1.0);
            self.per_bin_factor[i] = 1.0 / k_lin.sqrt();
        }
    }

    fn recompute_log_bin_geom(&mut self) {
        let nyq = (self.sr / 2.0).max(1e-3);
        let log_oct_total = (nyq / self.fft_f_min).max(1.0 + 1e-6).log2();
        let log_bins_per_oct = self.fft_n_bins as f64 / log_oct_total.max(1e-3);
        for i in 0..3 {
            let (lo, hi) = self.bands[i];
            let oct_band = (hi.max(1e-3) / lo.max(1e-3)).log2();
            self.log_bins_per_band[i] = log_bins_per_oct * oct_band.max(0.0);
        }
    }

    fn recompute_noise_pwr(&mut self) {
        let nf2 = self.noise_floor * self.noise_floor;
        let scale = self.n_fft_window as f64 / self.sr.max(1e-3);
        for i in 0..3 {
            let mut v = self.band_widths[i] * scale;
            v /= self.log_bins_per_band[i].max(1e-3);
            v = v.max(1.0);
            self.noise_pwr_per_band[i] = v * nf2;
        }
    }

    pub fn set_taus(&mut self, tau_attack_s: f64, tau_release_s: f64) {
        let dt = self.blocksize as f64 / self.sr;
        self.a_atk = tau_to_alpha(dt, tau_attack_s);
        self.a_rel = tau_to_alpha(dt, tau_release_s);
        self.tau_attack_s = tau_attack_s;
        self.tau_release_s = tau_release_s;
    }

    /// Also raises the existing peaks to the new floor (parity trap 11).
    pub fn set_noise_floor(&mut self, floor: f64) {
        self.noise_floor = floor.max(0.0);
        let f = self.noise_floor.max(EPS);
        for p in &mut self.peak {
            *p = p.max(f);
        }
        self.recompute_noise_pwr();
    }

    pub fn set_tilt(&mut self, tilt_db_per_oct: f64) {
        self.tilt_db_per_oct = tilt_db_per_oct;
        self.recompute_tilt_lin();
    }

    pub fn set_bands(&mut self, bands: Bands) {
        self.bands = bands;
        self.recompute_band_geom();
        self.recompute_tilt_lin();
        self.recompute_log_bin_geom();
        self.recompute_per_bin_factor();
        self.recompute_noise_pwr();
    }

    pub fn set_n_fft_window(&mut self, n: usize) {
        self.n_fft_window = n.max(1);
        self.recompute_per_bin_factor();
        self.recompute_noise_pwr();
    }

    pub fn set_fft_geometry(&mut self, n_bins: Option<usize>, f_min: Option<f64>) {
        if let Some(n) = n_bins {
            self.fft_n_bins = n.max(1);
        }
        if let Some(f) = f_min {
            self.fft_f_min = f.max(1e-3);
        }
        self.recompute_log_bin_geom();
        self.recompute_noise_pwr();
    }

    pub fn set_strength(&mut self, strength: f64) {
        self.strength = strength.min(1.0).max(0.0);
    }

    pub fn reset(&mut self) {
        self.peak = [self.noise_floor.max(EPS); 3];
        self.warmed = false;
    }

    // Derived-state accessors (goldens `derived_as_*`).
    pub fn tilt_lin(&self) -> [f64; 3] {
        self.tilt_lin
    }
    pub fn noise_pwr_per_band(&self) -> [f64; 3] {
        self.noise_pwr_per_band
    }
    pub fn per_bin_factor(&self) -> [f64; 3] {
        self.per_bin_factor
    }
    pub fn log_bins_per_band(&self) -> [f64; 3] {
        self.log_bins_per_band
    }
    pub fn peak_alphas(&self) -> [f64; 2] {
        [self.a_atk, self.a_rel]
    }
    pub fn noise_floor(&self) -> f64 {
        self.noise_floor
    }
    pub fn strength(&self) -> f64 {
        self.strength
    }

    /// One block. `values_in` = smoothed RMS; returns the scaled `[0, 1]`
    /// values (after the strength blend, before master_gain).
    pub fn update(&mut self, values_in: [f64; 3]) -> [f64; 3] {
        let nf2 = self.noise_floor; // gate threshold (may be 0)
        let nf = if nf2 > EPS { nf2 } else { EPS }; // divisor floor

        let r = values_in;
        let mut v = [0.0; 3];
        for i in 0..3 {
            let mut c = r[i] * r[i] - self.noise_pwr_per_band[i];
            if c < 0.0 {
                c = 0.0;
            }
            v[i] = c.sqrt() * self.tilt_lin[i];
        }

        if !self.warmed {
            self.peak = [nf; 3];
            self.warmed = true;
        }
        let mut sc = [0.0; 3];
        for i in 0..3 {
            let mut p = self.peak[i];
            let d = v[i] - p;
            p += (if d > 0.0 { self.a_atk } else { self.a_rel }) * d;
            self.peak[i] = p;
            let mut g = v[i] - nf2;
            if g < 0.0 {
                g = 0.0;
            }
            sc[i] = (g / (if p > nf { p } else { nf })).tanh();
        }

        let s = self.strength;
        if s >= 1.0 {
            return sc;
        }

        let mut span = self.db_ceiling - self.db_floor;
        if span < 1.0 {
            span = 1.0;
        }
        let inv_span = 1.0 / span;
        let df = self.db_floor;
        let baseline = |raw: f64, factor: f64| -> f64 {
            let mut pb = raw * factor;
            if pb < nf2 {
                return 0.0;
            }
            if pb < EPS {
                pb = EPS;
            }
            let v = (20.0 * pb.log10() - df) * inv_span;
            if v < 0.0 {
                return 0.0;
            }
            if v > 1.0 {
                return 1.0;
            }
            v
        };
        let b = [
            baseline(r[0], self.per_bin_factor[0]),
            baseline(r[1], self.per_bin_factor[1]),
            baseline(r[2], self.per_bin_factor[2]),
        ];
        if s <= 0.0 {
            b
        } else {
            let inv_s = 1.0 - s;
            [
                s * sc[0] + inv_s * b[0],
                s * sc[1] + inv_s * b[1],
                s * sc[2] + inv_s * b[2],
            ]
        }
    }
}
