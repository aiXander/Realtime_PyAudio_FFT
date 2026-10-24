//! Per-bin port of the AutoScaler for the FFT log bins (port of
//! `server/dsp/fft_postprocess.py`). `process` and the `update_*` setters
//! don't allocate, except `update_smear` (the kernel length varies):
//! realtime callers build that with `Smear::build` on the control thread
//! and `install_smear` it. `reconfigure` reallocates (control thread).

use crate::Bands;
use crate::math::{
    EPS, interp1, ln10_over_20, log_bin_center, pairwise_sum, tau_to_alpha, tau_to_alpha_into,
    tilt_db,
};

/// Construction parameters (Python keyword arguments of `FFTPostProcessor`).
#[derive(Clone, Copy, Debug)]
pub struct FftPostParams {
    pub n_bins: usize,
    pub f_min: f64,
    pub sr: f64,
    pub bands: Bands,
    pub tau: [f64; 3],
    pub tau_attack: [f64; 3],
    pub tau_release_s: f64,
    pub noise_floor: f64,
    pub strength: f64,
    pub db_floor: f64,
    pub db_ceiling: f64,
    pub hop_period_s: f64,
    pub tau_attack_s: f64,
    pub peak_smear_oct: f64,
    pub tilt_db_per_oct: f64,
}

/// Partial update for `update_autoscale` (None = unchanged).
#[derive(Clone, Copy, Debug, Default)]
pub struct AutoscaleUpdate {
    pub tau_attack_s: Option<f64>,
    pub tau_release_s: Option<f64>,
    pub noise_floor: Option<f64>,
    pub strength: Option<f64>,
}

/// Partial update for `reconfigure` (None = unchanged).
#[derive(Clone, Copy, Debug, Default)]
pub struct PostReconfigure {
    pub n_bins: Option<usize>,
    pub f_min: Option<f64>,
    pub sr: Option<f64>,
    pub bands: Option<Bands>,
    pub tau: Option<[f64; 3]>,
    pub tau_attack: Option<[f64; 3]>,
    pub hop_period_s: Option<f64>,
    pub db_floor: Option<f64>,
    pub db_ceiling: Option<f64>,
}

/// Sentinel-fill LUT (parity trap 6): built from the first frame after
/// (re)allocation, survives `reset()`.
struct SentinelLut {
    empty_idx: Vec<usize>,
    left_idx: Vec<usize>,
    right_idx: Vec<usize>,
    w_left: Vec<f64>,
    w_right: Vec<f64>,
}

/// Gaussian peak smear: kernel + the reflect-mode padding index table.
/// Built off the hot path (`Smear::build`), installed with `install_smear`.
pub struct Smear {
    kernel: Vec<f64>,
    radius: usize,
    /// `padded[i] = peak[pad_idx[i]]`, length n + 2·radius.
    pad_idx: Vec<usize>,
    padded: Vec<f64>,
    sigma_bins: f64,
}

impl Smear {
    /// `_recompute_smear`: σ = `peak_smear_oct × bins_per_octave` (bin
    /// units), radius `max(1, int(4σ + 0.5))`, normalised Gaussian. `None`
    /// when the smear is off (`peak_smear_oct ≤ 0`, `f_max ≤ f_min`, σ ≤ 0).
    /// Returns `(smear, sigma_bins)`; σ is reported even when it is ≤ 0.
    pub fn build(
        n_bins: usize,
        f_min: f64,
        sr: f64,
        peak_smear_oct: f64,
    ) -> (Option<Box<Smear>>, f64) {
        let peak_smear_oct = peak_smear_oct.max(0.0);
        if peak_smear_oct <= 0.0 {
            return (None, 0.0);
        }
        let f_max = sr / 2.0;
        if f_max <= f_min {
            return (None, 0.0);
        }
        let bins_per_octave = n_bins as f64 / (f_max / f_min).log2().max(1e-6);
        let sigma = peak_smear_oct * bins_per_octave;
        if sigma <= 0.0 {
            return (None, sigma);
        }
        let radius = ((4.0 * sigma + 0.5) as i64).max(1) as usize;
        let mut k: Vec<f64> = (0..2 * radius + 1)
            .map(|i| {
                let x = i as f64 - radius as f64;
                let q = x / sigma;
                (-0.5 * (q * q)).exp()
            })
            .collect();
        let sum = pairwise_sum(&k);
        for v in &mut k {
            *v /= sum;
        }
        let pad_idx = (0..n_bins + 2 * radius)
            .map(|i| reflect_index(i as i64 - radius as i64, n_bins))
            .collect();
        let smear = Smear {
            kernel: k,
            radius,
            pad_idx,
            padded: vec![0.0; n_bins + 2 * radius],
            sigma_bins: sigma,
        };
        (Some(Box::new(smear)), sigma)
    }
}

pub struct FftPostProcessor {
    n_bins: usize,
    f_min: f64,
    sr: f64,
    bands: Bands,
    tau: [f64; 3],
    tau_attack: [f64; 3],
    tau_release_s: f64,
    noise_floor: f64,
    strength: f64,
    db_floor: f64,
    db_ceiling: f64,
    hop_period_s: f64,
    tau_attack_s: f64,
    peak_smear_oct: f64,
    tilt_db_per_oct: f64,

    smooth_lin: Vec<f64>,
    peak_lin: Vec<f64>,
    peak_smoothed: Vec<f64>,
    interp_db: Vec<f64>,
    out: Vec<f32>,
    lin: Vec<f64>,
    diff: Vec<f64>,
    alpha: Vec<f64>,
    scratch: Vec<f64>,
    lut: Option<SentinelLut>,
    tau_per_bin: Vec<f64>,
    tau_attack_per_bin: Vec<f64>,
    alpha_smooth_release: Vec<f64>,
    alpha_smooth_attack: Vec<f64>,
    alpha_attack: f64,
    alpha_release: f64,
    tilt_db: Vec<f64>,
    smear: Option<Box<Smear>>,
    smear_sigma_bins: f64,
    warmed: bool,
}

/// `scipy.ndimage` "reflect" (half-sample symmetric, `d c b a | a b c d |
/// d c b a`), repeated for offsets past the edge (parity trap 8).
fn reflect_index(i: i64, n: usize) -> usize {
    let n2 = 2 * n as i64;
    let m = i.rem_euclid(n2);
    if m < n as i64 {
        m as usize
    } else {
        (n2 - 1 - m) as usize
    }
}

impl FftPostProcessor {
    pub fn new(p: FftPostParams) -> Self {
        let mut s = FftPostProcessor {
            n_bins: p.n_bins,
            f_min: p.f_min,
            sr: p.sr,
            bands: p.bands,
            tau: p.tau,
            tau_attack: p.tau_attack,
            tau_release_s: p.tau_release_s,
            noise_floor: p.noise_floor.max(0.0),
            strength: p.strength.min(1.0).max(0.0),
            db_floor: p.db_floor,
            db_ceiling: p.db_ceiling,
            hop_period_s: p.hop_period_s,
            tau_attack_s: p.tau_attack_s,
            peak_smear_oct: p.peak_smear_oct.max(0.0),
            tilt_db_per_oct: p.tilt_db_per_oct,
            smooth_lin: Vec::new(),
            peak_lin: Vec::new(),
            peak_smoothed: Vec::new(),
            interp_db: Vec::new(),
            out: Vec::new(),
            lin: Vec::new(),
            diff: Vec::new(),
            alpha: Vec::new(),
            scratch: Vec::new(),
            lut: None,
            tau_per_bin: Vec::new(),
            tau_attack_per_bin: Vec::new(),
            alpha_smooth_release: Vec::new(),
            alpha_smooth_attack: Vec::new(),
            alpha_attack: 0.0,
            alpha_release: 0.0,
            tilt_db: Vec::new(),
            smear: None,
            smear_sigma_bins: 0.0,
            warmed: false,
        };
        s.allocate();
        s
    }

    fn allocate(&mut self) {
        let n = self.n_bins;
        let floor = self.noise_floor.max(EPS);
        self.smooth_lin = vec![0.0; n];
        self.peak_lin = vec![floor; n];
        self.peak_smoothed = vec![floor; n];
        self.interp_db = vec![0.0; n];
        self.out = vec![0.0; n];
        self.lin = vec![0.0; n];
        self.diff = vec![0.0; n];
        self.alpha = vec![0.0; n];
        self.scratch = vec![0.0; n];
        self.lut = None;
        self.tau_per_bin = vec![0.0; n];
        self.tau_attack_per_bin = vec![0.0; n];
        self.fill_per_bin_taus();
        self.alpha_smooth_release = vec![0.0; n];
        self.alpha_smooth_attack = vec![0.0; n];
        self.tilt_db = vec![0.0; n];
        self.fill_tilt_db();
        self.recompute_alphas();
        let (smear, sigma) = Smear::build(n, self.f_min, self.sr, self.peak_smear_oct);
        self.smear = smear;
        self.smear_sigma_bins = sigma;
        self.warmed = false;
    }

    /// `_build_tilt_db`, in place.
    fn fill_tilt_db(&mut self) {
        let n = self.n_bins;
        for (k, t) in self.tilt_db.iter_mut().enumerate() {
            let f_center = log_bin_center(k, n, self.f_min, self.sr / 2.0);
            *t = tilt_db(f_center, self.tilt_db_per_oct);
        }
    }

    fn build_sentinel_interp_lut(&self, db_in: &[f32]) -> SentinelLut {
        let mut valid_idx = Vec::new();
        let mut empty_idx = Vec::new();
        for (k, &v) in db_in.iter().enumerate() {
            if v >= -500.0 {
                valid_idx.push(k);
            } else {
                empty_idx.push(k);
            }
        }
        if empty_idx.is_empty() || valid_idx.is_empty() {
            return SentinelLut {
                empty_idx: Vec::new(),
                left_idx: Vec::new(),
                right_idx: Vec::new(),
                w_left: Vec::new(),
                w_right: Vec::new(),
            };
        }
        let nv = valid_idx.len();
        let mut left_idx = Vec::with_capacity(empty_idx.len());
        let mut right_idx = Vec::with_capacity(empty_idx.len());
        let mut w_left = Vec::with_capacity(empty_idx.len());
        let mut w_right = Vec::with_capacity(empty_idx.len());
        for &e in &empty_idx {
            // searchsorted(valid_idx, e, side="left")
            let right_pos = valid_idx.partition_point(|&v| v < e);
            let has_left = right_pos >= 1;
            let has_right = right_pos < nv;
            let l = valid_idx[if has_left { right_pos - 1 } else { 0 }];
            let r = valid_idx[if has_right { right_pos } else { nv - 1 }];
            let span = r as f64 - l as f64;
            let safe_span = if span > 0.0 { span } else { 1.0 };
            let w_default = (r as i64 - e as i64) as f64 / safe_span;
            let wl = if has_left && has_right {
                w_default
            } else if has_left {
                1.0
            } else {
                0.0
            };
            left_idx.push(l);
            right_idx.push(r);
            w_left.push(wl);
            w_right.push(1.0 - wl);
        }
        SentinelLut {
            empty_idx,
            left_idx,
            right_idx,
            w_left,
            w_right,
        }
    }

    /// `_build_per_bin_tau_for` for both tau sets, in place: `np.interp` of
    /// log10(bin centre) over the band-centre anchors sorted by log-centre
    /// (stable, like `list.sort`), clamped outside.
    fn fill_per_bin_taus(&mut self) {
        let n = self.n_bins;
        for (tau, out) in [
            (self.tau, &mut self.tau_per_bin),
            (self.tau_attack, &mut self.tau_attack_per_bin),
        ] {
            let mut anchors = [(0.0f64, 0.0f64); 3];
            for i in 0..3 {
                let (lo, hi) = self.bands[i];
                let cen = (lo * hi).sqrt();
                anchors[i] = (cen.max(1e-3).log10(), tau[i]);
            }
            anchors.sort_by(|a, b| a.0.total_cmp(&b.0));
            let xp = anchors.map(|a| a.0);
            let fp = anchors.map(|a| a.1);
            for (k, o) in out.iter_mut().enumerate() {
                let log_f = log_bin_center(k, n, self.f_min, self.sr / 2.0).log10();
                *o = interp1(log_f, &xp, &fp);
            }
        }
    }

    fn recompute_alphas(&mut self) {
        let h = self.hop_period_s;
        tau_to_alpha_into(h, &self.tau_per_bin, &mut self.alpha_smooth_release);
        tau_to_alpha_into(h, &self.tau_attack_per_bin, &mut self.alpha_smooth_attack);
        self.alpha_attack = tau_to_alpha(h, self.tau_attack_s);
        self.alpha_release = tau_to_alpha(h, self.tau_release_s);
    }

    // ----------------- live retune (control thread) -----------------

    /// Resets smooth + peak + warm-up only; `peak_smoothed` and the LUT
    /// survive (parity trap 22).
    pub fn reset(&mut self) {
        self.smooth_lin.fill(0.0);
        self.peak_lin.fill(self.noise_floor.max(EPS));
        self.warmed = false;
    }

    pub fn reconfigure(&mut self, r: PostReconfigure) {
        if let Some(v) = r.n_bins {
            self.n_bins = v;
        }
        if let Some(v) = r.f_min {
            self.f_min = v;
        }
        if let Some(v) = r.sr {
            self.sr = v;
        }
        if let Some(v) = r.bands {
            self.bands = v;
        }
        if let Some(v) = r.tau {
            self.tau = v;
        }
        if let Some(v) = r.tau_attack {
            self.tau_attack = v;
        }
        if let Some(v) = r.hop_period_s {
            self.hop_period_s = v;
        }
        if let Some(v) = r.db_floor {
            self.db_floor = v;
        }
        if let Some(v) = r.db_ceiling {
            self.db_ceiling = v;
        }
        self.allocate();
    }

    pub fn update_smoothing(&mut self, tau: [f64; 3], tau_attack: [f64; 3]) {
        self.tau = tau;
        self.tau_attack = tau_attack;
        self.fill_per_bin_taus();
        self.recompute_alphas();
    }

    /// Re-anchors the per-bin taus only (parity trap 11).
    pub fn update_bands(&mut self, bands: Bands) {
        self.bands = bands;
        self.fill_per_bin_taus();
        self.recompute_alphas();
    }

    /// Allocates (builds the kernel); see `install_smear` for the
    /// realtime path.
    pub fn update_smear(&mut self, peak_smear_oct: f64) {
        let (smear, sigma) = Smear::build(self.n_bins, self.f_min, self.sr, peak_smear_oct);
        drop(self.install_smear(peak_smear_oct, smear, sigma));
    }

    /// Smear build parameters for `Smear::build` on another thread:
    /// `(n_bins, f_min, sr)`.
    pub fn smear_geometry(&self) -> (usize, f64, f64) {
        (self.n_bins, self.f_min, self.sr)
    }

    /// Install a smear built by `Smear::build` with this processor's
    /// `smear_geometry()`. Returns the previous one, for the caller to drop
    /// off the realtime thread. No allocation.
    pub fn install_smear(
        &mut self,
        peak_smear_oct: f64,
        smear: Option<Box<Smear>>,
        sigma_bins: f64,
    ) -> Option<Box<Smear>> {
        if let Some(s) = &smear {
            assert_eq!(
                s.pad_idx.len(),
                self.n_bins + 2 * s.radius,
                "smear built for another n_bins"
            );
            debug_assert_eq!(s.sigma_bins, sigma_bins);
        }
        self.peak_smear_oct = peak_smear_oct.max(0.0);
        self.smear_sigma_bins = sigma_bins;
        std::mem::replace(&mut self.smear, smear)
    }

    pub fn update_tilt(&mut self, tilt_db_per_oct: f64) {
        self.tilt_db_per_oct = tilt_db_per_oct;
        self.fill_tilt_db();
    }

    pub fn update_autoscale(&mut self, u: AutoscaleUpdate) {
        if let Some(v) = u.tau_attack_s {
            self.tau_attack_s = v;
        }
        if let Some(v) = u.tau_release_s {
            self.tau_release_s = v;
        }
        if let Some(v) = u.noise_floor {
            self.noise_floor = v.max(0.0);
            let f = self.noise_floor.max(EPS);
            for p in &mut self.peak_lin {
                *p = p.max(f);
            }
        }
        if let Some(v) = u.strength {
            self.strength = v.min(1.0).max(0.0);
        }
        self.recompute_alphas();
    }

    // Derived-state accessors (goldens `derived_post_*`).
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
    pub fn tilt_db(&self) -> &[f64] {
        &self.tilt_db
    }
    pub fn tau_per_bin(&self) -> &[f64] {
        &self.tau_per_bin
    }
    pub fn tau_attack_per_bin(&self) -> &[f64] {
        &self.tau_attack_per_bin
    }
    pub fn alpha_smooth_release(&self) -> &[f64] {
        &self.alpha_smooth_release
    }
    pub fn alpha_smooth_attack(&self) -> &[f64] {
        &self.alpha_smooth_attack
    }
    pub fn peak_alphas(&self) -> [f64; 2] {
        [self.alpha_attack, self.alpha_release]
    }
    pub fn smear_kernel(&self) -> &[f64] {
        self.smear
            .as_ref()
            .map(|s| s.kernel.as_slice())
            .unwrap_or(&[])
    }
    pub fn smear_sigma_bins(&self) -> f64 {
        self.smear_sigma_bins
    }

    // ----------------- hot path (FFT worker) -----------------

    /// Run the pipeline on one wire frame (raw dB, -1000 sentinels). Returns
    /// the `[0, 1]` processed frame, valid until the next call. No
    /// allocation after the first frame of a layout (which builds the LUT).
    pub fn process(&mut self, db_in: &[f32]) -> &[f32] {
        assert_eq!(db_in.len(), self.n_bins, "frame length != n_bins");
        let n = self.n_bins;

        // 1. Sentinel interpolation.
        for (d, &s) in self.interp_db.iter_mut().zip(db_in) {
            *d = s as f64;
        }
        if self.lut.is_none() {
            self.lut = Some(self.build_sentinel_interp_lut(db_in));
        }
        let lut = self.lut.as_ref().expect("built above");
        if !lut.empty_idx.is_empty() {
            // All takes happen before the scatter (indices are valid bins only).
            for k in 0..lut.empty_idx.len() {
                let l = self.interp_db[lut.left_idx[k]] * lut.w_left[k];
                let r = self.interp_db[lut.right_idx[k]] * lut.w_right[k];
                self.scratch[k] = l + r;
            }
            for k in 0..lut.empty_idx.len() {
                self.interp_db[lut.empty_idx[k]] = self.scratch[k];
            }
        }

        // 2. Pre-tilt (dB) + dB → linear.
        let k20 = ln10_over_20();
        for i in 0..n {
            self.lin[i] = ((self.interp_db[i] + self.tilt_db[i]) * k20).exp();
        }

        // 3. Per-bin asymmetric EMA smoother.
        if !self.warmed {
            self.smooth_lin.copy_from_slice(&self.lin);
            self.peak_lin.fill(self.noise_floor.max(EPS));
            self.warmed = true;
        } else {
            for i in 0..n {
                let d = self.lin[i] - self.smooth_lin[i];
                let a = if d > 0.0 {
                    self.alpha_smooth_attack[i]
                } else {
                    self.alpha_smooth_release[i]
                };
                self.smooth_lin[i] += d * a;
            }
        }

        // 4. Asymmetric peak follower (still runs on the warm-up frame; trap 9).
        for i in 0..n {
            let d = self.smooth_lin[i] - self.peak_lin[i];
            let a = if d > 0.0 {
                self.alpha_attack
            } else {
                self.alpha_release
            };
            self.peak_lin[i] += d * a;
        }

        // 4b. Spatial Gaussian smear (scipy correlate1d symmetric-kernel path).
        if let Some(sm) = self.smear.as_mut() {
            for (p, &j) in sm.padded.iter_mut().zip(&sm.pad_idx) {
                *p = self.peak_lin[j];
            }
            // Per output: acc = p[c]·w[r], then += (p[c-j] + p[c+j])·w[r-j] for
            // j = r, r-1, …, 1 (outermost first), exactly scipy's order. The
            // loops are interchanged (offset outer, output inner) so the
            // inner loop vectorises; each output's summation order is the
            // same, so the result is bit-identical.
            let r = sm.radius;
            let fw = &sm.kernel;
            let p = &sm.padded;
            let out = &mut self.peak_smoothed[..n];
            for (o, &c) in out.iter_mut().zip(&p[r..r + n]) {
                *o = c * fw[r];
            }
            for j in (1..=r).rev() {
                let w = fw[r - j];
                let left = &p[r - j..r - j + n];
                let right = &p[r + j..r + j + n];
                for ((o, &a), &b) in out.iter_mut().zip(left).zip(right) {
                    *o += (a + b) * w;
                }
            }
        } else {
            self.peak_smoothed.copy_from_slice(&self.peak_lin);
        }

        // 5. tanh( max(0, smooth - nf) / max(peak, nf) ).
        let nf = self.noise_floor.max(EPS);
        for i in 0..n {
            let mut g = self.smooth_lin[i] - nf;
            if g < 0.0 {
                g = 0.0;
            }
            let den = self.peak_smoothed[i].max(nf);
            self.diff[i] = (g / den).tanh();
        }

        // 6. Strength blend.
        let s = self.strength;
        if s >= 1.0 {
            for (o, &d) in self.out.iter_mut().zip(&self.diff) {
                *o = d as f32;
            }
            return &self.out;
        }
        let disp_floor = self.db_floor;
        let disp_span = (self.db_ceiling - disp_floor).max(1.0);
        let inv_span = 1.0 / disp_span;
        let noise_floor_db = 20.0 * nf.log10();
        let inv_s = 1.0 - s;
        for i in 0..n {
            let x = self.interp_db[i];
            let mut b = ((x - disp_floor) * inv_span).clamp(0.0, 1.0);
            if x < noise_floor_db {
                b = 0.0;
            }
            self.out[i] = (self.diff[i] * s + b * inv_s) as f32;
        }
        &self.out
    }
}
