//! Per-band onset detector + low-band BPM estimator (port of
//! `server/dsp/onset.py`). Comparisons are strict exactly as in Python
//! (parity trap 12); the BPM ring is a fixed array, so nothing allocates.

pub const DEFAULT_K_HIGH: f64 = 1.8;
pub const DEFAULT_MIN_IOI_S: f64 = 0.25;
pub const DEFAULT_SLOW_TAU_S: f64 = 0.30;
pub const DEFAULT_ABS_FLOOR: f64 = 0.10;

pub const K_LOW: f64 = 0.6;
pub const FAST_TAU_S: f64 = 0.012;
pub const THR_TAU_S: f64 = 1.000;

pub const BPM_MIN: f64 = 60.0;
pub const BPM_MAX: f64 = 180.0;
pub const BPM_RING: usize = 12;
pub const BPM_DECAY_AFTER_S: f64 = 5.0;
pub const BPM_SMOOTH_ALPHA: f64 = 0.3;
pub const BPM_BAND_IDX: usize = 0;

/// One band's user-facing detection parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnsetParams {
    pub sensitivity: f64,
    pub refractory_s: f64,
    pub slow_tau_s: f64,
    pub abs_floor: f64,
}

impl Default for OnsetParams {
    fn default() -> Self {
        OnsetParams {
            sensitivity: DEFAULT_K_HIGH,
            refractory_s: DEFAULT_MIN_IOI_S,
            slow_tau_s: DEFAULT_SLOW_TAU_S,
            abs_floor: DEFAULT_ABS_FLOOR,
        }
    }
}

/// Partial update for `set_params` (Python keyword args left at None).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OnsetParamsUpdate {
    pub sensitivity: Option<f64>,
    pub refractory_s: Option<f64>,
    pub slow_tau_s: Option<f64>,
    pub abs_floor: Option<f64>,
}

fn one_pole(dt: f64, tau: f64) -> f64 {
    1.0 - (-dt / tau.max(1e-3)).exp()
}

pub struct OnsetTracker {
    k_high: [f64; 3],
    min_ioi_s: [f64; 3],
    slow_tau_s: [f64; 3],
    abs_floor: [f64; 3],
    alpha_slow: [f64; 3],
    alpha_fast: f64,
    alpha_thr: f64,
    dt: f64,
    /// Monotonic per-band counters; never reset (parity trap 12).
    onset_count: [u64; 3],
    fast: [f64; 3],
    slow: [f64; 3],
    thr_avg: [f64; 3],
    armed: [bool; 3],
    t_now: f64,
    last_fire_t: [f64; 3],
    onset_times: [f64; BPM_RING],
    n_onset_times: usize,
    bpm_smoothed: f64,
}

impl OnsetTracker {
    pub fn new(sr: f64, blocksize: usize, params: [OnsetParams; 3]) -> Self {
        let mut t = OnsetTracker {
            k_high: params.map(|p| p.sensitivity),
            min_ioi_s: params.map(|p| p.refractory_s),
            slow_tau_s: params.map(|p| p.slow_tau_s),
            abs_floor: params.map(|p| p.abs_floor),
            alpha_slow: [0.0; 3],
            alpha_fast: 0.0,
            alpha_thr: 0.0,
            dt: 0.0,
            onset_count: [0; 3],
            fast: [0.0; 3],
            slow: [0.0; 3],
            thr_avg: [0.0; 3],
            armed: [false; 3],
            t_now: 0.0,
            last_fire_t: [-1e9; 3],
            onset_times: [0.0; BPM_RING],
            n_onset_times: 0,
            bpm_smoothed: 0.0,
        };
        t.configure(sr, blocksize);
        t.reset();
        t
    }

    fn configure(&mut self, sr: f64, blocksize: usize) {
        self.dt = blocksize as f64 / sr.max(1.0);
        self.alpha_fast = one_pole(self.dt, FAST_TAU_S);
        self.alpha_thr = one_pole(self.dt, THR_TAU_S);
        for i in 0..3 {
            self.alpha_slow[i] = one_pole(self.dt, self.slow_tau_s[i]);
        }
    }

    /// Device hot-switch: re-derive dt / alphas; tuned params persist.
    pub fn reconfigure(&mut self, sr: f64, blocksize: usize) {
        self.configure(sr, blocksize);
    }

    /// `band` is 0 = low, 1 = mid, 2 = high.
    pub fn set_params(&mut self, band: usize, u: OnsetParamsUpdate) {
        if let Some(v) = u.sensitivity {
            self.k_high[band] = v;
        }
        if let Some(v) = u.refractory_s {
            self.min_ioi_s[band] = v;
        }
        if let Some(v) = u.slow_tau_s {
            self.slow_tau_s[band] = v;
            self.alpha_slow[band] = one_pole(self.dt, v);
        }
        if let Some(v) = u.abs_floor {
            self.abs_floor[band] = v;
        }
    }

    pub fn reset(&mut self) {
        self.fast = [0.0; 3];
        self.slow = [0.0; 3];
        self.thr_avg = [0.0; 3];
        self.armed = [false; 3];
        self.t_now = 0.0;
        self.last_fire_t = [-1e9; 3];
        // onset_count is deliberately NOT reset.
        self.n_onset_times = 0;
        self.bpm_smoothed = 0.0;
    }

    /// Advance the clock by blocks that were dropped unprocessed.
    pub fn advance(&mut self, n_blocks: u64) {
        if n_blocks > 0 {
            self.t_now += n_blocks as f64 * self.dt;
        }
    }

    pub fn onset_count(&self) -> [u64; 3] {
        self.onset_count
    }

    /// `[alpha_fast, alpha_thr, alpha_slow × 3]` (golden `derived_onset_alphas`).
    pub fn alphas(&self) -> [f64; 5] {
        [
            self.alpha_fast,
            self.alpha_thr,
            self.alpha_slow[0],
            self.alpha_slow[1],
            self.alpha_slow[2],
        ]
    }

    /// One block of post-processed L/M/H. Returns (per-band onset pulses,
    /// smoothed BPM).
    pub fn update(&mut self, lmh: [f64; 3]) -> ([i8; 3], f64) {
        self.t_now += self.dt;
        let t = self.t_now;
        let a_fast = self.alpha_fast;
        let a_thr = self.alpha_thr;

        let mut nov = [0.0; 3];
        let mut ta = [0.0; 3];
        for i in 0..3 {
            let x = lmh[i];
            let f = self.fast[i] + a_fast * (x - self.fast[i]);
            self.fast[i] = f;
            let s = self.slow[i] + self.alpha_slow[i] * (x - self.slow[i]);
            self.slow[i] = s;
            let mut n = f - s;
            if n < 0.0 {
                n = 0.0;
            }
            nov[i] = n;
        }
        for i in 0..3 {
            let v = self.thr_avg[i] + a_thr * (nov[i] - self.thr_avg[i]);
            self.thr_avg[i] = v;
            ta[i] = v;
        }

        let mut out = [0i8; 3];
        for i in 0..3 {
            let af = self.abs_floor[i];
            let mut th = ta[i] * self.k_high[i];
            if th < af {
                th = af;
            }
            let mut tl = ta[i] * K_LOW;
            let af_lo = af * 0.5;
            if tl < af_lo {
                tl = af_lo;
            }
            if !self.armed[i] {
                if nov[i] > th && (t - self.last_fire_t[i]) > self.min_ioi_s[i] {
                    out[i] = 1;
                    self.armed[i] = true;
                    self.last_fire_t[i] = t;
                    self.onset_count[i] += 1;
                    if i == BPM_BAND_IDX {
                        self.on_low_fire(t);
                    }
                }
            } else if nov[i] < tl {
                self.armed[i] = false;
            }
        }

        if (t - self.last_fire_t[BPM_BAND_IDX]) > BPM_DECAY_AFTER_S
            && (self.bpm_smoothed != 0.0 || self.n_onset_times > 0)
        {
            self.bpm_smoothed = 0.0;
            self.n_onset_times = 0;
        }
        (out, self.bpm_smoothed)
    }

    fn on_low_fire(&mut self, t: f64) {
        if self.n_onset_times == BPM_RING {
            self.onset_times.copy_within(1.., 0);
            self.onset_times[BPM_RING - 1] = t;
        } else {
            self.onset_times[self.n_onset_times] = t;
            self.n_onset_times += 1;
        }
        let n = self.n_onset_times;
        if n < 4 {
            return;
        }
        let ot = &self.onset_times[..n];
        let mut iois = [0.0f64; BPM_RING - 1];
        let ni = n - 1;
        for i in 0..ni {
            iois[i] = ot[i + 1] - ot[i];
        }
        let iois = &mut iois[..ni];
        iois.sort_unstable_by(f64::total_cmp);
        let m = iois[ni / 2]; // upper median
        if m <= 0.0 {
            return;
        }
        let mut kept = [0.0f64; BPM_RING - 1];
        let mut nk = 0;
        for &v in iois.iter() {
            if 0.5 * m < v && v < 2.0 * m {
                kept[nk] = v;
                nk += 1;
            }
        }
        if nk < 2 {
            return;
        }
        let med = kept[nk / 2];
        if med <= 0.0 {
            return;
        }
        let mut bpm = 60.0 / med;
        while bpm < BPM_MIN {
            bpm *= 2.0;
        }
        while bpm > BPM_MAX {
            bpm *= 0.5;
        }
        if self.bpm_smoothed <= 0.0 {
            self.bpm_smoothed = bpm;
        } else {
            self.bpm_smoothed += BPM_SMOOTH_ALPHA * (bpm - self.bpm_smoothed);
        }
    }
}
