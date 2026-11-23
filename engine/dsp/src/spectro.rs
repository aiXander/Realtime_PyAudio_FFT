//! Multi-resolution spectrogram for *looking at* sound (the UI's 3D FFT
//! view). Not part of the Python reference and not parity-tested: nothing
//! downstream (OSC, L/M/H, LEDs) reads it.
//!
//! One STFT can't resolve both a bass note and a hi-hat: a long window
//! smears time, a short one smears frequency. So every hop runs a ladder of
//! Hann windows (×2 sizes, 512 … 8192 samples at 48 kHz) over the same
//! instant, and each log bin reads the window whose resolution suits its
//! frequency, crossfading continuously between neighbours in log-frequency
//! (no seams). The window wanted at f is `sr / (f · rel_res)`: the rfft bin
//! spacing is a fixed fraction `rel_res` of f, i.e. constant Q, clamped to
//! the ladder at both ends.
//!
//! - **Aligned in time.** All windows are centred on the middle of the
//!   longest one, so an onset lines up vertically across frequency (the
//!   price is half the longest window of latency, ~85 ms at 48 kHz).
//! - **Amplitude-normalised.** A sine reads its dBFS level (0 dB = full
//!   scale) whatever window it lands in. With constant-Q windows that also
//!   makes pink noise (≈ music's long-term spectrum) read flat.
//! - **Zero-padded 2×**, so a log bin narrower than an rfft bin interpolates
//!   between half-spaced bins. A log bin wider than an rfft bin takes the
//!   max over the bins it covers: narrow tonal peaks survive.
//! - **No smoothing, no autoscale.** Raw dB per frame; display mapping is the
//!   viewer's job.
//!
//! `process` doesn't allocate; `new` does (it runs on a non-realtime thread).

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

/// Floor of the dB output (also the value of silence).
pub const DB_FLOOR: f32 = -140.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectroParams {
    pub sr: f64,
    pub n_bins: usize,
    pub f_min: f64,
    pub f_max: f64,
    /// Wanted rfft bin spacing as a fraction of the frequency (smaller =
    /// sharper pitch, blurrier time). Hann's main lobe spans ~2 bins, so
    /// two tones resolve when they're ≳ 2.5·rel_res apart: 0.02 splits a
    /// semitone (5.9 %), 0.045 the low harmonics of a note.
    pub rel_res: f64,
    /// Largest window the input can supply, samples (a power of two).
    pub max_window: usize,
}

struct Win {
    size: usize,
    hann: Vec<f32>,
    /// Amplitude normalisation, squared: (2 / Σw)².
    norm2: f32,
    plan: Arc<dyn RealToComplex<f32>>,
    input: Vec<f32>,
    spec: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    mag2: Vec<f32>,
}

/// How one log bin reads one window's (zero-padded) spectrum.
#[derive(Clone, Copy)]
struct Tap {
    win: u8,
    /// Max over `mag2[lo..=hi]`, or (when `interp`) lerp `mag2[lo]` → `mag2[lo + 1]` by `frac`.
    lo: u32,
    hi: u32,
    interp: bool,
    frac: f32,
}

#[derive(Clone, Copy)]
struct BinPlan {
    a: Tap,
    b: Tap,
    /// dB = (1 − t)·dB_a + t·dB_b.
    t: f32,
}

pub struct Spectro {
    params: SpectroParams,
    /// Samples each `process` call needs (the longest window).
    span: usize,
    wins: Vec<Win>,
    bins: Vec<BinPlan>,
    db: Vec<f32>,
}

/// Smallest ladder window: the power of two nearest sr / 93.75 (512 at 44.1–48 kHz).
fn min_window(sr: f64) -> usize {
    let w = sr / 93.75;
    let p = w.log2().round().clamp(6.0, 14.0);
    1usize << (p as u32)
}

impl Spectro {
    pub fn new(p: SpectroParams) -> Self {
        let sr = p.sr.max(1000.0);
        let n_bins = p.n_bins.max(2);
        let f_max = p.f_max.min(sr / 2.0);
        let f_min = p.f_min.clamp(1.0, f_max / 2.0);
        let w_min = min_window(sr).min(p.max_window);
        let w_max = (w_min * 16).min(p.max_window.max(w_min));
        let mut planner = RealFftPlanner::<f32>::new();
        let mut wins = Vec::new();
        let mut size = w_min;
        while size <= w_max {
            let nfft = 2 * size;
            let plan = planner.plan_fft_forward(nfft);
            let hann: Vec<f32> = (0..size)
                .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * (i as f64 + 0.5) / size as f64).cos()) as f32)
                .collect();
            let sum: f64 = hann.iter().map(|&w| w as f64).sum();
            let norm = (2.0 / sum) as f32;
            wins.push(Win {
                size,
                hann,
                norm2: norm * norm,
                input: plan.make_input_vec(),
                spec: plan.make_output_vec(),
                scratch: plan.make_scratch_vec(),
                mag2: vec![0.0; nfft / 2 + 1],
                plan,
            });
            size *= 2;
        }
        let k_max = (wins.len() - 1) as f64;
        let log_span = (f_max / f_min).log2();
        let edge = |x: f64| f_min * 2f64.powf(x / n_bins as f64 * log_span);
        let tap = |w: usize, wins: &[Win], f_lo: f64, f_c: f64, f_hi: f64| -> Tap {
            let bin_hz = sr / (2 * wins[w].size) as f64; // zero-padded spacing
            let last = (wins[w].mag2.len() - 1) as f64;
            let lo = (f_lo / bin_hz).ceil();
            let hi = (f_hi / bin_hz).floor().min(last);
            if hi >= lo {
                Tap { win: w as u8, lo: lo as u32, hi: hi as u32, interp: false, frac: 0.0 }
            } else {
                let x = (f_c / bin_hz).clamp(0.0, last - 1.0);
                let i = x.floor();
                Tap { win: w as u8, lo: i as u32, hi: i as u32, interp: true, frac: (x - i) as f32 }
            }
        };
        let bins = (0..n_bins)
            .map(|j| {
                let (f_lo, f_hi) = (edge(j as f64), edge(j as f64 + 1.0));
                let f_c = (f_lo * f_hi).sqrt();
                // Continuous ladder position of the wanted window.
                let wanted = sr / (f_c * p.rel_res.max(1e-4));
                let k = (wanted / w_min as f64).log2().clamp(0.0, k_max);
                let a = (k.floor() as usize).min(wins.len().saturating_sub(2));
                let b = (a + 1).min(wins.len() - 1);
                let t = if a == b { 0.0 } else { (k - a as f64) as f32 };
                BinPlan { a: tap(a, &wins, f_lo, f_c, f_hi), b: tap(b, &wins, f_lo, f_c, f_hi), t }
            })
            .collect();
        Spectro {
            params: SpectroParams { sr, n_bins, f_min, f_max, ..p },
            span: w_max,
            wins,
            bins,
            db: vec![DB_FLOOR; n_bins],
        }
    }

    pub fn params(&self) -> &SpectroParams {
        &self.params
    }

    /// Samples `process` reads.
    pub fn span(&self) -> usize {
        self.span
    }

    /// Window sizes in the ladder, ascending.
    pub fn window_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        self.wins.iter().map(|w| w.size)
    }

    /// One frame from the newest `span()` samples (oldest first). Returns
    /// `n_bins` dB values (amplitude dBFS), ≥ `DB_FLOOR`.
    pub fn process(&mut self, samples: &[f64]) -> &[f32] {
        let span = self.span;
        let x = &samples[samples.len().saturating_sub(span)..];
        for w in &mut self.wins {
            // Centred on the middle of the longest window.
            let off = (span - w.size) / 2;
            let src = &x[off..off + w.size];
            for ((d, &s), &h) in w.input[..w.size].iter_mut().zip(src).zip(&w.hann) {
                *d = s as f32 * h;
            }
            w.input[w.size..].fill(0.0);
            if w.plan.process_with_scratch(&mut w.input, &mut w.spec, &mut w.scratch).is_err() {
                w.mag2.fill(0.0);
                continue;
            }
            for (m, c) in w.mag2.iter_mut().zip(&w.spec) {
                *m = c.norm_sqr() * w.norm2;
            }
        }
        let read = |wins: &[Win], tp: &Tap| -> f32 {
            let m = &wins[tp.win as usize].mag2;
            let p = if tp.interp {
                let (a, b) = (m[tp.lo as usize], m[tp.lo as usize + 1]);
                a + (b - a) * tp.frac
            } else {
                m[tp.lo as usize..=tp.hi as usize].iter().fold(0.0f32, |acc, &v| acc.max(v))
            };
            (10.0 * (p + 1e-20).log10()).max(DB_FLOOR)
        };
        for (out, bp) in self.db.iter_mut().zip(&self.bins) {
            let a = read(&self.wins, &bp.a);
            *out = if bp.t == 0.0 { a } else { a + (read(&self.wins, &bp.b) - a) * bp.t };
        }
        &self.db
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SpectroParams {
        SpectroParams { sr: 48_000.0, n_bins: 1024, f_min: 30.0, f_max: 20_000.0, rel_res: 0.047, max_window: 16384 }
    }

    fn bin_of(p: &SpectroParams, f: f64) -> usize {
        ((f / p.f_min).log2() / (p.f_max / p.f_min).log2() * p.n_bins as f64) as usize
    }

    /// A full-scale sine reads ≈ 0 dBFS at its own frequency, in every part
    /// of the ladder, and the peak is narrow (constant Q).
    #[test]
    fn sine_level_and_peak() {
        let p = params();
        let mut s = Spectro::new(p);
        assert_eq!(s.window_sizes().collect::<Vec<_>>(), [512, 1024, 2048, 4096, 8192]);
        for f in [55.0, 220.0, 1000.0, 3000.0, 12_000.0] {
            let x: Vec<f64> = (0..s.span()).map(|i| (std::f64::consts::TAU * f * i as f64 / p.sr).sin()).collect();
            let db = s.process(&x).to_vec();
            let j = bin_of(&p, f);
            let peak = db[j.saturating_sub(3)..(j + 4).min(p.n_bins)].iter().cloned().fold(f32::MIN, f32::max);
            assert!((-2.0..=0.5).contains(&peak), "{f} Hz: peak {peak} dB");
            // Half an octave away it's far down.
            let off = bin_of(&p, f * 1.41).min(p.n_bins - 1);
            assert!(db[off] < peak - 30.0, "{f} Hz: {} dB half an octave up", db[off]);
        }
    }

    /// In pitch mode, two notes a semitone apart at 440 Hz resolve into two peaks.
    #[test]
    fn resolves_a_semitone_at_440() {
        let p = SpectroParams { rel_res: 0.02, ..params() };
        let mut s = Spectro::new(p);
        let (f1, f2) = (440.0, 466.16);
        let x: Vec<f64> = (0..s.span())
            .map(|i| {
                let t = i as f64 / p.sr;
                (std::f64::consts::TAU * f1 * t).sin() + (std::f64::consts::TAU * f2 * t).sin()
            })
            .collect();
        let db = s.process(&x).to_vec();
        let (j1, j2) = (bin_of(&p, f1), bin_of(&p, f2));
        let dip = db[j1..=j2].iter().cloned().fold(f32::MAX, f32::min);
        let peaks = db[j1 - 2..=j1 + 2].iter().cloned().fold(f32::MIN, f32::max)
            .min(db[j2 - 2..=j2 + 2].iter().cloned().fold(f32::MIN, f32::max));
        assert!(dip < peaks - 6.0, "no dip between the notes: peaks {peaks}, dip {dip}");
    }

    #[test]
    fn silence_is_floor() {
        let mut s = Spectro::new(params());
        let x = vec![0.0; s.span()];
        assert!(s.process(&x).iter().all(|&v| v == DB_FLOOR));
    }
}
