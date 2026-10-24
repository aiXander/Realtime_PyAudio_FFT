//! Three independent Butterworth bandpass filters (port of
//! `server/dsp/filters.py`). Design goes through sci-rs (matches scipy's
//! `iirfilter(..., output="sos")` to ~1e-15); the per-sample kernel is
//! scipy's DF2T `_sosfilt`.

use sci_rs::signal::filter::design::{
    DigitalFilter, FilterBandType, FilterOutputType, FilterType, iirfilter_dyn,
};

use crate::Bands;

/// Bandpass order N gives N biquad sections; `validate_filter_order` caps N at 8.
pub const MAX_SECTIONS: usize = 8;
pub const MAX_ORDER: usize = MAX_SECTIONS;

const HI_MAX_FRAC_OF_SR: f64 = 0.45;
const LO_MIN_HZ: f64 = 1.0;
const LO_MAX_FRAC_OF_HI: f64 = 0.9;

/// Clamp a bandpass `[lo, hi]` (Hz) so it can always be designed at `sr`:
/// 1 Hz ≤ lo ≤ 0.9·hi, hi ≤ 0.45·sr. Pure.
pub fn clamp_band_edges(lo: f64, hi: f64, sr: f64) -> (f64, f64) {
    let hi = hi.min(HI_MAX_FRAC_OF_SR * sr).max(2.0 * LO_MIN_HZ);
    let lo = lo.max(LO_MIN_HZ);
    let lo = lo.min(LO_MAX_FRAC_OF_HI * hi);
    (lo, hi)
}

/// One band's second-order sections, scipy row layout `[b0 b1 b2 a0 a1 a2]`.
/// Fixed capacity so installing a new design never allocates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sos {
    pub sections: [[f64; 6]; MAX_SECTIONS],
    pub n: usize,
}

impl Sos {
    pub fn rows(&self) -> &[[f64; 6]] {
        &self.sections[..self.n]
    }
}

/// `FilterBank._design_one`: clamp, then
/// `iirfilter(order, [lo/nyq, hi/nyq], btype="bandpass", ftype="butter", output="sos")`.
/// Returns the SOS and the clamped edges actually used. Allocates (control thread).
pub fn design_bandpass(order: usize, lo: f64, hi: f64, sr: f64) -> (Sos, (f64, f64)) {
    assert!(
        (1..=MAX_ORDER).contains(&order),
        "filter order must be in 1..=8"
    );
    let nyq = sr / 2.0;
    let (lo_c, hi_c) = clamp_band_edges(lo, hi, sr);
    let filt = iirfilter_dyn::<f64>(
        order,
        vec![lo_c / nyq, hi_c / nyq],
        None,
        None,
        Some(FilterBandType::Bandpass),
        Some(FilterType::Butterworth),
        Some(false),
        Some(FilterOutputType::Sos),
        None,
    );
    let DigitalFilter::Sos(sos) = filt else {
        unreachable!("iirfilter_dyn(output=Sos) returned a non-SOS filter")
    };
    assert!(sos.sos.len() <= MAX_SECTIONS);
    let mut out = Sos {
        sections: [[0.0; 6]; MAX_SECTIONS],
        n: sos.sos.len(),
    };
    for (row, s) in out.sections.iter_mut().zip(&sos.sos) {
        *row = [s.b[0], s.b[1], s.b[2], s.a[0], s.a[1], s.a[2]];
    }
    (out, (lo_c, hi_c))
}

/// A full 3-band design: built on the control thread, installed on the
/// DSP worker with `FilterBank::install` (no allocation).
#[derive(Clone, Copy, Debug)]
pub struct FilterDesign {
    pub sr: f64,
    pub order: usize,
    /// Requested edges.
    pub bands: Bands,
    /// Clamped edges actually designed.
    pub effective_bands: Bands,
    pub sos: [Sos; 3],
}

impl FilterDesign {
    pub fn new(sr: f64, bands: Bands, order: usize) -> Self {
        let mut sos = [Sos {
            sections: [[0.0; 6]; MAX_SECTIONS],
            n: 0,
        }; 3];
        let mut effective_bands = bands;
        for i in 0..3 {
            let (s, eff) = design_bandpass(order, bands[i].0, bands[i].1, sr);
            sos[i] = s;
            effective_bands[i] = eff;
        }
        FilterDesign {
            sr,
            order,
            bands,
            effective_bands,
            sos,
        }
    }
}

/// Three bandpass filters owned by the DSP worker. `process` filters one
/// f32 block (cast to f64) through each band in place into preallocated
/// buffers; the zi state is f64 DF2T, one `[z0, z1]` per section.
pub struct FilterBank {
    design: FilterDesign,
    zi: [[[f64; 2]; MAX_SECTIONS]; 3],
    out: [Vec<f64>; 3],
}

impl FilterBank {
    pub fn new(sr: f64, bands: Bands, blocksize: usize, order: usize) -> Self {
        Self::from_design(FilterDesign::new(sr, bands, order), blocksize)
    }

    pub fn from_design(design: FilterDesign, blocksize: usize) -> Self {
        FilterBank {
            design,
            zi: [[[0.0; 2]; MAX_SECTIONS]; 3],
            out: [
                vec![0.0; blocksize],
                vec![0.0; blocksize],
                vec![0.0; blocksize],
            ],
        }
    }

    pub fn design(&self) -> &FilterDesign {
        &self.design
    }

    pub fn bands(&self) -> Bands {
        self.design.bands
    }

    pub fn effective_bands(&self) -> Bands {
        self.design.effective_bands
    }

    pub fn order(&self) -> usize {
        self.design.order
    }

    /// Swap in a prebuilt design and zero the state (Python `retune` /
    /// `set_order`: "brief click acceptable"). No allocation.
    pub fn install(&mut self, design: FilterDesign) {
        self.design = design;
        self.reset_state();
    }

    /// `retune(bands)`: redesign at the current order. Allocates (design).
    pub fn retune(&mut self, bands: Bands) {
        self.install(FilterDesign::new(self.design.sr, bands, self.design.order));
    }

    /// `set_order(order)`: redesign the current bands. Allocates (design).
    pub fn set_order(&mut self, order: usize) {
        self.install(FilterDesign::new(self.design.sr, self.design.bands, order));
    }

    pub fn reset_state(&mut self) {
        self.zi = [[[0.0; 2]; MAX_SECTIONS]; 3];
    }

    /// Filter one block. Returns the three filtered buffers (valid until the
    /// next call). Allocation-free unless the block length changes.
    pub fn process(&mut self, x: &[f32]) -> [&[f64]; 3] {
        if x.len() != self.out[0].len() {
            for o in &mut self.out {
                o.resize(x.len(), 0.0); // off the steady-state path
            }
        }
        // All three bands share the order, so one section count.
        match self.design.sos[0].n {
            1 => self.run::<1>(x),
            2 => self.run::<2>(x),
            3 => self.run::<3>(x),
            4 => self.run::<4>(x),
            5 => self.run::<5>(x),
            6 => self.run::<6>(x),
            7 => self.run::<7>(x),
            _ => self.run::<MAX_SECTIONS>(x),
        }
        [&self.out[0], &self.out[1], &self.out[2]]
    }

    /// The DF2T kernel for `N` sections, the three bands interleaved per
    /// sample. A compile-time `N` keeps coefficients and state in registers
    /// and the three independent recurrences overlap in the pipeline (~4×
    /// faster than band-by-band at order 2). Each band's arithmetic is
    /// unchanged, so the output is bit-identical.
    #[inline]
    fn run<const N: usize>(&mut self, x: &[f32]) {
        let mut s = [[[0.0f64; 6]; N]; 3];
        let mut z = [[[0.0f64; 2]; N]; 3];
        for b in 0..3 {
            s[b].copy_from_slice(&self.design.sos[b].sections[..N]);
            z[b].copy_from_slice(&self.zi[b][..N]);
        }
        let [o0, o1, o2] = &mut self.out;
        for (i, &xs) in x.iter().enumerate() {
            let x_in = xs as f64;
            let mut y = [0.0f64; 3];
            for b in 0..3 {
                let mut x_cur = x_in;
                for k in 0..N {
                    let (sk, zk) = (&s[b][k], &mut z[b][k]);
                    let x_new = sk[0] * x_cur + zk[0];
                    zk[0] = sk[1] * x_cur - sk[4] * x_new + zk[1];
                    zk[1] = sk[2] * x_cur - sk[5] * x_new;
                    x_cur = x_new;
                }
                y[b] = x_cur;
            }
            o0[i] = y[0];
            o1[i] = y[1];
            o2[i] = y[2];
        }
        for b in 0..3 {
            self.zi[b][..N].copy_from_slice(&z[b]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interleaved const-N kernel must stay bit-identical to scipy's
    /// band-by-band `_sosfilt` loop (the L/M/H goldens only check 1e-9).
    #[test]
    fn interleaved_kernel_matches_reference_loop_bit_for_bit() {
        let (sr, bs) = (48_000.0, 256);
        let bands = [(40.0, 250.0), (250.0, 2000.0), (2000.0, 16000.0)];
        let mut state = 12345u64;
        let sig: Vec<f32> = (0..bs * 200)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.5
            })
            .collect();
        for order in 1..=MAX_ORDER {
            let design = FilterDesign::new(sr, bands, order);
            let mut fb = FilterBank::from_design(design, bs);
            let mut zi = [[[0.0f64; 2]; MAX_SECTIONS]; 3];
            for block in sig.chunks_exact(bs) {
                let got = fb.process(block);
                for band in 0..3 {
                    let sos = design.sos[band].rows();
                    for (i, &xs) in block.iter().enumerate() {
                        let mut x_cur = xs as f64;
                        for (s, z) in sos.iter().zip(zi[band].iter_mut()) {
                            let x_new = s[0] * x_cur + z[0];
                            z[0] = s[1] * x_cur - s[4] * x_new + z[1];
                            z[1] = s[2] * x_cur - s[5] * x_new;
                            x_cur = x_new;
                        }
                        assert_eq!(
                            got[band][i].to_bits(),
                            x_cur.to_bits(),
                            "order {order} band {band}"
                        );
                    }
                }
            }
        }
    }
}
