//! Inputs no golden contains: non-finite audio and degenerate FFT geometry.

mod common;

use common::example_params;
use dsp::DspChain;
use dsp::fft_post::Smear;

/// One NaN sample must not poison L/M/H: the block reads as silence and the
/// next finite blocks produce finite levels again.
#[test]
fn nan_input_sample_does_not_stick_in_lmh() {
    let p = example_params();
    let mut dsp = DspChain::new(&p);
    let tone: Vec<f32> = (0..p.blocksize)
        .map(|i| (2.0 * std::f32::consts::PI * 120.0 * i as f32 / 48_000.0).sin() * 0.5)
        .collect();
    for _ in 0..50 {
        dsp.process_block(&tone);
    }
    let mut bad = tone.clone();
    bad[17] = f32::NAN;
    let out = dsp.process_block(&bad);
    assert!(
        out.raw.iter().chain(&out.scaled).all(|v| v.is_finite()),
        "{out:?}"
    );
    for _ in 0..50 {
        dsp.process_block(&tone);
    }
    let out = dsp.process_block(&tone);
    assert!(
        out.raw.iter().chain(&out.scaled).all(|v| v.is_finite()),
        "{out:?}"
    );
    assert!(
        out.raw[0] > 0.0,
        "the low band should see the 120 Hz tone again"
    );
}

/// An f_min just below Nyquist (accepted by validation) used to make σ ~1e8
/// bins and the kernel GBs; σ is capped at n_bins.
#[test]
fn smear_kernel_is_bounded_for_f_min_near_nyquist() {
    for (n_bins, oct) in [(128, 0.3), (1024, 4.0)] {
        let (smear, sigma) = Smear::build(n_bins, 23_999.99, 48_000.0, oct);
        assert!(smear.is_some());
        assert_eq!(sigma, n_bins as f64);
    }
}
