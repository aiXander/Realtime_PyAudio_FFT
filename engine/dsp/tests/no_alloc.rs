//! `process_block` / `process_hop` must not allocate (the realtime contract).
//! Uses `assert_no_alloc`'s allocator, which aborts the test binary on any
//! allocation inside `assert_no_alloc(..)` in debug builds. No goldens needed.

use assert_no_alloc::{AllocDisabler, assert_no_alloc};
use dsp::onset::OnsetParams;
use dsp::{DspChain, EngineParams, FftChain};

#[cfg(debug_assertions)]
#[global_allocator]
static A: AllocDisabler = AllocDisabler;

/// `configs/main.example.yaml` values at 48 kHz.
fn example_params() -> EngineParams {
    EngineParams {
        sr: 48000.0,
        blocksize: 256,
        bands: [(88.0, 184.0), (304.0, 1760.0), (2944.0, 13952.0)],
        filter_order: 2,
        tau: [0.12, 0.09, 0.05],
        tau_attack: [0.015, 0.005, 0.003],
        autoscale_tau_attack_s: 0.255,
        autoscale_tau_release_s: 30.0,
        noise_floor: 3.2e-5,
        strength: 0.7,
        fft_n_bins: 128,
        fft_window_size: 1024,
        fft_hop: 512,
        fft_f_min: 30.0,
        fft_db_floor: -90.0,
        fft_db_ceiling: -10.0,
        fft_peak_smear_oct: 0.3,
        fft_tilt_db_per_oct: 3.5,
        onset: [OnsetParams::default(); 3],
    }
}

/// Deterministic test signal: a decaying 60 Hz "kick" every 0.5 s + LCG noise.
fn signal(n: usize, sr: f64) -> Vec<f32> {
    let mut state: u32 = 12345;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let t = (i % 24000) as f64 / sr;
            let kick = (2.0 * std::f64::consts::PI * 60.0 * t).sin() * (-t * 20.0).exp();
            (0.8 * kick) as f32 + 0.05 * noise
        })
        .collect()
}

#[test]
fn hot_paths_do_not_allocate() {
    for strength in [0.0, 0.7, 1.0] {
        let p = EngineParams {
            strength,
            ..example_params()
        };
        let bs = p.blocksize;
        let ws = p.fft_window_size;
        let sig = signal(48000 * 3, p.sr);
        let mut dsp = DspChain::new(&p);
        let mut fft = FftChain::new(&p).unwrap();

        // First hop of a layout builds the sentinel LUT (allocates, by design).
        fft.window_mut()
            .iter_mut()
            .zip(&sig)
            .for_each(|(w, &x)| *w = x as f64);
        fft.process_hop();

        let mut onsets = 0;
        assert_no_alloc(|| {
            for b in 0..sig.len() / bs {
                let out = dsp.process_block(&sig[b * bs..(b + 1) * bs]);
                onsets += out.onsets.iter().map(|&o| o as i32).sum::<i32>();
                if b * bs >= ws && (b * bs).is_multiple_of(p.fft_hop) {
                    let s0 = b * bs - ws;
                    for (w, &x) in fft.window_mut().iter_mut().zip(&sig[s0..s0 + ws]) {
                        *w = x as f64;
                    }
                    let (db, proc) = fft.process_hop();
                    assert_eq!(db.len(), proc.len());
                }
            }
        });
        // Every setter the realtime workers apply must be alloc-free too
        // (the smear kernel and filter designs are built off-thread first).
        let design = Box::new(dsp::filters::FilterDesign::new(
            p.sr,
            [(100.0, 200.0), (400.0, 2000.0), (3000.0, 12000.0)],
            4,
        ));
        let (n, f_min, sr) = fft.post.smear_geometry();
        let (smear, sigma) = dsp::fft_post::Smear::build(n, f_min, sr, 0.8);
        let mut old_smear = None;
        assert_no_alloc(|| {
            dsp.filter_bank.install(*design);
            dsp.filter_bank.reset_state();
            dsp.smoother.set_tau([0.1, 0.1, 0.1], [0.01, 0.01, 0.01]);
            dsp.auto_scaler.set_bands(design.bands);
            dsp.auto_scaler.set_noise_floor(1e-4);
            dsp.auto_scaler.set_tilt(-3.0);
            dsp.auto_scaler.set_taus(0.1, 5.0);
            dsp.auto_scaler.set_n_fft_window(2048);
            dsp.auto_scaler.set_fft_geometry(Some(64), Some(40.0));
            dsp.auto_scaler.set_strength(0.5);
            dsp.onset_tracker.set_params(
                2,
                dsp::onset::OnsetParamsUpdate {
                    slow_tau_s: Some(0.08),
                    ..Default::default()
                },
            );
            fft.post.update_bands(design.bands);
            fft.post
                .update_smoothing([0.1, 0.1, 0.1], [0.01, 0.01, 0.01]);
            fft.post.update_tilt(-3.0);
            fft.post.update_autoscale(dsp::fft_post::AutoscaleUpdate {
                noise_floor: Some(0.0),
                strength: Some(0.5),
                ..Default::default()
            });
            fft.post.reset();
            old_smear = Some(fft.post.install_smear(0.8, smear, sigma));
            for b in 0..64 {
                dsp.process_block(&sig[b * bs..(b + 1) * bs]);
            }
            fft.process_hop();
        });
        drop(old_smear);
        assert!(
            onsets > 0,
            "the test signal should trigger onsets (exercises the BPM ring)"
        );
    }
}
