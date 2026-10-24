//! Per-stage cost of the Rust DSP, using the stage names of
//! `tools/bench/stage_bench.py` so the two tables line up
//! (`docs/reference/perf_baseline.md`). Configs + signal come from the
//! golden cases (`default_drums`, `bins256_drums`, ...).
//!
//!     cargo bench -p dsp --bench stages
//!
//! `fft.power_logbin_db` is not measured directly: like the Python bench it
//! is `fft_hop_total - window_rfft - postprocess`.

#[path = "../tests/common/mod.rs"]
mod common;

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use dsp::features::block_rms;
use dsp::fft::hanning;
use dsp::{DspChain, FftChain};
use realfft::RealFftPlanner;

const CONFIGS: [(&str, &str); 7] = [
    ("default", "default_drums"),
    ("bins256", "bins256_drums"),
    ("bins1024", "bins1024_drums"),
    ("order8", "order8_drums"),
    ("win4096", "win4096_drums"),
    ("strength1", "strength1_drums"),
    ("block128", "block128_drums"),
];

fn stages(c: &mut Criterion) {
    let Some(m) = common::manifest() else { return };
    let sig = common::load_npy::<f32>(m["signals"]["drums_48000"]["file"].as_str().unwrap()).data;
    for (name, case_id) in CONFIGS {
        let case = m["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == case_id)
            .unwrap_or_else(|| panic!("golden case {case_id} missing"));
        let p = common::engine_params(case);
        let bs = p.blocksize;
        let blocks: Vec<&[f32]> = sig.chunks_exact(bs).collect();
        let nb = blocks.len();

        // Representative inputs for the stages after the filter bank.
        let mut chain = DspChain::new(&p);
        let mut rms_in = Vec::with_capacity(nb);
        let mut raw_in = Vec::with_capacity(nb);
        let mut scaled_in = Vec::with_capacity(nb);
        for x in &blocks {
            let [lo, md, hi] = chain.filter_bank.process(x);
            let rms = [block_rms(lo), block_rms(md), block_rms(hi)];
            let raw = chain.smoother.update(rms);
            let scaled = chain.auto_scaler.update(raw);
            chain.onset_tracker.update(scaled);
            rms_in.push(rms);
            raw_in.push(raw);
            scaled_in.push(scaled);
        }

        let mut g = c.benchmark_group(name);
        g.warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(2))
            .sample_size(50);

        let mut i = 0usize;
        let mut fb = DspChain::new(&p).filter_bank;
        g.bench_function("dsp.filter_bank", |b| {
            b.iter(|| {
                i = (i + 1) % nb;
                black_box(fb.process(blocks[i]));
            })
        });
        let filtered: Vec<[Vec<f64>; 3]> = blocks
            .iter()
            .take(64)
            .map(|x| fb.process(x).map(|s| s.to_vec()))
            .collect();
        g.bench_function("dsp.rms_x3", |b| {
            b.iter(|| {
                i = (i + 1) % filtered.len();
                let f = &filtered[i];
                black_box([block_rms(&f[0]), block_rms(&f[1]), block_rms(&f[2])])
            })
        });
        let mut sm = DspChain::new(&p).smoother;
        g.bench_function("dsp.smoother", |b| {
            b.iter(|| {
                i = (i + 1) % nb;
                black_box(sm.update(rms_in[i]))
            })
        });
        let mut asc = DspChain::new(&p).auto_scaler;
        g.bench_function("dsp.autoscaler", |b| {
            b.iter(|| {
                i = (i + 1) % nb;
                black_box(asc.update(raw_in[i]))
            })
        });
        let mut ot = DspChain::new(&p).onset_tracker;
        g.bench_function("dsp.onset", |b| {
            b.iter(|| {
                i = (i + 1) % nb;
                black_box(ot.update(scaled_in[i]))
            })
        });
        let mut dsp = DspChain::new(&p);
        g.bench_function("dsp.dsp_block_total", |b| {
            b.iter(|| {
                i = (i + 1) % nb;
                black_box(dsp.process_block(blocks[i]))
            })
        });

        // FFT path: windows at hop spacing.
        let ws = p.fft_window_size;
        let n_win = (sig.len() - ws) / p.fft_hop;
        let windows: Vec<Vec<f64>> = (0..n_win.min(800))
            .map(|k| {
                sig[k * p.fft_hop..k * p.fft_hop + ws]
                    .iter()
                    .map(|&x| x as f64)
                    .collect()
            })
            .collect();
        let hann = hanning(ws);
        let plan = RealFftPlanner::<f64>::new().plan_fft_forward(ws);
        let (mut fin, mut fout, mut fscr) = (
            plan.make_input_vec(),
            plan.make_output_vec(),
            plan.make_scratch_vec(),
        );
        g.bench_function("fft.window_rfft", |b| {
            b.iter(|| {
                i = (i + 1) % windows.len();
                for ((o, &w), &h) in fin.iter_mut().zip(&windows[i]).zip(&hann) {
                    *o = w * h;
                }
                plan.process_with_scratch(&mut fin, &mut fout, &mut fscr)
                    .unwrap();
                black_box(&fout);
            })
        });
        let mut fft = FftChain::new(&p).unwrap();
        g.bench_function("fft.fft_hop_total", |b| {
            b.iter(|| {
                i = (i + 1) % windows.len();
                fft.window_mut().copy_from_slice(&windows[i]);
                black_box(fft.process_hop());
            })
        });
        let db_in = fft.analyzer.wire().to_vec();
        let mut post = FftChain::new(&p).unwrap().post;
        g.bench_function("fft.postprocess", |b| {
            b.iter(|| {
                black_box(post.process(&db_in));
            })
        });
        g.finish();
    }
}

criterion_group!(benches, stages);
criterion_main!(benches);
