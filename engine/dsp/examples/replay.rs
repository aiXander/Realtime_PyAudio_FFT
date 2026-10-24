//! Offline replay: WAV in → the golden per-case dump format out (raw.npy,
//! scaled.npy, onsets.npy, bpm.npy, hops.npy, fft_offsets.npy,
//! fft_raw_db.npy, fft_processed.npy). For debugging a divergence against
//! the Python engine on arbitrary audio.
//!
//!     cargo run -p dsp --release --example replay -- in.wav config.json out_dir/
//!
//! `config.json` has the `configs/main.yaml` shape (e.g. a golden case's
//! `config` object; band edges already clamped for the WAV's rate). The WAV
//! is mixed to mono (channel mean) and converted to f32. Hops follow the
//! live worker's drain-all schedule with no backlog: window k starts at
//! block k·hop_blocks and is processed after its last block.

#[path = "../tests/common/mod.rs"]
mod common;

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use dsp::{DspChain, FftChain};
use npyz::WriterBuilder;

fn write_npy<T: npyz::AutoSerialize + Copy>(dir: &Path, name: &str, shape: &[u64], data: &[T]) {
    let f = BufWriter::new(File::create(dir.join(format!("{name}.npy"))).expect("create npy"));
    let mut w = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .writer(f)
        .begin_nd()
        .expect("npy header");
    w.extend(data.iter().copied()).expect("npy data");
    w.finish().expect("npy finish");
}

fn read_wav_mono(path: &str) -> (Vec<f32>, f64) {
    let mut r = hound::WavReader::open(path).expect("open wav");
    let spec = r.spec();
    let ch = spec.channels as usize;
    let inter: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.unwrap() as f32 * scale)
                .collect()
        }
    };
    let mono = inter
        .chunks_exact(ch)
        .map(|c| c.iter().sum::<f32>() / ch as f32)
        .collect();
    (mono, spec.sample_rate as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: replay <in.wav> <config.json> <out_dir>");
        std::process::exit(2);
    }
    let (sig, sr) = read_wav_mono(&args[1]);
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&args[2]).expect("read config"))
            .expect("parse config");
    let case = serde_json::json!({ "sr": sr, "config": cfg });
    let p = common::engine_params(&case);
    let out = Path::new(&args[3]);
    std::fs::create_dir_all(out).expect("create out dir");

    let bs = p.blocksize;
    let n_blocks = sig.len() / bs;
    let mut dsp = DspChain::new(&p);
    let mut fft = FftChain::new(&p).expect("fft config");
    let wb = fft.analyzer.geometry().n_blocks_per_window();
    let hb = fft.analyzer.geometry().hop_blocks();

    let (mut raw, mut scaled, mut onsets, mut bpm) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut hops, mut offsets, mut db, mut proc) =
        (Vec::new(), vec![0i64], Vec::new(), Vec::new());
    let mut next_start = 0usize;
    for b in 0..n_blocks {
        let o = dsp.process_block(&sig[b * bs..(b + 1) * bs]);
        raw.extend(o.raw);
        scaled.extend(o.scaled);
        onsets.extend(o.onsets);
        bpm.push(o.bpm);
        while next_start + wb <= b + 1 {
            let s0 = next_start * bs;
            let win = fft.window_mut();
            let n = win.len();
            for (w, &x) in win.iter_mut().zip(&sig[s0..s0 + n]) {
                *w = x as f64;
            }
            let (d, pr) = fft.process_hop();
            db.extend_from_slice(d);
            proc.extend_from_slice(pr);
            offsets.push(*offsets.last().unwrap() + d.len() as i64);
            hops.extend([next_start as i64, b as i64]);
            next_start += hb;
        }
    }
    let nb = n_blocks as u64;
    write_npy(out, "raw", &[nb, 3], &raw);
    write_npy(out, "scaled", &[nb, 3], &scaled);
    write_npy(out, "onsets", &[nb, 3], &onsets);
    write_npy(out, "bpm", &[nb], &bpm);
    write_npy(out, "hops", &[(hops.len() / 2) as u64, 2], &hops);
    write_npy(out, "fft_offsets", &[offsets.len() as u64], &offsets);
    write_npy(out, "fft_raw_db", &[db.len() as u64], &db);
    write_npy(out, "fft_processed", &[proc.len() as u64], &proc);
    eprintln!(
        "replay: {n_blocks} blocks, {} hops → {}",
        hops.len() / 2,
        out.display()
    );
}
