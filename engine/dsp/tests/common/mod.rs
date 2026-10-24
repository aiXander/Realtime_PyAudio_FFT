//! Shared golden-loading helpers for the parity tests. Format + replay
//! contract: `testdata/golden/README.md`.
#![allow(dead_code)]

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use dsp::onset::OnsetParams;
use dsp::{Bands, EngineParams};
use npyz::{DType, NpyFile};
use serde_json::Value;

/// `testdata/golden/`, or `$GOLDEN_DIR` when set.
pub fn golden_dir() -> PathBuf {
    match std::env::var_os("GOLDEN_DIR") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden"),
    }
}

/// The manifest, or `None` (with a message) when the goldens haven't been
/// generated: they are gitignored and regenerated locally.
pub fn manifest() -> Option<Value> {
    let path = golden_dir().join("manifest.json");
    if !path.exists() {
        eprintln!(
            "SKIP: goldens not generated ({} missing); run `uv run python tools/parity/make_goldens.py`",
            path.display()
        );
        return None;
    }
    let text = std::fs::read_to_string(&path).expect("read manifest.json");
    Some(serde_json::from_str(&text).expect("parse manifest.json"))
}

/// Return early from a test when the goldens are missing.
#[macro_export]
macro_rules! require_goldens {
    () => {
        match common::manifest() {
            Some(m) => m,
            None => return,
        }
    };
}

pub struct Npy<T> {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Vec<T>,
}

pub fn load_npy<T: npyz::Deserialize>(rel: &str) -> Npy<T> {
    let path = golden_dir().join(rel);
    let f = File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let npy =
        NpyFile::new(BufReader::new(f)).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let shape = npy.shape().iter().map(|&s| s as usize).collect();
    let dtype = match npy.dtype() {
        DType::Plain(ts) => ts.to_string(),
        other => panic!("{}: unexpected dtype {other:?}", path.display()),
    };
    assert_eq!(
        npy.order(),
        npyz::Order::C,
        "{}: not C-order",
        path.display()
    );
    let data = npy
        .into_vec::<T>()
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    Npy { shape, dtype, data }
}

pub fn f(v: &Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("expected number, got {v}"))
}

pub fn u(v: &Value) -> usize {
    v.as_u64()
        .unwrap_or_else(|| panic!("expected uint, got {v}")) as usize
}

pub fn band3(v: &Value) -> [f64; 3] {
    [f(&v["low"]), f(&v["mid"]), f(&v["high"])]
}

/// `{"low": [lo, hi], ...}` (FilterBank / AutoScaler) or
/// `{"low": {"lo_hz", "hi_hz"}, ...}` (FFTPostProcessor / config).
pub fn bands(v: &Value) -> Bands {
    let one = |b: &Value| -> (f64, f64) {
        if let Some(a) = b.as_array() {
            (f(&a[0]), f(&a[1]))
        } else {
            (f(&b["lo_hz"]), f(&b["hi_hz"]))
        }
    };
    [one(&v["low"]), one(&v["mid"]), one(&v["high"])]
}

pub fn engine_params(case: &Value) -> EngineParams {
    let c = &case["config"];
    let dsp = &c["dsp"];
    let a = &c["autoscale"];
    let fft = &c["fft"];
    let onset = |b: &str| OnsetParams {
        sensitivity: f(&c["onset"][b]["sensitivity"]),
        refractory_s: f(&c["onset"][b]["refractory_s"]),
        slow_tau_s: f(&c["onset"][b]["slow_tau_s"]),
        abs_floor: f(&c["onset"][b]["abs_floor"]),
    };
    EngineParams {
        sr: f(&case["sr"]),
        blocksize: u(&c["audio"]["blocksize"]),
        bands: bands(dsp),
        filter_order: u(&dsp["filter_order"]),
        tau: band3(&dsp["tau"]),
        tau_attack: band3(&dsp["tau_attack"]),
        autoscale_tau_attack_s: f(&a["tau_attack_s"]),
        autoscale_tau_release_s: f(&a["tau_release_s"]),
        noise_floor: f(&a["noise_floor"]),
        strength: f(&a["strength"]),
        fft_n_bins: u(&fft["n_bins"]),
        fft_window_size: u(&fft["window_size"]),
        fft_hop: u(&fft["hop"]),
        fft_f_min: f(&fft["f_min"]),
        fft_db_floor: f(&fft["db_floor"]),
        fft_db_ceiling: f(&fft["db_ceiling"]),
        fft_peak_smear_oct: f(&fft["peak_smear_oct"]),
        fft_tilt_db_per_oct: f(&fft["tilt_db_per_oct"]),
        onset: [onset("low"), onset("mid"), onset("high")],
    }
}

/// Max abs error tracker that remembers where the worst one was.
#[derive(Default, Debug, Clone)]
pub struct MaxErr {
    pub n: usize,
    pub max: f64,
    pub at: String,
    pub first_over: Option<String>,
}

impl MaxErr {
    pub fn add(&mut self, got: f64, want: f64, tol: f64, at: impl FnOnce() -> String) {
        self.n += 1;
        let e = (got - want).abs();
        let bad = e.is_nan() || e > tol || got.is_nan() != want.is_nan();
        if e > self.max || (e.is_nan() && !self.max.is_nan()) {
            self.max = e;
            self.at = at();
            if bad && self.first_over.is_none() {
                self.first_over = Some(format!("{} (got {got:e}, want {want:e})", self.at));
            }
        } else if bad && self.first_over.is_none() {
            self.first_over = Some(format!("{} (got {got:e}, want {want:e})", at()));
        }
    }
}

/// Max relative error of `got` vs `want` (|g-w| / max(|w|, tiny)).
pub fn max_rel(got: &[f64], want: &[f64]) -> (f64, usize) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut worst = (0.0, 0);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let e = if g == w {
            0.0
        } else {
            (g - w).abs() / w.abs().max(1e-300)
        };
        if e > worst.0 || e.is_nan() {
            worst = (e, i);
        }
    }
    worst
}

/// `configs/main.example.yaml` values at 48 kHz.
pub fn example_params() -> EngineParams {
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
