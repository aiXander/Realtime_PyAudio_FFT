//! Parity against the Python reference goldens (`testdata/golden/`).
//! Replay contract: `testdata/golden/README.md`. Skips when the goldens
//! haven't been generated.

mod common;

use common::{MaxErr, band3, bands, engine_params, f, load_npy, max_rel, u};
use dsp::fft_post::AutoscaleUpdate;
use dsp::filters::design_bandpass;
use dsp::onset::OnsetParamsUpdate;
use dsp::{DspChain, FftChain};
use serde_json::Value;

fn opt_f(v: &Value) -> Option<f64> {
    if v.is_null() { None } else { Some(f(v)) }
}
fn opt_u(v: &Value) -> Option<usize> {
    if v.is_null() { None } else { Some(u(v)) }
}

fn tol(m: &Value, key: &str) -> f64 {
    f(&m["tolerances"][key]["max_abs"])
}

#[test]
fn filter_design_matches_scipy() {
    let m = require_goldens!();
    let fd = &m["filter_design"];
    let sos = load_npy::<f64>(fd["file"].as_str().unwrap());
    let max_abs = tol(&m, "filter_design_sos");
    let mut worst = 0.0f64;
    for row in fd["rows"].as_array().unwrap() {
        let (order, sr) = (u(&row["order"]), f(&row["sr"]));
        let (got, (lo_c, hi_c)) = design_bandpass(order, f(&row["lo_hz"]), f(&row["hi_hz"]), sr);
        assert_eq!(
            (lo_c, hi_c),
            (f(&row["lo_clamped"]), f(&row["hi_clamped"])),
            "clamp {row}"
        );
        assert_eq!(got.n, u(&row["n_sections"]), "section count {row}");
        let base = u(&row["row"]) * 48;
        for (s, sec) in got.rows().iter().enumerate() {
            for (c, &g) in sec.iter().enumerate() {
                let want = sos.data[base + s * 6 + c];
                let e = (g - want).abs();
                worst = worst.max(e);
                assert!(
                    e <= max_abs,
                    "{row}: section {s} coef {c}: got {g} want {want} (|e|={e:e})"
                );
            }
        }
    }
    eprintln!("filter design: worst |coef err| = {worst:e}");
}

/// Internal state right after construction, per case (compare these first
/// when an output diverges).
#[test]
fn derived_state_matches() {
    let m = require_goldens!();
    let max_rel_tol = f(&m["tolerances"]["derived"]["max_rel"]);
    let mut failures = Vec::new();
    for case in m["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let p = engine_params(case);
        let dsp = DspChain::new(&p);
        let fft = FftChain::new(&p).expect("fft chain");
        let files = &case["files"];
        let get =
            |name: &str| load_npy::<f64>(files[format!("derived_{name}")].as_str().unwrap()).data;

        let mut check = |name: &str, got: Vec<f64>| {
            let want = get(name);
            if got.len() != want.len() {
                failures.push(format!("{id}/{name}: len {} vs {}", got.len(), want.len()));
                return;
            }
            let (e, i) = max_rel(&got, &want);
            if e.is_nan() || e > max_rel_tol {
                failures.push(format!(
                    "{id}/{name}[{i}]: rel {e:e} (got {:e}, want {:e})",
                    got[i], want[i]
                ));
            }
        };

        let a = &fft.analyzer;
        let assign = load_npy::<i64>(files["derived_fft_bin_assign"].as_str().unwrap()).data;
        let assign_ok = assign == a.bin_map().assign;
        check("fft_bin_counts", a.bin_map().counts.clone());
        check("fft_hann", a.hann().to_vec());
        check("fft_win_power_corr", vec![a.win_power_corr()]);
        let pp = &fft.post;
        check("post_tilt_db", pp.tilt_db().to_vec());
        check("post_tau_per_bin", pp.tau_per_bin().to_vec());
        check("post_tau_attack_per_bin", pp.tau_attack_per_bin().to_vec());
        check(
            "post_alpha_smooth_release",
            pp.alpha_smooth_release().to_vec(),
        );
        check(
            "post_alpha_smooth_attack",
            pp.alpha_smooth_attack().to_vec(),
        );
        check("post_peak_alphas", pp.peak_alphas().to_vec());
        check("post_smear_kernel", pp.smear_kernel().to_vec());
        let s = &dsp.auto_scaler;
        check("as_tilt_lin", s.tilt_lin().to_vec());
        check("as_noise_pwr_per_band", s.noise_pwr_per_band().to_vec());
        check("as_per_bin_factor", s.per_bin_factor().to_vec());
        check("as_log_bins_per_band", s.log_bins_per_band().to_vec());
        check("as_peak_alphas", s.peak_alphas().to_vec());
        check("smoother_alphas", dsp.smoother.alphas().concat());
        check("onset_alphas", dsp.onset_tracker.alphas().to_vec());
        let fb = dsp.filter_bank.design();
        for (i, name) in ["sos_low", "sos_mid", "sos_high"].iter().enumerate() {
            check(name, fb.sos[i].rows().concat());
        }
        if !assign_ok {
            failures.push(format!("{id}/fft_bin_assign differs"));
        }
    }
    assert!(
        failures.is_empty(),
        "derived mismatches:\n{}",
        failures.join("\n")
    );
}

/// Apply one depth-0 DSP-level setter call recorded by the Python tracer.
fn apply_call(call: &Value, dsp: &mut DspChain, fft: &mut FftChain) {
    let target = call["target"].as_str().unwrap();
    let method = call["method"].as_str().unwrap();
    let args = call["args"].as_array().unwrap();
    let kw = &call["kwargs"];
    match (target, method) {
        ("filter_bank", "retune") => dsp.filter_bank.retune(bands(&args[0])),
        ("filter_bank", "set_order") => dsp.filter_bank.set_order(u(&args[0])),
        ("filter_bank", "reset_state") => dsp.filter_bank.reset_state(),
        ("smoother", "set_tau") => dsp.smoother.set_tau(band3(&args[0]), band3(&args[1])),
        ("smoother", "reset") => dsp.smoother.reset(),
        ("auto_scaler", "set_taus") => dsp.auto_scaler.set_taus(f(&args[0]), f(&args[1])),
        ("auto_scaler", "set_noise_floor") => dsp.auto_scaler.set_noise_floor(f(&args[0])),
        ("auto_scaler", "set_tilt") => dsp.auto_scaler.set_tilt(f(&args[0])),
        ("auto_scaler", "set_bands") => dsp.auto_scaler.set_bands(bands(&args[0])),
        ("auto_scaler", "set_n_fft_window") => dsp.auto_scaler.set_n_fft_window(u(&args[0])),
        ("auto_scaler", "set_fft_geometry") => dsp
            .auto_scaler
            .set_fft_geometry(opt_u(&kw["n_bins"]), opt_f(&kw["f_min"])),
        ("auto_scaler", "set_strength") => dsp.auto_scaler.set_strength(f(&args[0])),
        ("auto_scaler", "reset") => dsp.auto_scaler.reset(),
        ("onset_tracker", "set_params") => {
            let band = dsp::BAND_NAMES
                .iter()
                .position(|&b| b == args[0].as_str().unwrap())
                .unwrap();
            dsp.onset_tracker.set_params(
                band,
                OnsetParamsUpdate {
                    sensitivity: opt_f(&kw["sensitivity"]),
                    refractory_s: opt_f(&kw["refractory_s"]),
                    slow_tau_s: opt_f(&kw["slow_tau_s"]),
                    abs_floor: opt_f(&kw["abs_floor"]),
                },
            )
        }
        ("onset_tracker", "reset") => dsp.onset_tracker.reset(),
        ("onset_tracker", "advance") => dsp.onset_tracker.advance(u(&args[0]) as u64),
        ("onset_tracker", "reconfigure") => dsp.onset_tracker.reconfigure(f(&args[0]), u(&args[1])),
        ("fft_worker", "reconfigure") => fft
            .reconfigure(
                opt_u(&kw["n_bins"]),
                opt_f(&kw["sr"]),
                opt_u(&kw["window_size"]),
                opt_u(&kw["hop"]),
                opt_f(&kw["f_min"]),
            )
            .expect("fft reconfigure"),
        // Read-pointer moves only; the hops table already encodes them.
        ("fft_worker", "reset") | ("fft_worker", "resync") => {}
        ("fft_postprocessor", "update_smoothing") => {
            fft.post.update_smoothing(band3(&args[0]), band3(&args[1]))
        }
        ("fft_postprocessor", "update_bands") => fft.post.update_bands(bands(&args[0])),
        ("fft_postprocessor", "update_smear") => fft.post.update_smear(f(&args[0])),
        ("fft_postprocessor", "update_tilt") => fft.post.update_tilt(f(&args[0])),
        ("fft_postprocessor", "update_autoscale") => fft.post.update_autoscale(AutoscaleUpdate {
            tau_attack_s: opt_f(&kw["tau_attack_s"]),
            tau_release_s: opt_f(&kw["tau_release_s"]),
            noise_floor: opt_f(&kw["noise_floor"]),
            strength: opt_f(&kw["strength"]),
        }),
        ("fft_postprocessor", "reset") => fft.post.reset(),
        _ => panic!("unhandled traced call {target}.{method}"),
    }
}

struct CaseResult {
    id: String,
    raw: MaxErr,
    scaled: MaxErr,
    bpm: MaxErr,
    fft_raw_db: MaxErr,
    fft_processed: MaxErr,
    onset_mismatches: Vec<String>,
    sentinel_mismatches: usize,
}

fn replay_case(m: &Value, case: &Value) -> CaseResult {
    let id = case["id"].as_str().unwrap().to_string();
    let p = engine_params(case);
    let bs = p.blocksize;
    let n_blocks = u(&case["n_blocks"]);
    let files = &case["files"];
    let path = |k: &str| files[k].as_str().unwrap().to_string();
    let sig = load_npy::<f32>(
        m["signals"][case["signal"].as_str().unwrap()]["file"]
            .as_str()
            .unwrap(),
    )
    .data;
    let raw = load_npy::<f64>(&path("raw")).data;
    let scaled = load_npy::<f64>(&path("scaled")).data;
    let onsets = load_npy::<i8>(&path("onsets")).data;
    let bpm = load_npy::<f64>(&path("bpm")).data;
    let hops = load_npy::<i64>(&path("hops")).data;
    let offsets = load_npy::<i64>(&path("fft_offsets")).data;
    let want_db = load_npy::<f32>(&path("fft_raw_db")).data;
    let want_proc = load_npy::<f32>(&path("fft_processed")).data;

    let (t_lmh, t_bpm) = (tol(m, "raw"), tol(m, "bpm"));
    let (t_db, t_proc) = (tol(m, "fft_raw_db"), tol(m, "fft_processed"));

    let mut dsp = DspChain::new(&p);
    let mut fft = FftChain::new(&p).expect("fft chain");
    let events = case["events"].as_array().cloned().unwrap_or_default();
    let mut ev = 0;
    let mut hop = 0;
    let n_hops = hops.len() / 2;

    let mut r = CaseResult {
        id: id.clone(),
        raw: MaxErr::default(),
        scaled: MaxErr::default(),
        bpm: MaxErr::default(),
        fft_raw_db: MaxErr::default(),
        fft_processed: MaxErr::default(),
        onset_mismatches: Vec::new(),
        sentinel_mismatches: 0,
    };

    for b in 0..n_blocks {
        while ev < events.len() && u(&events[ev]["block"]) == b {
            for call in events[ev]["calls"].as_array().unwrap() {
                if call["depth"].as_u64() == Some(0) {
                    apply_call(call, &mut dsp, &mut fft);
                }
            }
            ev += 1;
        }
        let out = dsp.process_block(&sig[b * bs..(b + 1) * bs]);
        for i in 0..3 {
            r.raw.add(out.raw[i], raw[b * 3 + i], t_lmh, || {
                format!("block {b} band {i}")
            });
            r.scaled.add(out.scaled[i], scaled[b * 3 + i], t_lmh, || {
                format!("block {b} band {i}")
            });
            if out.onsets[i] != onsets[b * 3 + i] {
                r.onset_mismatches.push(format!(
                    "block {b} band {i}: got {} want {}",
                    out.onsets[i],
                    onsets[b * 3 + i]
                ));
            }
        }
        r.bpm.add(out.bpm, bpm[b], t_bpm, || format!("block {b}"));

        while hop < n_hops && hops[hop * 2 + 1] as usize == b {
            let start = hops[hop * 2] as usize;
            let win = fft.window_mut();
            let s0 = start * bs;
            let n = win.len();
            for (w, &x) in win.iter_mut().zip(&sig[s0..s0 + n]) {
                *w = x as f64;
            }
            let (db, proc) = fft.process_hop();
            let (o0, o1) = (offsets[hop] as usize, offsets[hop + 1] as usize);
            assert_eq!(db.len(), o1 - o0, "{id}: hop {hop} frame length");
            for k in 0..db.len() {
                let (g, w) = (db[k], want_db[o0 + k]);
                if (g == dsp::fft::EMPTY_BIN_SENTINEL) != (w == dsp::fft::EMPTY_BIN_SENTINEL) {
                    r.sentinel_mismatches += 1;
                }
                r.fft_raw_db
                    .add(g as f64, w as f64, t_db, || format!("hop {hop} bin {k}"));
                r.fft_processed
                    .add(proc[k] as f64, want_proc[o0 + k] as f64, t_proc, || {
                        format!("hop {hop} bin {k}")
                    });
            }
            hop += 1;
        }
    }
    assert_eq!(hop, n_hops, "{id}: not every hop row was replayed");
    assert_eq!(ev, events.len(), "{id}: not every event was replayed");
    r
}

#[test]
fn replay_matrix_within_tolerance() {
    let m = require_goldens!();
    let only = std::env::var("PARITY_CASE").ok();
    let mut bad = Vec::new();
    eprintln!(
        "{:<16} {:>9} {:>9} {:>9} {:>9} {:>9} {:>4} {:>4} {:>8}",
        "case", "raw", "scaled", "bpm", "fft_db", "fft_proc", "ons", "sent", "n_fft"
    );
    for case in m["cases"].as_array().unwrap() {
        if only
            .as_deref()
            .is_some_and(|o| case["id"].as_str() != Some(o))
        {
            continue;
        }
        let r = replay_case(&m, case);
        eprintln!(
            "{:<16} {:>9.1e} {:>9.1e} {:>9.1e} {:>9.1e} {:>9.1e} {:>4} {:>4} {:>8}",
            r.id,
            r.raw.max,
            r.scaled.max,
            r.bpm.max,
            r.fft_raw_db.max,
            r.fft_processed.max,
            r.onset_mismatches.len(),
            r.sentinel_mismatches,
            r.fft_raw_db.n
        );
        for (name, e) in [
            ("raw", &r.raw),
            ("scaled", &r.scaled),
            ("bpm", &r.bpm),
            ("fft_raw_db", &r.fft_raw_db),
            ("fft_processed", &r.fft_processed),
        ] {
            if let Some(first) = &e.first_over {
                bad.push(format!(
                    "{}/{name}: max {:e} at {}; first over tol at {first}",
                    r.id, e.max, e.at
                ));
            }
        }
        if !r.onset_mismatches.is_empty() {
            bad.push(format!(
                "{}/onsets: {} mismatches, first {}",
                r.id,
                r.onset_mismatches.len(),
                r.onset_mismatches[0]
            ));
        }
        if r.sentinel_mismatches > 0 {
            bad.push(format!(
                "{}/fft sentinel positions differ in {} bins",
                r.id, r.sentinel_mismatches
            ));
        }
    }
    assert!(bad.is_empty(), "parity failures:\n{}", bad.join("\n"));
}
