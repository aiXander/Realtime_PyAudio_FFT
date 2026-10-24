//! Phase-0 "empty parity test": the goldens load, and every per-case array
//! has the shape and dtype the README documents.

mod common;

use common::{load_npy, u};

#[test]
fn manifest_and_case_arrays_have_documented_shapes() {
    let m = require_goldens!();
    assert_eq!(m["format"], 1);
    let cases = m["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 27);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let files = &case["files"];
        let n_blocks = u(&case["n_blocks"]);
        let n_hops = u(&case["n_hops"]);

        for name in ["raw", "scaled"] {
            let a = load_npy::<f64>(files[name].as_str().unwrap());
            assert_eq!(
                (a.shape.as_slice(), a.dtype.as_str()),
                (&[n_blocks, 3][..], "<f8"),
                "{id}/{name}"
            );
        }
        let a = load_npy::<i8>(files["onsets"].as_str().unwrap());
        assert_eq!(
            (a.shape.as_slice(), a.dtype.as_str()),
            (&[n_blocks, 3][..], "|i1"),
            "{id}/onsets"
        );
        let a = load_npy::<f64>(files["bpm"].as_str().unwrap());
        assert_eq!(
            (a.shape.as_slice(), a.dtype.as_str()),
            (&[n_blocks][..], "<f8"),
            "{id}/bpm"
        );
        let hops = load_npy::<i64>(files["hops"].as_str().unwrap());
        assert_eq!(
            (hops.shape.as_slice(), hops.dtype.as_str()),
            (&[n_hops, 2][..], "<i8"),
            "{id}/hops"
        );
        let off = load_npy::<i64>(files["fft_offsets"].as_str().unwrap());
        assert_eq!(off.shape, vec![n_hops + 1], "{id}/fft_offsets");
        let total = *off.data.last().unwrap() as usize;
        for name in ["fft_raw_db", "fft_processed"] {
            let a = load_npy::<f32>(files[name].as_str().unwrap());
            assert_eq!(
                (a.shape.as_slice(), a.dtype.as_str()),
                (&[total][..], "<f4"),
                "{id}/{name}"
            );
        }

        // The signal covers the case's blocks.
        let sig_key = case["signal"].as_str().unwrap();
        let sig = load_npy::<f32>(m["signals"][sig_key]["file"].as_str().unwrap());
        assert!(
            sig.data.len() >= n_blocks * u(&case["blocksize"]),
            "{id}: signal too short"
        );
    }

    let sos = load_npy::<f64>(m["filter_design"]["file"].as_str().unwrap());
    let rows = m["filter_design"]["rows"].as_array().unwrap().len();
    assert_eq!(sos.shape, vec![rows, 8, 6]);
}
