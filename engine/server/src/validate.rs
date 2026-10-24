//! Pure validators for control messages and persisted config values
//! (`server/control/validate.py`). Single source of truth for value ranges.
//!
//! Philosophy (unchanged from Python): only enforce invariants that would
//! crash the server, poison state with NaN/inf, or break a security boundary
//! (preset names hit the filesystem). Errors are the Python `ValueError`
//! messages, word for word.

use crate::py::{as_int, fmt_g, is_number, repr_str};
use serde_json::Value;

pub type VResult<T> = Result<T, String>;

pub const BAND_NAMES: [&str; 3] = ["low", "mid", "high"];

pub fn band_index(name: &str) -> Option<usize> {
    BAND_NAMES.iter().position(|b| *b == name)
}

fn finite_float(x: Option<&Value>, name: &str, gt: Option<f64>, ge: Option<f64>) -> VResult<f64> {
    let v = match x {
        Some(v) if is_number(v) => v.as_f64().unwrap_or(f64::NAN),
        _ => return Err(format!("{name} must be numeric")),
    };
    if !v.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    if let Some(g) = gt
        && v <= g
    {
        return Err(format!("{name} must be > {}", fmt_g(g)));
    }
    if let Some(g) = ge
        && v < g
    {
        return Err(format!("{name} must be >= {}", fmt_g(g)));
    }
    Ok(v)
}

/// `validate_band(name, lo, hi)` (the `sr=None` form every caller uses).
pub fn validate_band(name: &str, lo: Option<&Value>, hi: Option<&Value>) -> VResult<(f64, f64)> {
    let lo = finite_float(lo, &format!("{name}.lo_hz"), None, None)?;
    let hi = finite_float(hi, &format!("{name}.hi_hz"), None, None)?;
    if lo <= 0.0 {
        return Err(format!("{name}: lo_hz must be > 0"));
    }
    if hi <= lo {
        return Err(format!("{name}: hi_hz must be > lo_hz"));
    }
    Ok((lo, hi))
}

/// `{low: {lo_hz, hi_hz}, mid: …, high: …}` → `[(lo, hi); 3]`.
pub fn validate_bands(bands: &Value) -> VResult<[(f64, f64); 3]> {
    let Value::Object(m) = bands else {
        return Err("bands must be a dict".into());
    };
    let mut out = [(0.0, 0.0); 3];
    for (i, name) in BAND_NAMES.iter().enumerate() {
        let b = match m.get(*name) {
            Some(b @ Value::Object(_)) => b,
            _ => return Err(format!("missing band {}", repr_str(name))),
        };
        out[i] = validate_band(name, crate::py::get(b, "lo_hz"), crate::py::get(b, "hi_hz"))?;
    }
    Ok(out)
}

/// Partial per-band tau dict: only the bands present, in the given order.
pub fn validate_tau(tau: &Value) -> VResult<Vec<(usize, f64)>> {
    let Value::Object(m) = tau else {
        return Err("tau must be a dict".into());
    };
    let mut out = Vec::new();
    for (k, v) in m {
        let Some(i) = band_index(k) else {
            return Err(format!("unknown band {}", repr_str(k)));
        };
        out.push((i, finite_float(Some(v), &format!("tau[{k}]"), Some(0.0), None)?));
    }
    Ok(out)
}

pub fn validate_filter_order(n: Option<&Value>) -> VResult<usize> {
    let Some(n) = n.and_then(as_int) else {
        return Err("filter_order must be an int".into());
    };
    if !(1..=8).contains(&n) {
        return Err("filter_order must be in 1..8".into());
    }
    Ok(n as usize)
}

/// macOS caps outgoing UDP datagrams at 9216 bytes (`net.inet.udp.maxdgram`).
pub const OSC_MAX_UDP_DATAGRAM: usize = 9216;

/// Size in bytes of the `/audio/fft ,f…f <f32·n>` OSC packet.
pub const fn osc_fft_packet_bytes(n_bins: usize) -> usize {
    12 + 4 * ((n_bins + 1) / 4 + 1) + 4 * n_bins
}

pub const MIN_N_FFT_BINS: usize = 16;
pub const MAX_N_FFT_BINS: usize = 1024;
const _: () = assert!(osc_fft_packet_bytes(MAX_N_FFT_BINS) <= OSC_MAX_UDP_DATAGRAM);

/// Positive int, clamped into `[MIN_N_FFT_BINS, MAX_N_FFT_BINS]`.
pub fn validate_n_fft_bins(n: Option<&Value>) -> VResult<usize> {
    let Some(n) = n.and_then(as_int) else {
        return Err("n_fft_bins must be an int".into());
    };
    if n < 1 {
        return Err("n_fft_bins must be >= 1".into());
    }
    Ok(n.clamp(MIN_N_FFT_BINS as i128, MAX_N_FFT_BINS as i128) as usize)
}

/// Largest FFT window (samples); the slot ring holds ≥ 2× this.
pub const MAX_FFT_WINDOW: usize = 16384;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FftWindowOk {
    pub window_size: Option<usize>,
    pub hop: Option<usize>,
    pub f_min: Option<f64>,
}

fn block_multiple(v: &Value, name: &str, blocksize: usize) -> VResult<usize> {
    let Some(n) = as_int(v) else {
        return Err(format!("{name} must be an int"));
    };
    if n < blocksize as i128 || n % blocksize as i128 != 0 {
        return Err(format!("{name} must be a positive multiple of blocksize ({blocksize})"));
    }
    if n > MAX_FFT_WINDOW as i128 {
        return Err(format!("{name} must be <= {MAX_FFT_WINDOW}"));
    }
    Ok(n as usize)
}

/// FFT window geometry; only the keys provided (`None` = not provided).
pub fn validate_fft_window(
    window_size: Option<&Value>,
    hop: Option<&Value>,
    f_min: Option<&Value>,
    blocksize: usize,
) -> VResult<FftWindowOk> {
    let mut out = FftWindowOk::default();
    if let Some(w) = window_size.filter(|v| !v.is_null()) {
        out.window_size = Some(block_multiple(w, "window_size", blocksize)?);
    }
    if let Some(h) = hop.filter(|v| !v.is_null()) {
        out.hop = Some(block_multiple(h, "hop", blocksize)?);
    }
    if let Some(f) = f_min.filter(|v| !v.is_null()) {
        out.f_min = Some(finite_float(Some(f), "f_min", Some(0.0), None)?);
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AutoscaleOk {
    pub tau_attack_s: Option<f64>,
    pub tau_release_s: Option<f64>,
    pub noise_floor: Option<f64>,
    pub strength: Option<f64>,
    pub master_gain: Option<f64>,
}

impl AutoscaleOk {
    pub fn is_empty(&self) -> bool {
        *self == AutoscaleOk::default()
    }
}

/// Each argument: `None` = not provided (Python passes `msg.get(k)`).
pub fn validate_autoscale(
    tau_attack_s: Option<&Value>,
    tau_release_s: Option<&Value>,
    noise_floor: Option<&Value>,
    strength: Option<&Value>,
    master_gain: Option<&Value>,
) -> VResult<AutoscaleOk> {
    let opt = |v: Option<&Value>, name: &str, gt: Option<f64>| -> VResult<Option<f64>> {
        match v.filter(|v| !v.is_null()) {
            None => Ok(None),
            Some(v) => finite_float(Some(v), name, gt, None).map(Some),
        }
    };
    Ok(AutoscaleOk {
        tau_attack_s: opt(tau_attack_s, "tau_attack_s", Some(0.0))?,
        tau_release_s: opt(tau_release_s, "tau_release_s", Some(0.0))?,
        noise_floor: opt(noise_floor, "noise_floor", None)?,
        strength: opt(strength, "strength", None)?,
        master_gain: opt(master_gain, "master_gain", None)?,
    })
}

pub fn validate_peak_smear_oct(v: Option<&Value>) -> VResult<f64> {
    finite_float(v, "peak_smear_oct", None, None)
}

pub const FFT_TILT_RANGE: (f64, f64) = (-6.0, 12.0);
pub const PEAK_DECAY_RANGE: (f64, f64) = (0.05, 3.0);

fn clamp(v: f64, r: (f64, f64)) -> f64 {
    v.max(r.0).min(r.1)
}

pub fn validate_fft_tilt_db_per_oct(v: Option<&Value>) -> VResult<f64> {
    Ok(clamp(finite_float(v, "tilt_db_per_oct", None, None)?, FFT_TILT_RANGE))
}

pub fn validate_peak_decay_per_s(v: Option<&Value>) -> VResult<f64> {
    Ok(clamp(finite_float(v, "peak_decay_per_s", None, None)?, PEAK_DECAY_RANGE))
}

pub const FFT_VIEWS: [&str; 2] = ["2d", "3d"];
pub const FFT3D_STYLES: [&str; 4] = ["cloud", "topo", "spectrogram", "partials"];
pub const PALETTES: [&str; 7] = ["classic", "neon", "ember", "ocean", "sunset", "mono", "okabe"];

fn py_list(items: &[&str]) -> String {
    let q: Vec<String> = items.iter().map(|s| repr_str(s)).collect();
    format!("[{}]", q.join(", "))
}

fn one_of(v: Option<&Value>, allowed: &[&str], what: &str) -> VResult<String> {
    match v {
        Some(Value::String(s)) if allowed.contains(&s.as_str()) => Ok(s.clone()),
        _ => Err(format!("{what} must be one of {}", py_list(allowed))),
    }
}

pub fn validate_fft_view(v: Option<&Value>) -> VResult<String> {
    one_of(v, &FFT_VIEWS, "fft_view")
}

pub fn validate_fft3d_style(v: Option<&Value>) -> VResult<String> {
    one_of(v, &FFT3D_STYLES, "fft3d_style")
}

pub fn validate_palette(v: Option<&Value>) -> VResult<String> {
    one_of(v, &PALETTES, "palette")
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OnsetOk {
    pub sensitivity: Option<f64>,
    pub refractory_s: Option<f64>,
    pub slow_tau_s: Option<f64>,
    pub abs_floor: Option<f64>,
}

impl OnsetOk {
    pub fn is_empty(&self) -> bool {
        *self == OnsetOk::default()
    }
}

pub fn validate_onset(
    sensitivity: Option<&Value>,
    refractory_s: Option<&Value>,
    slow_tau_s: Option<&Value>,
    abs_floor: Option<&Value>,
) -> VResult<OnsetOk> {
    let opt = |v: Option<&Value>, name: &str, gt: Option<f64>, ge: Option<f64>| -> VResult<Option<f64>> {
        match v.filter(|v| !v.is_null()) {
            None => Ok(None),
            Some(v) => finite_float(Some(v), name, gt, ge).map(Some),
        }
    };
    Ok(OnsetOk {
        sensitivity: opt(sensitivity, "sensitivity", Some(1.0), None)?,
        refractory_s: opt(refractory_s, "refractory_s", Some(0.0), None)?,
        slow_tau_s: opt(slow_tau_s, "slow_tau_s", Some(0.0), None)?,
        abs_floor: opt(abs_floor, "abs_floor", None, Some(0.0))?,
    })
}

pub fn validate_onset_band(band: Option<&Value>) -> VResult<usize> {
    match band {
        Some(Value::String(s)) if band_index(s).is_some() => Ok(band_index(s).unwrap_or(0)),
        _ => Err("band must be one of ('low', 'mid', 'high')".into()),
    }
}

/// Rounded (half to even, like Python's `round`), then clamped to 15..240.
pub fn validate_ws_snapshot_hz(hz: Option<&Value>) -> VResult<i64> {
    let v = finite_float(hz, "hz", None, None)?;
    Ok((v.round_ties_even() as i64).clamp(15, 240))
}

/// Preset names hit the filesystem: `^[A-Za-z0-9_\- ]+$`, 1–64 chars after
/// strip, `main` reserved. Returns the stripped name.
pub fn validate_preset_name(name: Option<&Value>) -> VResult<String> {
    let Some(Value::String(s)) = name else {
        return Err("preset name must be a string".into());
    };
    let n = s.trim();
    let len = n.chars().count();
    if !(1..=64).contains(&len) {
        return Err("preset name must be 1-64 characters".into());
    }
    if n.to_lowercase() == "main" {
        return Err("'main' is reserved for the active config file".into());
    }
    if !n
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == ' ')
    {
        return Err(
            "preset name may only contain letters, digits, spaces, hyphens, underscores".into(),
        );
    }
    Ok(n.to_string())
}

pub const CARD_IDS: [&str; 4] = ["bars", "lines", "scene", "fft"];

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct UiLayout {
    pub split_x: f64,
    pub split_y: f64,
    pub quadrants: Vec<String>,
}

/// Tiling 2×2 layout; quadrants must be a permutation of `CARD_IDS`.
pub fn validate_ui_layout(layout: &Value) -> VResult<UiLayout> {
    if !layout.is_object() {
        return Err("layout must be a dict".into());
    }
    let sx = finite_float(crate::py::get(layout, "split_x"), "split_x", None, None)?;
    let sy = finite_float(crate::py::get(layout, "split_y"), "split_y", None, None)?;
    let quads = match layout.get("quadrants") {
        Some(Value::Array(a)) if a.len() == 4 => a,
        _ => return Err("quadrants must be a 4-item list (TL, TR, BL, BR)".into()),
    };
    let mut got: Vec<&str> = Vec::with_capacity(4);
    for q in quads {
        match q {
            Value::String(s) => got.push(s),
            _ => {
                return Err(format!(
                    "quadrants must be a permutation of {}",
                    py_list(&CARD_IDS)
                ));
            }
        }
    }
    let mut sorted = got.clone();
    sorted.sort_unstable();
    let mut want = CARD_IDS.to_vec();
    want.sort_unstable();
    if sorted != want {
        return Err(format!("quadrants must be a permutation of {}", py_list(&CARD_IDS)));
    }
    Ok(UiLayout {
        split_x: sx,
        split_y: sy,
        quadrants: got.into_iter().map(String::from).collect(),
    })
}

pub fn validate_blocksize(n: Option<&Value>) -> VResult<usize> {
    let Some(n) = n.and_then(as_int) else {
        return Err("blocksize must be an int".into());
    };
    if !(16..=MAX_FFT_WINDOW as i128).contains(&n) {
        return Err(format!("blocksize must be in 16..{MAX_FFT_WINDOW}"));
    }
    Ok(n as usize)
}

pub fn validate_device_index(idx: Option<&Value>) -> VResult<usize> {
    let Some(i) = idx.and_then(as_int) else {
        return Err("device index must be an int".into());
    };
    if i < 0 {
        return Err("device index must be >= 0".into());
    }
    Ok(i as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn messages_match_python() {
        assert_eq!(
            validate_band("low", Some(&json!(500.0)), Some(&json!(100.0))).unwrap_err(),
            "low: hi_hz must be > lo_hz"
        );
        assert_eq!(
            validate_band("low", None, Some(&json!(100.0))).unwrap_err(),
            "low.lo_hz must be numeric"
        );
        assert_eq!(
            validate_tau(&json!({"low": 0})).unwrap_err(),
            "tau[low] must be > 0"
        );
        assert_eq!(
            validate_tau(&json!({"bass": 1})).unwrap_err(),
            "unknown band 'bass'"
        );
        assert_eq!(validate_n_fft_bins(Some(&json!(5000))).unwrap(), 1024);
        assert_eq!(validate_n_fft_bins(Some(&json!(3))).unwrap(), 16);
        assert!(validate_n_fft_bins(Some(&json!(128.0))).is_err());
        assert!(validate_n_fft_bins(Some(&json!(true))).is_err());
        assert_eq!(validate_ws_snapshot_hz(Some(&json!(60.5))).unwrap(), 60);
        assert_eq!(validate_ws_snapshot_hz(Some(&json!(61.5))).unwrap(), 62);
        assert_eq!(validate_ws_snapshot_hz(Some(&json!(1000))).unwrap(), 240);
        assert_eq!(
            validate_fft_view(Some(&json!("4d"))).unwrap_err(),
            "fft_view must be one of ['2d', '3d']"
        );
        assert_eq!(
            validate_onset(Some(&json!(1.0)), None, None, None).unwrap_err(),
            "sensitivity must be > 1"
        );
        assert_eq!(
            validate_onset(None, None, None, Some(&json!(-1))).unwrap_err(),
            "abs_floor must be >= 0"
        );
        assert_eq!(validate_preset_name(Some(&json!("  a b "))).unwrap(), "a b");
        assert!(validate_preset_name(Some(&json!("Main"))).is_err());
        assert!(validate_preset_name(Some(&json!("../x"))).is_err());
        assert!(
            validate_fft_window(Some(&json!(1000)), None, None, 256)
                .unwrap_err()
                .contains("multiple of blocksize (256)")
        );
        assert_eq!(
            validate_ui_layout(&json!({"split_x": 0.5, "split_y": 0.5, "quadrants": ["fft", "bars", "lines", "scene"]}))
                .unwrap()
                .quadrants[0],
            "fft"
        );
    }
}
