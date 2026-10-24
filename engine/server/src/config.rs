//! Config schema, strict YAML load, atomic write (`server/config.py`).
//!
//! The loader reproduces the reference's split: a missing section/key, a
//! YAML parse error, an empty file or a non-mapping root is *corrupt* (backed
//! up and reseeded from the example); a parseable file with an invalid value
//! is a fatal startup error. A short list of optional keys take defaults.
//! No legacy migrations (decided by Xander: `beat:` and the old layout are
//! not read).

use crate::py::{self, CoerceError};
use crate::validate::{self as v, UiLayout};
use serde::Serialize;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DeviceCfg {
    pub name: Option<String>,
    pub index: Option<i64>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct AudioCfg {
    pub device: DeviceCfg,
    pub blocksize: usize,
    pub channels: i64,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct BandCfg {
    pub lo_hz: f64,
    pub hi_hz: f64,
}

/// A `{low, mid, high}` mapping (the reference's per-band dicts).
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct PerBand {
    pub low: f64,
    pub mid: f64,
    pub high: f64,
}

impl PerBand {
    pub fn arr(&self) -> [f64; 3] {
        [self.low, self.mid, self.high]
    }
    pub fn set(&mut self, i: usize, x: f64) {
        match i {
            0 => self.low = x,
            1 => self.mid = x,
            _ => self.high = x,
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DspCfg {
    pub low: BandCfg,
    pub mid: BandCfg,
    pub high: BandCfg,
    pub tau: PerBand,
    pub tau_attack: PerBand,
    pub filter_order: usize,
}

impl DspCfg {
    pub fn bands(&self) -> dsp::Bands {
        [
            (self.low.lo_hz, self.low.hi_hz),
            (self.mid.lo_hz, self.mid.hi_hz),
            (self.high.lo_hz, self.high.hi_hz),
        ]
    }
    pub fn band_mut(&mut self, i: usize) -> &mut BandCfg {
        match i {
            0 => &mut self.low,
            1 => &mut self.mid,
            _ => &mut self.high,
        }
    }
    pub fn set_bands(&mut self, b: dsp::Bands) {
        for (i, (lo, hi)) in b.into_iter().enumerate() {
            *self.band_mut(i) = BandCfg { lo_hz: lo, hi_hz: hi };
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct AutoscaleCfg {
    pub tau_attack_s: f64,
    pub tau_release_s: f64,
    pub noise_floor: f64,
    pub strength: f64,
    pub master_gain: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct FftCfg {
    pub enabled: bool,
    pub n_bins: usize,
    pub window_size: usize,
    pub hop: usize,
    pub f_min: f64,
    pub db_floor: f64,
    pub db_ceiling: f64,
    pub peak_smear_oct: f64,
    pub tilt_db_per_oct: f64,
    pub send_raw_db: bool,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct OnsetBandCfg {
    pub sensitivity: f64,
    pub refractory_s: f64,
    pub slow_tau_s: f64,
    pub abs_floor: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OnsetCfg {
    pub low: OnsetBandCfg,
    pub mid: OnsetBandCfg,
    pub high: OnsetBandCfg,
}

impl OnsetCfg {
    pub fn band_mut(&mut self, i: usize) -> &mut OnsetBandCfg {
        match i {
            0 => &mut self.low,
            1 => &mut self.mid,
            _ => &mut self.high,
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OscDest {
    pub host: String,
    pub port: i64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OscCfg {
    pub destinations: Vec<OscDest>,
    pub send_fft: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct WsCfg {
    pub enabled: bool,
    pub host: String,
    pub port: i64,
    pub snapshot_hz: i64,
    pub http_port: i64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct UiCfg {
    pub peak_decay_per_s: f64,
    pub layout: UiLayout,
    pub show_onsets: bool,
    pub fft_view: String,
    pub fft3d_style: String,
    pub palette: String,
}

/// Field order = the reference dataclasses, so the YAML a persister writes
/// has the same key order as PyYAML's `safe_dump(sort_keys=False)`.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Config {
    pub audio: AudioCfg,
    pub dsp: DspCfg,
    pub autoscale: AutoscaleCfg,
    pub fft: FftCfg,
    pub onset: OnsetCfg,
    pub osc: OscCfg,
    pub ws: WsCfg,
    pub ui: UiCfg,
}

/// Per-band onset defaults for absent keys (`config.py` `onset_defaults`).
pub const ONSET_DEFAULTS: [OnsetBandCfg; 3] = [
    OnsetBandCfg { sensitivity: 1.8, refractory_s: 0.25, slow_tau_s: 0.30, abs_floor: 0.10 },
    OnsetBandCfg { sensitivity: 1.8, refractory_s: 0.18, slow_tau_s: 0.20, abs_floor: 0.10 },
    OnsetBandCfg { sensitivity: 1.8, refractory_s: 0.12, slow_tau_s: 0.15, abs_floor: 0.10 },
];

#[derive(Debug)]
pub enum LoadError {
    /// Unreadable as a config (parse error, empty, missing key): back up + reseed.
    Corrupt(String),
    /// Parseable but an invalid value: fatal.
    Invalid(String),
    /// The file is missing or unreadable.
    Io(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Corrupt(s) | LoadError::Invalid(s) | LoadError::Io(s) => f.write_str(s),
        }
    }
}

/// Error while building a Config from a parsed value. `Missing` mirrors the
/// reference's KeyError/TypeError (→ corrupt), `Invalid` its ValueError.
enum BuildErr {
    Missing(String),
    Invalid(String),
}

impl From<String> for BuildErr {
    fn from(s: String) -> Self {
        BuildErr::Invalid(s)
    }
}

impl From<CoerceError> for BuildErr {
    fn from(e: CoerceError) -> Self {
        match e {
            CoerceError::Type(s) => BuildErr::Missing(s),
            CoerceError::Value(s) => BuildErr::Invalid(s),
        }
    }
}

/// `d[key]` on a mapping: KeyError / TypeError → Missing.
fn req<'a>(d: &'a Value, key: &str) -> Result<&'a Value, BuildErr> {
    match d {
        Value::Object(m) => m.get(key).ok_or_else(|| BuildErr::Missing(py::repr_str(key))),
        other => Err(BuildErr::Missing(format!(
            "'{}' object is not subscriptable",
            py::type_name(other)
        ))),
    }
}

/// `d.get(key)` where `d` must be a mapping (an AttributeError otherwise,
/// which the reference doesn't catch; reported as an invalid value).
fn opt<'a>(d: &'a Value, key: &str) -> Result<Option<&'a Value>, BuildErr> {
    match d {
        Value::Object(m) => Ok(m.get(key)),
        other => Err(BuildErr::Invalid(format!(
            "'{}' object has no attribute 'get'",
            py::type_name(other)
        ))),
    }
}

/// A validator that returned no value for a required field (the reference
/// then fails a dataclass constructor with TypeError → corrupt).
fn need<T>(x: Option<T>, what: &str) -> Result<T, BuildErr> {
    x.ok_or_else(|| BuildErr::Missing(format!("missing required argument: {}", py::repr_str(what))))
}

fn per_band_full(parts: Vec<(usize, f64)>, what: &str) -> Result<PerBand, BuildErr> {
    let mut have = [None; 3];
    for (i, x) in parts {
        have[i] = Some(x);
    }
    let mut out = PerBand { low: 0.0, mid: 0.0, high: 0.0 };
    for (i, h) in have.into_iter().enumerate() {
        match h {
            Some(x) => out.set(i, x),
            None => {
                return Err(BuildErr::Invalid(format!(
                    "{what} missing band {}",
                    py::repr_str(v::BAND_NAMES[i])
                )));
            }
        }
    }
    Ok(out)
}

fn build_config(d: &Value) -> Result<Config, BuildErr> {
    let a = req(d, "audio")?;
    let dev = match opt(a, "device")? {
        Some(x) if py::truthy(x) => x.clone(),
        _ => Value::Object(Default::default()),
    };
    let name = match opt(&dev, "name")? {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(BuildErr::Invalid("audio.device.name must be a string".into())),
    };
    let index = match opt(&dev, "index")? {
        None | Some(Value::Null) => None,
        Some(x) => match py::as_int(x) {
            Some(i) => Some(i as i64),
            None => return Err(BuildErr::Invalid("audio.device.index must be an int".into())),
        },
    };
    let blocksize = v::validate_blocksize(Some(req(a, "blocksize")?))?;
    let audio = AudioCfg {
        device: DeviceCfg { name, index },
        blocksize,
        channels: py::py_int(req(a, "channels")?)?,
    };

    let ds = req(d, "dsp")?;
    let mut bands_raw = serde_json::Map::new();
    for k in v::BAND_NAMES {
        bands_raw.insert(k.into(), req(ds, k)?.clone());
    }
    let ok = v::validate_bands(&Value::Object(bands_raw))?;
    let band = |i: usize| BandCfg { lo_hz: ok[i].0, hi_hz: ok[i].1 };
    let dsp = DspCfg {
        low: band(0),
        mid: band(1),
        high: band(2),
        tau: per_band_full(v::validate_tau(req(ds, "tau")?)?, "tau")?,
        tau_attack: per_band_full(v::validate_tau(req(ds, "tau_attack")?)?, "tau_attack")?,
        filter_order: v::validate_filter_order(Some(
            opt(ds, "filter_order")?.unwrap_or(&Value::from(2)),
        ))?,
    };

    let asd = req(d, "autoscale")?;
    let ok = v::validate_autoscale(
        Some(req(asd, "tau_attack_s")?),
        Some(req(asd, "tau_release_s")?),
        Some(req(asd, "noise_floor")?),
        Some(req(asd, "strength")?),
        Some(req(asd, "master_gain")?),
    )?;
    let autoscale = AutoscaleCfg {
        tau_attack_s: need(ok.tau_attack_s, "tau_attack_s")?,
        tau_release_s: need(ok.tau_release_s, "tau_release_s")?,
        noise_floor: need(ok.noise_floor, "noise_floor")?,
        strength: need(ok.strength, "strength")?,
        master_gain: need(ok.master_gain, "master_gain")?,
    };

    let fd = req(d, "fft")?;
    let win = v::validate_fft_window(
        Some(req(fd, "window_size")?),
        Some(req(fd, "hop")?),
        Some(req(fd, "f_min")?),
        audio.blocksize,
    )?;
    let fft = FftCfg {
        enabled: py::truthy(req(fd, "enabled")?),
        n_bins: v::validate_n_fft_bins(Some(req(fd, "n_bins")?))?,
        window_size: need(win.window_size, "window_size")?,
        hop: need(win.hop, "hop")?,
        f_min: need(win.f_min, "f_min")?,
        db_floor: py::py_float(req(fd, "db_floor")?)?,
        db_ceiling: py::py_float(req(fd, "db_ceiling")?)?,
        peak_smear_oct: v::validate_peak_smear_oct(Some(req(fd, "peak_smear_oct")?))?,
        tilt_db_per_oct: v::validate_fft_tilt_db_per_oct(Some(req(fd, "tilt_db_per_oct")?))?,
        send_raw_db: py::truthy(req(fd, "send_raw_db")?),
    };

    // A missing / non-mapping `onset:` uses the per-band defaults.
    let onset_raw = match d.get("onset") {
        Some(o @ Value::Object(_)) => o.clone(),
        _ => Value::Object(Default::default()),
    };
    let mut bands = ONSET_DEFAULTS;
    for (i, name) in v::BAND_NAMES.iter().enumerate() {
        let b = match onset_raw.get(*name) {
            Some(x) if py::truthy(x) => x.clone(),
            _ => Value::Object(Default::default()),
        };
        let dflt = ONSET_DEFAULTS[i];
        let field = |k: &str, dv: f64| -> Result<Value, BuildErr> {
            Ok(opt(&b, k)?.cloned().unwrap_or(Value::from(dv)))
        };
        let (s, r, t, f) = (
            field("sensitivity", dflt.sensitivity)?,
            field("refractory_s", dflt.refractory_s)?,
            field("slow_tau_s", dflt.slow_tau_s)?,
            field("abs_floor", dflt.abs_floor)?,
        );
        let ok = v::validate_onset(Some(&s), Some(&r), Some(&t), Some(&f))?;
        bands[i] = OnsetBandCfg {
            sensitivity: need(ok.sensitivity, "sensitivity")?,
            refractory_s: need(ok.refractory_s, "refractory_s")?,
            slow_tau_s: need(ok.slow_tau_s, "slow_tau_s")?,
            abs_floor: need(ok.abs_floor, "abs_floor")?,
        };
    }
    let onset = OnsetCfg { low: bands[0], mid: bands[1], high: bands[2] };

    let od = req(d, "osc")?;
    let dests_raw = req(od, "destinations")?;
    let Value::Array(items) = dests_raw else {
        return Err(BuildErr::Missing("osc.destinations must be a list".into()));
    };
    let mut destinations = Vec::with_capacity(items.len());
    for x in items {
        destinations.push(OscDest {
            host: py::py_str(req(x, "host")?),
            port: py::py_int(req(x, "port")?)?,
        });
    }
    let osc = OscCfg { destinations, send_fft: py::truthy(req(od, "send_fft")?) };

    let wd = req(d, "ws")?;
    let ws = WsCfg {
        enabled: py::truthy(req(wd, "enabled")?),
        host: py::py_str(req(wd, "host")?),
        port: py::py_int(req(wd, "port")?)?,
        snapshot_hz: v::validate_ws_snapshot_hz(Some(req(wd, "snapshot_hz")?))?,
        http_port: py::py_int(req(wd, "http_port")?)?,
    };

    let ud = req(d, "ui")?;
    let ui = UiCfg {
        peak_decay_per_s: v::validate_peak_decay_per_s(Some(req(ud, "peak_decay_per_s")?))?,
        layout: v::validate_ui_layout(req(ud, "layout")?)?,
        show_onsets: opt(ud, "show_onsets")?.is_some_and(py::truthy),
        fft_view: v::validate_fft_view(Some(
            opt(ud, "fft_view")?.unwrap_or(&Value::from("2d")),
        ))?,
        // Look-only UI choices: an unknown name falls back to the default.
        fft3d_style: v::validate_fft3d_style(opt(ud, "fft3d_style")?)
            .unwrap_or_else(|_| "cloud".into()),
        palette: v::validate_palette(opt(ud, "palette")?).unwrap_or_else(|_| "classic".into()),
    };

    Ok(Config { audio, dsp, autoscale, fft, onset, osc, ws, ui })
}

/// YAML → JSON value with the reference's scalar resolution. Non-finite
/// floats come through as strings, so `.inf` is "must be numeric" (invalid),
/// not a parse error.
pub fn parse_yaml(text: &str) -> Result<Value, String> {
    let mut opts = serde_saphyr::Options::default();
    opts.non_finite_float_policy = serde_saphyr::NonFiniteFloatPolicy::AsString;
    serde_saphyr::from_str_with_options::<Value>(text, opts).map_err(|e| e.to_string())
}

/// `load_config(path)`: no fallbacks, no merging.
pub fn load_config(path: &Path) -> Result<Config, LoadError> {
    if !path.exists() {
        return Err(LoadError::Io(format!("config file not found: {}", path.display())));
    }
    let text = std::fs::read(path)
        .map_err(|e| LoadError::Io(format!("{}: {e}", path.display())))?;
    let text = String::from_utf8(text).map_err(|e| {
        LoadError::Corrupt(format!("config {} failed to parse: {e}", path.display()))
    })?;
    let raw = parse_yaml(&text)
        .map_err(|e| LoadError::Corrupt(format!("config {} failed to parse: {e}", path.display())))?;
    if !raw.is_object() {
        return Err(LoadError::Corrupt(format!(
            "config root must be a YAML mapping (empty file?): {}",
            path.display()
        )));
    }
    build_config(&raw).map_err(|e| match e {
        BuildErr::Missing(k) => LoadError::Corrupt(format!(
            "config {} missing/malformed required key: {k}",
            path.display()
        )),
        BuildErr::Invalid(s) => LoadError::Invalid(s),
    })
}

/// Local-time `strftime` (the reference stamps backups / presets in local time).
pub fn local_strftime(fmt: &str, t: libc::time_t) -> String {
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: localtime_r writes into the provided struct; strftime writes at
    // most `buf.len()` bytes and NUL-terminates.
    unsafe {
        libc::localtime_r(&t, &mut tm);
        let cfmt = std::ffi::CString::new(fmt).unwrap_or_default();
        let mut buf = [0u8; 64];
        let n = libc::strftime(buf.as_mut_ptr().cast(), buf.len(), cfmt.as_ptr(), &tm);
        String::from_utf8_lossy(&buf[..n]).into_owned()
    }
}

pub fn now_time_t() -> libc::time_t {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as libc::time_t)
        .unwrap_or(0)
}

/// `load_config_or_fallback`: a corrupt file is moved aside as
/// `<name>.corrupt-<YYYYmmdd-HHMMSS>[-n]`, reseeded from `example`, and the
/// copy loaded instead. Invalid values still fail.
pub fn load_config_or_fallback(path: &Path, example: &Path) -> Result<Config, LoadError> {
    match load_config(path) {
        Err(LoadError::Corrupt(msg)) => {
            if !example.exists() {
                return Err(LoadError::Corrupt(msg));
            }
            let stamp = local_strftime("%Y%m%d-%H%M%S", now_time_t());
            let fname = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let mut backup = path.with_file_name(format!("{fname}.corrupt-{stamp}"));
            let mut n = 1;
            while backup.exists() {
                backup = path.with_file_name(format!("{fname}.corrupt-{stamp}-{n}"));
                n += 1;
            }
            tracing::warn!(
                "{msg} — backing it up as {} and falling back to {} defaults",
                backup.file_name().map(|s| s.to_string_lossy()).unwrap_or_default(),
                example.file_name().map(|s| s.to_string_lossy()).unwrap_or_default()
            );
            std::fs::rename(path, &backup).map_err(|e| LoadError::Io(e.to_string()))?;
            std::fs::copy(example, path).map_err(|e| LoadError::Io(e.to_string()))?;
            load_config(path)
        }
        other => other,
    }
}

/// The YAML the persister / preset writer produce: field order of the
/// dataclasses, floats that PyYAML reads back as floats.
pub fn to_yaml(cfg: &Config) -> Result<String, String> {
    serde_saphyr::to_string(cfg).map_err(|e| e.to_string())
}

/// Atomic, durable write: tmp file in the same dir → fsync → rename → fsync
/// the directory.
pub fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    std::fs::create_dir_all(&dir)?;
    let fname = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut tmp = None;
    for attempt in 0..100u32 {
        let cand = dir.join(format!("{fname}.{}{attempt}.tmp", std::process::id()));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&cand) {
            Ok(f) => {
                tmp = Some((cand, f));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    let Some((tmp_path, mut f)) = tmp else {
        return Err(std::io::Error::other("could not create a temp file"));
    };
    let res = (|| {
        f.write_all(data)?;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp_path, path)?;
        if let Ok(d) = std::fs::File::open(&dir) {
            let _ = d.sync_all();
        }
        Ok(())
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    res
}

pub fn write_yaml_atomic(path: &Path, cfg: &Config) -> std::io::Result<()> {
    let text = to_yaml(cfg).map_err(std::io::Error::other)?;
    write_atomic(path, text.as_bytes())
}

/// `_deep_merge`: overlay wins; nested mappings merge key by key.
pub fn deep_merge(base: &Value, overlay: &Value) -> Value {
    let (Value::Object(b), Value::Object(o)) = (base, overlay) else {
        return overlay.clone();
    };
    let mut out = b.clone();
    for (k, v) in o {
        let merged = match (v, out.get(k)) {
            (Value::Object(_), Some(bv @ Value::Object(_))) => deep_merge(bv, v),
            _ => v.clone(),
        };
        out.insert(k.clone(), merged);
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn example_text() -> String {
        std::fs::read_to_string(repo().join("configs/main.example.yaml")).unwrap()
    }

    fn load_str(text: &str) -> Result<Config, LoadError> {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("main.yaml");
        std::fs::write(&p, text).unwrap();
        load_config(&p)
    }

    #[test]
    fn loads_example_and_round_trips() {
        let mut cfg = load_str(&example_text()).unwrap();
        // The golden was captured with a named device.
        cfg.audio.device.name = Some("BlackHole 2ch".into());
        assert_eq!(cfg.audio.blocksize, 256);
        assert_eq!(cfg.dsp.filter_order, 2);
        assert_eq!(cfg.onset.mid.refractory_s, 0.18);
        let y = to_yaml(&cfg).unwrap();
        let back = load_str(&y).unwrap();
        assert_eq!(cfg, back);
        // Same key set / types as the Python-written golden.
        let golden: Value =
            parse_yaml(&std::fs::read_to_string(repo().join("testdata/golden/wire/persisted_main.yaml")).unwrap())
                .unwrap();
        let mine = parse_yaml(&y).unwrap();
        assert_eq!(shape(&golden), shape(&mine));
    }

    fn shape(v: &Value) -> Value {
        match v {
            Value::Object(m) => {
                let mut keys: Vec<_> = m.keys().cloned().collect();
                keys.sort();
                Value::Object(keys.into_iter().map(|k| (k.clone(), shape(&m[&k]))).collect())
            }
            Value::Array(a) => Value::Array(a.first().map(shape).into_iter().collect()),
            Value::Number(n) if n.is_f64() => "number".into(),
            Value::Number(_) => "int".into(),
            Value::String(_) => "str".into(),
            Value::Bool(_) => "bool".into(),
            Value::Null => "null".into(),
        }
    }

    /// Python-written files (PyYAML `safe_dump`) load here with every value
    /// intact, and what we write back parses to the same values and types.
    #[test]
    fn loads_python_written_yaml() {
        for f in ["persisted_main.yaml", "preset_saved.yaml"] {
            let p = repo().join("testdata/golden/wire").join(f);
            let cfg = load_config(&p).unwrap();
            let py: Value = parse_yaml(&std::fs::read_to_string(&p).unwrap()).unwrap();
            let mine = parse_yaml(&to_yaml(&cfg).unwrap()).unwrap();
            assert_eq!(py, mine, "{f}");
        }
    }

    #[test]
    fn corrupt_vs_invalid() {
        assert!(matches!(load_str(""), Err(LoadError::Corrupt(_))));
        assert!(matches!(load_str("a: [1"), Err(LoadError::Corrupt(_))));
        let t = example_text();
        assert!(matches!(
            load_str(&t.replace("  hop: 512\n", "")),
            Err(LoadError::Corrupt(_))
        ));
        assert!(matches!(
            load_str(&t.replace("blocksize: 256", "blocksize: 7")),
            Err(LoadError::Invalid(_))
        ));
        assert!(matches!(
            load_str(&t.replace("strength: 0.7", "strength: .inf")),
            Err(LoadError::Invalid(_))
        ));
        assert!(matches!(
            load_str(&t.replace("noise_floor: 3.213660538640315e-05", "noise_floor: null")),
            Err(LoadError::Corrupt(_))
        ));
        // Optional keys + clamps.
        let c = load_str(
            &t.replace("  filter_order: 2\n", "")
                .replace("n_bins: 128", "n_bins: 4000")
                .replace("palette: classic", "palette: nope")
                .replace("snapshot_hz: 60", "snapshot_hz: 500"),
        )
        .unwrap();
        assert_eq!(c.dsp.filter_order, 2);
        assert_eq!(c.fft.n_bins, 1024);
        assert_eq!(c.ui.palette, "classic");
        assert_eq!(c.ws.snapshot_hz, 240);
        // Missing onset block → defaults.
        let i = t.find("onset:").unwrap();
        let j = t.find("osc:").unwrap();
        let c = load_str(&format!("{}{}", &t[..i], &t[j..])).unwrap();
        assert_eq!(c.onset.high, ONSET_DEFAULTS[2]);
    }

    #[test]
    fn fallback_backs_up_and_reseeds() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("main.yaml");
        let ex = dir.path().join("main.example.yaml");
        std::fs::write(&ex, example_text()).unwrap();
        std::fs::write(&p, "").unwrap();
        let cfg = load_config_or_fallback(&p, &ex).unwrap();
        assert_eq!(cfg.fft.n_bins, 128);
        let backups: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("main.yaml.corrupt-"))
            .collect();
        assert_eq!(backups.len(), 1);
    }
}
