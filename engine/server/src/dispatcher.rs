//! Inbound WS message dispatcher (`server/control/dispatcher.py`):
//! validate → apply → persist → reply. Returns `(targeted, broadcasts)`.
//!
//! Meta is not returned per message: state changes mark it dirty and the
//! broadcast loop sends one coalesced meta per tick. A handler validation
//! error replies `error` + an immediate targeted meta (so the UI snaps its
//! controls back); an unknown type gets the error only.

use crate::app::App;
use crate::config::{self, deep_merge};
use crate::py::{self, get, get_bool};
use crate::validate as v;
use dsp::filters::clamp_band_edges;
use serde_json::{Value, json};

/// The reference's `ValueError` (→ error + snap-back meta) vs any other
/// exception (→ `internal: …` + meta).
pub enum HErr {
    Value(String),
    Internal(String),
}

impl From<String> for HErr {
    fn from(s: String) -> Self {
        HErr::Value(s)
    }
}

pub(crate) type Out = (Vec<Value>, Vec<Value>);
type HRes = Result<Out, HErr>;

const NONE: Out = (Vec::new(), Vec::new());

/// `_preset_section`: run one preset section; `Skip` is silent, an error is
/// logged and the section skipped.
enum Section {
    Applied,
    Skip,
}

fn section(applied: &mut Vec<&'static str>, name: &'static str, r: Result<Section, String>) {
    match r {
        Ok(Section::Applied) => applied.push(name),
        Ok(Section::Skip) => {}
        Err(e) => tracing::warn!("preset {name} invalid: {e}"),
    }
}

fn as_map(v: Option<&Value>) -> Value {
    match v {
        Some(m @ Value::Object(_)) => m.clone(),
        _ => Value::Object(Default::default()),
    }
}

impl App {
    pub fn dispatch(&mut self, msg: &Value, client_id: u64) -> Out {
        let t = msg.get("type").cloned().unwrap_or(Value::Null);
        let name = t.as_str().unwrap_or("");
        let res: HRes = match name {
            "set_fft" => self.h_set_fft(msg),
            "set_band" => self.h_set_band(msg),
            "set_smoothing" => self.h_set_smoothing(msg),
            "set_autoscale" => self.h_set_autoscale(msg),
            "list_devices" => self.h_list_devices(msg, client_id),
            "set_device" => self.h_set_device(msg),
            "set_n_fft_bins" => self.h_set_n_fft_bins(msg),
            "set_ws_snapshot_hz" => self.h_set_ws_snapshot_hz(msg),
            "set_fft_send_raw_db" => self.h_set_fft_send_raw_db(msg),
            "set_fft_peak_smear" => self.h_set_fft_peak_smear(msg),
            "set_fft_tilt" => self.h_set_fft_tilt(msg),
            "set_onset" => self.h_set_onset(msg),
            "set_filter_order" => self.h_set_filter_order(msg),
            "set_peak_decay" => self.h_set_peak_decay(msg),
            "set_show_onsets" => self.h_set_show_onsets(msg),
            "set_fft_view" => self.h_set_fft_view(msg),
            "set_fft3d_style" => self.h_set_fft3d_style(msg),
            "set_palette" => self.h_set_palette(msg),
            "set_ui_layout" => self.h_set_ui_layout(msg),
            "list_presets" => Ok((vec![json!({"type": "presets", "items": self.list_presets()})], vec![])),
            "save_preset" => self.h_save_preset(msg),
            "load_preset" => self.h_load_preset(msg),
            _ => {
                return (
                    vec![json!({"type": "error", "reason": format!("unknown type {}", py::repr(&t))})],
                    vec![],
                );
            }
        };
        match res {
            Ok(out) => out,
            Err(HErr::Value(reason)) => self.error_reply(reason),
            Err(HErr::Internal(e)) => {
                tracing::error!("dispatcher handler {name} failed: {e}");
                self.error_reply(format!("internal: {e}"))
            }
        }
    }

    pub(crate) fn error_reply(&mut self, reason: String) -> Out {
        let replies = vec![json!({"type": "error", "reason": reason}), self.meta_message()];
        self.mark_meta_dirty();
        (replies, vec![])
    }

    fn meta_changed(&mut self) -> HRes {
        self.mark_meta_dirty();
        Ok(NONE)
    }

    fn h_set_fft(&mut self, msg: &Value) -> HRes {
        let enabled = get_bool(msg, "enabled", false);
        // The ENABLE toggle also drives OSC FFT transmission.
        self.set_fft_enabled(enabled);
        self.persister.request(true);
        self.meta_changed()
    }

    fn h_set_band(&mut self, msg: &Value) -> HRes {
        let band = match msg.get("band").and_then(|b| b.as_str()).and_then(v::band_index) {
            Some(i) => i,
            None => return Err("band must be 'low', 'mid', or 'high'".to_string().into()),
        };
        let commit = get_bool(msg, "commit", true);
        let sr = self.current_sr();
        let name = v::BAND_NAMES[band];
        let (lo, hi) = v::validate_band(name, get(msg, "lo_hz"), get(msg, "hi_hz"))?;
        let (lo, hi) = clamp_band_edges(lo, hi, sr);
        let b = self.cfg.dsp.band_mut(band);
        b.lo_hz = lo;
        b.hi_hz = hi;
        // IIR retune is debounced; the band-geometry knobs update now.
        self.schedule_filter_retune();
        self.apply_bands();
        self.persister.request(commit);
        self.meta_changed()
    }

    fn h_set_smoothing(&mut self, msg: &Value) -> HRes {
        let commit = get_bool(msg, "commit", true);
        let or_empty = |k: &str| match msg.get(k) {
            Some(x) if py::truthy(x) => x.clone(),
            _ => Value::Object(Default::default()),
        };
        let tau = v::validate_tau(&or_empty("tau"))?;
        let tau_atk = v::validate_tau(&or_empty("tau_attack"))?;
        for (i, x) in tau {
            self.cfg.dsp.tau.set(i, x);
        }
        for (i, x) in tau_atk {
            self.cfg.dsp.tau_attack.set(i, x);
        }
        self.apply_smoothing();
        self.persister.request(commit);
        self.meta_changed()
    }

    fn h_set_autoscale(&mut self, msg: &Value) -> HRes {
        let commit = get_bool(msg, "commit", true);
        let ok = v::validate_autoscale(
            get(msg, "tau_attack_s"),
            get(msg, "tau_release_s"),
            get(msg, "noise_floor"),
            get(msg, "strength"),
            get(msg, "master_gain"),
        )?;
        self.apply_autoscale(&ok);
        self.persister.request(commit);
        self.meta_changed()
    }

    fn h_list_devices(&mut self, msg: &Value, client_id: u64) -> HRes {
        let probe = get_bool(msg, "probe", false);
        match self.refresh_devices(probe, client_id) {
            Some(items) => Ok((vec![json!({"type": "devices", "items": items})], vec![])),
            None => Ok(NONE), // the probe replies when it finishes
        }
    }

    fn h_set_device(&mut self, msg: &Value) -> HRes {
        let idx = v::validate_device_index(get(msg, "index"))?;
        self.hot_switch_device(idx)?;
        self.persister.request(true);
        self.mark_meta_dirty();
        let items = self.list_devices_json(&crate::audio::list_input_devices());
        Ok((vec![], vec![json!({"type": "devices", "items": items})]))
    }

    fn h_set_n_fft_bins(&mut self, msg: &Value) -> HRes {
        let n = v::validate_n_fft_bins(get(msg, "n"))?;
        self.apply_fft_n_bins(n).map_err(HErr::Internal)?;
        self.persister.request(true);
        self.meta_changed()
    }

    fn h_set_ws_snapshot_hz(&mut self, msg: &Value) -> HRes {
        let commit = get_bool(msg, "commit", true);
        let hz = v::validate_ws_snapshot_hz(get(msg, "hz"))?;
        match self.ws.as_mut() {
            Some(ws) => ws.set_snapshot_hz(hz),
            None => return Err(HErr::Internal("'NoneType' object has no attribute 'set_snapshot_hz'".into())),
        }
        self.cfg.ws.snapshot_hz = hz;
        self.persister.request(commit);
        self.meta_changed()
    }

    fn h_set_fft_send_raw_db(&mut self, msg: &Value) -> HRes {
        self.set_send_raw_db(get_bool(msg, "send_raw_db", false));
        self.persister.request(true);
        self.meta_changed()
    }

    fn h_set_fft_peak_smear(&mut self, msg: &Value) -> HRes {
        let commit = get_bool(msg, "commit", true);
        let x = v::validate_peak_smear_oct(get(msg, "peak_smear_oct"))?;
        self.apply_fft_peak_smear(x);
        self.persister.request(commit);
        self.meta_changed()
    }

    fn h_set_fft_tilt(&mut self, msg: &Value) -> HRes {
        let commit = get_bool(msg, "commit", true);
        let x = v::validate_fft_tilt_db_per_oct(get(msg, "tilt_db_per_oct"))?;
        self.apply_fft_tilt(x);
        self.persister.request(commit);
        self.meta_changed()
    }

    fn h_set_filter_order(&mut self, msg: &Value) -> HRes {
        let commit = get_bool(msg, "commit", true);
        let n = v::validate_filter_order(get(msg, "order"))?;
        self.apply_filter_order(n);
        self.persister.request(commit);
        self.meta_changed()
    }

    fn h_set_onset(&mut self, msg: &Value) -> HRes {
        let commit = get_bool(msg, "commit", true);
        let band = v::validate_onset_band(get(msg, "band"))?;
        let ok = v::validate_onset(
            get(msg, "sensitivity"),
            get(msg, "refractory_s"),
            get(msg, "slow_tau_s"),
            get(msg, "abs_floor"),
        )?;
        if ok.is_empty() {
            return Err(
                "set_onset needs at least one of sensitivity / refractory_s / slow_tau_s / abs_floor"
                    .to_string()
                    .into(),
            );
        }
        self.apply_onset(band, &ok);
        self.persister.request(commit);
        self.meta_changed()
    }

    fn h_set_peak_decay(&mut self, msg: &Value) -> HRes {
        let commit = get_bool(msg, "commit", true);
        self.cfg.ui.peak_decay_per_s = v::validate_peak_decay_per_s(get(msg, "peak_decay_per_s"))?;
        self.persister.request(commit);
        self.meta_changed()
    }

    fn h_set_show_onsets(&mut self, msg: &Value) -> HRes {
        self.cfg.ui.show_onsets = get_bool(msg, "show_onsets", false);
        self.persister.request(true);
        self.meta_changed()
    }

    fn h_set_fft_view(&mut self, msg: &Value) -> HRes {
        self.cfg.ui.fft_view = v::validate_fft_view(get(msg, "view"))?;
        self.persister.request(true);
        self.meta_changed()
    }

    fn h_set_fft3d_style(&mut self, msg: &Value) -> HRes {
        self.cfg.ui.fft3d_style = v::validate_fft3d_style(get(msg, "style"))?;
        self.persister.request(true);
        self.meta_changed()
    }

    fn h_set_palette(&mut self, msg: &Value) -> HRes {
        self.cfg.ui.palette = v::validate_palette(get(msg, "palette"))?;
        self.persister.request(true);
        self.meta_changed()
    }

    fn h_set_ui_layout(&mut self, msg: &Value) -> HRes {
        let commit = get_bool(msg, "commit", true);
        let layout = match msg.get("layout") {
            Some(x) if py::truthy(x) => x.clone(),
            _ => Value::Object(Default::default()),
        };
        self.cfg.ui.layout = v::validate_ui_layout(&layout)?;
        self.persister.request(commit);
        self.meta_changed()
    }

    // ------------------------------------------------------------ presets

    fn preset_name(msg: &Value) -> Result<String, HErr> {
        let name = msg.get("name").cloned().unwrap_or(Value::from(""));
        Ok(v::validate_preset_name(Some(&name))?)
    }

    fn h_save_preset(&mut self, msg: &Value) -> HRes {
        let name = Self::preset_name(msg)?;
        let path = self.preset_path(&name);
        config::write_yaml_atomic(&path, &self.cfg).map_err(|e| HErr::Internal(e.to_string()))?;
        Ok((vec![json!({"type": "presets", "items": self.list_presets()})], vec![]))
    }

    /// Deep-merge the preset over main.yaml *as on disk*, then apply it
    /// section by section in the reference's order, each validated alone.
    /// The FFT is always reconfigured (parity trap 23).
    fn h_load_preset(&mut self, msg: &Value) -> HRes {
        let name = Self::preset_name(msg)?;
        let path = self.preset_path(&name);
        if !path.exists() {
            return Err(format!("preset {} not found", py::repr_str(&name)).into());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("preset {} parse failed: {e}", py::repr_str(&name)))?;
        let mut data = config::parse_yaml(&text)
            .map_err(|e| format!("preset {} parse failed: {e}", py::repr_str(&name)))?;
        if !py::truthy(&data) {
            data = Value::Object(Default::default());
        }
        match std::fs::read_to_string(&self.config_path).map_err(|e| e.to_string()).and_then(|t| config::parse_yaml(&t)) {
            Ok(main) => {
                if main.is_object() && data.is_object() {
                    data = deep_merge(&main, &data);
                }
            }
            Err(e) => tracing::warn!("preset {}: main.yaml fallback unavailable ({e})", py::repr_str(&name)),
        }
        if !data.is_object() {
            return Err(HErr::Internal(format!("'{}' object has no attribute 'get'", py::type_name(&data))));
        }

        let sr = self.current_sr();
        let d = as_map(get(&data, "dsp"));
        let f = as_map(get(&data, "fft"));
        let a = as_map(get(&data, "autoscale"));
        let onset = as_map(data.get("onset"));
        let mut applied: Vec<&'static str> = Vec::new();

        let r = (|| {
            let mut bands_raw = serde_json::Map::new();
            for k in v::BAND_NAMES {
                if let Some(b @ Value::Object(_)) = d.get(k) {
                    bands_raw.insert(k.into(), b.clone());
                }
            }
            if bands_raw.len() != 3 {
                return Ok(Section::Skip);
            }
            let ok = v::validate_bands(&Value::Object(bands_raw))?;
            for (i, (lo, hi)) in ok.into_iter().enumerate() {
                let (lo, hi) = clamp_band_edges(lo, hi, sr);
                let b = self.cfg.dsp.band_mut(i);
                b.lo_hz = lo;
                b.hi_hz = hi;
            }
            self.schedule_filter_retune();
            self.apply_bands();
            Ok(Section::Applied)
        })();
        section(&mut applied, "dsp.bands", r);

        let r = (|| {
            let Some(x) = d.get("filter_order") else { return Ok(Section::Skip) };
            let n = v::validate_filter_order(Some(x))?;
            if n != self.cfg.dsp.filter_order {
                self.apply_filter_order(n);
            }
            Ok(Section::Applied)
        })();
        section(&mut applied, "dsp.filter_order", r);

        let r = (|| {
            if d.get("tau").is_none() && d.get("tau_attack").is_none() {
                return Ok(Section::Skip);
            }
            if let Some(t) = d.get("tau") {
                for (i, x) in v::validate_tau(t)? {
                    self.cfg.dsp.tau.set(i, x);
                }
            }
            if let Some(t) = d.get("tau_attack") {
                for (i, x) in v::validate_tau(t)? {
                    self.cfg.dsp.tau_attack.set(i, x);
                }
            }
            self.apply_smoothing();
            Ok(Section::Applied)
        })();
        section(&mut applied, "dsp.tau", r);

        let r = (|| {
            let ok = v::validate_autoscale(
                get(&a, "tau_attack_s"),
                get(&a, "tau_release_s"),
                get(&a, "noise_floor"),
                get(&a, "strength"),
                get(&a, "master_gain"),
            )?;
            if ok.is_empty() {
                return Ok(Section::Skip);
            }
            self.apply_autoscale(&ok);
            Ok(Section::Applied)
        })();
        section(&mut applied, "autoscale", r);

        let r = (|| {
            let Some(x) = f.get("n_bins") else { return Ok(Section::Skip) };
            let n = v::validate_n_fft_bins(Some(x))?;
            self.apply_fft_n_bins(n)?;
            Ok(Section::Applied)
        })();
        section(&mut applied, "fft.n_bins", r);

        let r = (|| {
            // isinstance(int) / isinstance((int, float)) as in the reference.
            let ws = f.get("window_size").filter(|x| py::as_int(x).is_some());
            let hop = f.get("hop").filter(|x| py::as_int(x).is_some());
            let fmin = f.get("f_min").filter(|x| py::is_number(x));
            let truthy = |x: Option<&Value>| x.is_some_and(py::truthy);
            if !(truthy(ws) || truthy(hop) || truthy(fmin)) {
                return Ok(Section::Skip);
            }
            self.apply_fft_window(ws, hop, fmin)?;
            Ok(Section::Applied)
        })();
        section(&mut applied, "fft.window", r);

        let r = (|| {
            let Some(x) = f.get("peak_smear_oct") else { return Ok(Section::Skip) };
            let s = v::validate_peak_smear_oct(Some(x))?;
            self.apply_fft_peak_smear(s);
            Ok(Section::Applied)
        })();
        section(&mut applied, "fft.peak_smear_oct", r);

        let r = (|| {
            let Some(x) = f.get("tilt_db_per_oct") else { return Ok(Section::Skip) };
            let t = v::validate_fft_tilt_db_per_oct(Some(x))?;
            self.apply_fft_tilt(t);
            Ok(Section::Applied)
        })();
        section(&mut applied, "fft.tilt_db_per_oct", r);

        let r = (|| {
            let mut any = false;
            for (i, band) in v::BAND_NAMES.iter().enumerate() {
                let Some(b @ Value::Object(_)) = onset.get(*band) else { continue };
                let ok = v::validate_onset(
                    get(b, "sensitivity"),
                    get(b, "refractory_s"),
                    get(b, "slow_tau_s"),
                    get(b, "abs_floor"),
                )?;
                if !ok.is_empty() {
                    self.apply_onset(i, &ok);
                    any = true;
                }
            }
            Ok(if any { Section::Applied } else { Section::Skip })
        })();
        section(&mut applied, "onset", r);

        if applied.is_empty() {
            return Err(format!("preset {} produced no valid fields", py::repr_str(&name)).into());
        }
        self.persister.request(true);
        self.meta_changed()
    }
}
