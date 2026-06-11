// Sliders, dropdowns, presets. All numeric controls are drag-aware
// (commit:false during drag, commit:true on release).

import { store } from "./store.js";
import { send } from "./ws.js";
import { makeFreqAxis } from "./freq_axis.js";

const PRESET_NAME_RE = /^[a-zA-Z0-9_\- ]+$/;

// log-mapped slider helpers
function logSliderToValue(el) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  const t = (parseFloat(el.value) - min) / (max - min);
  return Math.exp(Math.log(min) + t * (Math.log(max) - Math.log(min)));
}
function valueToLogSlider(el, v) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  const t = (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min));
  el.value = String(min + t * (max - min));
}
function readSlider(el) {
  return el.dataset.log ? logSliderToValue(el) : parseFloat(el.value);
}
function writeSlider(el, v) {
  if (el.dataset.log) valueToLogSlider(el, v);
  else el.value = String(v);
}

// Snap a frequency to a "nice" multiple that scales with f. Step doubles each
// octave: ~8 below 320 Hz, 16 by 640 Hz, 32 by 1.3k, 64 by 2.5k, 128 above 5k.
export function snapHz(f) {
  const step = Math.min(128, Math.max(8, Math.pow(2, Math.round(Math.log2(f / 40)))));
  return Math.max(step, Math.round(f / step) * step);
}
// Snap seconds to nearest multiple of 10 (min 10s).
export function snapSec(s) {
  return Math.max(10, Math.round(s / 10) * 10);
}

function bindDragAware(el, label, fmt, getMsg, snapFn) {
  let dragging = false;
  const read = () => {
    const v = readSlider(el);
    return snapFn ? snapFn(v) : v;
  };
  const update = (commit) => {
    const v = read();
    label.textContent = fmt(v);
    const msg = getMsg(v);
    if (msg) send({ ...msg, commit });
  };
  el.addEventListener("input", () => { dragging = true; update(false); });
  const commit = () => {
    if (dragging) { update(true); dragging = false; }
    else update(true); // keyboard step
  };
  el.addEventListener("change", commit);
  el.addEventListener("pointerup", commit);
  return {
    read,
    // Ignored mid-drag: every control change triggers a meta broadcast →
    // syncMeta() → setValue(), and rewriting the slider the user is
    // currently dragging makes the thumb fight the server echo. Same guard
    // the freq-axis / FFT band overlays use (`if (drag) return`).
    setValue: (v) => {
      if (dragging) return;
      writeSlider(el, v); label.textContent = fmt(v);
    },
  };
}

export function setupControls() {
  // Visual frequency-axis picker (drag colored regions / their edges).
  const freqAxis = makeFreqAxis(
    document.getElementById("freq-axis"),
    () => store.meta.sr || 48000,
  );

  // Bandpass edges are now exclusively controlled via the freq_axis widget.

  // Tau (UI in ms; protocol in seconds). The L/M/H envelope follower is
  // ASYMMETRIC: separate per-band release τ (slow, smooths sustained content)
  // and attack τ (fast, lets transients through).
  const tauEls = {
    low:  document.getElementById("tau-low"),
    mid:  document.getElementById("tau-mid"),
    high: document.getElementById("tau-high"),
  };
  const tauLabs = {
    low:  document.getElementById("tau-low-val"),
    mid:  document.getElementById("tau-mid-val"),
    high: document.getElementById("tau-high-val"),
  };
  const fmtMs = (v) => `${Math.round(v)} ms`;
  const tauCtls = {};
  for (const k of ["low", "mid", "high"]) {
    tauCtls[k] = bindDragAware(tauEls[k], tauLabs[k], fmtMs, () => {
      const tau = {
        low:  readSlider(tauEls.low)  / 1000,
        mid:  readSlider(tauEls.mid)  / 1000,
        high: readSlider(tauEls.high) / 1000,
      };
      return { type: "set_smoothing", tau };
    });
  }
  const tauAtkBandEls = {
    low:  document.getElementById("tau-attack-low"),
    mid:  document.getElementById("tau-attack-mid"),
    high: document.getElementById("tau-attack-high"),
  };
  const tauAtkBandLabs = {
    low:  document.getElementById("tau-attack-low-val"),
    mid:  document.getElementById("tau-attack-mid-val"),
    high: document.getElementById("tau-attack-high-val"),
  };
  const tauAtkBandCtls = {};
  for (const k of ["low", "mid", "high"]) {
    tauAtkBandCtls[k] = bindDragAware(tauAtkBandEls[k], tauAtkBandLabs[k], fmtMs, () => {
      const tau_attack = {
        low:  readSlider(tauAtkBandEls.low)  / 1000,
        mid:  readSlider(tauAtkBandEls.mid)  / 1000,
        high: readSlider(tauAtkBandEls.high) / 1000,
      };
      return { type: "set_smoothing", tau_attack };
    });
  }

  // Peak follower (asymmetric attack/release)
  const tauAtkEl  = document.getElementById("autoscale-attack");
  const tauAtkLab = document.getElementById("autoscale-attack-val");
  const tauAtkCtl = bindDragAware(tauAtkEl, tauAtkLab, (v) => `${Math.round(v)} ms`,
    (v) => ({ type: "set_autoscale", tau_attack_s: v / 1000 }));

  const tauRelEl  = document.getElementById("autoscale-tau");
  const tauRelLab = document.getElementById("autoscale-tau-val");
  const tauRelCtl = bindDragAware(tauRelEl, tauRelLab, (v) => `${Math.round(v)} s`,
    (v) => ({ type: "set_autoscale", tau_release_s: v }), snapSec);

  // FFT peak-smear slider — slider value is hundredths of an octave (0..200 → 0..2.0 oct).
  const smearEl  = document.getElementById("fft-peak-smear");
  const smearLab = document.getElementById("fft-peak-smear-val");
  let smearDragging = false;
  const updateSmear = (commit) => {
    const v = parseFloat(smearEl.value) / 100;
    smearLab.textContent = v <= 0 ? "off" : `${v.toFixed(2)} oct`;
    send({ type: "set_fft_peak_smear", peak_smear_oct: v, commit });
  };
  smearEl.addEventListener("input",   () => { smearDragging = true; updateSmear(false); });
  smearEl.addEventListener("change",  () => { updateSmear(true); smearDragging = false; });
  smearEl.addEventListener("pointerup", () => { if (smearDragging) { updateSmear(true); smearDragging = false; } });
  const smearCtl = { setValue: (v) => {
    if (smearDragging) return;
    smearEl.value = String(Math.round(v * 100));
    smearLab.textContent = v <= 0 ? "off" : `${v.toFixed(2)} oct`;
  } };

  // FFT spectral-tilt slider — slider value is tenths of a dB/oct (-60..120 → -6..12 dB/oct).
  const tiltEl  = document.getElementById("fft-tilt");
  const tiltLab = document.getElementById("fft-tilt-val");
  let tiltDragging = false;
  const updateTilt = (commit) => {
    const v = parseFloat(tiltEl.value) / 10;
    tiltLab.textContent = v === 0 ? "flat" : `${v >= 0 ? "+" : ""}${v.toFixed(1)} dB/oct`;
    send({ type: "set_fft_tilt", tilt_db_per_oct: v, commit });
  };
  tiltEl.addEventListener("input",   () => { tiltDragging = true; updateTilt(false); });
  tiltEl.addEventListener("change",  () => { updateTilt(true); tiltDragging = false; });
  tiltEl.addEventListener("pointerup", () => { if (tiltDragging) { updateTilt(true); tiltDragging = false; } });
  const tiltCtl = { setValue: (v) => {
    if (tiltDragging) return;
    tiltEl.value = String(Math.round(v * 10));
    tiltLab.textContent = v === 0 ? "flat" : `${v >= 0 ? "+" : ""}${v.toFixed(1)} dB/oct`;
  } };

  const floorEl  = document.getElementById("autoscale-floor");
  const floorLab = document.getElementById("autoscale-floor-val");
  // Slider is linear in dBFS across [FLOOR_DB_MIN, FLOOR_DB_MAX]; convert to linear amplitude.
  const FLOOR_DB_MIN = -140, FLOOR_DB_MAX = -25;
  const sliderToFloor = (s) => {
    const db = FLOOR_DB_MIN + (s / 1000) * (FLOOR_DB_MAX - FLOOR_DB_MIN);
    return Math.pow(10, db / 20);
  };
  const floorToSlider = (f) => {
    if (f <= 0) return 0;
    const db = 20 * Math.log10(f);
    const clamped = Math.max(FLOOR_DB_MIN, Math.min(FLOOR_DB_MAX, db));
    return 1000 * (clamped - FLOOR_DB_MIN) / (FLOOR_DB_MAX - FLOOR_DB_MIN);
  };
  const fmtFloor = (f) => (f <= 0 ? "off" : `${(20 * Math.log10(f)).toFixed(0)} dBFS`);
  let floorDragging = false;
  const updateFloor = (commit) => {
    const v = sliderToFloor(parseFloat(floorEl.value));
    floorLab.textContent = fmtFloor(v);
    send({ type: "set_autoscale", noise_floor: v, commit });
  };
  floorEl.addEventListener("input",   () => { floorDragging = true; updateFloor(false); });
  floorEl.addEventListener("change",  () => { updateFloor(true); floorDragging = false; });
  floorEl.addEventListener("pointerup", () => { if (floorDragging) { updateFloor(true); floorDragging = false; } });
  const floorCtl = { setValue: (f) => {
    if (floorDragging) return;
    floorEl.value = String(floorToSlider(f));
    floorLab.textContent = fmtFloor(f);
  } };

  // Master gain — final post-processing multiplier. Slider range 50..150 →
  // 0.5..1.5 (1.0 centered). At gain ≤ 1.0 the slider uses the default UI
  // accent styling. Above 1.0 it switches to a tint that lerps from orange
  // (just past 1.0) to red (at 1.5) to flag values that overflow the
  // conventional [0,1] output range.
  const masterEl  = document.getElementById("autoscale-master");
  const masterLab = document.getElementById("autoscale-master-val");
  const lerp = (a, b, t) => a + (b - a) * t;
  const lerpRgb = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
  const rgbStr = (c) => `rgb(${Math.round(c[0])}, ${Math.round(c[1])}, ${Math.round(c[2])})`;
  const ORANGE = [245, 166,  35];
  const RED    = [230,  74,  74];
  const darken = (c) => [c[0] * 0.62, c[1] * 0.62, c[2] * 0.62];
  const paintMaster = (gain) => {
    if (gain <= 1.0) {
      masterEl.classList.remove("overflow");
      masterEl.style.removeProperty("--master-color");
      masterEl.style.removeProperty("--master-color-dark");
      return;
    }
    const t = Math.min(1, (gain - 1.0) / 0.5);
    const c = lerpRgb(ORANGE, RED, t);
    masterEl.classList.add("overflow");
    masterEl.style.setProperty("--master-color", rgbStr(c));
    masterEl.style.setProperty("--master-color-dark", rgbStr(darken(c)));
  };
  let masterDragging = false;
  const updateMaster = (commit) => {
    const v = parseFloat(masterEl.value) / 100;
    masterLab.textContent = `${v.toFixed(2)}×`;
    paintMaster(v);
    send({ type: "set_autoscale", master_gain: v, commit });
  };
  masterEl.addEventListener("input",     () => { masterDragging = true; updateMaster(false); });
  masterEl.addEventListener("change",    () => { updateMaster(true); masterDragging = false; });
  masterEl.addEventListener("pointerup", () => { if (masterDragging) { updateMaster(true); masterDragging = false; } });
  const masterCtl = { setValue: (v) => {
    if (masterDragging) return;
    masterEl.value = String(Math.round(v * 100));
    masterLab.textContent = `${v.toFixed(2)}×`;
    paintMaster(v);
  } };

  const strengthEl  = document.getElementById("autoscale-strength");
  const strengthLab = document.getElementById("autoscale-strength-val");
  let strDragging = false;
  const updateStrength = (commit) => {
    const v = parseFloat(strengthEl.value) / 100;
    strengthLab.textContent = `${Math.round(v * 100)}%`;
    send({ type: "set_autoscale", strength: v, commit });
  };
  strengthEl.addEventListener("input", () => { strDragging = true; updateStrength(false); });
  strengthEl.addEventListener("change", () => { updateStrength(true); strDragging = false; });
  strengthEl.addEventListener("pointerup", () => { if (strDragging) { updateStrength(true); strDragging = false; } });
  const strengthCtl = { setValue: (v) => {
    if (strDragging) return;
    strengthEl.value = String(Math.round(v * 100));
    strengthLab.textContent = `${Math.round(v * 100)}%`;
  } };

  // Onset detection — sensitivity / refractory / slow envelope τ, PER BAND.
  // One set of sliders edits the currently selected band; the band selector
  // (3 tabs) switches which band's params they read/write. Per-band state is
  // kept in `onsetCfg` and pushed to the server with a `set_onset` message
  // tagged with the current band. Server is authoritative — meta payloads
  // refresh `onsetCfg` and re-render the sliders.
  // Slider encoding:
  // - sensitivity: hundredths (105..1000 → 1.05..10.00)
  // - refractory:  milliseconds (50..600 → 0.05..0.60 s)
  // - slow τ:      log-scale ms (50..1000 → 0.05..1.00 s)
  // - abs_floor:   hundredths of full-scale novelty (0..50 → 0.00..0.50)
  const onsetSensEl = document.getElementById("onset-sensitivity");
  const onsetSensLab = document.getElementById("onset-sensitivity-val");
  const onsetRefrEl = document.getElementById("onset-refractory");
  const onsetRefrLab = document.getElementById("onset-refractory-val");
  const onsetTauEl = document.getElementById("onset-slow-tau");
  const onsetTauLab = document.getElementById("onset-slow-tau-val");
  const onsetFloorEl = document.getElementById("onset-abs-floor");
  const onsetFloorLab = document.getElementById("onset-abs-floor-val");
  const fmtSens = (v) => `${(v / 100).toFixed(2)}`;
  const fmtOnsetFloor = (v) => (v <= 0 ? "off" : `${(v / 100).toFixed(2)}`);
  // Per-band cached params (live mirror of server-side state, updated on
  // every `meta` payload). All four are seeded from cfg defaults until the
  // first meta arrives.
  const onsetCfg = {
    low:  { sensitivity: 1.8, refractory_s: 0.25, slow_tau_s: 0.30, abs_floor: 0.10 },
    mid:  { sensitivity: 1.8, refractory_s: 0.18, slow_tau_s: 0.20, abs_floor: 0.10 },
    high: { sensitivity: 1.8, refractory_s: 0.12, slow_tau_s: 0.15, abs_floor: 0.10 },
  };
  let onsetBand = "low";
  const onsetBandBtns = Array.from(document.querySelectorAll(".onset-band-btn"));
  // Sliders currently being dragged — renderOnsetSliders skips these so the
  // meta echo can't fight the user's in-flight drag (same guard as the
  // other sliders' setValue).
  const onsetDragging = new Set();
  function renderOnsetSliders() {
    const c = onsetCfg[onsetBand];
    if (!onsetDragging.has(onsetSensEl)) {
      onsetSensEl.value = String(Math.round(c.sensitivity * 100));
      onsetSensLab.textContent = fmtSens(onsetSensEl.value);
    }
    if (!onsetDragging.has(onsetRefrEl)) {
      onsetRefrEl.value = String(Math.round(c.refractory_s * 1000));
      onsetRefrLab.textContent = `${Math.round(parseFloat(onsetRefrEl.value))} ms`;
    }
    if (!onsetDragging.has(onsetTauEl)) {
      writeSlider(onsetTauEl, Math.round(c.slow_tau_s * 1000));
      onsetTauLab.textContent = `${Math.round(readSlider(onsetTauEl))} ms`;
    }
    if (!onsetDragging.has(onsetFloorEl)) {
      onsetFloorEl.value = String(Math.round(c.abs_floor * 100));
      onsetFloorLab.textContent = fmtOnsetFloor(onsetFloorEl.value);
    }
  }
  function setOnsetBand(b) {
    if (b === onsetBand) return;
    onsetBand = b;
    for (const btn of onsetBandBtns) {
      const active = btn.dataset.band === b;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    }
    renderOnsetSliders();
  }
  for (const btn of onsetBandBtns) {
    btn.addEventListener("click", () => setOnsetBand(btn.dataset.band));
  }
  // Drag-aware slider wiring. We can't use bindDragAware directly because
  // the message body depends on `onsetBand` at SEND time (not at bind time).
  function bindOnsetSlider(el, label, fmt, paramKey, toServer) {
    let dragging = false;
    const apply = (commit) => {
      const sliderVal = readSlider(el);
      label.textContent = fmt(sliderVal);
      const serverVal = toServer(sliderVal);
      onsetCfg[onsetBand][paramKey] = serverVal;
      send({ type: "set_onset", band: onsetBand, [paramKey]: serverVal, commit });
    };
    const endDrag = () => { dragging = false; onsetDragging.delete(el); };
    el.addEventListener("input", () => { dragging = true; onsetDragging.add(el); apply(false); });
    el.addEventListener("change", () => { apply(true); endDrag(); });
    el.addEventListener("pointerup", () => { if (dragging) { apply(true); endDrag(); } });
  }
  bindOnsetSlider(onsetSensEl, onsetSensLab, fmtSens, "sensitivity", (v) => v / 100);
  bindOnsetSlider(onsetRefrEl, onsetRefrLab, (v) => `${Math.round(v)} ms`,
                  "refractory_s", (v) => v / 1000);
  bindOnsetSlider(onsetTauEl, onsetTauLab, (v) => `${Math.round(v)} ms`,
                  "slow_tau_s", (v) => v / 1000);
  bindOnsetSlider(onsetFloorEl, onsetFloorLab, fmtOnsetFloor,
                  "abs_floor", (v) => v / 100);

  // UI refresh rate — discrete slider snapped to industry-standard frame rates.
  // The same value drives the server-side WS snapshot rate AND the browser's
  // RAF render throttle (set in store.target_ui_fps; main.js reads it).
  const UI_FPS_STEPS = [15, 24, 30, 40, 60];
  const wsEl  = document.getElementById("ws-hz");
  const wsLab = document.getElementById("ws-hz-val");
  const nearestFpsIdx = (hz) => {
    let best = 0, bestD = Infinity;
    for (let i = 0; i < UI_FPS_STEPS.length; i++) {
      const d = Math.abs(UI_FPS_STEPS[i] - hz);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  };
  let wsDragging = false;
  const updateWs = (commit) => {
    const idx = parseInt(wsEl.value, 10);
    const hz = UI_FPS_STEPS[Math.max(0, Math.min(UI_FPS_STEPS.length - 1, idx))];
    wsLab.textContent = `${hz} fps`;
    store.target_ui_fps = hz;
    send({ type: "set_ws_snapshot_hz", hz, commit });
  };
  wsEl.addEventListener("input",   () => { wsDragging = true; updateWs(false); });
  wsEl.addEventListener("change",  () => { updateWs(true); wsDragging = false; });
  wsEl.addEventListener("pointerup", () => { if (wsDragging) { updateWs(true); wsDragging = false; } });
  const wsCtl = {
    setValue: (hz) => {
      if (wsDragging) return;
      const idx = nearestFpsIdx(hz);
      wsEl.value = String(idx);
      const snapped = UI_FPS_STEPS[idx];
      wsLab.textContent = `${snapped} fps`;
      store.target_ui_fps = snapped;
    },
  };

  // Visual peak-hold decay rate (UI-only; affects bars + FFT viz). Slider value
  // is hundredths of a unit/sec (1..500 → 0.01..5.0 /s).
  const peakDecayEl  = document.getElementById("peak-decay");
  const peakDecayLab = document.getElementById("peak-decay-val");
  let peakDecayDragging = false;
  const updatePeakDecay = (commit) => {
    const v = parseFloat(peakDecayEl.value) / 100;
    peakDecayLab.textContent = `${v.toFixed(2)}/s`;
    store.peak_decay_per_s = v;
    send({ type: "set_peak_decay", peak_decay_per_s: v, commit });
  };
  peakDecayEl.addEventListener("input",   () => { peakDecayDragging = true; updatePeakDecay(false); });
  peakDecayEl.addEventListener("change",  () => { updatePeakDecay(true); peakDecayDragging = false; });
  peakDecayEl.addEventListener("pointerup", () => { if (peakDecayDragging) { updatePeakDecay(true); peakDecayDragging = false; } });
  const peakDecayCtl = { setValue: (v) => {
    if (peakDecayDragging) return;
    peakDecayEl.value = String(Math.round(v * 100));
    peakDecayLab.textContent = `${v.toFixed(2)}/s`;
    store.peak_decay_per_s = v;
  } };

  // History window for the L/M/H rolling-lines chart (UI-only, not persisted).
  // Slider is log-scale (2..30s) but snaps to integer seconds.
  const historyEl  = document.getElementById("history-s");
  const historyLab = document.getElementById("history-s-val");
  const updateHistory = () => {
    const raw = readSlider(historyEl);
    const v = Math.max(2, Math.min(30, Math.round(raw)));
    historyLab.textContent = `${v}s`;
    store.lines_history_s = v;
  };
  historyEl.addEventListener("input", updateHistory);
  writeSlider(historyEl, Math.max(2, Math.min(30, store.lines_history_s ?? 5)));
  updateHistory();

  // Bandpass filter order. Slider snaps to {2, 4} only (min=2, max=4, step=2).
  // Server-authoritative; reflected from meta.
  const filterOrderCtl = bindDragAware(
    document.getElementById("filter-order"),
    document.getElementById("filter-order-val"),
    (v) => `${v} (~${6 * v} dB/oct)`,
    (v) => ({ type: "set_filter_order", order: v }),
  );

  // Onset-squares display toggle. Visibility-only — server detection / OSC /
  // WS payload are unaffected. Server-authoritative: toggle sends a
  // set_show_onsets message; the value reflected back via meta.ui_show_onsets
  // drives the actual store flag (so reload + persistence behave correctly).
  const showOnsetsToggle = document.getElementById("show-onsets");
  showOnsetsToggle.addEventListener("change", () => {
    send({ type: "set_show_onsets", show_onsets: showOnsetsToggle.checked });
  });
  const showOnsetsCtl = { setValue: (v) => { showOnsetsToggle.checked = !!v; } };

  // FFT toggle
  const fftToggle = document.getElementById("fft-toggle");
  fftToggle.addEventListener("change", () => {
    send({ type: "set_fft", enabled: fftToggle.checked });
  });

  // FFT raw-dB toggle. The server is authoritative — toggling sends a
  // set_fft_send_raw_db message; the server reflects the new value back in
  // meta. Both the viz AND the OSC payload switch in lockstep so what's
  // shown is byte-identical to what's sent.
  const fftRawDb = document.getElementById("fft-raw-db");
  fftRawDb.addEventListener("change", () => {
    send({ type: "set_fft_send_raw_db", send_raw_db: fftRawDb.checked });
  });
  const fftRawDbCtl = { setValue: (v) => { fftRawDb.checked = !!v; } };

  // Devices
  const deviceSelect = document.getElementById("device-select");
  document.getElementById("device-refresh").addEventListener("click", () => send({ type: "list_devices", probe: false }));
  document.getElementById("device-probe").addEventListener("click", () => send({ type: "list_devices", probe: true }));
  deviceSelect.addEventListener("change", () => {
    const idx = parseInt(deviceSelect.value, 10);
    if (Number.isFinite(idx)) send({ type: "set_device", index: idx });
  });

  // Presets
  const presetName  = document.getElementById("preset-name");
  const presetSave  = document.getElementById("preset-save");
  const presetList  = document.getElementById("preset-list");
  const presetLoad  = document.getElementById("preset-load");
  const validatePresetName = () => {
    const v = presetName.value.trim();
    const ok = v.length >= 1 && v.length <= 64 && PRESET_NAME_RE.test(v);
    presetSave.disabled = !ok;
    presetName.style.borderColor = (v.length > 0 && !ok) ? "#e64a4a" : "";
    return ok ? v : null;
  };
  presetName.addEventListener("input", validatePresetName);
  validatePresetName();
  presetSave.addEventListener("click", () => {
    const v = validatePresetName();
    if (!v) return;
    send({ type: "save_preset", name: v });
    presetName.value = "";
    validatePresetName();
  });
  presetLoad.addEventListener("click", () => {
    const name = presetList.value;
    if (!name) return;
    send({ type: "load_preset", name });
  });

  return {
    syncMeta() {
      const m = store.meta;
      const bands = m.bands || {};
      freqAxis.syncBands(bands);
      tauCtls.low.setValue((m.tau.low || 0) * 1000);
      tauCtls.mid.setValue((m.tau.mid || 0) * 1000);
      tauCtls.high.setValue((m.tau.high || 0) * 1000);
      const ta = m.tau_attack || {};
      tauAtkBandCtls.low.setValue((ta.low  || 0) * 1000);
      tauAtkBandCtls.mid.setValue((ta.mid  || 0) * 1000);
      tauAtkBandCtls.high.setValue((ta.high || 0) * 1000);
      tauAtkCtl.setValue((m.autoscale.tau_attack_s ?? 0.05) * 1000);
      tauRelCtl.setValue(m.autoscale.tau_release_s || 60);
      smearCtl.setValue(m.fft_peak_smear_oct ?? 0.3);
      tiltCtl.setValue(m.fft_tilt_db_per_oct ?? 3.0);
      floorCtl.setValue(m.autoscale.noise_floor || 0);
      strengthCtl.setValue(m.autoscale.strength ?? 1.0);
      masterCtl.setValue(m.autoscale.master_gain ?? 1.0);
      wsCtl.setValue(m.ws_snapshot_hz || 60);
      peakDecayCtl.setValue(m.ui_peak_decay_per_s ?? 0.6);
      if (m.ui_show_onsets !== undefined) {
        store.show_onsets = !!m.ui_show_onsets;
        showOnsetsCtl.setValue(!!m.ui_show_onsets);
      }
      const onset = m.onset || {};
      for (const b of ["low", "mid", "high"]) {
        const src = onset[b];
        if (src) {
          onsetCfg[b].sensitivity  = src.sensitivity  ?? onsetCfg[b].sensitivity;
          onsetCfg[b].refractory_s = src.refractory_s ?? onsetCfg[b].refractory_s;
          onsetCfg[b].slow_tau_s   = src.slow_tau_s   ?? onsetCfg[b].slow_tau_s;
          onsetCfg[b].abs_floor    = src.abs_floor    ?? onsetCfg[b].abs_floor;
        }
      }
      renderOnsetSliders();
      if (m.filter_order !== undefined) filterOrderCtl.setValue(m.filter_order);
      fftToggle.checked = !!m.fft_enabled;
      if (m.fft_send_raw_db !== undefined) fftRawDbCtl.setValue(!!m.fft_send_raw_db);
    },
    syncDevices() {
      const cur = store.meta?.device?.index;
      deviceSelect.innerHTML = "";
      const sorted = [...store.devices].sort((a, b) => a.index - b.index);
      for (const d of sorted) {
        const opt = document.createElement("option");
        opt.value = String(d.index);
        const probed = d.probed_signal ? " ★" : "";
        opt.textContent = `[${d.index}] ${d.name} (${d.hostapi})${probed}`;
        if (d.index === cur) opt.selected = true;
        deviceSelect.appendChild(opt);
      }
    },
    syncPresets() {
      const cur = presetList.value;
      presetList.innerHTML = "";
      const counts = {};
      for (const p of store.presets) counts[p.name] = (counts[p.name] || 0) + 1;
      // Already sorted by saved_at desc on the server.
      for (const p of store.presets) {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = counts[p.name] > 1 ? `${p.name}  (${p.saved_at})` : p.name;
        presetList.appendChild(opt);
      }
      if (cur) presetList.value = cur;
    },
    snapBackOnError(_reason) {
      // Re-sync sliders from the last known meta — simple & always correct.
      this.syncMeta();
    }
  };
}
