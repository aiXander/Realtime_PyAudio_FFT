// Sliders, dropdowns, presets. Every range slider goes through bindSlider():
// commit:false while the pointer holds the thumb, commit:true on release (or
// immediately for keyboard steps). The server is authoritative — meta echoes
// flow back through syncMeta() → setValue(), which is idempotent (no DOM
// writes when nothing changed) and never fights an in-flight drag.

import { store } from "./store.js";
import { send } from "./ws.js";
import { makeFreqAxis } from "./freq_axis.js";
import { toast } from "./toast.js";

const PRESET_NAME_RE = /^[a-zA-Z0-9_\- ]+$/;

// log-mapped slider helpers
function logSliderToValue(el) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  const t = (parseFloat(el.value) - min) / (max - min);
  return Math.exp(Math.log(min) + t * (Math.log(max) - Math.log(min)));
}
function logPosition(el, v) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  const c = Math.min(max, Math.max(min, v));
  const t = (Math.log(c) - Math.log(min)) / (Math.log(max) - Math.log(min));
  return min + t * (max - min);
}
function valueToLogSlider(el, v) {
  el.value = String(logPosition(el, v));
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

// ---------------------------------------------------------------------------
// Drag-aware slider binding (the ONE place drag state lives).
//
// "Held" = a pointer is down on the slider. It is cleared by pointerup /
// pointercancel / lostpointercapture / blur on the slider, and by any
// window-level pointerup or window blur — so it can't stick if the release
// happens outside the element, the tab loses focus, or touch turns into a
// scroll. While held, server echoes are ignored so the thumb doesn't fight
// the meta round-trip; on release any uncommitted value is committed.
// ---------------------------------------------------------------------------
const heldSliders = new Set();
function releaseAll() { for (const s of Array.from(heldSliders)) s.release(); }
window.addEventListener("pointerup", releaseAll);
window.addEventListener("pointercancel", releaseAll);
window.addEventListener("blur", releaseAll);

/**
 * @param el     range input
 * @param label  readout element
 * @param o.fmt       value → readout text
 * @param o.send      (value, commit) → void (sends the WS message)
 * @param o.toValue   el → value (default: readSlider, log-aware)
 * @param o.toSlider  value → slider position (default: writeSlider)
 * @param o.snap      optional value snapper applied on read
 * @param o.onShow    optional side effect whenever a value is displayed
 */
function bindSlider(el, label, o) {
  let held = false;
  let dirty = false;      // sent a commit:false value that isn't committed yet
  let shown;              // last value displayed (for idempotent setValue)

  const read = () => {
    const v = o.toValue ? o.toValue(el) : readSlider(el);
    return o.snap ? o.snap(v) : v;
  };
  const show = (v) => {
    shown = v;
    const t = o.fmt(v);
    if (label && label.textContent !== t) label.textContent = t;
    if (o.onShow) o.onShow(v);
  };
  const emit = (commit) => {
    const v = read();
    show(v);
    o.send(v, commit);
    dirty = !commit;
  };
  const api = {
    release() {
      heldSliders.delete(api);
      if (!held) return;
      held = false;
      if (dirty) emit(true);
    },
    read,
    /** Reflect a server value. Skipped while held; no-op if unchanged
     *  unless `force` (error snap-back rewrites the thumb regardless). */
    setValue(v, force) {
      if (held) return;
      if (!force && v === shown) return;
      if (o.toSlider) el.value = String(o.toSlider(v));
      else writeSlider(el, v);
      show(v);
    },
  };
  el.addEventListener("pointerdown", () => { held = true; heldSliders.add(api); });
  // Keyboard steps (no pointer held) commit immediately; drags stream
  // commit:false and commit once on release.
  el.addEventListener("input", () => emit(!held));
  el.addEventListener("change", () => { if (dirty && !held) emit(true); });
  for (const ev of ["pointerup", "pointercancel", "lostpointercapture", "blur"]) {
    el.addEventListener(ev, api.release);
  }
  return api;
}

export function setupControls() {
  // Visual frequency-axis picker (drag colored regions / their edges).
  const freqAxis = makeFreqAxis(
    document.getElementById("freq-axis"),
    () => store.meta.sr || 48000,
  );
  const $ = (id) => document.getElementById(id);
  const fmtMs = (v) => `${Math.round(v)} ms`;

  // Tau (UI in ms; protocol in seconds). The L/M/H envelope follower is
  // ASYMMETRIC: per-band release τ (slow) and attack τ (fast). Each message
  // carries all three bands, read from the sliders at send time.
  const BANDS = ["low", "mid", "high"];
  const tauEls = { low: $("tau-low"), mid: $("tau-mid"), high: $("tau-high") };
  const tauAtkEls = { low: $("tau-attack-low"), mid: $("tau-attack-mid"), high: $("tau-attack-high") };
  const allTau = (els) => ({
    low: readSlider(els.low) / 1000, mid: readSlider(els.mid) / 1000, high: readSlider(els.high) / 1000,
  });
  const tauCtls = {}, tauAtkCtls = {};
  for (const k of BANDS) {
    tauCtls[k] = bindSlider(tauEls[k], $(`tau-${k}-val`), {
      fmt: fmtMs,
      send: (_v, commit) => send({ type: "set_smoothing", tau: allTau(tauEls), commit }),
    });
    tauAtkCtls[k] = bindSlider(tauAtkEls[k], $(`tau-attack-${k}-val`), {
      fmt: fmtMs,
      send: (_v, commit) => send({ type: "set_smoothing", tau_attack: allTau(tauAtkEls), commit }),
    });
  }

  // Peak follower (asymmetric attack/release)
  const asAttackCtl = bindSlider($("autoscale-attack"), $("autoscale-attack-val"), {
    fmt: fmtMs,
    send: (v, commit) => send({ type: "set_autoscale", tau_attack_s: v / 1000, commit }),
  });
  const asReleaseCtl = bindSlider($("autoscale-tau"), $("autoscale-tau-val"), {
    fmt: (v) => `${Math.round(v)} s`,
    snap: snapSec,
    send: (v, commit) => send({ type: "set_autoscale", tau_release_s: v, commit }),
  });

  // FFT peak smear — slider in hundredths of an octave (0..200 → 0..2.0 oct).
  const smearCtl = bindSlider($("fft-peak-smear"), $("fft-peak-smear-val"), {
    toValue: (el) => parseFloat(el.value) / 100,
    toSlider: (v) => Math.round(v * 100),
    fmt: (v) => (v <= 0 ? "off" : `${v.toFixed(2)} oct`),
    send: (v, commit) => send({ type: "set_fft_peak_smear", peak_smear_oct: v, commit }),
  });

  // Spectral tilt — slider in tenths of a dB/oct. Range -60..120 → -6..+12
  // dB/oct, exactly the server validator's clamp range.
  const tiltCtl = bindSlider($("fft-tilt"), $("fft-tilt-val"), {
    toValue: (el) => parseFloat(el.value) / 10,
    toSlider: (v) => Math.round(v * 10),
    fmt: (v) => (v === 0 ? "flat" : `${v > 0 ? "+" : ""}${v.toFixed(1)} dB/oct`),
    send: (v, commit) => send({ type: "set_fft_tilt", tilt_db_per_oct: v, commit }),
  });

  // Noise gate — slider linear in dBFS across [FLOOR_DB_MIN, FLOOR_DB_MAX].
  const FLOOR_DB_MIN = -140, FLOOR_DB_MAX = -25;
  const floorCtl = bindSlider($("autoscale-floor"), $("autoscale-floor-val"), {
    toValue: (el) => Math.pow(10, (FLOOR_DB_MIN + (parseFloat(el.value) / 1000) * (FLOOR_DB_MAX - FLOOR_DB_MIN)) / 20),
    toSlider: (f) => {
      if (f <= 0) return 0;
      const db = Math.max(FLOOR_DB_MIN, Math.min(FLOOR_DB_MAX, 20 * Math.log10(f)));
      return 1000 * (db - FLOOR_DB_MIN) / (FLOOR_DB_MAX - FLOOR_DB_MIN);
    },
    fmt: (f) => (f <= 0 ? "off" : `${(20 * Math.log10(f)).toFixed(0)} dBFS`),
    send: (v, commit) => send({ type: "set_autoscale", noise_floor: v, commit }),
  });

  // Master gain — 50..150 → 0.5..1.5 (1.0 centered). Above 1.0 the slider
  // tints from orange to red to flag output beyond the conventional [0,1].
  const masterEl = $("autoscale-master");
  const ORANGE = [245, 166, 35], RED = [230, 74, 74];
  const rgbStr = (c, k = 1) => `rgb(${Math.round(c[0] * k)}, ${Math.round(c[1] * k)}, ${Math.round(c[2] * k)})`;
  const paintMaster = (gain) => {
    if (gain <= 1.0) {
      masterEl.classList.remove("overflow");
      masterEl.style.removeProperty("--master-color");
      masterEl.style.removeProperty("--master-color-dark");
      return;
    }
    const t = Math.min(1, (gain - 1.0) / 0.5);
    const c = ORANGE.map((a, i) => a + (RED[i] - a) * t);
    masterEl.classList.add("overflow");
    masterEl.style.setProperty("--master-color", rgbStr(c));
    masterEl.style.setProperty("--master-color-dark", rgbStr(c, 0.62));
  };
  const masterCtl = bindSlider(masterEl, $("autoscale-master-val"), {
    toValue: (el) => parseFloat(el.value) / 100,
    toSlider: (v) => Math.round(v * 100),
    fmt: (v) => `${v.toFixed(2)}×`,
    onShow: paintMaster,
    send: (v, commit) => send({ type: "set_autoscale", master_gain: v, commit }),
  });

  const strengthCtl = bindSlider($("autoscale-strength"), $("autoscale-strength-val"), {
    toValue: (el) => parseFloat(el.value) / 100,
    toSlider: (v) => Math.round(v * 100),
    fmt: (v) => `${Math.round(v * 100)}%`,
    send: (v, commit) => send({ type: "set_autoscale", strength: v, commit }),
  });

  // Onset detection — per band. One set of sliders edits the selected band;
  // the band tabs switch which band's params they read/write. `onsetCfg` is
  // the live mirror of meta.onset.
  // Slider encoding:
  // - sensitivity: hundredths (105..1000 → 1.05..10.00)
  // - refractory:  milliseconds (50..600 → 0.05..0.60 s)
  // - slow τ:      log-scale ms (50..1000 → 0.05..1.00 s)
  // - abs_floor:   hundredths (0..50 → 0.00..0.50) — "min jump" in the UI
  const onsetCfg = {
    low:  { sensitivity: 1.8, refractory_s: 0.25, slow_tau_s: 0.30, abs_floor: 0.10 },
    mid:  { sensitivity: 1.8, refractory_s: 0.18, slow_tau_s: 0.20, abs_floor: 0.10 },
    high: { sensitivity: 1.8, refractory_s: 0.12, slow_tau_s: 0.15, abs_floor: 0.10 },
  };
  let onsetBand = "low";
  const onsetBandBtns = Array.from(document.querySelectorAll(".onset-band-btn"));
  const onsetSlider = (id, key, o) => bindSlider($(id), $(`${id}-val`), {
    ...o,
    send: (v, commit) => {
      onsetCfg[onsetBand][key] = v;
      send({ type: "set_onset", band: onsetBand, [key]: v, commit });
    },
  });
  const onsetCtls = {
    sensitivity: onsetSlider("onset-sensitivity", "sensitivity", {
      toValue: (el) => parseFloat(el.value) / 100,
      toSlider: (v) => Math.round(v * 100),
      fmt: (v) => `${v.toFixed(2)}×`,
    }),
    refractory_s: onsetSlider("onset-refractory", "refractory_s", {
      toValue: (el) => parseFloat(el.value) / 1000,
      toSlider: (v) => Math.round(v * 1000),
      fmt: (v) => `${Math.round(v * 1000)} ms`,
    }),
    slow_tau_s: onsetSlider("onset-slow-tau", "slow_tau_s", {
      // Log slider whose min/max are in ms; the wire value is seconds.
      toValue: (el) => Math.round(logSliderToValue(el)) / 1000,
      toSlider: (v) => logPosition($("onset-slow-tau"), v * 1000),
      fmt: (v) => `${Math.round(v * 1000)} ms`,
    }),
    abs_floor: onsetSlider("onset-abs-floor", "abs_floor", {
      toValue: (el) => parseFloat(el.value) / 100,
      toSlider: (v) => Math.round(v * 100),
      fmt: (v) => (v <= 0 ? "off" : v.toFixed(2)),
    }),
  };
  function renderOnsetSliders(force) {
    const c = onsetCfg[onsetBand];
    for (const k of Object.keys(onsetCtls)) onsetCtls[k].setValue(c[k], force);
  }
  function setOnsetBand(b) {
    if (b === onsetBand) return;
    onsetBand = b;
    for (const btn of onsetBandBtns) {
      const active = btn.dataset.band === b;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
      btn.tabIndex = active ? 0 : -1;
    }
    renderOnsetSliders(true);
  }
  for (const btn of onsetBandBtns) {
    btn.addEventListener("click", () => setOnsetBand(btn.dataset.band));
    // Arrow keys move between tabs (WAI-ARIA tabs pattern).
    btn.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const i = onsetBandBtns.indexOf(btn);
      const next = onsetBandBtns[(i + (e.key === "ArrowRight" ? 1 : -1) + onsetBandBtns.length) % onsetBandBtns.length];
      next.focus();
      setOnsetBand(next.dataset.band);
    });
    btn.tabIndex = btn.dataset.band === onsetBand ? 0 : -1;
  }

  // UI refresh rate — snapped to standard frame rates. Drives both the
  // server's WS snapshot rate and the browser's RAF throttle.
  const UI_FPS_STEPS = [15, 24, 30, 40, 60];
  const nearestFpsIdx = (hz) => {
    let best = 0, bestD = Infinity;
    for (let i = 0; i < UI_FPS_STEPS.length; i++) {
      const d = Math.abs(UI_FPS_STEPS[i] - hz);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  };
  const wsCtl = bindSlider($("ws-hz"), $("ws-hz-val"), {
    toValue: (el) => UI_FPS_STEPS[Math.max(0, Math.min(UI_FPS_STEPS.length - 1, parseInt(el.value, 10)))],
    toSlider: nearestFpsIdx,
    fmt: (hz) => `${UI_FPS_STEPS[nearestFpsIdx(hz)]} fps`,
    onShow: (hz) => { store.target_ui_fps = UI_FPS_STEPS[nearestFpsIdx(hz)]; },
    send: (hz, commit) => send({ type: "set_ws_snapshot_hz", hz, commit }),
  });

  // Visual peak-hold decay rate (affects bars + FFT viz). Log slider over
  // 0.05..3.0 /s — the server validator's clamp range.
  const peakDecayCtl = bindSlider($("peak-decay"), $("peak-decay-val"), {
    snap: (v) => Math.round(v * 100) / 100,
    fmt: (v) => `${v.toFixed(2)}/s`,
    onShow: (v) => { store.peak_decay_per_s = v; },
    send: (v, commit) => send({ type: "set_peak_decay", peak_decay_per_s: v, commit }),
  });

  // History window for the L/M/H rolling-lines chart (UI-only, not
  // persisted). Log slider 2..30 s, snapped to whole seconds.
  const historyEl  = $("history-s");
  const historyLab = $("history-s-val");
  const updateHistory = () => {
    const v = Math.max(2, Math.min(30, Math.round(readSlider(historyEl))));
    historyLab.textContent = `${v}s`;
    store.lines_history_s = v;
  };
  historyEl.addEventListener("input", updateHistory);
  writeSlider(historyEl, Math.max(2, Math.min(30, store.lines_history_s ?? 5)));
  updateHistory();

  // Bandpass filter order: {2, 4}.
  const filterOrderCtl = bindSlider($("filter-order"), $("filter-order-val"), {
    fmt: (v) => `${v} · ${6 * v} dB/oct`,
    send: (v, commit) => send({ type: "set_filter_order", order: v, commit }),
  });

  // Checkboxes: send, then reflect the server's value from meta.
  const showOnsetsToggle = $("show-onsets");
  showOnsetsToggle.addEventListener("change", () => {
    send({ type: "set_show_onsets", show_onsets: showOnsetsToggle.checked });
  });
  const fftToggle = $("fft-toggle");
  fftToggle.addEventListener("change", () => send({ type: "set_fft", enabled: fftToggle.checked }));
  const fftRawDb = $("fft-raw-db");
  fftRawDb.addEventListener("change", () => {
    send({ type: "set_fft_send_raw_db", send_raw_db: fftRawDb.checked });
  });
  const setChecked = (el, v) => { if (el.checked !== v) el.checked = v; };

  // ---------------- Devices ----------------
  const deviceSelect = $("device-select");
  const deviceStatus = $("device-status");
  let pendingDevice = null;
  let pendingDeviceTimer = null;
  $("device-refresh").addEventListener("click", () => send({ type: "list_devices", probe: false }));
  const probeBtn = $("device-probe");
  probeBtn.addEventListener("click", () => {
    if (!send({ type: "list_devices", probe: true })) return;
    probeBtn.disabled = true;
    probeBtn.classList.add("busy");
    setDeviceStatus("probing…");
  });
  function setDeviceStatus(text) {
    deviceStatus.textContent = text;
    deviceStatus.hidden = !text;
  }
  function clearPendingDevice() {
    pendingDevice = null;
    clearTimeout(pendingDeviceTimer);
    deviceSelect.disabled = false;
    deviceSelect.classList.remove("busy");
    setDeviceStatus("");
  }
  function selectDeviceIndex(idx) {
    const v = idx === null || idx === undefined ? "" : String(idx);
    let opt = deviceSelect.querySelector(`option[value="${CSS.escape(v)}"]`);
    if (!opt) {
      // Current device isn't in the list (or there is none): show a
      // placeholder instead of silently displaying the first device.
      opt = deviceSelect.querySelector('option[data-placeholder]') || document.createElement("option");
      opt.dataset.placeholder = "1";
      opt.value = v;
      opt.textContent = v === "" ? "— no input device —" : `[${v}] ${store.meta?.device?.name || "current device"}`;
      if (!opt.isConnected) deviceSelect.prepend(opt);
    }
    if (deviceSelect.value !== v) deviceSelect.value = v;
  }
  deviceSelect.addEventListener("change", () => {
    const idx = parseInt(deviceSelect.value, 10);
    if (!Number.isFinite(idx)) return;
    if (!send({ type: "set_device", index: idx })) return;
    pendingDevice = idx;
    deviceSelect.disabled = true;
    deviceSelect.classList.add("busy");
    setDeviceStatus("switching…");
    // Safety net: a switch that never reports back must not lock the select.
    clearTimeout(pendingDeviceTimer);
    pendingDeviceTimer = setTimeout(() => {
      clearPendingDevice();
      selectDeviceIndex(store.meta?.device?.index);
      toast("Device switch did not confirm — showing the server's current device", "err");
    }, 12000);
  });

  // ---------------- Presets ----------------
  const presetName = $("preset-name");
  const presetSave = $("preset-save");
  const presetList = $("preset-list");
  const presetLoad = $("preset-load");
  let overwriteArmed = null;      // name awaiting overwrite confirmation
  let overwriteTimer = null;
  let pendingSave = null;         // name we asked the server to save
  let pendingLoad = null;
  const disarmOverwrite = () => {
    overwriteArmed = null;
    clearTimeout(overwriteTimer);
    presetSave.textContent = "save";
    presetSave.classList.remove("warn");
  };
  const validatePresetName = () => {
    const v = presetName.value.trim();
    const ok = v.length >= 1 && v.length <= 64 && PRESET_NAME_RE.test(v);
    presetSave.disabled = !ok;
    presetName.classList.toggle("invalid", v.length > 0 && !ok);
    presetName.setAttribute("aria-invalid", v.length > 0 && !ok ? "true" : "false");
    return ok ? v : null;
  };
  presetName.addEventListener("input", () => { disarmOverwrite(); validatePresetName(); });
  presetName.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); presetSave.click(); } });
  validatePresetName();
  presetSave.addEventListener("click", () => {
    const v = validatePresetName();
    if (!v) return;
    const exists = store.presets.some((p) => p.name === v);
    if (exists && overwriteArmed !== v) {
      // Two-step confirm: first click arms, second click (within 4 s) saves.
      overwriteArmed = v;
      presetSave.textContent = "overwrite?";
      presetSave.classList.add("warn");
      clearTimeout(overwriteTimer);
      overwriteTimer = setTimeout(disarmOverwrite, 4000);
      return;
    }
    disarmOverwrite();
    if (!send({ type: "save_preset", name: v })) return;
    pendingSave = v;
    presetName.value = "";
    validatePresetName();
  });
  presetLoad.addEventListener("click", () => {
    const name = presetList.value;
    if (!name) return;
    if (send({ type: "load_preset", name })) pendingLoad = name;
  });

  // Everything that needs a live server: disabled while offline / unsynced.
  const aside = document.querySelector(".controls");
  const liveInputs = ["ws-hz", "peak-decay", "show-onsets", "fft-toggle", "fft-raw-db"].map($);

  function syncMeta(force = false) {
    const m = store.meta;
    freqAxis.syncBands(m.bands || {});
    const tau = m.tau || {}, ta = m.tau_attack || {}, as = m.autoscale || {};
    for (const k of BANDS) {
      tauCtls[k].setValue((tau[k] || 0) * 1000, force);
      tauAtkCtls[k].setValue((ta[k] || 0) * 1000, force);
    }
    asAttackCtl.setValue((as.tau_attack_s ?? 0.05) * 1000, force);
    asReleaseCtl.setValue(as.tau_release_s || 60, force);
    smearCtl.setValue(m.fft_peak_smear_oct ?? 0.3, force);
    tiltCtl.setValue(m.fft_tilt_db_per_oct ?? 3.0, force);
    floorCtl.setValue(as.noise_floor || 0, force);
    strengthCtl.setValue(as.strength ?? 1.0, force);
    masterCtl.setValue(as.master_gain ?? 1.0, force);
    wsCtl.setValue(m.ws_snapshot_hz || 60, force);
    peakDecayCtl.setValue(m.ui_peak_decay_per_s ?? 0.6, force);
    if (m.ui_show_onsets !== undefined) {
      store.show_onsets = !!m.ui_show_onsets;
      setChecked(showOnsetsToggle, !!m.ui_show_onsets);
    }
    const onset = m.onset || {};
    for (const b of BANDS) {
      const src = onset[b];
      if (!src) continue;
      for (const k of Object.keys(onsetCfg[b])) onsetCfg[b][k] = src[k] ?? onsetCfg[b][k];
    }
    renderOnsetSliders(force);
    if (m.filter_order !== undefined) filterOrderCtl.setValue(m.filter_order, force);
    setChecked(fftToggle, !!m.fft_enabled);
    if (m.fft_send_raw_db !== undefined) setChecked(fftRawDb, !!m.fft_send_raw_db);

    // Device: the switch is done once meta reports the requested index.
    const devIdx = m.device?.index ?? null;
    if (pendingDevice !== null && devIdx === pendingDevice) {
      clearPendingDevice();
      toast(`Input: ${m.device?.name || `device ${devIdx}`}`, "ok");
    }
    if (pendingDevice === null) selectDeviceIndex(devIdx);

    if (pendingLoad !== null) {
      toast(`Loaded preset “${pendingLoad}”`, "ok");
      pendingLoad = null;
    }
  }

  return {
    syncMeta,
    syncDevices() {
      const cur = pendingDevice ?? store.meta?.device?.index ?? null;
      const frag = document.createDocumentFragment();
      const sorted = [...store.devices].sort((a, b) => a.index - b.index);
      for (const d of sorted) {
        const opt = document.createElement("option");
        opt.value = String(d.index);
        const probed = d.probed_signal ? " ★" : "";
        opt.textContent = `[${d.index}] ${d.name} (${d.hostapi})${probed}`;
        frag.appendChild(opt);
      }
      deviceSelect.replaceChildren(frag);
      selectDeviceIndex(cur);
      if (probeBtn.disabled) {
        probeBtn.disabled = false;
        probeBtn.classList.remove("busy");
        if (pendingDevice === null) setDeviceStatus("");
        const n = store.devices.filter((d) => d.probed_signal).length;
        toast(n ? `Probe: ${n} input${n > 1 ? "s" : ""} with signal (★)` : "Probe: no input is producing signal", n ? "ok" : "info");
      }
    },
    syncPresets() {
      const cur = pendingSave ?? presetList.value;
      const counts = {};
      for (const p of store.presets) counts[p.name] = (counts[p.name] || 0) + 1;
      const frag = document.createDocumentFragment();
      // Already sorted by saved_at desc on the server.
      for (const p of store.presets) {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = counts[p.name] > 1 ? `${p.name}  (${p.saved_at})` : p.name;
        frag.appendChild(opt);
      }
      if (!store.presets.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "— no presets yet —";
        frag.appendChild(opt);
      }
      presetList.replaceChildren(frag);
      presetList.disabled = !store.presets.length;
      presetLoad.disabled = !store.presets.length;
      if (cur && store.presets.some((p) => p.name === cur)) presetList.value = cur;
      if (pendingSave !== null && store.presets.some((p) => p.name === pendingSave)) {
        toast(`Saved preset “${pendingSave}”`, "ok");
        pendingSave = null;
      }
    },
    /** Error from the server: drop any pending optimistic state and re-sync
     *  every control from the last confirmed meta (forced rewrite). */
    snapBackOnError(_reason) {
      pendingSave = null;
      pendingLoad = null;
      if (probeBtn.disabled) { probeBtn.disabled = false; probeBtn.classList.remove("busy"); }
      if (pendingDevice !== null) clearPendingDevice();
      syncMeta(true);
    },
    /** Enable/disable everything that talks to the server. */
    setLive(live) {
      aside.inert = !live;
      aside.classList.toggle("is-offline", !live);
      for (const el of liveInputs) if (el) el.disabled = !live;
      if (!live) {
        releaseAll();
        pendingSave = null;
        pendingLoad = null;
        disarmOverwrite();
        if (pendingDevice !== null) clearPendingDevice();
        if (probeBtn.disabled) { probeBtn.disabled = false; probeBtn.classList.remove("busy"); setDeviceStatus(""); }
      }
    },
  };
}
