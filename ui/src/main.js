// Entry point. Wires WS handlers -> store, sets up controls, runs RAF loop.

import { connect, onMessage, onError, onStatus, retryNow, snapshotRate, resetSnapshotRate } from "./ws.js";
import { store, avgRing, p95Ring, clearLive } from "./store.js";
import { setupControls } from "./controls.js";
import { makeLines } from "./viz/lmh_lines.js";
import { makeBars }  from "./viz/lmh_bars.js";
import { makeScene } from "./viz/lmh_scene.js";
import { makeFft }   from "./viz/fft_2d.js";
import { setupTooltips } from "./tooltips.js";
import { setupLayout, applyLayout } from "./layout.js";
import { setupSidebar } from "./sidebar.js";
import { toast } from "./toast.js";

setupSidebar();
setupLayout();
const controls = setupControls();

const lines = makeLines(document.getElementById("viz-lines"));
const bars  = makeBars(document.getElementById("viz-bars"));
const scene = makeScene(document.getElementById("viz-scene"));
const fft   = makeFft(document.getElementById("viz-fft"));

const bpmEl = document.getElementById("bpm-readout");
const bpmVal = bpmEl ? bpmEl.querySelector(".bpm-value") : null;
const fftTitleMode = document.getElementById("fft-title-mode");
const bandpassFs = document.getElementById("fieldset-bandpass");
const uiFpsEl = document.getElementById("ui-fps");
const srvFpsEl = document.getElementById("server-fps");
const overlay = document.getElementById("conn-overlay");
const overlayMsg = document.getElementById("conn-overlay-msg");

let bpmText = "";
function setBpm(bpm) {
  const text = bpm > 0 ? bpm.toFixed(1) : "—";
  if (text === bpmText || !bpmVal) return;
  bpmText = text;
  bpmVal.textContent = text;
  bpmEl.classList.toggle("locked", bpm > 0);
}
setBpm(0);

// ----- Connection state -----
// Offline: wipe live data (no frozen bars/spectrum/BPM), dim the canvases
// behind an overlay, and disable every control until the server greets us
// with a fresh meta — which then re-syncs everything.
let syncFallback = null;
function setLive(live) {
  document.body.classList.toggle("offline", !live);
  if (overlay) overlay.hidden = live;
  controls.setLive(live);
}

onStatus((state) => {
  store.conn = state;
  if (retryBtn) retryBtn.hidden = state !== "reconnecting";
  if (state === "connected") {
    // Wait for the greeting meta before enabling controls. Safety net: if
    // it never arrives, enable anyway after 3 s rather than lock the UI.
    if (overlayMsg) overlayMsg.textContent = "Syncing with server…";
    clearTimeout(syncFallback);
    syncFallback = setTimeout(() => { if (!store.synced) { store.synced = true; setLive(true); } }, 3000);
    return;
  }
  clearTimeout(syncFallback);
  const wasSynced = store.synced;
  store.synced = false;
  clearLive();
  resetSnapshotRate();
  lines.reset();
  bars.reset();
  fft.reset();
  setBpm(0);
  if (srvFpsEl) srvFpsEl.textContent = "srv — Hz";
  if (overlayMsg) {
    overlayMsg.textContent = state === "connecting"
      ? "Connecting to the audio server…"
      : "Lost connection to the audio server. Retrying…";
  }
  setLive(false);
  if (wasSynced) toast("Disconnected from the audio server", "err");
});
const retryBtn = document.getElementById("conn-overlay-retry");
retryBtn?.addEventListener("click", retryNow);
setLive(false); // until the first meta arrives

// ----- WS handlers -----
onMessage("snapshot", (m) => {
  store.low = m.low; store.mid = m.mid; store.high = m.high;
  store.low_raw = m.low_raw; store.mid_raw = m.mid_raw; store.high_raw = m.high_raw;
  const now = performance.now();
  if (m.low_onset)  store.low_onset_pulse_t  = now;
  if (m.mid_onset)  store.mid_onset_pulse_t  = now;
  if (m.high_onset) store.high_onset_pulse_t = now;
  if (typeof m.bpm === "number") {
    store.bpm = m.bpm;
    setBpm(m.bpm); // touches the DOM only when the rendered text changes
  }
});

let lastFftEnabled = null, lastRawDb = null;
onMessage("meta", (m) => {
  store.meta = { ...store.meta, ...m };
  if (m.fft_db_floor !== undefined) store.fft_db_floor = m.fft_db_floor;
  if (m.fft_db_ceiling !== undefined) store.fft_db_ceiling = m.fft_db_ceiling;
  if (m.fft_send_raw_db !== undefined) store.fft_send_raw_db = !!m.fft_send_raw_db;
  if (m.ui_layout) applyLayout(m.ui_layout); // no-op when unchanged
  // When FFT is disabled, drop the last frame so the viz shows "FFT disabled"
  // instead of a frozen spectrum from the moment of toggle-off.
  if (m.fft_enabled === false) store.fft_bins = null;
  if (m.bands) fft.syncBands(m.bands);
  if (m.fft_enabled !== undefined && m.fft_enabled !== lastFftEnabled) {
    lastFftEnabled = m.fft_enabled;
    // The FFT canvas band overlay replaces the side-panel "Bandpass edges"
    // widget while FFT is on.
    fft.setInteractive(!!m.fft_enabled);
    if (bandpassFs) bandpassFs.hidden = !!m.fft_enabled;
  }
  const rawDb = !!store.meta.fft_send_raw_db;
  if (rawDb !== lastRawDb && fftTitleMode) {
    lastRawDb = rawDb;
    fftTitleMode.textContent = rawDb ? "log-x · raw dB" : "log-x · scaled 0..1";
  }
  controls.syncMeta();
  if (!store.synced && store.conn === "connected") {
    store.synced = true;
    clearTimeout(syncFallback);
    setLive(true);
  }
});

onMessage("devices", (m) => {
  store.devices = m.items || [];
  controls.syncDevices();
});

onMessage("presets", (m) => {
  store.presets = m.items || [];
  controls.syncPresets();
});

const cbOverrunsEl = document.getElementById("cb_overruns");
const dspDropsEl = document.getElementById("dsp_drops");
const fftDropsEl = document.getElementById("fft_drops");
onMessage("server_status", (m) => {
  store.status = m;
  setCounter(cbOverrunsEl, m.cb_overruns);
  setCounter(dspDropsEl, m.dsp_drops);
  setCounter(fftDropsEl, m.fft_drops);
  renderPerfPanel();
});
function setCounter(el, v) {
  if (!el) return;
  const t = String(v ?? 0);
  if (el.textContent !== t) el.textContent = t;
  el.classList.toggle("nonzero", !!v);
}

const errLog = document.getElementById("err-log");
onError((reason) => {
  const ts = new Date().toLocaleTimeString();
  errLog.textContent = `[${ts}] ${reason}\n` + errLog.textContent;
  if (errLog.textContent.length > 4000) errLog.textContent = errLog.textContent.slice(0, 4000);
  toast(reason, "err");
  controls.snapBackOnError(reason);
});

// ----- Perf panel rendering -----
const PERF_ROWS = [
  { key: "lmh_e2e", label: "lmh e2e", tooltip: [
      "**End-to-end L/M/H latency.** Wall-clock time between the audio callback receiving an audio block and the corresponding `/audio/lmh` packet being dispatched over OSC.",
      "",
      "Includes: ring write, DSP worker wakeup + filter/RMS/smoother/auto-scaler, asyncio sender hop, and the OSC UDP send call.",
      "",
      "*Excludes* the PortAudio input buffer fill itself (~one block) and any kernel/driver delay before the callback fires.",
      "",
      "Numbers are `avg / p95 ms` (same convention as the other rows). The bar is scaled against one audio block period — 100% = one block of latency.",
    ].join("\n") },
  { key: "fft_e2e", label: "fft e2e", tooltip: [
      "**End-to-end FFT latency.** Wall-clock time between the audio block that *completes* an FFT hop landing in the ring and the corresponding `/audio/fft` packet being dispatched over OSC.",
      "",
      "Includes: FFT worker wakeup, windowed rfft + log-binning + post-processing, asyncio sender hop, and the OSC UDP send call.",
      "",
      "Only sampled when an `/audio/fft` packet actually goes out. The ENABLE toggle drives both FFT computation and OSC transmission, so this row is greyed out exactly when FFT is off.",
      "",
      "Numbers are `avg / p95 ms`. The bar is scaled against one hop period — 100% = one hop of latency.",
    ].join("\n") },
  { key: "cb",  label: "cb",  tooltip: [
      "**PortAudio callback cost.** Time spent inside the audio C-thread callback per audio block (mono-mix + ring write + event signal).",
      "",
      "Numbers are `avg / p95 ms`. Bar is scaled against one audio block period.",
    ].join("\n") },
  { key: "dsp", label: "dsp", tooltip: [
      "**DSP worker cost.** Time spent per block in the L/M/H pipeline: IIR bandpass, RMS, exponential smoother, auto-scaler, store publish.",
      "",
      "Numbers are `avg / p95 ms`. Bar is scaled against one audio block period.",
    ].join("\n") },
  { key: "fft", label: "fft", tooltip: [
      "**FFT worker cost.** Time spent per hop: window read, Hann + rfft, log-bin aggregation, per-bin post-processing, store publish.",
      "",
      "Numbers are `avg / p95 ms`. Bar is scaled against one hop period.",
    ].join("\n") },
  { key: "ws",  label: "ws",  tooltip: [
      "**WebSocket broadcast cost.** Time spent assembling and queueing one snapshot fan-out (JSON L/M/H + binary FFT) to all connected clients.",
      "",
      "Numbers are `avg / p95 ms`. Bar is scaled against one snapshot interval (`1 / ws_snapshot_hz`).",
    ].join("\n") },
];
const VIZ_NAMES = ["lines", "bars", "scene", "fft"];
const BROWSER_ROWS = [
  { key: "raf", tooltip: [
      "**Browser inter-draw interval.** Wall-clock time between consecutive canvas redraws. Throttled to the UI refresh rate slider.",
      "",
      "Numbers are `avg / p95 ms`. Bar shows how far the average exceeds the target frame period (0% = on target).",
    ].join("\n") },
  ...VIZ_NAMES.map((name) => ({ key: name, tooltip: [
      `**\`${name}\` canvas paint cost.** Time spent inside its draw() call per frame.`,
      "",
      "Numbers are `avg / p95 ms`. Bar is scaled against one UI frame period (1 / refresh rate).",
    ].join("\n") })),
];
const perfContainer = document.getElementById("perf-rows");

function ensurePerfRows() {
  if (perfContainer.children.length > 0) return;
  for (const r of PERF_ROWS) addPerfRow(r.key, r.label, r.tooltip);
  for (const r of BROWSER_ROWS) addPerfRow("b_" + r.key, r.key, r.tooltip);
}

const perfRowEls = {};

function addPerfRow(id, label, tooltip) {
  const row = document.createElement("div");
  row.className = "perf-row";
  row.id = "perf-" + id;
  if (tooltip) row.setAttribute("data-tooltip", tooltip);
  row.innerHTML = `<span class="perf-label">${label}</span><div class="perf-bar"><div class="perf-bar-fill"></div></div><span class="perf-num">– / –</span>`;
  perfContainer.appendChild(row);
  perfRowEls[id] = { row, fill: row.querySelector(".perf-bar-fill"), num: row.querySelector(".perf-num") };
}

const fmtMs = (x) => (x >= 10 ? x.toFixed(1) : x.toFixed(2));
function setPerfRow(id, avg_ms, p95_ms, load_pct, disabled) {
  const r = perfRowEls[id];
  if (!r) return;
  r.row.classList.toggle("disabled", !!disabled);
  r.fill.style.width = `${Math.min(100, load_pct).toFixed(1)}%`;
  r.fill.classList.toggle("red", load_pct >= 80);
  r.fill.classList.toggle("amber", load_pct >= 50 && load_pct < 80);
  const t = `${fmtMs(avg_ms)} / ${fmtMs(p95_ms)} ms`;
  if (r.num.textContent !== t) r.num.textContent = t;
}

function renderPerfPanel() {
  ensurePerfRows();
  const p = store.status?.perf;
  if (p) {
    for (const r of PERF_ROWS) {
      const stage = p[r.key] || {};
      const fftDisabled = (r.key === "fft" || r.key === "fft_e2e") && stage.enabled === false;
      setPerfRow(r.key, stage.avg_ms || 0, stage.p95_ms || 0, stage.load_pct || 0, fftDisabled);
    }
  }
  // Browser side. raf_ms here is the inter-DRAW interval (we throttle draws
  // to the UI refresh rate). Compare against the target period rather than a
  // fixed 60 fps budget so reducing the refresh rate doesn't light the bar red.
  const targetFps = Math.max(1, store.target_ui_fps || 60);
  const targetPeriod = 1000 / targetFps;
  const raf_avg = avgRing(store.raf_ms_ring);
  const raf_p95 = p95Ring(store.raf_ms_ring);
  const raf_load = Math.max(0, (raf_avg - targetPeriod) / targetPeriod * 100);
  setPerfRow("b_raf", raf_avg, raf_p95, raf_load, false);
  // Per-canvas paint cost, scaled against the frame budget.
  for (const name of VIZ_NAMES) {
    const v = store.viz_perf[name];
    if (!v) continue;
    const avg = avgRing(v.ring);
    setPerfRow("b_" + name, avg, p95Ring(v.ring), avg / targetPeriod * 100, false);
  }
}

// ----- RAF loop -----
// We always run at requestAnimationFrame cadence (driven by the monitor) but
// only redraw when at least `1000 / target_ui_fps` ms have elapsed since the
// previous draw. The badge measures the actual draw rate, so changing the UI
// refresh rate slider is reflected directly in the "ui X fps" indicator.
let lastDrawT = performance.now();
let offlineFramePainted = false;

// Tooltips for the top-right badges so it's clear what each number means.
{
  const srv = srvFpsEl;
  if (srv) srv.title = [
    "**Server snapshot rate.** How often the server is pushing L/M/H snapshot JSON over the WebSocket.",
    "",
    "Tracks the UI refresh rate slider.",
    "",
    "*Independent of the FFT enable toggle — FFT frames are sent as separate binary messages and are not counted here.*",
  ].join("\n");
  const ui = uiFpsEl;
  if (ui) ui.title = [
    "**Browser render rate.** How often the canvases are actually being redrawn.",
    "",
    "The render loop is throttled to the UI refresh rate slider, so this should track that value (capped by the monitor's refresh rate).",
  ].join("\n");
}

setupTooltips();

function frame(now) {
  const targetFps = Math.max(1, store.target_ui_fps || 60);
  // Slack of ~half a vsync (8 ms) absorbs rAF jitter — otherwise a single
  // 15.x-ms inter-frame interval (well within normal scheduling noise on a
  // 60Hz panel) drops us to the next vsync and locks us at 30 fps. Half-period
  // is the standard Nyquist-style threshold: draw if we're closer to the
  // target tick than to the previous one.
  const period = 1000 / targetFps;
  const minPeriod = period - Math.min(8, period * 0.49);
  const elapsed = now - lastDrawT;
  if (elapsed >= minPeriod) {
    // Record the inter-draw interval (not the inter-RAF interval) so the
    // "ui fps" badge reflects actual draw cadence, which the UI refresh rate
    // slider controls.
    store.raf_ms_ring[store.raf_idx % store.raf_ms_ring.length] = elapsed;
    store.raf_idx++;
    lastDrawT = now;

    if ((store.raf_idx & 15) === 0) {
      const avg = avgRing(store.raf_ms_ring);
      const fps = avg > 0 ? Math.round(1000 / avg) : 0;
      const uiText = `ui ${fps} fps`;
      if (uiFpsEl.textContent !== uiText) uiFpsEl.textContent = uiText;
      // Server snapshot rate (snapshot JSON only; binary FFT frames excluded
      // so the FFT enable toggle doesn't move this number).
      if (store.conn === "connected") {
        const srvText = `srv ${Math.round(snapshotRate())} Hz`;
        if (srvFpsEl.textContent !== srvText) srvFpsEl.textContent = srvText;
      }
    }
    // Offline: paint one cleared frame, then stop redrawing until live again
    // (the overlay covers the canvases and there's no data to show).
    if (store.conn === "connected" || !offlineFramePainted) {
      offlineFramePainted = store.conn !== "connected";
      lines.draw();
      bars.draw();
      scene.draw();
      fft.draw();
    }
  }
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
connect();
