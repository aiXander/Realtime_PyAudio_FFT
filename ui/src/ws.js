// WebSocket connection with exponential backoff (cap 2s), connection-state
// fan-out, and message routing.
//
// Connection states (emitted to onStatus listeners and shown on the badge):
//   "connecting"   — first attempt after page load
//   "connected"    — socket open (the server greets with meta/devices/presets)
//   "reconnecting" — socket dropped; retrying with backoff. Clicking the
//                    badge retries immediately.

import { store } from "./store.js";

// Pick WS URL based on how the page was loaded:
//  - https:// (e.g. served via `tailscale serve` over the tailnet) → use a
//    same-origin wss://host/ws so the WebSocket upgrade rides the same
//    HTTPS proxy. The Pi has a `tailscale serve --https=10000 --set-path=/ws
//    http://localhost:8765` mount that forwards /ws to the raw WS server.
//    This avoids mixed-content blocks AND avoids needing a separate
//    tailscale port for 8765 (only 443/8443/10000 are allowed anyway).
//  - http:// on LAN (the audio-server's own UI on :8766) → keep the legacy
//    direct ws://host:8765 so dev / hotspot access still works unchanged.
//  - file:// or unknown → fall back to 127.0.0.1:8765.
// `?ws_port=N` overrides the port (e.g. a second dev server on other ports).
const WS_URL = (() => {
  if (location.protocol === "https:") {
    return `wss://${location.host}/ws`;
  }
  const port = Number(new URLSearchParams(location.search).get("ws_port")) || 8765;
  if (location.protocol === "http:") {
    return `ws://${location.hostname}:${port}`;
  }
  return `ws://127.0.0.1:${port}`;
})();

let socket = null;
let backoffMs = 250;
let retryTimer = null;
let retryAt = 0;
let countdownTimer = null;
let attempts = 0;
let state = "connecting";
const handlers = {};
const errSinks = [];
const statusSinks = [];
const fftSinks = [];

export function onMessage(type, fn) {
  handlers[type] = fn;
}

export function onError(fn) {
  errSinks.push(fn);
}

/** fn(Float32Array) on every accepted binary FFT frame, at arrival time
 *  (the FFT history for the 3D view must not depend on draw cadence). */
export function onFftFrame(fn) {
  fftSinks.push(fn);
}

/** fn(state) on every connection-state change: "connecting" | "connected" | "reconnecting". */
export function onStatus(fn) {
  statusSinks.push(fn);
}

export function send(obj) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(obj));
  return true;
}

export function isConnected() {
  return !!socket && socket.readyState === WebSocket.OPEN;
}

const badge = document.getElementById("ws-status");
const badgeText = badge ? badge.querySelector(".badge-text") || badge : null;

function renderBadge() {
  if (!badge) return;
  let text;
  if (state === "connected") text = "live";
  else if (state === "connecting") text = "connecting…";
  else {
    const s = Math.max(0, Math.ceil((retryAt - performance.now()) / 1000));
    text = retryTimer && s > 0 ? `reconnecting · ${s}s` : "reconnecting…";
  }
  if (badgeText.textContent !== text) badgeText.textContent = text;
  badge.className = "badge badge-conn " + state;
  badge.setAttribute("aria-disabled", state === "reconnecting" ? "false" : "true");
  badge.setAttribute("aria-label",
    state === "connected" ? "WebSocket connected"
    : state === "connecting" ? "Connecting to the audio server"
    : "Disconnected from the audio server. Activate to retry now.");
}

function setState(next) {
  if (next === state) { renderBadge(); return; }
  state = next;
  renderBadge();
  for (const f of statusSinks) f(state);
}

function onOpen() {
  backoffMs = 250;
  attempts = 0;
  clearTimeout(countdownTimer);
  setState("connected");
}

function scheduleReconnect() {
  clearTimeout(retryTimer);
  attempts++;
  retryAt = performance.now() + backoffMs;
  retryTimer = setTimeout(() => { retryTimer = null; connect(); }, backoffMs);
  backoffMs = Math.min(backoffMs * 2, 2000);
  tickCountdown();
}

function tickCountdown() {
  clearTimeout(countdownTimer);
  renderBadge();
  if (retryTimer) countdownTimer = setTimeout(tickCountdown, 250);
}

function onClose(ev) {
  if (ev.target !== socket) return; // stale socket from a manual retry
  socket = null;
  setState("reconnecting");
  scheduleReconnect();
}

function onWsError(_e) {
  // close handler will fire
}

/** Retry right away (badge click). No-op while connected or connecting. */
export function retryNow() {
  if (socket) return;
  clearTimeout(retryTimer);
  retryTimer = null;
  backoffMs = 250;
  connect();
}

if (badge) badge.addEventListener("click", retryNow);

function decodeFftBinary(buf) {
  // [type=1:u8][reserved:u8][n_bins:u16][float32 * n_bins] LE
  if (buf.byteLength < 4) return null;
  const dv = new DataView(buf);
  const type = dv.getUint8(0);
  if (type !== 1) return null;
  const n = dv.getUint16(2, true);
  if (buf.byteLength < 4 + 4 * n) return null;
  return new Float32Array(buf, 4, n);
}

// Snapshot arrival times for the "srv Hz" badge — fixed ring, no shifting.
const SNAP_RING = 64;
const snapTimes = new Float64Array(SNAP_RING);
let snapIdx = 0, snapCount = 0;

/** Measured server snapshot rate (Hz) over the last ~second of arrivals. */
export function snapshotRate() {
  if (snapCount < 2) return 0;
  const newest = snapTimes[(snapIdx - 1 + SNAP_RING) % SNAP_RING];
  // Stale: nothing for a second → report 0 rather than the last rate.
  if (performance.now() - newest > 1000) return 0;
  const n = Math.min(snapCount, SNAP_RING);
  const oldest = snapTimes[(snapIdx - n + SNAP_RING) % SNAP_RING];
  const span = (newest - oldest) / 1000;
  return span > 0 ? (n - 1) / span : 0;
}

export function resetSnapshotRate() { snapIdx = 0; snapCount = 0; }

function onMsg(ev) {
  if (ev.target !== socket) return;
  if (typeof ev.data !== "string") {
    // Binary -> FFT frame
    const buf = ev.data instanceof ArrayBuffer ? ev.data : null;
    if (buf) {
      const f32 = decodeFftBinary(buf);
      // Defensive: ignore stray FFT frames if the server has reported FFT as
      // disabled, so the UI can never paint live bars while the toggle /
      // side panel reflect the disabled state.
      if (f32 && store.meta?.fft_enabled !== false) {
        store.fft_bins = f32;
        for (const fn of fftSinks) fn(f32);
      }
    }
    return;
  }

  let msg;
  try { msg = JSON.parse(ev.data); }
  catch { return; }
  // Track snapshot rate independently of FFT binary frames so the badge
  // reflects "how often does the server push L/M/H state".
  if (msg.type === "snapshot") {
    snapTimes[snapIdx] = performance.now();
    snapIdx = (snapIdx + 1) % SNAP_RING;
    snapCount++;
  }
  const h = handlers[msg.type];
  if (h) h(msg);
  if (msg.type === "error") {
    for (const f of errSinks) f(msg.reason || "(no reason)");
  }
}

export function connect() {
  if (socket) return;
  if (state !== "connected" && attempts > 0) setState("reconnecting");
  try {
    const s = new WebSocket(WS_URL);
    s.binaryType = "arraybuffer";
    s.addEventListener("open", onOpen);
    s.addEventListener("close", onClose);
    s.addEventListener("error", onWsError);
    s.addEventListener("message", onMsg);
    socket = s;
    renderBadge();
  } catch (e) {
    socket = null;
    setState("reconnecting");
    scheduleReconnect();
  }
}
