// Single source of truth for live data and meta. WS handlers update; UI reads.

export const store = {
  // Live signal
  low: 0, mid: 0, high: 0,
  low_raw: 0, mid_raw: 0, high_raw: 0,
  // Per-band onset detection (server-side onset detector running
  // independently on each of low/mid/high). `*_onset_pulse_t` is the
  // performance.now() timestamp of the most recent <band>_onset=1 snapshot
  // — visualizers use these to drive band-specific decay-based flashes.
  // `bpm` is derived from the low-band onset stream.
  low_onset_pulse_t:  -Infinity,
  mid_onset_pulse_t:  -Infinity,
  high_onset_pulse_t: -Infinity,
  bpm: 0,
  fft_bins: null, // Float32Array
  fft_db_floor: -60,
  fft_db_ceiling: 0,
  // FFT mode mirror — server-driven via meta. The UI checkbox just sends a
  // set_fft_send_raw_db WS message; the value here is reflected from meta on
  // every meta payload so OSC and the viz can never disagree.
  fft_send_raw_db: false,

  // Meta
  meta: {
    sr: 48000,
    blocksize: 256,
    n_fft_bins: 128,
    bands: {
      low:  { lo_hz: 30,   hi_hz: 250   },
      mid:  { lo_hz: 250,  hi_hz: 4000  },
      high: { lo_hz: 4000, hi_hz: 16000 },
    },
    fft_enabled: false,
    tau: { low: 0.15, mid: 0.06, high: 0.02 },
    autoscale: { tau_attack_s: 0.05, tau_release_s: 60, noise_floor: 0.001, strength: 1.0 },
    ws_snapshot_hz: 60,
    device: { index: null, name: null }
  },

  devices: [],
  presets: [],

  status: {
    cb_overruns: 0, dsp_drops: 0, fft_drops: 0,
    perf: null
  },

  // Target browser render rate (Hz). Mirrors the UI refresh rate slider; the
  // RAF loop in main.js throttles draws to this rate.
  target_ui_fps: 60,

  // Visual peak-hold decay rate (units / second) for both the L/M/H bars and
  // the FFT viz. Server-authoritative — mirrored from meta.ui_peak_decay_per_s.
  peak_decay_per_s: 0.6,

  // Whether to render the per-band onset squares above the L/M/H bars. UI
  // visibility toggle — does NOT affect server-side detection, OSC, or WS
  // payload (those run unconditionally). Server-authoritative — mirrored
  // from meta.ui_show_onsets and persisted to ui.show_onsets in YAML.
  show_onsets: false,

  // Visual history window (seconds) for the L/M/H rolling-lines chart. UI-only
  // — not persisted, not sent to the server. Range: 2..30s.
  lines_history_s: 5,

  // Connection state mirror ("connecting" | "connected" | "reconnecting")
  // and whether the first meta after (re)connect has arrived. Controls stay
  // disabled until `synced` so they never show stale values as live.
  conn: "connecting",
  synced: false,

  // Browser-side perf
  raf_ms_ring: new Array(60).fill(0),
  raf_idx: 0,

  // Per-viz render time rings (ms)
  viz_perf: {
    lines: { ring: new Array(60).fill(0), idx: 0 },
    bars:  { ring: new Array(60).fill(0), idx: 0 },
    scene: { ring: new Array(60).fill(0), idx: 0 },
    fft:   { ring: new Array(60).fill(0), idx: 0 }
  },

  // Last-confirmed values for slider snap-back on error
  lastConfirmed: {}
};

export function recordVizPerf(name, ms) {
  const v = store.viz_perf[name];
  if (!v) return;
  v.ring[v.idx % v.ring.length] = ms;
  v.idx++;
}

/** Drop all live signal state (on disconnect) so nothing frozen is shown. */
export function clearLive() {
  store.low = store.mid = store.high = 0;
  store.low_raw = store.mid_raw = store.high_raw = 0;
  store.low_onset_pulse_t = store.mid_onset_pulse_t = store.high_onset_pulse_t = -Infinity;
  store.bpm = 0;
  store.fft_bins = null;
}

export function avgRing(ring) {
  let s = 0, n = 0;
  for (const v of ring) { if (v > 0) { s += v; n++; } }
  return n ? s / n : 0;
}

export function p95Ring(ring) {
  const sorted = ring.filter(v => v > 0).sort((a, b) => a - b);
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
}
