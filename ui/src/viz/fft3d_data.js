// Data side of the 3D FFT view, free of DOM and WebGL so it runs in node
// too: decoding the spectro stream, and the three pieces that turn raw
// frames into a steady picture.
//
//   makeIntegrator  raw frames → frames averaged in power over the last
//                   few rows. The raw stream has no time integration: a
//                   short-window bin of anything noise-like (hats, reverb,
//                   air, a mic floor) is a new random draw every 5 ms
//                   (~±5 dB), which reads as speckle and as shimmer when
//                   the picture scrolls. Averaging power over ~40 ms is
//                   Welch's method: the variance drops, a tone keeps its
//                   level, an onset still rises within a frame or two.
//                   The same step blurs the power across frequency (a
//                   Gaussian, a fixed fraction of an octave wide): hearing
//                   pools the power inside a critical band, so detail finer
//                   than that (the comb of a cymbal, noise ripple) is
//                   nothing the ear can follow, and drawn it hides what it can.
//                   It also measures how tonal each bin is. Height is
//                   energy, and a melody sits 15–25 dB under the drums; the
//                   ear still picks it out because it is narrow in frequency
//                   and steady in time. A bin that stands clear of its half
//                   octave for a few frames is tonal: that goes out as a
//                   second byte per bin, which the display mapping lifts
//                   (toneLift) and the styles light up.
//   makePlayhead    the front edge's clock. Frames arrive in bursts; the
//                   playhead runs at exactly one row per hop of wall-clock
//                   time, as far behind the newest frame as the bursts
//                   have needed so far, so it never waits for data.
//   makeMoment      which bands are sounding in each frame: low / mid /
//                   high, each measured against its own recent swing. Goes
//                   out as one RGB texel per row; the styles turn it into
//                   the row's hue (the Color picker), so a kick, a snare and
//                   a hat are told apart by colour as well as by place.
//   makeLevel       the dB that maps to full height. Holds the loudest
//                   level of the last LEVEL_HOLD_S seconds, so a kick as
//                   loud as the previous ones moves nothing: the mapping
//                   is global, and every change re-maps the whole history.

/** Parse a spectro binary message (type 2). Returns null if it isn't one. */
export function decodeSpectro(buf) {
  if (buf.byteLength < 32) return null;
  const dv = new DataView(buf);
  if (dv.getUint8(0) !== 2 || dv.getUint8(1) !== 1) return null;
  const nBins = dv.getUint16(2, true), nFrames = dv.getUint16(4, true);
  if (buf.byteLength < 32 + nBins * nFrames) return null;
  return {
    nBins, nFrames,
    firstSeq: dv.getUint32(8, true),
    hopS: dv.getFloat32(12, true),
    fMin: dv.getFloat32(16, true),
    fMax: dv.getFloat32(20, true),
    db0: dv.getFloat32(24, true),
    dbStep: dv.getFloat32(28, true),
    data: new Uint8Array(buf, 32, nBins * nFrames),
  };
}

// ---- power integration ----

const MAX_INTEGRATE_ROWS = 32;
const MAX_BLUR_BINS = 96;
// Tonalness (0..1 per bin): how far a bin stands above the mean of the half
// octave around it, held for a moment so a drum hit doesn't count.
const TONE_NARROW_OCT = 1 / 24;       // the bin's own level: power blurred this wide (bin jitter out)
const TONE_AROUND_OCT = 1 / 2;        // its surround: box mean this wide
const TONE_PROM_DB = [3, 8];          // prominence over the surround: none … full
const TONE_RISE_S = 0.05, TONE_FALL_S = 0.03;   // has to last to count; ends with the note

/**
 * The lift a tonal bin gets, on the dB-linear level d (0 floor … 1 top):
 * up to `lift` of the way to the top, nothing at the floor (noise that
 * happens to be narrow stays down) or at the top (the loudest thing needs
 * no help, and the level reference is left alone). GLSL twin: specDb.
 */
export function toneLift(d, tone, lift) {
  const g = d <= 0 ? 0 : d >= 0.3 ? 1 : (d / 0.3) * (d / 0.3) * (3 - (2 * d) / 0.3);
  return d + tone * lift * (1 - d) * g;
}

/**
 * Boxcar average, in power, of the last `rows` frames per bin (u8 dB in,
 * u8 dB out, same quantisation). A boxcar rather than a one-pole: its
 * support is finite, so a click ends `rows` frames later instead of
 * dragging a tail (a 25 ms one-pole in power needs ~350 ms to fall 60 dB).
 */
/** Half of a normalised Gaussian kernel: weights 0 … r for a sigma in bins. */
function gaussKernel(sigma) {
  const r = Math.min(MAX_BLUR_BINS, Math.ceil(2.5 * sigma));
  const k = new Float64Array(r + 1);
  let s = 0;
  for (let j = 0; j <= r; j++) { k[j] = Math.exp(-0.5 * (j / sigma) ** 2); s += (j ? 2 : 1) * k[j]; }
  for (let j = 0; j <= r; j++) k[j] /= s;
  return k;
}

/** dst = src blurred with a half kernel. Edge bins are repeated outward, so the kernel keeps its full weight there. */
function blurInto(src, dst, kern, n) {
  const r = kern.length - 1;
  for (let b = 0; b < n; b++) {
    let a = kern[0] * src[b];
    for (let j = 1; j <= r; j++) {
      const lo = b - j, hi = b + j;
      a += kern[j] * (src[lo < 0 ? 0 : lo] + src[hi >= n ? n - 1 : hi]);
    }
    dst[b] = a;
  }
}

/**
 * Raw frames in, display frames out: two bytes per bin, [level q (u8 dB,
 * same quantisation as the input), tonalness 0..255].
 */
export function makeIntegrator() {
  let n = 0, rows = 1, hist = null, sum = null, tmp = null, head = 0, count = 0;
  let kern = null;                        // frequency blur (null: off)
  let binsPerOct = 100, hopS = 0.005, blurSigma = 0;
  let tonal = 0, kernN = null, nar = null, acc = null, held = null, tone = null, tone2 = null;
  const lut = new Float64Array(256);      // q → power, relative to q = 0
  let qPerLog10 = 20;                     // q steps per decade of power

  return {
    configure(nBins, dbStep, bpo, hop) {
      n = nBins; binsPerOct = bpo; hopS = hop;
      kernN = gaussKernel((TONE_NARROW_OCT * bpo) / 2.355);
      nar = new Float64Array(n); acc = new Float64Array(n + 1);
      held = new Float32Array(n); tone = new Float32Array(n); tone2 = new Float32Array(n);
      hist = new Uint8Array(MAX_INTEGRATE_ROWS * n);
      sum = new Float64Array(n);
      tmp = new Float64Array(n);
      for (let q = 0; q < 256; q++) lut[q] = Math.pow(10, (q * dbStep) / 10);
      qPerLog10 = 10 / dbStep;
      head = 0; count = 0;
    },
    setRows(r) { rows = Math.max(1, Math.min(MAX_INTEGRATE_ROWS, Math.round(r))); },
    /** Blur the power across frequency with a Gaussian of `sigma` bins (< 0.3: off). */
    setBlur(sigma) { blurSigma = sigma; kern = sigma >= 0.3 ? gaussKernel(sigma) : null; },
    /** Measure tonalness (the second byte) or not. */
    setTonal(on) { tonal = on ? 1 : 0; if (!tonal && held) held.fill(0); },
    /** Forget the past (a gap in the stream, a reset). */
    clear() { head = 0; count = 0; if (held) held.fill(0); },
    /** One raw frame in, the display frame (2 bytes per bin) into dst[dstOff …]. */
    push(src, dst, dstOff) {
      hist.set(src, head * n);
      head = (head + 1) % MAX_INTEGRATE_ROWS;
      if (count < MAX_INTEGRATE_ROWS) count++;
      const k = Math.min(rows, count);
      // Summed afresh each frame (k × n lookups): a running sum would drift,
      // and its error floor sits ~70 dB under the loudest value it has held.
      sum.fill(0);
      for (let i = 1; i <= k; i++) {
        const o = ((head - i + MAX_INTEGRATE_ROWS) % MAX_INTEGRATE_ROWS) * n;
        for (let b = 0; b < n; b++) sum[b] += lut[hist[o + b]];
      }
      let p = sum;
      if (kern) { blurInto(sum, tmp, kern, n); p = tmp; }
      if (tonal) {
        // Prominence of each bin over the half octave around it, held over time.
        blurInto(sum, nar, kernN, n);
        acc[0] = 0;
        for (let b = 0; b < n; b++) acc[b + 1] = acc[b] + sum[b];
        const hw = Math.max(2, Math.round((TONE_AROUND_OCT * binsPerOct) / 2));
        const up = 1 - Math.exp(-hopS / TONE_RISE_S), dn = 1 - Math.exp(-hopS / TONE_FALL_S);
        const lo10 = TONE_PROM_DB[0] / 10, span10 = (TONE_PROM_DB[1] - TONE_PROM_DB[0]) / 10;
        for (let b = 0; b < n; b++) {
          const a = b - hw < 0 ? 0 : b - hw, z = b + hw + 1 > n ? n : b + hw + 1;
          const around = (acc[z] - acc[a]) / (z - a);
          let t = (Math.log10(nar[b] / around) - lo10) / span10;
          t = t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
          held[b] += (t - held[b]) * (t > held[b] ? up : dn);
        }
        // As wide as the drawn peak: spread to the neighbours, then round the edges off.
        const d = Math.max(2, Math.round(blurSigma * 2.355 * 0.5));
        for (let b = 0; b < n; b++) {
          let m = 0;
          for (let j = b - d < 0 ? 0 : b - d, e = b + d >= n ? n - 1 : b + d; j <= e; j++) if (held[j] > m) m = held[j];
          tone[b] = m;
        }
        blurInto(tone, tone2, kern || kernN, n);
      }
      const inv = 1 / k;
      for (let b = 0; b < n; b++) {
        const q = Math.round(Math.log10(p[b] * inv) * qPerLog10);
        dst[dstOff + 2 * b] = q < 0 ? 0 : q > 255 ? 255 : q;
        dst[dstOff + 2 * b + 1] = tonal ? Math.round(Math.min(1, tone2[b] * 1.3) * 255) : 0;
      }
    },
  };
}

// ---- the moment's colour: which bands are sounding ----

// A band's activity is where its level sits between its own recent trough
// and peak. Measured on music: a held sub moves ~5 dB between kicks, hats
// ~20 dB between hits, so one fixed range would leave the first always on
// and gate the second; following both ends gives every band its full swing.
const MOMENT_SPAN_DB = [6, 24];   // peak − trough is kept inside this (steady sound: on; hits over silence: not all noise)
const MOMENT_PEAK_FALL_DB_S = 1;  // how fast the peak comes down when the band gets quieter
const MOMENT_TROUGH_RISE_DB_S = 2;   // how fast the trough comes up when the band stops dipping
const MOMENT_SILENT_DB = -90;     // mean bin level under which a band is silent whatever its history
const MOMENT_CONTRAST = 2;        // > 1: the band nearest its peak wins the hue

/**
 * Per frame, how active each of the three bands is (0..255): the band's mean
 * power in dB between the lowest and the loudest it has recently been. Each
 * band is judged on its own, so hats colour a moment as strongly as a kick
 * does though they carry far less energy. The frame is a display frame from
 * makeIntegrator (level byte of each [level, tonalness] pair), so the
 * activity has the same time smoothing as the picture it colours.
 */
export function makeMoment() {
  let n = 0, hopS = 0.005, fMin = 20, octaves = 10, b1 = 0, b2 = 0;
  const lut = new Float64Array(256);
  const peak = new Float64Array(3), trough = new Float64Array(3);
  const peakMin = MOMENT_SILENT_DB + MOMENT_SPAN_DB[1];
  const forget = () => { peak.fill(peakMin); trough.fill(peakMin - MOMENT_SPAN_DB[0]); };

  return {
    /** `qDb`: the dB each level byte q stands for (256 entries). */
    configure(nBins, qDb, fLo, fHi, hop) {
      n = nBins; hopS = hop; fMin = fLo; octaves = Math.log2(fHi / fLo);
      for (let q = 0; q < 256; q++) lut[q] = Math.pow(10, qDb[q] / 10);
      forget();
    },
    /** The two crossover frequencies between low / mid / high, Hz. */
    setSplits(f1, f2) {
      const bin = (hz) => Math.round((Math.log2(hz / fMin) / octaves) * n);
      b1 = Math.max(1, Math.min(n - 2, bin(f1)));
      b2 = Math.max(b1 + 1, Math.min(n - 1, bin(f2)));
    },
    reset: forget,
    /** One display frame (src[srcOff …], 2 bytes per bin) → 4 bytes [low, mid, high, 255] at dst[dstOff …]. */
    push(src, srcOff, dst, dstOff) {
      for (let k = 0; k < 3; k++) {
        const lo = k === 0 ? 0 : k === 1 ? b1 : b2, hi = k === 0 ? b1 : k === 1 ? b2 : n;
        let p = 0;
        for (let b = lo; b < hi; b++) p += lut[src[srcOff + 2 * b]];
        const db = 10 * Math.log10(p / (hi - lo));
        const pk = peak[k] = Math.max(db, peak[k] - MOMENT_PEAK_FALL_DB_S * hopS, peakMin);
        const tr = trough[k] = Math.min(pk - MOMENT_SPAN_DB[0],
          Math.max(pk - MOMENT_SPAN_DB[1], Math.min(db, trough[k] + MOMENT_TROUGH_RISE_DB_S * hopS)));
        const a = (db - tr) / (pk - tr);
        dst[dstOff + k] = a <= 0 ? 0 : Math.round(255 * Math.pow(a, MOMENT_CONTRAST));
      }
      dst[dstOff + 3] = 255;
    },
  };
}

// ---- the /audio/fft frames as rows (source "fft") ----

const FFT_GAP_S = 0.25;          // no frame for this long: a gap in the history, not a slow frame

/**
 * The 3D view drawing the /audio/fft stream the 2D view and the LEDs get,
 * instead of its own spectro stream. Each WS FFT frame (binary type 1)
 * becomes one row holding exactly the height the 2D graph draws for it:
 * processed → the value clamped to 0..1; raw dB → (dB − floor) / (ceiling −
 * floor) clamped, empty bins (−1000 sentinels) at 0. Quantised as h·255.
 *
 * The result has the spectro message's shape plus `linear: true` (the level
 * byte is the height itself, not dB: the view skips its averaging and its
 * dB mapping) and `qDb` (the dB each byte stands for, for the moment's
 * colour). The server's log bins span fft_f_min … sr/2, so they line up with
 * the view's bin layout. One frame reaches the UI per WS broadcast tick, or
 * per FFT hop when that is slower: that is the row period.
 */
export function makeFftRows() {
  let seq = 0, lastT = -Infinity, data = null, qDb = null, qKey = "";
  return {
    /** `bins`: the decoded frame (Float32Array); `meta`: store.meta; `now`: arrival time, ms. */
    frame(bins, meta, now) {
      const n = bins.length;
      const rawDb = !!meta.fft_send_raw_db;
      const floor = meta.fft_db_floor ?? -60, ceiling = meta.fft_db_ceiling ?? 0;
      const span = Math.max(1, ceiling - floor);
      const sr = meta.sr ?? 48000;
      const hopS = Math.max(1 / (meta.ws_snapshot_hz || 60), (meta.fft_hop ?? 512) / sr);
      if (!data || data.length !== n) data = new Uint8Array(n);
      for (let b = 0; b < n; b++) {
        const v = rawDb ? (bins[b] - floor) / span : bins[b];   // a sentinel lands far below 0
        data[b] = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(v * 255);
      }
      const key = rawDb ? `raw|${floor}|${span}` : "processed";
      if (key !== qKey) {
        qKey = key;
        qDb = new Float64Array(256);
        // Byte 0 is "nothing" (gated, or at the floor): far below MOMENT_SILENT_DB.
        // Processed values are tanh-normalised amplitudes, so they read as 20·log10.
        for (let q = 1; q < 256; q++) qDb[q] = rawDb ? floor + (q / 255) * span : 20 * Math.log10(q / 255);
        qDb[0] = -200;
      }
      const gap = (now - lastT) / 1000;
      seq += gap > FFT_GAP_S && lastT > -Infinity ? Math.round(gap / hopS) : 1;
      lastT = now;
      return {
        nBins: n, nFrames: 1, firstSeq: seq, hopS,
        fMin: meta.fft_f_min ?? 30, fMax: sr / 2,
        db0: 0, dbStep: 1 / 255, data, linear: true, kind: key, qDb,
      };
    },
  };
}

// ---- playhead ----

const SNAP_MS = 400;             // further off than this: jump (reconnect, tab return)
const LEAD_SAFETY_ROWS = 1.5;    // extra rows behind the measured need
const LEAD_MAX_MS = 250;         // never trail further than this; beyond it the front waits instead
const LEAD_DECAY_ROWS_S = 2;     // how fast the lead shrinks again after a late burst
const CLOCK_LEAK_ROWS_S = 0.05;  // lets the clock estimate follow a slow audio clock (≫ real drift)
const FOLLOW_TAU_MS = 500;       // how gently the playhead takes up a change of clock or lead

/**
 * The front edge's position (absolute seq, float).
 *
 * A frame's seq and its arrival time give one sample of "seq − now/hop".
 * Arrivals are only ever late, so the largest sample seen is the stream's
 * clock (`off`). At each arrival, how far that clock had run past the newest
 * frame held just before it is the lead the playhead needs (`need`): one
 * burst period plus that burst's lateness. The playhead trails the clock by
 * the largest recent need, and otherwise advances by wall-clock time alone,
 * so its speed is constant whatever the arrival pattern.
 */
export function makePlayhead() {
  let off = NaN, need = 0, play = NaN, lastT = 0, arrT = 0;

  return {
    reset() { off = NaN; need = 0; play = NaN; lastT = 0; },
    /** A message of `count` frames arrived at `now` ms: `prevSeq` = newest frame held before it (−1: none), `seq` = newest now. */
    arrive(now, prevSeq, seq, hopMs, count) {
      const a = seq - now / hopMs;
      if (!(off === off) || Math.abs(a - off) > SNAP_MS / hopMs) {
        // First guess for the lead: this burst's length (the next arrivals measure it).
        off = a; need = Math.min(count, LEAD_MAX_MS / hopMs); play = NaN;
      } else {
        const dt = (now - arrT) / 1000;
        if (prevSeq >= 0) {
          const ran = now / hopMs + off - prevSeq;
          need = Math.max(Math.min(ran, LEAD_MAX_MS / hopMs), need - LEAD_DECAY_ROWS_S * dt);
        }
        off = Math.max(a, off - CLOCK_LEAK_ROWS_S * dt);
      }
      arrT = now;
    },
    /** Position at `now`; never past `lastSeq` (rows beyond it hold stale data). */
    advance(now, lastSeq, hopMs) {
      if (!(off === off)) return lastSeq;
      const target = now / hopMs + off - need - LEAD_SAFETY_ROWS;
      const dt = lastT ? Math.min(200, now - lastT) : 0;
      lastT = now;
      if (!(play === play) || Math.abs(target - play) > SNAP_MS / hopMs) play = target;
      else play += dt / hopMs + (target - (play + dt / hopMs)) * (1 - Math.exp(-dt / FOLLOW_TAU_MS));
      if (play > lastSeq) play = lastSeq;
      return play;
    },
  };
}

// ---- level reference (the "auto top") ----

const LEVEL_MIN_DB = -50;   // never map anything quieter than this to full height (silence stays dark)
const LEVEL_HEADROOM_DB = 3;
const LEVEL_HOLD_S = 20;           // the loudest level of this long sets the top
const LEVEL_BUCKETS = 20;
// Toward a louder level. The mapping is one number for the whole history, so
// a fast rise squashes everything on screen at once when one loud hit lands.
// It glides instead, and what is louder than the top meanwhile stands at full
// height; only a level more than LEVEL_MAX_LAG_DB over the top (music after
// silence) is closed at once, down to that lag.
const LEVEL_RISE_TAU_S = 2;
const LEVEL_MAX_LAG_DB = 10;
const LEVEL_FALL_TAU_S = 4;        // toward a quieter one, once the hold has run out

/**
 * The dB mapped to full height. Fed the loudest bin of each batch of frames;
 * `value(now)` glides toward the largest peak of the last LEVEL_HOLD_S
 * seconds plus headroom. Steady music keeps hitting about the same peak, so
 * the top stands still; it only moves when the material gets louder than
 * anything in the hold, or stays quieter for the whole of it.
 */
export function makeLevel() {
  const bucket = new Float64Array(LEVEL_BUCKETS);
  let slot = -1, top = LEVEL_MIN_DB, lastT = 0, fresh = true;

  function roll(now) {
    const s = Math.floor(now / 1000 / (LEVEL_HOLD_S / LEVEL_BUCKETS));
    if (slot < 0 || s - slot >= LEVEL_BUCKETS) bucket.fill(-Infinity);
    else for (let i = slot + 1; i <= s; i++) bucket[i % LEVEL_BUCKETS] = -Infinity;
    slot = s;
  }

  return {
    /** Start over; the top holds until the next peak arrives, then jumps to it. */
    reset() { slot = -1; fresh = true; },
    push(peakDb, now) {
      roll(now);
      const i = slot % LEVEL_BUCKETS;
      if (peakDb > bucket[i]) bucket[i] = peakDb;
    },
    value(now) {
      roll(now);
      let m = -Infinity;
      for (let i = 0; i < LEVEL_BUCKETS; i++) if (bucket[i] > m) m = bucket[i];
      const dt = Math.min(1, Math.max(0, now - lastT) / 1000);
      lastT = now;
      if (m === -Infinity) return top;             // nothing heard yet: hold
      const target = Math.max(LEVEL_MIN_DB, m + LEVEL_HEADROOM_DB);
      if (fresh) { fresh = false; top = target; }  // no glide up from the floor
      else top += (target - top) * (1 - Math.exp(-dt / (target > top ? LEVEL_RISE_TAU_S : LEVEL_FALL_TAU_S)));
      if (top < target - LEVEL_MAX_LAG_DB) top = target - LEVEL_MAX_LAG_DB;
      return top;
    },
  };
}
