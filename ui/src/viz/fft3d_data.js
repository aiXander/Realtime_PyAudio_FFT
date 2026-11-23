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
//   makePlayhead    the front edge's clock. Frames arrive in bursts; the
//                   playhead runs at exactly one row per hop of wall-clock
//                   time, as far behind the newest frame as the bursts
//                   have needed so far, so it never waits for data.
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

/**
 * Boxcar average, in power, of the last `rows` frames per bin (u8 dB in,
 * u8 dB out, same quantisation). A boxcar rather than a one-pole: its
 * support is finite, so a click ends `rows` frames later instead of
 * dragging a tail (a 25 ms one-pole in power needs ~350 ms to fall 60 dB).
 */
export function makeIntegrator() {
  let n = 0, rows = 1, hist = null, sum = null, head = 0, count = 0;
  const lut = new Float64Array(256);      // q → power, relative to q = 0
  let qPerLog10 = 20;                     // q steps per decade of power

  return {
    configure(nBins, dbStep) {
      n = nBins;
      hist = new Uint8Array(MAX_INTEGRATE_ROWS * n);
      sum = new Float64Array(n);
      for (let q = 0; q < 256; q++) lut[q] = Math.pow(10, (q * dbStep) / 10);
      qPerLog10 = 10 / dbStep;
      head = 0; count = 0;
    },
    setRows(r) { rows = Math.max(1, Math.min(MAX_INTEGRATE_ROWS, Math.round(r))); },
    /** Forget the past (a gap in the stream, a reset). */
    clear() { head = 0; count = 0; },
    /** One raw frame in, the integrated frame into dst[dstOff …]. */
    push(src, dst, dstOff) {
      hist.set(src, head * n);
      head = (head + 1) % MAX_INTEGRATE_ROWS;
      if (count < MAX_INTEGRATE_ROWS) count++;
      const k = Math.min(rows, count);
      if (k === 1) { dst.set(src, dstOff); return; }
      // Summed afresh each frame (k × n lookups): a running sum would drift,
      // and its error floor sits ~70 dB under the loudest value it has held.
      sum.fill(0);
      for (let i = 1; i <= k; i++) {
        const o = ((head - i + MAX_INTEGRATE_ROWS) % MAX_INTEGRATE_ROWS) * n;
        for (let b = 0; b < n; b++) sum[b] += lut[hist[o + b]];
      }
      const inv = 1 / k;
      for (let b = 0; b < n; b++) {
        const q = Math.round(Math.log10(sum[b] * inv) * qPerLog10);
        dst[dstOff + b] = q < 0 ? 0 : q > 255 ? 255 : q;
      }
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
const LEVEL_RISE_TAU_S = 0.15;     // toward a louder level
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
      return top;
    },
  };
}
