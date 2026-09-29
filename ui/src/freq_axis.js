// Three-band frequency picker: log-scale axis with colored draggable regions.
//
// Per band:
//   - Body drag: shifts both edges together (preserves log-width).
//   - Edge handle drag: moves only that edge.
//   - Edge labels render the live frequency in Hz / .1f kHz.
//
// Bands may overlap; validation is per-band only (matches server: lo>=20,
// hi>lo+50, hi<=0.45*sr). UI clamps during drag so the server never errors.

import { send } from "./ws.js";

const F_AXIS_MIN = 20;
const F_AXIS_MAX = 22000;
const MIN_GAP_HZ = 50;

// SVG user units == CSS px: the viewBox width tracks the element's real width
// (ResizeObserver), so text is never stretched/squished horizontally the way
// a fixed viewBox + preserveAspectRatio="none" would. H is fixed.
let W = 600;
const H = 78;
const PAD_TOP = 28;       // room for two rows of edge labels
const PAD_BOT = 14;       // room for tick labels
const TRACK_Y = PAD_TOP;
const TRACK_H = H - PAD_TOP - PAD_BOT;
const HANDLE_W = 8;       // edge-handle hit width in viewBox units

// Stagger label rows so adjacent bands don't collide: low+high on inner row,
// mid on outer row.
const LABEL_Y = { low: 22, mid: 12, high: 22 };

// Colors are CSS custom properties (set by the palette), applied as inline
// styles so a palette switch recolors the widget without rebuilding it.
const bandFill = (name) => `color-mix(in srgb, var(--${name}) 32%, transparent)`;

const TICKS = [20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
const ORDER = ["low", "mid", "high"];

const NS = "http://www.w3.org/2000/svg";

function fmtHz(v) {
  return v >= 1000 ? `${(v / 1000).toFixed(1)} kHz` : `${Math.round(v)} Hz`;
}
function fmtTick(v) {
  return v >= 1000 ? `${v / 1000}k` : `${v}`;
}

// Quantize a freq to an octave-relative "nice" multiple. Mirrors snapHz()
// in controls.js so the widget and the sliders agree on values.
function snapHz(f) {
  const step = Math.min(128, Math.max(8, Math.pow(2, Math.round(Math.log2(f / 40)))));
  return Math.max(step, Math.round(f / step) * step);
}

function el(name, attrs) {
  const e = document.createElementNS(NS, name);
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

export function makeFreqAxis(container, getSr) {
  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`,
    role: "group",
    "aria-label": "Bandpass edge editor",
  });
  svg.style.width = "100%";
  svg.style.height = `${H}px`;
  svg.style.userSelect = "none";
  svg.style.touchAction = "none";
  svg.style.display = "block";

  const lmin = Math.log(F_AXIS_MIN);
  const lmax = Math.log(F_AXIS_MAX);
  const f2x = (f) => (Math.log(f) - lmin) / (lmax - lmin) * W;
  const x2f = (x) => Math.exp(lmin + (x / W) * (lmax - lmin));

  // Base track background.
  const track = el("rect", {
    x: 0.5, y: TRACK_Y, width: W - 1, height: TRACK_H,
    stroke: "#2a2e34", "stroke-width": 1, rx: 2,
  });
  track.style.fill = "var(--bg-0)";
  svg.appendChild(track);

  // Tick lines + labels (x positions set in layoutStatic()).
  const ticks = TICKS.map((f) => {
    const line = el("line", {
      y1: TRACK_Y, y2: TRACK_Y + TRACK_H, stroke: "#1f2227", "stroke-width": 1,
    });
    const t = el("text", {
      y: H - 3, "font-size": 9, "text-anchor": "middle", fill: "#6e757d",
    });
    t.textContent = fmtTick(f);
    svg.appendChild(line);
    svg.appendChild(t);
    return { f, line, t };
  });

  function layoutStatic() {
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    track.setAttribute("width", Math.max(1, W - 1));
    // Thin out tick labels when narrow so they never collide.
    let lastX = -Infinity;
    for (const { f, line, t } of ticks) {
      const x = f2x(f);
      line.setAttribute("x1", x); line.setAttribute("x2", x);
      t.setAttribute("x", x);
      const show = x - lastX >= 26 && x > 6 && x < W - 6;
      t.style.display = show ? "" : "none";
      if (show) lastX = x;
    }
  }
  layoutStatic();

  // Per-band shapes.
  const bands = {};
  for (const name of ORDER) {
    const rect = el("rect", { y: TRACK_Y, height: TRACK_H, "stroke-width": 1 });
    rect.style.fill = bandFill(name);
    rect.style.stroke = `var(--${name})`;
    rect.style.cursor = "pointer";
    rect.dataset.band = name;
    rect.dataset.role = "body";
    const handleLo = el("rect", { y: TRACK_Y, height: TRACK_H, width: HANDLE_W, opacity: 0.85 });
    handleLo.style.fill = `var(--${name})`;
    handleLo.style.cursor = "ew-resize";
    handleLo.dataset.band = name;
    handleLo.dataset.role = "lo";
    const handleHi = el("rect", { y: TRACK_Y, height: TRACK_H, width: HANDLE_W, opacity: 0.85 });
    handleHi.style.fill = `var(--${name})`;
    handleHi.style.cursor = "ew-resize";
    handleHi.dataset.band = name;
    handleHi.dataset.role = "hi";
    const labLo = el("text", {
      y: LABEL_Y[name], "font-size": 10, "text-anchor": "middle",
      fill: "#d6d9dc", "font-variant-numeric": "tabular-nums",
      "pointer-events": "none",
    });
    const labHi = el("text", {
      y: LABEL_Y[name], "font-size": 10, "text-anchor": "middle",
      fill: "#d6d9dc", "font-variant-numeric": "tabular-nums",
      "pointer-events": "none",
    });
    svg.appendChild(rect);
    svg.appendChild(handleLo);
    svg.appendChild(handleHi);
    svg.appendChild(labLo);
    svg.appendChild(labHi);
    bands[name] = { rect, handleLo, handleHi, labLo, labHi };
  }

  container.appendChild(svg);

  // State: server's last-known values (or local mid-drag values).
  let state = {
    low:  { lo_hz: 30,   hi_hz: 250   },
    mid:  { lo_hz: 250,  hi_hz: 4000  },
    high: { lo_hz: 4000, hi_hz: 16000 },
  };

  // null = no band selected; only the selected band's edges/body are draggable.
  let selected = null;

  function applySelectionStyles() {
    for (const name of ORDER) {
      const b = bands[name];
      const isSel = name === selected;
      b.rect.style.opacity = isSel ? "1" : "0.45";
      b.rect.setAttribute("stroke-width", isSel ? 2 : 1);
      b.rect.style.cursor = isSel ? "grab" : "pointer";
      b.handleLo.style.display = isSel ? "" : "none";
      b.handleHi.style.display = isSel ? "" : "none";
      b.labLo.style.display = isSel ? "" : "none";
      b.labHi.style.display = isSel ? "" : "none";
    }
    // Raise the selected band's edge handles to the top of the SVG so they
    // win hit-testing against any overlapping band body underneath.
    if (selected) {
      const b = bands[selected];
      svg.appendChild(b.handleLo);
      svg.appendChild(b.handleHi);
      svg.appendChild(b.labLo);
      svg.appendChild(b.labHi);
    }
  }

  function setSelected(name) {
    if (selected === name) return;
    selected = name;
    applySelectionStyles();
  }

  function render() {
    for (const name of ORDER) {
      const b = bands[name];
      const s = state[name];
      const xLo = f2x(s.lo_hz);
      const xHi = f2x(s.hi_hz);
      b.rect.setAttribute("x", xLo);
      b.rect.setAttribute("width", Math.max(1, xHi - xLo));
      b.handleLo.setAttribute("x", xLo - HANDLE_W / 2);
      b.handleHi.setAttribute("x", xHi - HANDLE_W / 2);
      b.labLo.setAttribute("x", xLo);
      b.labHi.setAttribute("x", xHi);
      b.labLo.textContent = fmtHz(s.lo_hz);
      b.labHi.textContent = fmtHz(s.hi_hz);
    }
  }
  render();
  applySelectionStyles();

  function fmaxHard() {
    const sr = getSr() || 48000;
    return Math.min(F_AXIS_MAX, 0.45 * sr);
  }

  function clampEdges(lo, hi) {
    if (lo < F_AXIS_MIN) lo = F_AXIS_MIN;
    if (hi > fmaxHard()) hi = fmaxHard();
    if (hi < lo + MIN_GAP_HZ) hi = lo + MIN_GAP_HZ;
    return [lo, hi];
  }

  // Translate a pointer event to viewBox-x in [0, W].
  function evtToX(evt) {
    const r = svg.getBoundingClientRect();
    if (r.width <= 0) return 0;
    return Math.max(0, Math.min(W, (evt.clientX - r.left) / r.width * W));
  }

  // Drag state. mode in {"lo","hi","body"}.
  let drag = null;

  function startDrag(name, mode, evt) {
    evt.preventDefault();
    drag = {
      name, mode,
      startX: evtToX(evt),
      startLo: state[name].lo_hz,
      startHi: state[name].hi_hz,
    };
    if (mode === "body") bands[name].rect.style.cursor = "grabbing";
    try { svg.setPointerCapture(evt.pointerId); } catch (e) {}
  }

  function onBandPointerDown(name, role, evt) {
    if (role === "body") {
      // First click on an unselected band only selects it; a second drag
      // gesture is needed to actually shift the band.
      if (selected !== name) {
        setSelected(name);
        evt.preventDefault();
        return;
      }
      startDrag(name, "body", evt);
    } else {
      // Edge handles only react when the owning band is selected; otherwise
      // they're hidden via display:none and unhittable.
      if (selected !== name) return;
      startDrag(name, role, evt);
    }
  }

  function onMove(evt) {
    if (!drag) return;
    const x = evtToX(evt);
    let lo = drag.startLo;
    let hi = drag.startHi;
    if (drag.mode === "lo") {
      lo = snapHz(x2f(x));
      if (lo > hi - MIN_GAP_HZ) lo = hi - MIN_GAP_HZ;
    } else if (drag.mode === "hi") {
      hi = snapHz(x2f(x));
      if (hi < lo + MIN_GAP_HZ) hi = lo + MIN_GAP_HZ;
    } else {
      // Body drag: shift both edges by the same log-space delta (= multiply
      // by ratio = freq-at-cursor / freq-at-press). Preserves log-width.
      const ratio = x2f(x) / x2f(drag.startX);
      lo = drag.startLo * ratio;
      hi = drag.startHi * ratio;
      // Clamp against axis edges by adjusting the ratio so width is preserved
      // exactly (no edge "compression" against the wall).
      const fmin = F_AXIS_MIN, fmax = fmaxHard();
      if (lo < fmin) { const k = fmin / lo; lo *= k; hi *= k; }
      if (hi > fmax) { const k = fmax / hi; lo *= k; hi *= k; }
      lo = snapHz(lo);
      hi = snapHz(hi);
    }
    [lo, hi] = clampEdges(lo, hi);
    state[drag.name] = { lo_hz: lo, hi_hz: hi };
    render();
    send({ type: "set_band", band: drag.name, lo_hz: lo, hi_hz: hi, commit: false });
  }

  function endDrag(evt) {
    if (!drag) return;
    const { name } = drag;
    drag = null;
    bands[name].rect.style.cursor = "grab";
    try { if (evt && svg.hasPointerCapture(evt.pointerId)) svg.releasePointerCapture(evt.pointerId); } catch (e) {}
    const s = state[name];
    send({ type: "set_band", band: name, lo_hz: s.lo_hz, hi_hz: s.hi_hz, commit: true });
  }

  for (const name of ORDER) {
    const b = bands[name];
    b.rect.addEventListener("pointerdown",     (e) => onBandPointerDown(name, "body", e));
    b.handleLo.addEventListener("pointerdown", (e) => onBandPointerDown(name, "lo",   e));
    b.handleHi.addEventListener("pointerdown", (e) => onBandPointerDown(name, "hi",   e));
  }

  // Clicks on the SVG outside any band → deselect. Per-band handlers run
  // first (target phase); this bubble-phase listener fires afterwards and
  // skips when the target carries a band tag.
  svg.addEventListener("pointerdown", (evt) => {
    const t = evt.target;
    if (t && t.dataset && t.dataset.band) return;
    setSelected(null);
  });

  svg.addEventListener("pointermove",   onMove);
  svg.addEventListener("pointerup",     endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("lostpointercapture", endDrag);
  window.addEventListener("blur", () => endDrag(null));

  // Track the real width so user units stay 1:1 with CSS px.
  new ResizeObserver((entries) => {
    const w = Math.round(entries[entries.length - 1].contentRect.width);
    if (w > 0 && w !== W) { W = w; layoutStatic(); render(); }
  }).observe(svg);

  // Clicks anywhere else on the page → deselect.
  document.addEventListener("pointerdown", (evt) => {
    if (selected === null) return;
    if (svg.contains(evt.target)) return;
    setSelected(null);
  });

  return {
    /** Push server-confirmed band values into the widget. Ignored mid-drag
     *  so we don't fight the user; values they're dragging are authoritative
     *  until pointerup, after which the meta echo will resync. */
    syncBands(metaBands) {
      if (drag) return;
      if (!metaBands) return;
      let changed = false;
      for (const name of ORDER) {
        const b = metaBands[name];
        if (b && typeof b.lo_hz === "number" && typeof b.hi_hz === "number") {
          const cur = state[name];
          if (cur.lo_hz !== b.lo_hz || cur.hi_hz !== b.hi_hz) {
            state[name] = { lo_hz: b.lo_hz, hi_hz: b.hi_hz };
            changed = true;
          }
        }
      }
      if (changed) render();
    },
  };
}
