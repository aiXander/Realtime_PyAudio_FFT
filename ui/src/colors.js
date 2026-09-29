// Single source of truth for UI colors: the L/M/H band colors, the accent,
// the background tint and the 2D spectrum ramp — all set by the active
// palette (server-persisted as ui.palette, meta `ui_palette`).
//
// `LMH`, `LMH_HEX` and `theme` are mutated IN PLACE by applyPalette(), so
// importers can hold references. Anything that caches derived colors (static
// canvas layers, gradients, sprite ramps) keys its cache on `theme.version`.
// DOM/CSS picks the palette up through the custom properties set on :root
// (--low / --mid / --high / --accent* / --bg-*).

// Keep the names in sync with PALETTES in server/control/validate.py.
export const PALETTES = {
  classic: {
    label: "Classic",
    low: "#5a8dee", mid: "#79d17a", high: "#e8a857",
    accent: "#7da6ff", accentHi: "#95b8ff", accentLo: "#6a96f0",
    bg: ["#0a0b0d", "#0e0f12", "#14161a", "#181b21"],
    ramp: null,   // null = the original blue → green → red spectrum ramp
  },
  neon: {
    label: "Neon",
    low: "#ff3dbb", mid: "#3de0ff", high: "#f4f36b",
    accent: "#c77dff",
    bg: ["#0b0710", "#0f0a16", "#16101f", "#1c1428"],
    ramp: ["#3a0f6e", "#ff3dbb", "#3de0ff", "#f9f8b0"],
  },
  ember: {
    label: "Ember",
    low: "#e5483a", mid: "#f39a3c", high: "#ffd98a",
    accent: "#ff8a4c",
    bg: ["#0d0a09", "#120e0c", "#1a1411", "#211915"],
    ramp: ["#4a0d08", "#d0371f", "#f59b38", "#fff2cf"],
  },
  ocean: {
    label: "Ocean",
    low: "#4174e8", mid: "#26c4c9", high: "#aef2d9",
    accent: "#4cc9f0",
    bg: ["#080b0f", "#0b1015", "#10171f", "#141d27"],
    ramp: ["#0c2257", "#2f6fe0", "#26c4c9", "#e0fff4"],
  },
  sunset: {
    label: "Sunset",
    low: "#8062ff", mid: "#ff5f8f", high: "#ffb74d",
    accent: "#ff7aa2",
    bg: ["#0c090e", "#110c13", "#18121b", "#1f1722"],
    ramp: ["#2a1260", "#8062ff", "#ff5f8f", "#ffd9a0"],
  },
  mono: {
    label: "Mono",
    low: "#6f7a88", mid: "#aab4c1", high: "#eef2f6",
    accent: "#d5dce5",
    bg: ["#0a0a0b", "#0e0e10", "#151517", "#1a1a1d"],
    ramp: ["#1d2127", "#5f6874", "#b3bcc7", "#ffffff"],
  },
  okabe: {
    label: "Colorblind-safe",
    low: "#56b4e9", mid: "#e69f00", high: "#cc79a7",
    accent: "#56b4e9",
    bg: ["#0a0b0d", "#0e0f12", "#14161a", "#181b21"],
    ramp: ["#0b2a4a", "#0072b2", "#e69f00", "#f0e442"],
  },
};
export const PALETTE_ORDER = ["classic", "neon", "ember", "ocean", "sunset", "mono", "okabe"];
export const DEFAULT_PALETTE = "classic";

export const LMH_ORDER = ["low", "mid", "high"];
export const LMH = {
  low:  { hex: "", rgb: "", hue: 0, sat: 0 },
  mid:  { hex: "", rgb: "", hue: 0, sat: 0 },
  high: { hex: "", rgb: "", hue: 0, sat: 0 },
};
export const LMH_HEX = ["", "", ""];

/** Active palette: name, canvas background, spectrum ramp; version bumps on every change. */
export const theme = { name: "", bg: "", ramp: null, version: 0 };

export function lmhRgba(name, alpha) {
  return `rgba(${LMH[name].rgb},${alpha})`;
}

export function hexRgb(h) {
  return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
}

/** HSL hue (degrees) and saturation (%) of an [r, g, b] color. */
function hueSat([r, g, b]) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  if (d === 0) return [0, 0];
  const h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  const l = (mx + mn) / 2;
  return [Math.round((h * 60 + 360) % 360), Math.round((100 * d) / (1 - Math.abs(2 * l - 1)))];
}

const mixHex = (h, target, t) => {
  const c = hexRgb(h);
  const m = c.map((v) => Math.round(v + (target - v) * t));
  return `rgb(${m.join(",")})`;
};

/** Switch the whole UI to a palette. Unknown names fall back to the default. */
export function applyPalette(name) {
  if (!PALETTES[name]) name = DEFAULT_PALETTE;
  if (name === theme.name) return false;
  const p = PALETTES[name];
  LMH_ORDER.forEach((k, i) => {
    const rgb = hexRgb(p[k]);
    LMH[k].hex = p[k];
    LMH[k].rgb = rgb.join(",");
    [LMH[k].hue, LMH[k].sat] = hueSat(rgb);
    LMH_HEX[i] = p[k];
  });
  theme.name = name;
  theme.bg = p.bg[0];
  theme.ramp = p.ramp;
  theme.version++;

  const root = document.documentElement.style;
  root.setProperty("--low", p.low);
  root.setProperty("--mid", p.mid);
  root.setProperty("--high", p.high);
  root.setProperty("--accent", p.accent);
  root.setProperty("--accent-hi", p.accentHi || mixHex(p.accent, 255, 0.18));
  root.setProperty("--accent-lo", p.accentLo || mixHex(p.accent, 0, 0.1));
  root.setProperty("--accent-glow", `rgba(${hexRgb(p.accent).join(",")},0.32)`);
  root.setProperty("--accent-soft", `rgba(${hexRgb(p.accent).join(",")},0.22)`);
  ["--bg-0", "--bg-1", "--bg-2", "--bg-2-top"].forEach((v, i) => root.setProperty(v, p.bg[i]));
  const tc = document.querySelector('meta[name="theme-color"]');
  if (tc) tc.setAttribute("content", p.bg[2]);
  return true;
}

applyPalette(DEFAULT_PALETTE);
