// Registry of 3D FFT styles. Keep names in sync with FFT3D_STYLES in
// server/control/validate.py (the server validates set_fft3d_style).

import cloud from "./cloud.js";
import aurora from "./aurora.js";
import topo from "./topo.js";
import synthwave from "./synthwave.js";
import engraving from "./engraving.js";
import embers from "./embers.js";

export const STYLE_ORDER = [cloud, aurora, topo, synthwave, engraving, embers];
export const STYLES = Object.fromEntries(STYLE_ORDER.map((s) => [s.name, s]));
export const DEFAULT_STYLE = "cloud";
