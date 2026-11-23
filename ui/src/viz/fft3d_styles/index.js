// Registry of 3D FFT styles. Keep names in sync with FFT3D_STYLES in
// server/control/validate.py (the server validates set_fft3d_style).

import cloud from "./cloud.js";
import topo from "./topo.js";
import spectrogram from "./spectrogram.js";

export const STYLE_ORDER = [cloud, topo, spectrogram];
export const STYLES = Object.fromEntries(STYLE_ORDER.map((s) => [s.name, s]));
export const DEFAULT_STYLE = "cloud";

/** The style to draw for a saved style name. */
export const styleFor = (name) => STYLES[name] || STYLES[DEFAULT_STYLE];
