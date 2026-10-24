//! Shared low-level math helpers (port of `server/dsp/_math.py`), plus the
//! handful of numpy primitives whose exact arithmetic the goldens depend on
//! (`linspace`, `logspace`, `interp`, pairwise summation).

pub const EPS: f64 = 1e-12;

/// `math.log(10.0) / 20.0`: 10^(x/20) ≡ exp(x · LN10_OVER_20).
pub fn ln10_over_20() -> f64 {
    10f64.ln() / 20.0
}

/// Reference frequency for spectral tilt: the offset is zero here.
pub const TILT_REF_HZ: f64 = 1000.0;

const TAU_FLOOR: f64 = 1e-3;

/// One-pole alpha: `1 - exp(-dt / max(tau, 1e-3))`.
pub fn tau_to_alpha(dt: f64, tau: f64) -> f64 {
    1.0 - (-dt / tau.max(TAU_FLOOR)).exp()
}

/// Vectorised `tau_to_alpha` into `out` (same formula, element-wise).
pub fn tau_to_alpha_into(dt: f64, tau: &[f64], out: &mut [f64]) {
    for (o, &t) in out.iter_mut().zip(tau) {
        *o = 1.0 - (-dt / t.max(TAU_FLOOR)).exp();
    }
}

/// Per-frequency tilt in dB: `db_per_oct · log2(max(f, 1e-3) / ref_hz)`.
pub fn tilt_db_curve(freqs: &[f64], db_per_oct: f64) -> Vec<f64> {
    freqs.iter().map(|&f| tilt_db(f, db_per_oct)).collect()
}

#[inline]
pub fn tilt_db(f: f64, db_per_oct: f64) -> f64 {
    db_per_oct * (f.max(1e-3) / TILT_REF_HZ).log2()
}

/// Geometric centers of `n_bins` log-spaced bins from f_min to f_max, using
/// `(k + 0.5) / n` in log10 space (a different formula from the bin-map
/// edges in `fft::build_log_bin_map`; parity trap 7).
pub fn log_bin_centers(n_bins: usize, f_min: f64, f_max: f64) -> Vec<f64> {
    (0..n_bins)
        .map(|k| log_bin_center(k, n_bins, f_min, f_max))
        .collect()
}

/// Element `k` of `log_bin_centers` (alloc-free form, same arithmetic).
#[inline]
pub fn log_bin_center(k: usize, n_bins: usize, f_min: f64, f_max: f64) -> f64 {
    let log_fmin = f_min.max(1e-3).log10();
    let log_span = (f_max.max(1e-3).log10() - log_fmin).max(1e-6);
    let log_f = log_fmin + ((k as f64 + 0.5) / n_bins as f64) * log_span;
    10f64.powf(log_f)
}

/// `np.linspace(start, stop, num)` (endpoint=True): `i·step + start`, the
/// last element set to `stop` exactly.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let delta = stop - start;
    let step = delta / div;
    let mut y: Vec<f64> = if step == 0.0 {
        (0..num).map(|i| (i as f64 / div) * delta + start).collect()
    } else {
        (0..num).map(|i| i as f64 * step + start).collect()
    };
    y[num - 1] = stop;
    y
}

/// `np.logspace(start, stop, num)` = `10 ** linspace(start, stop, num)`.
pub fn logspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    linspace(start, stop, num)
        .into_iter()
        .map(|v| 10f64.powf(v))
        .collect()
}

/// `np.interp(x, xp, fp)` for increasing `xp`: clamps to `fp[0]` / `fp[-1]`
/// outside the range, same slope formula and NaN fallback as numpy.
pub fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    x.iter().map(|&xv| interp1(xv, xp, fp)).collect()
}

/// One point of `interp` (alloc-free; numpy's slope formula
/// `(fp[j+1]-fp[j]) / (xp[j+1]-xp[j])` evaluated on the fly).
pub fn interp1(xv: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    assert!(n >= 1 && fp.len() == n);
    if xv.is_nan() {
        return xv;
    }
    if xv > xp[n - 1] {
        return fp[n - 1];
    }
    if xv < xp[0] {
        return fp[0];
    }
    if xv == xp[n - 1] {
        return fp[n - 1];
    }
    // largest j with xp[j] <= xv (then xv < xp[j+1])
    let j = xp.partition_point(|&p| p <= xv) - 1;
    if j == n - 1 {
        return fp[n - 1];
    }
    let s = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let r = s * (xv - xp[j]) + fp[j];
    if r.is_nan() {
        let r2 = s * (xv - xp[j + 1]) + fp[j + 1];
        if r2.is_nan() && fp[j] == fp[j + 1] {
            return fp[j];
        }
        return r2;
    }
    r
}

/// numpy's `pairwise_sum` (the float add-reduce inner kernel), starting
/// from 0. `np.sum(a)` of a contiguous f64 array equals `pairwise_sum(a)`.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        let end = n - (n % 8);
        while i < end {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.add.reduceat` over one segment: the first element, then the
/// pairwise sum of the rest added onto it.
#[inline]
pub fn reduce_segment(a: &[f64]) -> f64 {
    match a.split_first() {
        None => 0.0,
        Some((first, rest)) => {
            if rest.is_empty() {
                *first
            } else {
                first + pairwise_sum(rest)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_clamps_and_blends() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [10.0, 20.0, 40.0];
        let y = interp(&[-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0], &xp, &fp);
        assert_eq!(y, vec![10.0, 10.0, 15.0, 20.0, 30.0, 40.0, 40.0]);
    }

    #[test]
    fn linspace_endpoint_exact() {
        let y = linspace(0.0, 24000.0, 513);
        assert_eq!(y[0], 0.0);
        assert_eq!(y[512], 24000.0);
        assert_eq!(y[1], 24000.0 / 512.0);
    }
}
