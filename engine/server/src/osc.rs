//! OSC over UDP (`server/io/osc_sender.py` + `osc_publisher.py`):
//! byte-identical packets from pre-encoded templates, one non-blocking
//! socket shared by the workers (L/M/H, BPM, onsets, FFT) and the control
//! loop (meta, silence). Hot-path sends never allocate; send errors are
//! swallowed (logged at DEBUG).

use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};

/// "/audio/lmh ,fff <f><f><f>": address 12 B, tags 8 B, floats at 20. 32 B.
pub const LMH_TEMPLATE: [u8; 32] = *b"/audio/lmh\0\0,fff\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
const LMH_FLOAT_OFFSET: usize = 20;
/// "/audio/bpm ,f <f>": float at 16. 20 B.
pub const BPM_TEMPLATE: [u8; 20] = *b"/audio/bpm\0\0,f\0\0\0\0\0\0";
const BPM_FLOAT_OFFSET: usize = 16;
/// "/audio/onset/{low,mid,high} ,i 1": constant, 28 B each.
pub const ONSET_PACKETS: [[u8; 28]; 3] = [
    *b"/audio/onset/low\0\0\0\0,i\0\0\0\0\0\x01",
    *b"/audio/onset/mid\0\0\0\0,i\0\0\0\0\0\x01",
    *b"/audio/onset/high\0\0\0,i\0\0\0\0\0\x01",
];

const FFT_ADDR_FIELD: &[u8; 12] = b"/audio/fft\0\0";
/// Largest `/audio/fft` packet (n_bins = MAX_N_FFT_BINS).
pub const MAX_FFT_PACKET: usize =
    crate::validate::osc_fft_packet_bytes(crate::validate::MAX_N_FFT_BINS);

pub struct OscSender {
    sock: Option<UdpSocket>,
    addrs: Vec<SocketAddr>,
}

impl OscSender {
    /// Resolve every destination once (IPv4, like the reference's AF_INET
    /// socket). An unresolvable destination is skipped with a warning.
    pub fn new(dests: &[crate::config::OscDest]) -> Self {
        let mut addrs = Vec::new();
        for d in dests {
            let port = match u16::try_from(d.port) {
                Ok(p) => p,
                Err(_) => {
                    tracing::warn!("osc destination {}:{} has an invalid port; skipped", d.host, d.port);
                    continue;
                }
            };
            match (d.host.as_str(), port).to_socket_addrs() {
                Ok(it) => match it.into_iter().find(|a| a.is_ipv4()) {
                    Some(a) => addrs.push(a),
                    None => tracing::warn!("osc destination {}:{} has no IPv4 address; skipped", d.host, port),
                },
                Err(e) => tracing::warn!("osc destination {}:{} unresolvable ({e}); skipped", d.host, port),
            }
        }
        let sock = UdpSocket::bind(("0.0.0.0", 0))
            .and_then(|s| s.set_nonblocking(true).map(|_| s))
            .map_err(|e| tracing::warn!("osc socket unavailable: {e}"))
            .ok();
        OscSender { sock, addrs }
    }

    #[inline]
    pub fn send(&self, pkt: &[u8]) {
        let Some(sock) = &self.sock else { return };
        for a in &self.addrs {
            if let Err(e) = sock.send_to(pkt, a)
                && tracing::enabled!(tracing::Level::DEBUG)
            {
                assert_no_alloc::permit_alloc(|| tracing::debug!("osc send to {a} failed: {e}"));
            }
        }
    }

    /// `/audio/meta ,iiiffffff sr blocksize n_bins low_lo low_hi mid_lo mid_hi high_lo high_hi`.
    pub fn send_meta(&self, sr: i32, blocksize: i32, n_bins: i32, bands: dsp::Bands) {
        self.send(&meta_packet(sr, blocksize, n_bins, bands));
    }

    /// `/audio/lmh 0 0 0` + `/audio/bpm 0` from the immutable templates
    /// (safe from any thread).
    pub fn send_silence(&self) {
        self.send(&LMH_TEMPLATE);
        self.send(&BPM_TEMPLATE);
    }
}

pub fn meta_packet(sr: i32, blocksize: i32, n_bins: i32, bands: dsp::Bands) -> [u8; 60] {
    let mut p = [0u8; 60];
    p[..12].copy_from_slice(b"/audio/meta\0");
    p[12..24].copy_from_slice(b",iiiffffff\0\0");
    p[24..28].copy_from_slice(&sr.to_be_bytes());
    p[28..32].copy_from_slice(&blocksize.to_be_bytes());
    p[32..36].copy_from_slice(&n_bins.to_be_bytes());
    let mut o = 36;
    for (lo, hi) in bands {
        p[o..o + 4].copy_from_slice(&(lo as f32).to_be_bytes());
        p[o + 4..o + 8].copy_from_slice(&(hi as f32).to_be_bytes());
        o += 8;
    }
    p
}

/// The DSP worker's packet buffers (L/M/H + BPM are mutated in place).
pub struct LmhPacker {
    lmh: [u8; 32],
    bpm: [u8; 20],
}

impl Default for LmhPacker {
    fn default() -> Self {
        LmhPacker { lmh: LMH_TEMPLATE, bpm: BPM_TEMPLATE }
    }
}

impl LmhPacker {
    /// `OscPublisher.publish_lmh`: scaled × gain (f64 product, packed f32),
    /// BPM, then one constant packet per band that fired.
    #[inline]
    pub fn publish(&mut self, osc: &OscSender, scaled: [f64; 3], gain: f64, onsets: [i8; 3], bpm: f64) {
        for (i, s) in scaled.iter().enumerate() {
            let o = LMH_FLOAT_OFFSET + 4 * i;
            self.lmh[o..o + 4].copy_from_slice(&((s * gain) as f32).to_be_bytes());
        }
        osc.send(&self.lmh);
        self.bpm[BPM_FLOAT_OFFSET..].copy_from_slice(&(bpm as f32).to_be_bytes());
        osc.send(&self.bpm);
        for (i, &fired) in onsets.iter().enumerate() {
            if fired != 0 {
                osc.send(&ONSET_PACKETS[i]);
            }
        }
    }
}

/// The FFT worker's packet buffer: header rebuilt only when n_bins changes.
pub struct FftPacker {
    buf: Box<[u8; MAX_FFT_PACKET]>,
    n_bins: usize,
    payload_offset: usize,
}

impl Default for FftPacker {
    fn default() -> Self {
        FftPacker { buf: Box::new([0u8; MAX_FFT_PACKET]), n_bins: 0, payload_offset: 0 }
    }
}

impl FftPacker {
    fn ensure(&mut self, n: usize) {
        if n == self.n_bins {
            return;
        }
        // OSC strings: pad to the next multiple of 4 with ≥ 1 NUL.
        let tag_len = n + 1;
        let tag_padded = (tag_len / 4 + 1) * 4;
        self.buf[..12].copy_from_slice(FFT_ADDR_FIELD);
        let t = &mut self.buf[12..12 + tag_padded];
        t.fill(0);
        t[0] = b',';
        t[1..tag_len].fill(b'f');
        self.payload_offset = 12 + tag_padded;
        self.n_bins = n;
    }

    /// Raw dB: every value below `db_floor` (sentinels included) is raised
    /// to it (`np.maximum(bins, f32(db_floor))`). No master gain.
    #[inline]
    pub fn send_raw_db(&mut self, osc: &OscSender, bins: &[f32], db_floor: f64) {
        let n = bins.len().min(crate::validate::MAX_N_FFT_BINS);
        self.ensure(n);
        let floor = db_floor as f32;
        let p = self.payload_offset;
        for (k, &b) in bins[..n].iter().enumerate() {
            let v = if b < floor { floor } else { b };
            self.buf[p + 4 * k..p + 4 * k + 4].copy_from_slice(&v.to_be_bytes());
        }
        osc.send(&self.buf[..p + 4 * n]);
    }

    /// Processed `[0, 1]` bins × f32(gain) (copied as-is when gain == 1).
    #[inline]
    pub fn send_processed(&mut self, osc: &OscSender, bins: &[f32], gain: f64) {
        let n = bins.len().min(crate::validate::MAX_N_FFT_BINS);
        self.ensure(n);
        let g = gain as f32;
        let p = self.payload_offset;
        for (k, &b) in bins[..n].iter().enumerate() {
            let v = if gain != 1.0 { b * g } else { b };
            self.buf[p + 4 * k..p + 4 * k + 4].copy_from_slice(&v.to_be_bytes());
        }
        osc.send(&self.buf[..p + 4 * n]);
    }

    #[cfg(test)]
    fn packet(&self) -> &[u8] {
        &self.buf[..self.payload_offset + 4 * self.n_bins]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn golden(name: &str) -> Vec<u8> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/wire").join(name);
        std::fs::read(p).unwrap()
    }

    /// Header bytes (everything before the payload) match the Python capture.
    #[test]
    fn templates_match_python_capture() {
        let g = golden("osc_audio_lmh.bin");
        assert_eq!(g.len(), 32);
        assert_eq!(&g[..20], &LMH_TEMPLATE[..20]);
        let g = golden("osc_audio_bpm.bin");
        assert_eq!(&g[..16], &BPM_TEMPLATE[..16]);
        for (i, band) in ["low", "mid", "high"].iter().enumerate() {
            assert_eq!(golden(&format!("osc_audio_onset_{band}.bin")), ONSET_PACKETS[i]);
        }
        let g = golden("osc_audio_meta.bin");
        let m = meta_packet(48000, 256, 128, [(88.0, 184.0), (304.0, 1760.0), (2944.0, 13952.0)]);
        assert_eq!(g, m);
        let g = golden("osc_audio_fft.bin");
        let mut f = FftPacker::default();
        let bins = vec![0.5f32; 128];
        f.ensure(128);
        let p = f.payload_offset;
        assert_eq!(&g[..p], &f.buf[..p]);
        assert_eq!(g.len(), p + 4 * 128);
        assert_eq!(osc_fft_len(128), g.len());
        let _ = bins;
    }

    fn osc_fft_len(n: usize) -> usize {
        crate::validate::osc_fft_packet_bytes(n)
    }

    #[test]
    fn fft_values() {
        let osc = OscSender { sock: None, addrs: vec![] };
        let mut f = FftPacker::default();
        f.send_raw_db(&osc, &[-1000.0, -50.0, -95.0], -90.0);
        let pk = f.packet();
        let p = pk.len() - 12;
        let vals: Vec<f32> = (0..3).map(|k| f32::from_be_bytes(pk[p + 4 * k..p + 4 * k + 4].try_into().unwrap())).collect();
        assert_eq!(vals, vec![-90.0, -50.0, -90.0]);
        f.send_processed(&osc, &[0.5, 1.0], 2.0);
        let pk = f.packet();
        assert_eq!(pk.len(), osc_fft_len(2));
        let p = pk.len() - 8;
        assert_eq!(f32::from_be_bytes(pk[p..p + 4].try_into().unwrap()), 1.0);
    }
}
