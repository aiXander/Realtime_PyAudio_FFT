//! OSC over UDP (`server/io/osc_sender.py` + `osc_publisher.py`):
//! byte-identical packets from pre-encoded templates, one non-blocking
//! socket shared by the workers (L/M/H, BPM, level, onsets, FFT) and the control
//! loop (meta, silence). Hot-path sends never allocate; send errors are
//! swallowed (logged at DEBUG). Destinations given as host names are
//! resolved off those threads, by `spawn_resolver`, first right after start
//! and then every 5 s, so one can join the network after the server started.

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, ToSocketAddrs, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// "/audio/lmh ,fff <f><f><f>": address 12 B, tags 8 B, floats at 20. 32 B.
pub const LMH_TEMPLATE: [u8; 32] = *b"/audio/lmh\0\0,fff\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
const LMH_FLOAT_OFFSET: usize = 20;
/// "/audio/bpm ,f <f>": float at 16. 20 B.
pub const BPM_TEMPLATE: [u8; 20] = *b"/audio/bpm\0\0,f\0\0\0\0\0\0";
const BPM_FLOAT_OFFSET: usize = 16;
/// "/audio/level ,f <f>": address padded to 16, float at 20. 24 B.
pub const LEVEL_TEMPLATE: [u8; 24] = *b"/audio/level\0\0\0\0,f\0\0\0\0\0\0";
const LEVEL_FLOAT_OFFSET: usize = 20;
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

/// How often the resolver thread looks every host name up again.
const RESOLVE_PERIOD: Duration = Duration::from_secs(5);
const RESOLVE_SLEEP_SLICE: Duration = Duration::from_millis(250);

/// Bit 48 marks a resolved slot; below it the IPv4 address (32 bits) and
/// the port (16 bits). 0 = not resolved yet.
const SLOT_VALID: u64 = 1 << 48;

fn pack_addr(a: SocketAddrV4) -> u64 {
    SLOT_VALID | (u64::from(u32::from(*a.ip())) << 16) | u64::from(a.port())
}

fn unpack_addr(v: u64) -> Option<SocketAddrV4> {
    (v & SLOT_VALID != 0).then(|| SocketAddrV4::new(Ipv4Addr::from((v >> 16) as u32), v as u16))
}

/// A destination given as a host name: resolved by the `osc-resolve`
/// thread into `slots[slot]`.
struct NamedDest {
    slot: usize,
    host: String,
    port: u16,
}

pub struct OscSender {
    sock: Option<UdpSocket>,
    /// One per destination. Read lock-free by the DSP / FFT workers and
    /// the control loop; written only by `set_slot`.
    slots: Box<[AtomicU64]>,
    names: Vec<NamedDest>,
}

impl OscSender {
    /// IPv4 only (like the reference's AF_INET socket). A destination whose
    /// host is an IPv4 literal is usable at once and never looked up. A host
    /// name gets an empty slot that `spawn_resolver`'s thread fills: this
    /// constructor resolves nothing, so startup never waits for a name.
    pub fn new(dests: &[crate::config::OscDest]) -> Self {
        let mut slots = Vec::new();
        let mut names = Vec::new();
        for d in dests {
            let port = match u16::try_from(d.port) {
                Ok(p) => p,
                Err(_) => {
                    tracing::warn!("osc destination {}:{} has an invalid port; skipped", d.host, d.port);
                    continue;
                }
            };
            match d.host.parse::<Ipv4Addr>() {
                Ok(ip) => slots.push(AtomicU64::new(pack_addr(SocketAddrV4::new(ip, port)))),
                Err(_) => {
                    names.push(NamedDest { slot: slots.len(), host: d.host.clone(), port });
                    slots.push(AtomicU64::new(0));
                }
            }
        }
        let sock = UdpSocket::bind(("0.0.0.0", 0))
            .and_then(|s| s.set_nonblocking(true).map(|_| s))
            .map_err(|e| tracing::warn!("osc socket unavailable: {e}"))
            .ok();
        OscSender { sock, slots: slots.into_boxed_slice(), names }
    }

    /// Lock-free and allocation-free: destinations whose name has not
    /// resolved yet are skipped.
    #[inline]
    pub fn send(&self, pkt: &[u8]) {
        let Some(sock) = &self.sock else { return };
        for slot in self.slots.iter() {
            let Some(a) = unpack_addr(slot.load(Ordering::Relaxed)) else { continue };
            if let Err(e) = sock.send_to(pkt, a)
                && tracing::enabled!(tracing::Level::DEBUG)
            {
                assert_no_alloc::permit_alloc(|| tracing::debug!("osc send to {a} failed: {e}"));
            }
        }
    }

    /// Point destination `slot` at `addr`; the next `send` on any thread
    /// uses it.
    fn set_slot(&self, slot: usize, addr: SocketAddrV4) {
        self.slots[slot].store(pack_addr(addr), Ordering::Relaxed);
    }

    /// Look every host name up once. A name that resolves (or whose address
    /// changed) is stored; a failed lookup keeps the last address, so a
    /// destination that reboots keeps receiving once it is back. Blocking
    /// (an absent `.local` name takes ~5 s to fail): resolver thread only.
    /// Logs on change only; `failing[i]` remembers that name i was warned.
    fn resolve_names(&self, failing: &mut [bool]) {
        for (n, failing) in self.names.iter().zip(failing) {
            let found = (n.host.as_str(), n.port).to_socket_addrs().map(|it| {
                it.into_iter().find_map(|a| match a {
                    SocketAddr::V4(a) => Some(a),
                    SocketAddr::V6(_) => None,
                })
            });
            match found {
                Ok(Some(a)) => {
                    if unpack_addr(self.slots[n.slot].load(Ordering::Relaxed)) != Some(a) {
                        self.set_slot(n.slot, a);
                        tracing::info!("osc destination {}:{} resolved to {a}", n.host, n.port);
                    }
                    *failing = false;
                }
                other => {
                    if !*failing {
                        let why = match other {
                            Err(e) => e.to_string(),
                            _ => "no IPv4 address".to_string(),
                        };
                        let held = match unpack_addr(self.slots[n.slot].load(Ordering::Relaxed)) {
                            Some(a) => format!("keeping {a}"),
                            None => "skipped until it resolves".to_string(),
                        };
                        tracing::warn!(
                            "osc destination {}:{} does not resolve ({why}); {held}, retrying every {} s",
                            n.host,
                            n.port,
                            RESOLVE_PERIOD.as_secs()
                        );
                    }
                    *failing = true;
                }
            }
        }
    }

    /// `/audio/meta ,iiiffffff sr blocksize n_bins low_lo low_hi mid_lo mid_hi high_lo high_hi`.
    pub fn send_meta(&self, sr: i32, blocksize: i32, n_bins: i32, bands: dsp::Bands) {
        self.send(&meta_packet(sr, blocksize, n_bins, bands));
    }

    /// `/audio/lmh 0 0 0` + `/audio/bpm 0` + `/audio/level 0` from the
    /// immutable templates (safe from any thread).
    pub fn send_silence(&self) {
        self.send(&LMH_TEMPLATE);
        self.send(&BPM_TEMPLATE);
        self.send(&LEVEL_TEMPLATE);
    }
}

/// Start the `osc-resolve` thread when any destination is a host name
/// (none: no thread). It looks the names up right away and then every
/// `RESOLVE_PERIOD`, and returns once `shared.stop` is set. Never joined:
/// at shutdown it may be inside a lookup that takes seconds. Not on the
/// control loop for the same reason (a lookup there would freeze the WS
/// broadcast and the audio watchdog).
pub fn spawn_resolver(shared: Arc<crate::shared::Shared>) {
    if shared.osc.names.is_empty() {
        return;
    }
    let run = move || {
        let mut failing = vec![false; shared.osc.names.len()];
        loop {
            shared.osc.resolve_names(&mut failing);
            let mut slept = Duration::ZERO;
            while slept < RESOLVE_PERIOD {
                if shared.stop.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(RESOLVE_SLEEP_SLICE);
                slept += RESOLVE_SLEEP_SLICE;
            }
        }
    };
    if let Err(e) = std::thread::Builder::new().name("osc-resolve".into()).spawn(run) {
        tracing::warn!("could not start the osc-resolve thread ({e}); host-name destinations stay unresolved");
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

/// The DSP worker's packet buffers (L/M/H, BPM and level are mutated in
/// place).
pub struct LmhPacker {
    lmh: [u8; 32],
    bpm: [u8; 20],
    level: [u8; 24],
}

impl Default for LmhPacker {
    fn default() -> Self {
        LmhPacker { lmh: LMH_TEMPLATE, bpm: BPM_TEMPLATE, level: LEVEL_TEMPLATE }
    }
}

impl LmhPacker {
    /// `OscPublisher.publish_lmh`: scaled × gain (f64 product, packed f32),
    /// BPM, level (no gain), then one constant packet per band that fired
    /// — a consumer woken by an onset already holds that block's level.
    #[inline]
    pub fn publish(&mut self, osc: &OscSender, scaled: [f64; 3], gain: f64, onsets: [i8; 3], bpm: f64, level: f64) {
        for (i, s) in scaled.iter().enumerate() {
            let o = LMH_FLOAT_OFFSET + 4 * i;
            self.lmh[o..o + 4].copy_from_slice(&((s * gain) as f32).to_be_bytes());
        }
        osc.send(&self.lmh);
        self.bpm[BPM_FLOAT_OFFSET..].copy_from_slice(&(bpm as f32).to_be_bytes());
        osc.send(&self.bpm);
        self.level[LEVEL_FLOAT_OFFSET..].copy_from_slice(&(level as f32).to_be_bytes());
        osc.send(&self.level);
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
        let g = golden("osc_audio_level.bin");
        assert_eq!(g.len(), 24);
        assert_eq!(&g[..20], &LEVEL_TEMPLATE[..20]);
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

    /// A destination that appears late starts receiving without a restart:
    /// nothing is sent to an unresolved name, and the first send after the
    /// resolver's store reaches it.
    #[test]
    fn late_resolved_destination_starts_receiving() {
        let rx = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        let SocketAddr::V4(rx_addr) = rx.local_addr().unwrap() else { panic!("ipv4") };
        let dest = crate::config::OscDest { host: "late-host.invalid".into(), port: rx_addr.port() as i64 };
        let osc = OscSender::new(&[dest]);
        let mut buf = [0u8; 64];

        osc.send(&LEVEL_TEMPLATE);
        rx.set_read_timeout(Some(Duration::from_millis(100))).unwrap();
        assert!(rx.recv_from(&mut buf).is_err(), "sent to an unresolved destination");

        osc.set_slot(osc.names[0].slot, rx_addr);
        osc.send(&LEVEL_TEMPLATE);
        rx.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let (n, _) = rx.recv_from(&mut buf).expect("no packet after the slot was stored");
        assert_eq!(&buf[..n], &LEVEL_TEMPLATE);
    }

    #[test]
    fn fft_values() {
        let osc = OscSender::new(&[]);
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
