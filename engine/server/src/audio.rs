//! Audio input: device discovery, stream lifecycle, the realtime callback
//! (`server/audio/{devices,stream,callback}.py`).
//!
//! Device indices are positions in cpal's full device list (inputs and
//! outputs), which on CoreAudio is the same numbering PortAudio used. Names
//! are what's matched on reopen, since indices shift as devices come and go.
//! cpal enumerates live, so PortAudio's re-init hack isn't needed.

use crate::shared::{Shared, now_ns};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub hostapi: String,
    pub default_samplerate: f64,
    pub max_input_channels: u16,
}

impl DeviceInfo {
    pub fn to_json(&self) -> Value {
        json!({
            "index": self.index,
            "name": self.name,
            "hostapi": self.hostapi,
            "default_samplerate": self.default_samplerate,
            "max_input_channels": self.max_input_channels,
        })
    }
}

/// PortAudio-style host API names (what the reference reported).
fn hostapi_name(host: &cpal::Host) -> String {
    let n = host.id().name();
    match n.to_ascii_lowercase().as_str() {
        "coreaudio" => "Core Audio".into(),
        "alsa" => "ALSA".into(),
        "wasapi" => "Windows WASAPI".into(),
        "asio" => "ASIO".into(),
        "jack" => "JACK Audio Connection Kit".into(),
        _ => n.to_string(),
    }
}

fn device_name(d: &cpal::Device) -> String {
    d.description().map(|x| x.name().to_string()).unwrap_or_else(|_| d.to_string())
}

fn max_input_channels(d: &cpal::Device) -> u16 {
    d.supported_input_configs()
        .map(|it| it.map(|c| c.channels()).max().unwrap_or(0))
        .unwrap_or(0)
}

/// All devices with ≥ 1 input channel.
pub fn list_input_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let api = hostapi_name(&host);
    let Ok(devs) = host.devices() else { return Vec::new() };
    let mut out = Vec::new();
    for (i, d) in devs.enumerate() {
        if !d.supports_input() {
            continue;
        }
        let ch = max_input_channels(&d);
        if ch < 1 {
            continue;
        }
        let sr = d.default_input_config().map(|c| c.sample_rate() as f64).unwrap_or(0.0);
        out.push(DeviceInfo {
            index: i,
            name: device_name(&d),
            hostapi: api.clone(),
            default_samplerate: sr,
            max_input_channels: ch,
        });
    }
    out
}

/// The device at position `index` of the full list (if it's an input).
fn device_at(index: usize) -> Option<cpal::Device> {
    let host = cpal::default_host();
    let d = host.devices().ok()?.nth(index)?;
    d.supports_input().then_some(d)
}

pub fn find_input_by_name(name: Option<&str>) -> Option<usize> {
    let name = name.filter(|n| !n.is_empty())?;
    list_input_devices().into_iter().find(|d| d.name == name).map(|d| d.index)
}

/// Index of the system default input, if any.
pub fn default_input() -> Option<usize> {
    let host = cpal::default_host();
    let def = host.default_input_device()?;
    let id = def.id().ok();
    let name = device_name(&def);
    let devs = host.devices().ok()?;
    let mut by_name = None;
    for (i, d) in devs.enumerate() {
        if id.is_some() && d.id().ok() == id {
            return Some(i);
        }
        if by_name.is_none() && d.supports_input() && device_name(&d) == name {
            by_name = Some(i);
        }
    }
    by_name
}

pub fn device_info(index: usize) -> Option<DeviceInfo> {
    list_input_devices().into_iter().find(|d| d.index == index)
}

/// `resolve_initial_device`: CLI → cfg name → cfg index → system default.
pub fn resolve_initial_device(cfg_dev: &crate::config::DeviceCfg, cli: Option<usize>) -> Option<usize> {
    if cli.is_some() {
        return cli;
    }
    if let Some(name) = cfg_dev.name.as_deref().filter(|n| !n.is_empty()) {
        if let Some(i) = find_input_by_name(Some(name)) {
            return Some(i);
        }
        tracing::warn!("configured device {name:?} not found; falling back");
    }
    if let Some(idx) = cfg_dev.index {
        if idx >= 0 && list_input_devices().iter().any(|d| d.index as i64 == idx) {
            return Some(idx as usize);
        }
        tracing::warn!("configured device index {idx} not found; falling back");
    }
    default_input()
}

/// An open (possibly not yet started) input stream.
pub struct StreamHandle {
    stream: cpal::Stream,
    pub samplerate: f64,
    pub device: usize,
    pub device_name: String,
}

impl StreamHandle {
    pub fn start(&self) -> Result<(), String> {
        self.stream.play().map_err(|e| e.to_string())
    }
    /// Stop + close (drop). cpal has no separate abort; dropping never drains.
    pub fn close(self) {
        let _ = self.stream.pause();
        drop(self.stream);
    }
}

/// Sample conversion for non-f32 hosts (ALSA may only offer integers).
trait ToF32: Copy {
    fn to_f32(self) -> f32;
}
impl ToF32 for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}
impl ToF32 for i16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32 / 32768.0
    }
}
impl ToF32 for i32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32 / 2_147_483_648.0
    }
}

/// The realtime callback (`AudioCallback.__call__`): mono-mix → re-block
/// into fixed `blocksize` blocks → ring write (timestamped) → wake both
/// workers. No allocation, no locks, no logging.
/// The worker threads the callback wakes (set once they're spawned).
pub type WakeTargets = Arc<std::sync::OnceLock<[std::thread::Thread; 2]>>;

struct Callback {
    shared: Arc<Shared>,
    workers: WakeTargets,
    acc: Vec<f32>,
    fill: usize,
    /// Interleaved channels in the host buffer.
    stride: usize,
    /// Mono = `(ch0 + ch1) · 0.5` (f32) when ≥ 2 channels are used, else ch0.
    mix2: bool,
    /// Overrun detection by timestamp (see `check_gap`).
    sr: f64,
    last_ts: Option<cpal::StreamInstant>,
    last_frames: usize,
}

impl Callback {
    /// cpal's CoreAudio backend never reports `Xrun`, so on macOS an input
    /// overrun (the reference counts PortAudio's `input_overflow`) is
    /// detected as a gap in the buffers' host timestamps: more than 1.5× the
    /// previous buffer's duration means CoreAudio dropped input.
    #[inline]
    fn check_gap(&mut self, info: &cpal::InputCallbackInfo, frames: usize) {
        if !cfg!(target_os = "macos") {
            return;
        }
        let ts = info.timestamp().callback;
        if let Some(prev) = self.last_ts {
            let expected = self.last_frames as f64 / self.sr;
            if ts.duration_since(prev).as_secs_f64() > 1.5 * expected {
                self.shared.cb_overruns.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.last_ts = Some(ts);
        self.last_frames = frames;
    }

    #[inline]
    fn on_data<T: ToF32>(&mut self, data: &[T], info: &cpal::InputCallbackInfo) {
        assert_no_alloc::assert_no_alloc(|| {
            let t0 = now_ns();
            self.check_gap(info, data.len() / self.stride);
            let bs = self.acc.len();
            for frame in data.chunks_exact(self.stride) {
                let x0 = frame[0].to_f32();
                self.acc[self.fill] = if self.mix2 { (x0 + frame[1].to_f32()) * 0.5 } else { x0 };
                self.fill += 1;
                if self.fill == bs {
                    // Stamp the block the moment it becomes available.
                    self.shared.ring.write_block(&self.acc, now_ns());
                    if let Some(w) = self.workers.get() {
                        w[0].unpark();
                        w[1].unpark();
                    }
                    self.fill = 0;
                }
            }
            self.shared.perf.cb.record(now_ns() - t0);
        })
    }
}

fn build<T: ToF32 + cpal::SizedSample>(
    dev: &cpal::Device,
    cfg: cpal::StreamConfig,
    mut cb: Callback,
    shared: Arc<Shared>,
) -> Result<cpal::Stream, cpal::Error> {
    dev.build_input_stream::<T, _, _>(
        cfg,
        move |data: &[T], info: &cpal::InputCallbackInfo| cb.on_data(data, info),
        move |e: cpal::Error| {
            if e.kind() == cpal::ErrorKind::Xrun {
                shared.cb_overruns.fetch_add(1, Ordering::Relaxed);
            } else {
                tracing::warn!("audio stream error: {e}");
            }
        },
        None,
    )
}

/// `open_input_stream`: open (not start) device `index` (None = system
/// default) at its default sample rate, `channels` clamped to what it has.
pub fn open_input_stream(
    index: Option<usize>,
    blocksize: usize,
    channels: i64,
    shared: &Arc<Shared>,
    workers: &WakeTargets,
) -> Result<StreamHandle, String> {
    let index = match index {
        Some(i) => i,
        None => default_input().ok_or("no default input device")?,
    };
    let dev = device_at(index).ok_or_else(|| format!("Invalid device {index} (not an input device)"))?;
    let name = device_name(&dev);
    let def = dev.default_input_config().map_err(|e| e.to_string())?;
    let max_in = max_input_channels(&dev).max(1) as i64;
    let mut want = channels;
    if want > max_in {
        tracing::warn!("requested channels={want} but device exposes {max_in}; clamping");
        want = max_in;
    }
    if want < 1 {
        return Err(format!("Invalid number of channels: {want}"));
    }
    let sr = def.sample_rate();
    let bs = blocksize as u32;
    let buffer_size = match def.buffer_size() {
        cpal::SupportedBufferSize::Range { min, max } if *min <= bs && bs <= *max => {
            cpal::BufferSize::Fixed(bs)
        }
        _ => cpal::BufferSize::Default,
    };
    // Prefer f32 at the requested channel count; fall back to the device's
    // default layout (and take the first `want` channels from it).
    let formats = [cpal::SampleFormat::F32, def.sample_format()];
    let mut last_err = String::from("no supported input format");
    for stride in [want as u16, def.channels()] {
        for &fmt in &formats {
            let cfg = cpal::StreamConfig { channels: stride, sample_rate: sr, buffer_size };
            let cb = Callback {
                shared: shared.clone(),
                workers: workers.clone(),
                acc: vec![0.0; blocksize],
                fill: 0,
                stride: stride as usize,
                mix2: want >= 2 && stride >= 2,
                sr: sr as f64,
                last_ts: None,
                last_frames: 0,
            };
            let res = match fmt {
                cpal::SampleFormat::F32 => build::<f32>(&dev, cfg, cb, shared.clone()),
                cpal::SampleFormat::I16 => build::<i16>(&dev, cfg, cb, shared.clone()),
                cpal::SampleFormat::I32 => build::<i32>(&dev, cfg, cb, shared.clone()),
                other => {
                    last_err = format!("unsupported sample format {other}");
                    continue;
                }
            };
            match res {
                Ok(stream) => {
                    tracing::info!(
                        "stream open: device={index} ({name}) sr={sr} blocksize={blocksize} channels={want} format={fmt}"
                    );
                    return Ok(StreamHandle { stream, samplerate: sr as f64, device: index, device_name: name });
                }
                Err(e) => last_err = e.to_string(),
            }
        }
    }
    Err(last_err)
}

/// `signal_active_probe`: open each device for `duration` s and report
/// whether it delivered any signal (RMS > 1e-4). Blocking; run off the loop.
pub fn signal_active_probe(devices: &[DeviceInfo], duration: f64) -> Vec<(usize, bool)> {
    devices.iter().map(|d| (d.index, probe_one(d, duration).unwrap_or(false))).collect()
}

fn probe_stream<T: ToF32 + cpal::SizedSample>(
    dev: &cpal::Device,
    cfg: cpal::StreamConfig,
    buf: &Arc<std::sync::Mutex<Vec<f32>>>,
    n: usize,
) -> Option<cpal::Stream> {
    let b2 = buf.clone();
    dev.build_input_stream::<T, _, _>(
        cfg,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            if let Ok(mut v) = b2.lock() {
                let room = n - v.len().min(n);
                v.extend(data[..data.len().min(room)].iter().map(|x| x.to_f32()));
            }
        },
        |_e| {},
        None,
    )
    .ok()
}

fn probe_one(d: &DeviceInfo, duration: f64) -> Option<bool> {
    let dev = device_at(d.index)?;
    let def = dev.default_input_config().ok()?;
    let sr = def.sample_rate();
    let ch = d.max_input_channels.clamp(1, 2);
    let n = ((sr as f64 * duration) as usize).max(256) * ch as usize;
    let buf = Arc::new(std::sync::Mutex::new(Vec::<f32>::with_capacity(n)));
    let cfg = cpal::StreamConfig { channels: ch, sample_rate: sr, buffer_size: cpal::BufferSize::Default };
    // f32 first, then the device's own format (integer-only ALSA devices),
    // like the main stream; samples are converted to f32 either way.
    let stream = [cpal::SampleFormat::F32, def.sample_format()].iter().find_map(|&fmt| match fmt {
        cpal::SampleFormat::F32 => probe_stream::<f32>(&dev, cfg, &buf, n),
        cpal::SampleFormat::I16 => probe_stream::<i16>(&dev, cfg, &buf, n),
        cpal::SampleFormat::I32 => probe_stream::<i32>(&dev, cfg, &buf, n),
        _ => None,
    })?;
    stream.play().ok()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(duration + 2.0);
    loop {
        std::thread::sleep(std::time::Duration::from_millis(10));
        let len = buf.lock().map(|v| v.len()).unwrap_or(n);
        if len >= n || std::time::Instant::now() > deadline {
            break;
        }
    }
    drop(stream);
    let v = buf.lock().ok()?;
    if v.is_empty() {
        return Some(false);
    }
    let ms = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64;
    Some(ms.sqrt() > 1e-4)
}
