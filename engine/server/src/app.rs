//! The orchestrator (`server/main.py:App`): owns the config, the stream,
//! the workers' command queues, the WS clients and every timer. One control
//! loop (`run`) does all mutation — the same invariant as the reference's
//! single asyncio loop — while the callback and the workers only touch
//! atomics, the ring and their own state.

use crate::audio::{self, StreamHandle, WakeTargets};
use crate::config::{self, Config};
use crate::osc::OscSender;
use crate::persist::Persister;
use crate::ring::{SlotRing, ring_slots_for};
use crate::shared::Shared;
use crate::workers::{DspCmd, DspCore, FftCmd, Workers};
use crate::ws::{Client, Event};
use axum::extract::ws::{Message, Utf8Bytes};
use dsp::fft_post::{AutoscaleUpdate, Smear};
use dsp::filters::{FilterDesign, clamp_band_edges};
use dsp::onset::{OnsetParams, OnsetParamsUpdate};
use dsp::{Bands, DspChain, EngineParams, FftChain};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const OSC_META_PERIOD: Duration = Duration::from_secs(1);
const OSC_META_MIN_GAP: Duration = Duration::from_millis(50);
const WATCHDOG_TICK: Duration = Duration::from_millis(250);
const AUDIO_STALL: Duration = Duration::from_secs(1);
const AUDIO_RECOVER_EVERY: Duration = Duration::from_secs(3);
const FILTER_RETUNE_DEBOUNCE: Duration = Duration::from_millis(50);
const DROP_REBASELINE_AFTER: Duration = Duration::from_millis(250);
const STATUS_PERIOD: Duration = Duration::from_millis(200);

#[derive(Clone, Debug)]
pub struct Args {
    pub no_ws: bool,
    pub open: bool,
    pub device: Option<i64>,
}

/// Expected startup failure: printed as `audio-server: error: …` / `hint: …`.
#[derive(Debug)]
pub struct StartupError {
    pub message: String,
    pub hint: String,
}

impl StartupError {
    fn new(message: impl Into<String>, hint: impl Into<String>) -> Self {
        StartupError { message: message.into(), hint: hint.into() }
    }
}

fn bind_error(what: &str, host: &str, port: i64, cfg_key: &str, e: &std::io::Error) -> StartupError {
    if e.kind() == std::io::ErrorKind::AddrInUse {
        return StartupError::new(
            format!("{what} port {port} on {host} is already in use"),
            format!(
                "another audio-server is probably running (check `lsof -i :{port}`); \
                 stop it, or change {cfg_key} in the config"
            ),
        );
    }
    StartupError::new(
        format!("{what} server could not bind {host}:{port}: {e}"),
        format!("check ws.host / {cfg_key} in the config"),
    )
}

fn bind_listener(host: &str, port: i64, what: &str, key: &str) -> Result<std::net::TcpListener, StartupError> {
    let p = u16::try_from(port).map_err(|_| {
        bind_error(what, host, port, key, &std::io::Error::other("port out of range"))
    })?;
    let l = std::net::TcpListener::bind((host, p)).map_err(|e| bind_error(what, host, port, key, &e))?;
    l.set_nonblocking(true).map_err(|e| bind_error(what, host, port, key, &e))?;
    Ok(l)
}

/// The cfg values the DSP reads (`_build_pipeline_for_sr` / `_build_fft_pipeline`).
pub fn engine_params(cfg: &Config, sr: f64) -> EngineParams {
    let ob = |b: &config::OnsetBandCfg| OnsetParams {
        sensitivity: b.sensitivity,
        refractory_s: b.refractory_s,
        slow_tau_s: b.slow_tau_s,
        abs_floor: b.abs_floor,
    };
    EngineParams {
        sr,
        blocksize: cfg.audio.blocksize,
        bands: cfg.dsp.bands(),
        filter_order: cfg.dsp.filter_order,
        tau: cfg.dsp.tau.arr(),
        tau_attack: cfg.dsp.tau_attack.arr(),
        autoscale_tau_attack_s: cfg.autoscale.tau_attack_s,
        autoscale_tau_release_s: cfg.autoscale.tau_release_s,
        noise_floor: cfg.autoscale.noise_floor,
        strength: cfg.autoscale.strength,
        fft_n_bins: cfg.fft.n_bins,
        fft_window_size: cfg.fft.window_size,
        fft_hop: cfg.fft.hop,
        fft_f_min: cfg.fft.f_min,
        fft_db_floor: cfg.fft.db_floor,
        fft_db_ceiling: cfg.fft.db_ceiling,
        fft_peak_smear_oct: cfg.fft.peak_smear_oct,
        fft_tilt_db_per_oct: cfg.fft.tilt_db_per_oct,
        onset: [ob(&cfg.onset.low), ob(&cfg.onset.mid), ob(&cfg.onset.high)],
    }
}

/// WS broadcaster + client registry (`WSServer`).
pub struct WsHub {
    pub clients: Vec<Arc<Client>>,
    pub snapshot_hz: i64,
    meta_dirty: bool,
    t0: Instant,
    bc_next: Option<Instant>,
    status_next: Option<Instant>,
    feat_seq: u64,
    fft_seq: u64,
    onset_counts: [u64; 3],
}

impl WsHub {
    fn new(snapshot_hz: i64) -> Self {
        WsHub {
            clients: Vec::new(),
            snapshot_hz: snapshot_hz.clamp(15, 240),
            meta_dirty: false,
            t0: Instant::now(),
            bc_next: None,
            status_next: None,
            feat_seq: 0,
            fft_seq: 0,
            onset_counts: [0; 3],
        }
    }

    fn has_streamers(&self) -> bool {
        self.clients.iter().any(|c| c.streaming())
    }

    fn period(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.snapshot_hz.max(1) as f64)
    }

    pub fn set_snapshot_hz(&mut self, hz: i64) {
        self.snapshot_hz = hz.clamp(15, 240);
    }

    pub fn broadcast_control(&self, text: &Utf8Bytes) {
        for c in &self.clients {
            c.put_control_text(text);
        }
    }

    pub fn client(&self, id: u64) -> Option<Arc<Client>> {
        self.clients.iter().find(|c| c.id == id).cloned()
    }
}

struct Watchdog {
    last_wi: u64,
    last_change: Instant,
    next_attempt: Instant,
    next_tick: Instant,
}

pub struct App {
    pub args: Args,
    pub cfg: Config,
    pub config_path: PathBuf,
    pub config_dir: PathBuf,
    ui_root: PathBuf,
    pub shared: Arc<Shared>,
    pub workers: Workers,
    wake: WakeTargets,
    pub stream: Option<StreamHandle>,
    active_device_name: Option<String>,
    pipeline_sr: f64,
    /// Bands + order of the filter design the DSP worker runs (the
    /// reference's `filter_bank.design`: `set_order` keeps its bands, a
    /// debounced `retune` keeps its order).
    filter_installed: (Bands, usize),
    audio_stalled_since: Option<Instant>,
    drops_base: [u64; 3],
    pub persister: Persister,
    pub ws: Option<WsHub>,
    osc_meta_dirty: bool,
    osc_meta_last: Instant,
    retune_at: Option<Instant>,
    rebaseline_at: Option<Instant>,
    wd: Watchdog,
    /// A device refresh / probe owns the input (the reference's
    /// `device_switch_lock` held across an await).
    pub switch_busy: bool,
    pub events_tx: mpsc::UnboundedSender<Event>,
    ws_listener: Option<std::net::TcpListener>,
    http_listener: Option<std::net::TcpListener>,
}

impl App {
    /// `App.start`: device → stream (opened) → DSP for its sr (bands
    /// clamped) → workers → `/audio/meta` → persister → HTTP bind → stream
    /// started last → WS bind → browser.
    pub fn start(
        args: Args,
        mut cfg: Config,
        config_path: PathBuf,
        ui_root: PathBuf,
        events_tx: mpsc::UnboundedSender<Event>,
    ) -> Result<App, StartupError> {
        let config_dir = config_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        let bs = cfg.audio.blocksize;
        let ring = SlotRing::new(ring_slots_for(bs), bs);
        let osc = OscSender::new(&cfg.osc.destinations);
        let shared = Arc::new(Shared::new(ring, osc, &cfg));
        let wake: WakeTargets = Arc::new(std::sync::OnceLock::new());

        let cli_dev = match args.device {
            Some(d) if d < 0 => {
                return Err(StartupError::new(
                    format!("could not open input device {d}: Invalid device"),
                    "plug in / enable a microphone, or pick another input with --device N \
                     (list them with `uv run python -m sounddevice`)",
                ));
            }
            d => d.map(|d| d as usize),
        };
        let device_idx = audio::resolve_initial_device(&cfg.audio.device, cli_dev);
        tracing::info!("initial device index: {device_idx:?}");
        let stream = audio::open_input_stream(device_idx, bs, cfg.audio.channels, &shared, &wake)
            .map_err(|e| {
                let which = match device_idx {
                    None => "the system default input".to_string(),
                    Some(i) => format!("input device {i}"),
                };
                StartupError::new(
                    format!("could not open {which}: {e}"),
                    "plug in / enable a microphone, or pick another input with --device N \
                     (list them with `uv run python -m sounddevice`)",
                )
            })?;
        let sr = stream.samplerate;
        let bands_clamped = clamp_bands_for_sr(&mut cfg, sr);

        let params = engine_params(&cfg, sr);
        let dsp = DspChain::new(&params);
        let fft = FftChain::new(&params).map_err(|e| StartupError::new(format!("invalid FFT geometry: {e}"), ""))?;
        let mut workers = Workers::spawn(shared.clone(), dsp, fft, bs)
            .map_err(|e| StartupError::new(format!("could not start worker threads: {e}"), ""))?;
        let _ = wake.set([workers.dsp_thread.clone(), workers.fft_thread.clone()]);
        workers.collect_garbage();

        let now = Instant::now();
        let mut app = App {
            persister: Persister::new(config_path.clone()),
            filter_installed: (cfg.dsp.bands(), cfg.dsp.filter_order),
            ws: None,
            args,
            config_path,
            config_dir,
            ui_root,
            shared,
            workers,
            wake,
            active_device_name: Some(stream.device_name.clone()),
            stream: Some(stream),
            pipeline_sr: sr,
            audio_stalled_since: None,
            drops_base: [0; 3],
            osc_meta_dirty: bands_clamped,
            osc_meta_last: now,
            retune_at: None,
            rebaseline_at: None,
            wd: Watchdog { last_wi: 0, last_change: now, next_attempt: now, next_tick: now + WATCHDOG_TICK },
            switch_busy: false,
            events_tx,
            ws_listener: None,
            http_listener: None,
            cfg,
        };
        app.send_osc_meta();
        app.persister.attach();
        if bands_clamped {
            app.persister.request(true);
        }

        let ws_on = app.cfg.ws.enabled && !app.args.no_ws;
        if ws_on {
            app.ws = Some(WsHub::new(app.cfg.ws.snapshot_hz));
            if app.ui_root.exists() {
                app.http_listener = Some(bind_listener(
                    &app.cfg.ws.host,
                    app.cfg.ws.http_port,
                    "UI HTTP",
                    "ws.http_port",
                )?);
            } else {
                tracing::warn!("ui directory not found at {}; static http server skipped", app.ui_root.display());
            }
        }

        if let Some(s) = &app.stream
            && let Err(e) = s.start()
        {
            return Err(StartupError::new(
                format!("could not start audio input device {}: {e}", s.device),
                "the device may be busy or unplugged; pick another with --device N",
            ));
        }

        if ws_on {
            app.ws_listener = Some(bind_listener(&app.cfg.ws.host, app.cfg.ws.port, "WebSocket", "ws.port")?);
        }

        if app.args.open {
            if app.ws.is_some() && app.http_listener.is_some() {
                let mut url = format!("http://{}:{}/index.html", app.cfg.ws.host, app.cfg.ws.http_port);
                if app.cfg.ws.port != 8765 {
                    url.push_str(&format!("?ws_port={}", app.cfg.ws.port));
                }
                if let Err(e) = webbrowser::open(&url) {
                    tracing::warn!("failed to open browser: {e}");
                }
            } else {
                tracing::info!("--open ignored: WS or HTTP server disabled");
            }
        }
        let now = Instant::now();
        app.osc_meta_last = now;
        app.wd = Watchdog {
            last_wi: app.shared.ring.write_idx(),
            last_change: now,
            next_attempt: now,
            next_tick: now + WATCHDOG_TICK,
        };
        Ok(app)
    }

    /// Hand the bound listeners to the axum servers (spawned on the runtime).
    pub fn spawn_servers(&mut self, ws_state: crate::ws::WsState) -> Result<(), StartupError> {
        if let Some(l) = self.http_listener.take() {
            let l = tokio::net::TcpListener::from_std(l)
                .map_err(|e| bind_error("UI HTTP", &self.cfg.ws.host, self.cfg.ws.http_port, "ws.http_port", &e))?;
            let router = crate::http::router(&self.ui_root);
            tracing::info!(
                "static http server listening on http://{}:{} (serving {})",
                self.cfg.ws.host,
                self.cfg.ws.http_port,
                self.ui_root.display()
            );
            tokio::spawn(async move {
                let _ = axum::serve(l, router).await;
            });
        }
        if let Some(l) = self.ws_listener.take() {
            let l = tokio::net::TcpListener::from_std(l)
                .map_err(|e| bind_error("WebSocket", &self.cfg.ws.host, self.cfg.ws.port, "ws.port", &e))?;
            tracing::info!("ws server listening on ws://{}:{}", self.cfg.ws.host, self.cfg.ws.port);
            let router = crate::ws::router(ws_state);
            tokio::spawn(async move {
                let _ = axum::serve(l, router).await;
            });
        }
        Ok(())
    }

    // ----------------------------------------------------------- the loop

    pub async fn run(mut self, mut rx: mpsc::UnboundedReceiver<Event>) {
        #[cfg(unix)]
        let (mut sigint, mut sigterm) = {
            use tokio::signal::unix::{SignalKind, signal};
            (
                signal(SignalKind::interrupt()).expect("SIGINT handler"),
                signal(SignalKind::terminate()).expect("SIGTERM handler"),
            )
        };
        loop {
            let next = self.next_deadline();
            #[cfg(unix)]
            tokio::select! {
                biased;
                _ = sigint.recv() => break,
                _ = sigterm.recv() => break,
                ev = rx.recv() => if let Some(ev) = ev { self.on_event(ev) },
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(next)) => {},
            }
            #[cfg(not(unix))]
            tokio::select! {
                biased;
                _ = tokio::signal::ctrl_c() => break,
                ev = rx.recv() => if let Some(ev) = ev { self.on_event(ev) },
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(next)) => {},
            }
            self.run_timers(Instant::now());
            self.workers.collect_garbage();
        }
        self.shutdown();
    }

    fn next_deadline(&self) -> Instant {
        let mut t = self.wd.next_tick;
        let mut consider = |x: Option<Instant>| {
            if let Some(x) = x
                && x < t
            {
                t = x;
            }
        };
        consider(Some(self.osc_meta_due()));
        consider(self.persister.deadline());
        consider(self.retune_at);
        consider(self.rebaseline_at);
        if let Some(ws) = &self.ws {
            consider(ws.bc_next);
            consider(ws.status_next);
        }
        t
    }

    fn osc_meta_due(&self) -> Instant {
        if self.osc_meta_dirty {
            self.osc_meta_last + OSC_META_MIN_GAP
        } else {
            self.osc_meta_last + OSC_META_PERIOD
        }
    }

    fn run_timers(&mut self, now: Instant) {
        if self.ws.as_ref().and_then(|w| w.bc_next).is_some_and(|t| t <= now) {
            self.broadcast_tick();
            self.reschedule_broadcast();
        }
        if self.ws.as_ref().and_then(|w| w.status_next).is_some_and(|t| t <= now) {
            self.status_tick(now);
        }
        if self.wd.next_tick <= now {
            self.watchdog_tick(now);
            self.wd.next_tick = Instant::now() + WATCHDOG_TICK;
        }
        if self.osc_meta_due() <= now {
            self.osc_meta_dirty = false;
            self.osc_meta_last = now;
            self.send_osc_meta();
        }
        if self.persister.deadline().is_some_and(|t| t <= now) {
            self.persister.fire(&self.cfg);
        }
        if self.retune_at.is_some_and(|t| t <= now) {
            self.retune_at = None;
            let bands = self.cfg.dsp.bands();
            let order = self.filter_installed.1;
            self.install_filter(bands, order);
        }
        if self.rebaseline_at.is_some_and(|t| t <= now) {
            self.rebaseline_at = None;
            self.capture_drop_baselines();
        }
    }

    fn on_event(&mut self, ev: Event) {
        match ev {
            Event::Connect(c) => {
                let Some(ws) = self.ws.as_mut() else { return };
                ws.clients.push(c.clone());
                tracing::info!("ws client connected (n={})", ws.clients.len());
                self.ws_wake();
                self.greet(&c);
            }
            Event::Disconnect(id) => {
                if let Some(ws) = self.ws.as_mut() {
                    ws.clients.retain(|c| c.id != id);
                    tracing::info!("ws client disconnected (n={})", ws.clients.len());
                }
            }
            Event::Message(id, text) => {
                let Some(client) = self.ws.as_ref().and_then(|w| w.client(id)) else { return };
                self.on_message(&client, &text);
            }
            Event::ProbeDone(id, items) => {
                self.switch_busy = false;
                if let Some(c) = self.ws.as_ref().and_then(|w| w.client(id)) {
                    c.put_control_text(&to_text(&json!({"type": "devices", "items": items})));
                }
            }
        }
    }

    fn greet(&mut self, c: &Client) {
        c.put_control_text(&to_text(&self.meta_message()));
        let devices = self.list_devices_json(&audio::list_input_devices());
        c.put_control_text(&to_text(&json!({"type": "devices", "items": devices})));
        c.put_control_text(&to_text(&json!({"type": "presets", "items": self.list_presets()})));
        c.put_status(&to_text(&self.server_status()));
    }

    /// `WSServer._on_message`: parse, per-connection `set_stream`, dispatch.
    fn on_message(&mut self, client: &Arc<Client>, raw: &str) {
        let msg: Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(_) => {
                client.put_control_text(&to_text(&json!({"type": "error", "reason": "invalid JSON"})));
                return;
            }
        };
        if !msg.is_object() || msg.get("type").is_none() {
            client.put_control_text(&to_text(&json!({"type": "error", "reason": "missing 'type'"})));
            return;
        }
        if msg.get("type") == Some(&Value::from("set_stream")) {
            let on = crate::py::get_bool(&msg, "enabled", true);
            client.set_streaming(on);
            if on {
                client.put_control_text(&to_text(&self.meta_message()));
                self.ws_wake();
            }
            return;
        }
        let (targeted, broadcasts) = self.dispatch(&msg, client.id);
        for r in targeted {
            client.put_control_text(&to_text(&r));
        }
        if let Some(ws) = &self.ws {
            for b in broadcasts {
                ws.broadcast_control(&to_text(&b));
            }
        }
    }

    // ------------------------------------------------------------- WS hub

    /// Meta changed: one coalesced broadcast on the next tick.
    pub fn mark_meta_dirty(&mut self) {
        if let Some(ws) = self.ws.as_mut() {
            ws.meta_dirty = true;
        }
        self.ws_wake();
    }

    /// Wake the broadcast loop if it idles (re-baselining onset counters so
    /// onsets that fired while idle don't flash) and start the status tick.
    fn ws_wake(&mut self) {
        let Some(ws) = self.ws.as_mut() else { return };
        let streamers = ws.has_streamers();
        if ws.bc_next.is_none() && (ws.meta_dirty || streamers) {
            let counts = self.workers.feat_out.read().onset_counts;
            let ws = self.ws.as_mut().unwrap_or_else(|| unreachable!());
            ws.onset_counts = counts;
            ws.bc_next = Some(Instant::now() + ws.period());
        }
        let ws = self.ws.as_mut().unwrap_or_else(|| unreachable!());
        if streamers && ws.status_next.is_none() {
            ws.status_next = Some(Instant::now() + STATUS_PERIOD);
        }
    }

    fn reschedule_broadcast(&mut self) {
        let Some(ws) = self.ws.as_mut() else { return };
        let Some(prev) = ws.bc_next else { return };
        if !ws.meta_dirty && !ws.has_streamers() {
            ws.bc_next = None; // idle until a client streams or meta changes
            return;
        }
        let period = ws.period();
        let now = Instant::now();
        let mut next = prev + period;
        // More than one period late (loop stalled): resync, don't burst.
        if next + period < now {
            next = now;
        }
        ws.bc_next = Some(next);
    }

    /// `WSServer._broadcast_tick`.
    fn broadcast_tick(&mut self) {
        let t_start = crate::shared::now_ns();
        let meta = match self.ws.as_mut() {
            Some(ws) if ws.meta_dirty => {
                ws.meta_dirty = false;
                !ws.clients.is_empty()
            }
            Some(_) => false,
            None => return,
        };
        if meta {
            let text = to_text(&self.meta_message());
            if let Some(ws) = &self.ws {
                ws.broadcast_control(&text);
            }
        }
        let streamers: Vec<Arc<Client>> = match &self.ws {
            Some(ws) => ws.clients.iter().filter(|c| c.streaming()).cloned().collect(),
            None => return,
        };
        if streamers.is_empty() {
            return;
        }
        let snap = *self.workers.feat_out.read();
        let gain = self.cfg.autoscale.master_gain;
        let ws = self.ws.as_mut().unwrap_or_else(|| unreachable!());
        if snap.seq != ws.feat_seq {
            ws.feat_seq = snap.seq;
            let last = ws.onset_counts;
            let fired = |i: usize| (snap.onset_counts[i] != last[i]) as i64;
            let (o_lo, o_md, o_hi) = (fired(0), fired(1), fired(2));
            ws.onset_counts = snap.onset_counts;
            let msg = json!({
                "type": "snapshot",
                "seq": snap.seq,
                "low": snap.scaled[0] * gain,
                "mid": snap.scaled[1] * gain,
                "high": snap.scaled[2] * gain,
                "low_raw": snap.raw[0],
                "mid_raw": snap.raw[1],
                "high_raw": snap.raw[2],
                "low_onset": o_lo,
                "mid_onset": o_md,
                "high_onset": o_hi,
                "bpm": snap.bpm,
                "t": ws.t0.elapsed().as_secs_f64() * 1000.0,
            });
            let text = to_text(&msg);
            for c in &streamers {
                c.put_stream(Message::Text(text.clone()));
            }
        }
        if self.shared.fft_enabled.load(Ordering::Relaxed) {
            let raw = self.cfg.fft.send_raw_db;
            let frame = self.workers.fft_out.read();
            let ws = self.ws.as_mut().unwrap_or_else(|| unreachable!());
            if frame.seq != ws.fft_seq && frame.seq != 0 {
                ws.fft_seq = frame.seq;
                let n = frame.n;
                let (bins, g) = if raw { (&frame.raw_db[..n], 1.0) } else { (&frame.processed[..n], gain) };
                let payload = bytes_from(crate::ws::encode_fft_binary(bins, g));
                for c in &streamers {
                    c.put_stream(Message::Binary(payload.clone()));
                }
            }
        }
        self.shared.perf.ws.record(crate::shared::now_ns() - t_start);
    }

    fn status_tick(&mut self, now: Instant) {
        let streamers: Vec<Arc<Client>> = match &self.ws {
            Some(ws) => ws.clients.iter().filter(|c| c.streaming()).cloned().collect(),
            None => return,
        };
        let ws = self.ws.as_mut().unwrap_or_else(|| unreachable!());
        if streamers.is_empty() {
            ws.status_next = None;
            return;
        }
        let prev = ws.status_next.unwrap_or(now);
        let mut next = prev + STATUS_PERIOD;
        if next <= now {
            next = now + STATUS_PERIOD;
        }
        ws.status_next = Some(next);
        let text = to_text(&self.server_status());
        for c in &streamers {
            c.put_status(&text);
        }
    }

    // --------------------------------------------------- meta + status

    pub fn current_sr(&self) -> f64 {
        self.stream.as_ref().map(|s| s.samplerate).unwrap_or(48000.0)
    }

    fn bands_meta(&self) -> Value {
        let d = &self.cfg.dsp;
        json!({
            "low": {"lo_hz": d.low.lo_hz, "hi_hz": d.low.hi_hz},
            "mid": {"lo_hz": d.mid.lo_hz, "hi_hz": d.mid.hi_hz},
            "high": {"lo_hz": d.high.lo_hz, "hi_hz": d.high.hi_hz},
        })
    }

    /// `snapshot_meta` with `type` first (26 keys after it).
    pub fn meta_message(&self) -> Value {
        let c = &self.cfg;
        let device = match &self.stream {
            Some(s) => json!({"index": s.device, "name": s.device_name}),
            None => json!({"index": null, "name": null}),
        };
        let ob = |b: &config::OnsetBandCfg| {
            json!({"sensitivity": b.sensitivity, "refractory_s": b.refractory_s,
                   "slow_tau_s": b.slow_tau_s, "abs_floor": b.abs_floor})
        };
        json!({
            "type": "meta",
            "sr": self.current_sr() as i64,
            "blocksize": c.audio.blocksize,
            "n_fft_bins": c.fft.n_bins,
            "bands": self.bands_meta(),
            "fft_enabled": self.shared.fft_enabled.load(Ordering::Relaxed),
            "fft_db_floor": c.fft.db_floor,
            "fft_db_ceiling": c.fft.db_ceiling,
            "fft_f_min": c.fft.f_min,
            "fft_window_size": c.fft.window_size,
            "fft_send_raw_db": c.fft.send_raw_db,
            "fft_peak_smear_oct": c.fft.peak_smear_oct,
            "fft_tilt_db_per_oct": c.fft.tilt_db_per_oct,
            "tau": c.dsp.tau,
            "tau_attack": c.dsp.tau_attack,
            "filter_order": c.dsp.filter_order,
            "autoscale": {
                "tau_attack_s": c.autoscale.tau_attack_s,
                "tau_release_s": c.autoscale.tau_release_s,
                "noise_floor": c.autoscale.noise_floor,
                "strength": c.autoscale.strength,
                "master_gain": c.autoscale.master_gain,
            },
            "onset": {"low": ob(&c.onset.low), "mid": ob(&c.onset.mid), "high": ob(&c.onset.high)},
            "ws_snapshot_hz": self.ws.as_ref().map(|w| w.snapshot_hz).unwrap_or(c.ws.snapshot_hz),
            "ui_peak_decay_per_s": c.ui.peak_decay_per_s,
            "ui_show_onsets": c.ui.show_onsets,
            "ui_fft_view": c.ui.fft_view,
            "ui_fft3d_style": c.ui.fft3d_style,
            "ui_palette": c.ui.palette,
            "ui_layout": c.ui.layout,
            "device": device,
        })
    }

    /// `snapshot_server_status` (the UI Performance panel's schema).
    pub fn server_status(&self) -> Value {
        let sr = self.current_sr();
        let block_ms = self.cfg.audio.blocksize as f64 / sr * 1000.0;
        let hop_ms = self.cfg.fft.hop as f64 / sr * 1000.0;
        let ws_hz = self.ws.as_ref().map(|w| w.snapshot_hz).unwrap_or(self.cfg.ws.snapshot_hz);
        let ws_ms = 1000.0 / ws_hz.max(1) as f64;
        let p = &self.shared.perf;
        let load = |avg: f64, period: f64| if period <= 0.0 { 0.0 } else { avg / period * 100.0 };
        let stage = |r: &crate::shared::PerfRing, period: f64| {
            let (a, p95) = r.stats();
            json!({"avg_ms": a, "p95_ms": p95, "load_pct": load(a, period)})
        };
        let fft_on = self.shared.fft_enabled.load(Ordering::Relaxed);
        let mut fft_e2e = stage(&p.fft_e2e, hop_ms);
        fft_e2e["enabled"] = Value::from(fft_on && self.cfg.osc.send_fft);
        let mut fft = stage(&p.fft, hop_ms);
        fft["enabled"] = Value::from(fft_on);
        let sub = |v: &std::sync::atomic::AtomicU64, base: u64| v.load(Ordering::Relaxed).saturating_sub(base);
        json!({
            "type": "server_status",
            "audio_stalled": self.audio_stalled_since.is_some(),
            "cb_overruns": sub(&self.shared.cb_overruns, self.drops_base[0]),
            "dsp_drops": sub(&self.shared.dsp_drops, self.drops_base[1]),
            "fft_drops": sub(&self.shared.fft_drops, self.drops_base[2]),
            "perf": {
                "block_period_ms": block_ms,
                "hop_period_ms": hop_ms,
                "ws_period_ms": ws_ms,
                "lmh_e2e": stage(&p.lmh_e2e, block_ms),
                "fft_e2e": fft_e2e,
                "cb": stage(&p.cb, block_ms),
                "dsp": stage(&p.dsp, block_ms),
                "fft": fft,
                "ws": stage(&p.ws, ws_ms),
            },
        })
    }

    pub fn list_devices_json(&self, items: &[audio::DeviceInfo]) -> Vec<Value> {
        items.iter().map(|d| d.to_json()).collect()
    }

    pub fn list_presets(&self) -> Vec<Value> {
        let _ = std::fs::create_dir_all(&self.config_dir);
        let mut names: Vec<(String, String)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.config_dir) {
            let mut paths: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
            paths.sort();
            for p in paths {
                if p.extension().and_then(|e| e.to_str()) != Some("yaml") {
                    continue;
                }
                let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else { continue };
                if stem.to_lowercase() == "main" {
                    continue;
                }
                let Ok(meta) = std::fs::metadata(&p) else { continue };
                if !meta.is_file() {
                    continue;
                }
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as libc::time_t)
                    .unwrap_or(0);
                names.push((stem.to_string(), config::local_strftime("%Y-%m-%d %H:%M", mtime)));
            }
        }
        names.sort_by(|a, b| b.1.cmp(&a.1)); // stable: name order within a minute
        names.into_iter().map(|(n, t)| json!({"name": n, "saved_at": t})).collect()
    }

    pub fn preset_path(&self, name: &str) -> PathBuf {
        let _ = std::fs::create_dir_all(&self.config_dir);
        self.config_dir.join(format!("{name}.yaml"))
    }

    // ------------------------------------------------- OSC meta + drops

    pub fn mark_osc_meta_dirty(&mut self) {
        self.osc_meta_dirty = true;
    }

    fn send_osc_meta(&self) {
        let Some(s) = &self.stream else { return };
        self.shared.osc.send_meta(
            s.samplerate as i32,
            self.cfg.audio.blocksize as i32,
            self.cfg.fft.n_bins as i32,
            self.cfg.dsp.bands(),
        );
    }

    fn capture_drop_baselines(&mut self) {
        let s = &self.shared;
        self.drops_base = [
            s.cb_overruns.load(Ordering::Relaxed),
            s.dsp_drops.load(Ordering::Relaxed),
            s.fft_drops.load(Ordering::Relaxed),
        ];
    }

    fn after_stream_change(&mut self) {
        self.send_osc_meta();
        self.mark_meta_dirty();
        self.capture_drop_baselines();
        self.rebaseline_at = Some(Instant::now() + DROP_REBASELINE_AFTER);
    }

    // ------------------------------------- pipeline bus (apply_* in Python)

    fn install_filter(&mut self, bands: Bands, order: usize) {
        let d = FilterDesign::new(self.pipeline_sr, bands, order);
        self.filter_installed = (bands, order);
        self.workers.dsp_tx.send(DspCmd::InstallFilter(d));
    }

    /// 50 ms server-side debounce: reads cfg at fire time.
    pub fn schedule_filter_retune(&mut self) {
        self.retune_at = Some(Instant::now() + FILTER_RETUNE_DEBOUNCE);
    }

    pub fn apply_bands(&mut self) {
        let b = self.cfg.dsp.bands();
        self.workers.dsp_tx.send(DspCmd::SetBands(b));
        self.workers.fft_tx.send(FftCmd::UpdateBands(b));
        self.mark_osc_meta_dirty();
    }

    pub fn apply_smoothing(&mut self) {
        let (t, ta) = (self.cfg.dsp.tau.arr(), self.cfg.dsp.tau_attack.arr());
        self.workers.dsp_tx.send(DspCmd::SetTau(t, ta));
        self.workers.fft_tx.send(FftCmd::UpdateSmoothing(t, ta));
    }

    pub fn apply_autoscale(&mut self, ok: &crate::validate::AutoscaleOk) {
        let a = &mut self.cfg.autoscale;
        if let Some(v) = ok.tau_attack_s {
            a.tau_attack_s = v;
        }
        if let Some(v) = ok.tau_release_s {
            a.tau_release_s = v;
        }
        if let Some(v) = ok.noise_floor {
            a.noise_floor = v;
        }
        if let Some(v) = ok.strength {
            a.strength = v;
        }
        if let Some(v) = ok.master_gain {
            a.master_gain = v;
            self.shared.master_gain.store(v);
        }
        let (atk, rel) = (a.tau_attack_s, a.tau_release_s);
        if ok.tau_attack_s.is_some() || ok.tau_release_s.is_some() {
            self.workers.dsp_tx.send(DspCmd::SetAutoscaleTaus(atk, rel));
        }
        if let Some(v) = ok.noise_floor {
            self.workers.dsp_tx.send(DspCmd::SetNoiseFloor(v));
        }
        if let Some(v) = ok.strength {
            self.workers.dsp_tx.send(DspCmd::SetStrength(v));
        }
        let u = AutoscaleUpdate {
            tau_attack_s: ok.tau_attack_s,
            tau_release_s: ok.tau_release_s,
            noise_floor: ok.noise_floor,
            strength: ok.strength,
        };
        if u.tau_attack_s.is_some() || u.tau_release_s.is_some() || u.noise_floor.is_some() || u.strength.is_some() {
            self.workers.fft_tx.send(FftCmd::UpdateAutoscale(u));
        }
    }

    pub fn apply_onset(&mut self, band: usize, ok: &crate::validate::OnsetOk) {
        let b = self.cfg.onset.band_mut(band);
        if let Some(v) = ok.sensitivity {
            b.sensitivity = v;
        }
        if let Some(v) = ok.refractory_s {
            b.refractory_s = v;
        }
        if let Some(v) = ok.slow_tau_s {
            b.slow_tau_s = v;
        }
        if let Some(v) = ok.abs_floor {
            b.abs_floor = v;
        }
        if !ok.is_empty() {
            self.workers.dsp_tx.send(DspCmd::SetOnset(
                band,
                OnsetParamsUpdate {
                    sensitivity: ok.sensitivity,
                    refractory_s: ok.refractory_s,
                    slow_tau_s: ok.slow_tau_s,
                    abs_floor: ok.abs_floor,
                },
            ));
        }
    }

    /// Any FFT reconfigure: a fresh chain from the current cfg (allocates
    /// here, on the control thread). Resets all FFT post-processing state.
    fn replace_fft_chain(&mut self) -> Result<(), String> {
        let chain = FftChain::new(&engine_params(&self.cfg, self.pipeline_sr))?;
        self.workers.fft_tx.send(FftCmd::ReplaceChain(Box::new(chain)));
        Ok(())
    }

    pub fn apply_fft_n_bins(&mut self, n: usize) -> Result<(), String> {
        self.cfg.fft.n_bins = n;
        self.replace_fft_chain()?;
        self.workers.dsp_tx.send(DspCmd::SetFftGeometry(Some(n), None));
        self.mark_osc_meta_dirty();
        Ok(())
    }

    /// Validates first (a bad window/hop would leave a zero-block geometry).
    pub fn apply_fft_window(&mut self, ws: Option<&Value>, hop: Option<&Value>, f_min: Option<&Value>) -> Result<(), String> {
        let ok = crate::validate::validate_fft_window(ws, hop, f_min, self.cfg.audio.blocksize)?;
        let prev = self.cfg.fft.clone();
        if let Some(w) = ok.window_size {
            self.cfg.fft.window_size = w;
        }
        if let Some(h) = ok.hop {
            self.cfg.fft.hop = h;
        }
        if let Some(f) = ok.f_min {
            self.cfg.fft.f_min = f;
        }
        if let Err(e) = self.replace_fft_chain() {
            self.cfg.fft = prev;
            return Err(e);
        }
        if let Some(w) = ok.window_size {
            self.workers.dsp_tx.send(DspCmd::SetNFftWindow(w));
        }
        if let Some(f) = ok.f_min {
            self.workers.dsp_tx.send(DspCmd::SetFftGeometry(None, Some(f)));
        }
        Ok(())
    }

    pub fn apply_fft_tilt(&mut self, v: f64) {
        self.cfg.fft.tilt_db_per_oct = v;
        self.workers.dsp_tx.send(DspCmd::SetTilt(v));
        self.workers.fft_tx.send(FftCmd::UpdateTilt(v));
    }

    pub fn apply_filter_order(&mut self, n: usize) {
        self.cfg.dsp.filter_order = n;
        let bands = self.filter_installed.0;
        self.install_filter(bands, n);
    }

    pub fn apply_fft_peak_smear(&mut self, v: f64) {
        self.cfg.fft.peak_smear_oct = v;
        let (n, f_min, sr) = (self.cfg.fft.n_bins, self.cfg.fft.f_min, self.pipeline_sr);
        let (smear, sigma) = Smear::build(n, f_min, sr, v);
        self.workers.fft_tx.send(FftCmd::InstallSmear(v, smear, sigma, n));
    }

    pub fn set_fft_enabled(&mut self, enabled: bool) {
        if enabled {
            if !self.shared.fft_enabled.load(Ordering::Relaxed) {
                // Fresh post-processor state, and skip the disabled gap
                // without counting it as drops.
                self.workers.fft_tx.send(FftCmd::ResetPost);
                self.workers.fft_tx.send(FftCmd::Resync);
            }
            self.shared.fft_enabled.store(true, Ordering::Release);
            self.workers.fft_thread.unpark();
        } else {
            self.shared.fft_enabled.store(false, Ordering::Release);
        }
        self.cfg.fft.enabled = enabled;
        self.cfg.osc.send_fft = enabled;
        self.shared.send_fft.store(enabled, Ordering::Relaxed);
    }

    pub fn set_send_raw_db(&mut self, on: bool) {
        self.cfg.fft.send_raw_db = on;
        self.shared.send_raw_db.store(on, Ordering::Relaxed);
    }

    // ------------------------------------------------ devices + streams

    fn teardown_stream(&mut self) {
        if let Some(s) = self.stream.take() {
            s.close();
        }
    }

    /// Rewind ring + worker read pointers + perf rings for a fresh stream.
    fn reset_stream_state(&mut self) {
        self.shared.ring.reset();
        self.workers.dsp_tx.send(DspCmd::ResetReadPtr);
        self.workers.fft_tx.send(FftCmd::ResetReadPtr);
        self.shared.perf.reset_all();
    }

    /// (Re)build every sr-dependent DSP object and swap it into the workers.
    fn build_pipeline_for_sr(&mut self, sr: f64) -> Result<(), String> {
        let clamped = clamp_bands_for_sr(&mut self.cfg, sr);
        if clamped {
            self.mark_meta_dirty();
            self.mark_osc_meta_dirty();
            self.persister.request(true);
        }
        self.pipeline_sr = sr;
        let params = engine_params(&self.cfg, sr);
        let fft = FftChain::new(&params)?;
        let chain = DspChain::new(&params);
        let core = DspCore {
            filter_bank: chain.filter_bank,
            smoother: chain.smoother,
            auto_scaler: chain.auto_scaler,
        };
        self.filter_installed = (self.cfg.dsp.bands(), self.cfg.dsp.filter_order);
        self.workers.dsp_tx.send(DspCmd::ReplaceCore(Box::new(core), sr));
        self.workers.fft_tx.send(FftCmd::ReplaceChain(Box::new(fft)));
        Ok(())
    }

    /// `_install_stream`: open → rebuild for its sr (unless `keep_state` and
    /// the sr is unchanged) → start. On failure nothing stays open.
    fn install_stream(&mut self, idx: Option<usize>, keep_state: bool) -> Result<(), String> {
        let stream = audio::open_input_stream(
            idx,
            self.cfg.audio.blocksize,
            self.cfg.audio.channels,
            &self.shared,
            &self.wake,
        )?;
        if !(keep_state && stream.samplerate == self.pipeline_sr)
            && let Err(e) = self.build_pipeline_for_sr(stream.samplerate)
        {
            stream.close();
            return Err(e);
        }
        if let Err(e) = stream.start() {
            stream.close();
            return Err(e);
        }
        self.active_device_name = Some(stream.device_name.clone());
        self.stream = Some(stream);
        Ok(())
    }

    /// `_reopen_input`: same device by name, else the system default.
    fn reopen_input(&mut self) -> Result<(), String> {
        let name = self.active_device_name.clone();
        self.teardown_stream();
        self.reset_stream_state();
        if let Some(idx) = audio::find_input_by_name(name.as_deref()) {
            match self.install_stream(Some(idx), true) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!("reopening {name:?} failed ({e}); trying the system default");
                    self.reset_stream_state();
                }
            }
        }
        let Some(default) = audio::default_input() else {
            return Err("no audio input device available".into());
        };
        self.install_stream(Some(default), false)
    }

    /// The UI's refresh / probe: reopen the input, then list (± probe).
    /// Returns the list now, or None when a probe was started (its result
    /// arrives as `Event::ProbeDone`).
    pub fn refresh_devices(&mut self, probe: bool, client_id: u64) -> Option<Vec<Value>> {
        if let Err(e) = self.reopen_input() {
            tracing::warn!("input reopen after device refresh failed ({e}); the audio watchdog will keep retrying");
        }
        self.after_stream_change();
        let items = audio::list_input_devices();
        if !probe {
            return Some(self.list_devices_json(&items));
        }
        self.switch_busy = true;
        let tx = self.events_tx.clone();
        tokio::task::spawn_blocking(move || {
            let results = audio::signal_active_probe(&items, 0.2);
            let at = config::local_strftime("%H:%M:%S", config::now_time_t());
            let out: Vec<Value> = items
                .iter()
                .map(|d| {
                    let mut j = d.to_json();
                    let hit = results.iter().any(|(i, s)| *i == d.index && *s);
                    j["probed_signal"] = Value::from(hit);
                    j["probed_at"] = Value::from(at.clone());
                    j
                })
                .collect();
            let _ = tx.send(Event::ProbeDone(client_id, out));
        });
        None
    }

    /// Transactional hot-switch; on failure the previous device is restored
    /// and the error (a ValueError in the reference) is returned.
    pub fn hot_switch_device(&mut self, new_idx: usize) -> Result<(), String> {
        if !audio::list_input_devices().iter().any(|d| d.index == new_idx) {
            return Err(format!("device {new_idx} is not an available input device"));
        }
        let old_idx = match &self.stream {
            Some(s) => Some(s.device),
            None => self.cfg.audio.device.index.and_then(|i| usize::try_from(i).ok()),
        };
        let old_bands = self.cfg.dsp.bands();
        self.teardown_stream();
        self.reset_stream_state();
        if let Err(e) = self.install_stream(Some(new_idx), false) {
            tracing::warn!("switch to device {new_idx} failed ({e}); rolling back to device {old_idx:?}");
            self.cfg.dsp.set_bands(old_bands);
            self.reset_stream_state();
            if let Err(e2) = self.install_stream(old_idx, false) {
                tracing::error!(
                    "rollback to device {old_idx:?} also failed ({e2}); audio input is stopped — pick a device in the UI"
                );
            }
            self.after_stream_change();
            return Err(format!("could not switch to device {new_idx}: {e}"));
        }
        let info = audio::device_info(new_idx);
        self.cfg.audio.device.index = Some(new_idx as i64);
        self.cfg.audio.device.name = info.map(|d| d.name);
        self.after_stream_change();
        Ok(())
    }

    /// `_audio_watchdog_loop` body: no new block for 1 s → one warning,
    /// OSC silence, `audio_stalled`, then a reopen every 3 s.
    fn watchdog_tick(&mut self, now: Instant) {
        if self.switch_busy {
            self.wd.last_change = now;
            return;
        }
        let wi = self.shared.ring.write_idx();
        if wi != self.wd.last_wi {
            self.wd.last_wi = wi;
            self.wd.last_change = now;
            if let Some(since) = self.audio_stalled_since.take() {
                tracing::info!(
                    "audio input recovered on {:?} after {:.1} s",
                    self.active_device_name,
                    now.duration_since(since).as_secs_f64()
                );
            }
            return;
        }
        if now.duration_since(self.wd.last_change) < AUDIO_STALL {
            return;
        }
        if self.audio_stalled_since.is_none() {
            self.audio_stalled_since = Some(self.wd.last_change);
            tracing::warn!(
                "audio input stalled: no audio from {:?} for {:.1} s (unplugged?); trying to reopen every {:.0} s",
                self.active_device_name,
                now.duration_since(self.wd.last_change).as_secs_f64(),
                AUDIO_RECOVER_EVERY.as_secs_f64()
            );
            self.shared.osc.send_silence();
            self.wd.next_attempt = now;
        }
        if now >= self.wd.next_attempt {
            self.wd.next_attempt = now + AUDIO_RECOVER_EVERY;
            if let Err(e) = self.reopen_input() {
                tracing::debug!("audio input reopen failed: {e}");
            }
            self.after_stream_change();
            self.wd.last_wi = self.shared.ring.write_idx();
            self.wd.last_change = Instant::now();
        }
    }

    // ------------------------------------------------------- shutdown

    fn shutdown(&mut self) {
        tracing::info!("shutdown: stopping audio stream");
        self.teardown_stream();
        tracing::info!("shutdown: stopping workers");
        self.workers.stop(&self.shared);
        if let Some(ws) = &self.ws {
            for c in &ws.clients {
                c.put_control(Message::Close(None));
            }
        }
        self.persister.flush_now(&self.cfg);
    }
}

/// `_clamp_cfg_bands_for_sr`: clamp every band to the Nyquist-safe range
/// (hi ≤ 0.45·sr, 1 Hz ≤ lo ≤ 0.9·hi) and write the result back to cfg.
pub fn clamp_bands_for_sr(cfg: &mut Config, sr: f64) -> bool {
    let mut changed = false;
    for (i, name) in crate::validate::BAND_NAMES.iter().enumerate() {
        let b = cfg.dsp.band_mut(i);
        let (lo, hi) = clamp_band_edges(b.lo_hz, b.hi_hz, sr);
        if lo != b.lo_hz || hi != b.hi_hz {
            tracing::warn!(
                "band {name} [{:.1}, {:.1}] Hz not representable at sr={sr:.0}; clamped to [{lo:.1}, {hi:.1}] Hz",
                b.lo_hz,
                b.hi_hz
            );
            b.lo_hz = lo;
            b.hi_hz = hi;
            changed = true;
        }
    }
    changed
}

pub fn to_text(v: &Value) -> Utf8Bytes {
    Utf8Bytes::from(serde_json::to_string(v).unwrap_or_else(|_| "{}".into()))
}

fn bytes_from(v: Vec<u8>) -> axum::body::Bytes {
    axum::body::Bytes::from(v)
}
