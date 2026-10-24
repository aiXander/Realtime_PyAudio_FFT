//! WebSocket transport (`server/io/ws_server.py`): per-client outbound
//! lanes and the axum connection tasks. All state decisions (greeting,
//! broadcast, dispatch) happen in the control loop (`app.rs`); a connection
//! task only shovels frames between its socket and the loop.
//!
//! Three outbound lanes per client, drained in this priority:
//!   1. control — FIFO, never dropped; > 1024 queued closes the client (1008);
//!   2. status  — latest-only `server_status`;
//!   3. stream  — snapshots + binary FFT, drop-oldest, max 4.
//!
//! Liveness (the reference gets these from the `websockets` library): a
//! ping every 20 s, and a client that sends nothing back (no pong, no
//! message) within 20 s of a ping is dropped; a single send stuck for 20 s
//! drops the client; an overflow close fires even while a send is stuck.
//! Inbound: at most `INBOUND_MAX` messages per client wait for the control
//! loop before the connection stops reading (the reference awaits each
//! handler before reading the next frame); a message or frame over 1 MiB
//! drops the connection (the reference's `websockets` `max_size`).
//!
//! Origin check (Rust only; the reference accepts any origin): a browser
//! upgrade is refused unless the page comes from a loopback host or the host
//! this server was reached at, so a random website can't drive the control
//! plane (save presets, switch or probe devices). Non-browser clients send
//! no `Origin` and are unaffected.

use axum::extract::ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use std::collections::VecDeque;
use crate::shared::now_ns;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Notify, mpsc};

pub const STREAM_QUEUE_MAX: usize = 4;
pub const CONTROL_QUEUE_MAX: usize = 1024;
pub const INBOUND_MAX: usize = 64;
/// The reference's `websockets` default `max_size` (2**20).
const MAX_MESSAGE_BYTES: usize = 1 << 20;
const PING_EVERY: Duration = Duration::from_secs(20);
const SEND_TIMEOUT: Duration = Duration::from_secs(20);

/// Inbound to the control loop.
pub enum Event {
    Connect(Arc<Client>),
    Message(u64, String),
    Disconnect(u64),
    /// A `list_devices {probe: true}` finished on a blocking thread.
    ProbeDone(u64, Vec<serde_json::Value>),
}

#[derive(Default)]
struct Lanes {
    control: VecDeque<Message>,
    status: Option<Message>,
    stream: VecDeque<Message>,
    closing: bool,
    close_now: bool,
    /// False while the client's tab is hidden (`set_stream {enabled:false}`).
    paused: bool,
}

pub struct Client {
    pub id: u64,
    lanes: Mutex<Lanes>,
    notify: Notify,
    /// Overflow close: interrupts a stuck send.
    kill: Notify,
    /// `now_ns` of the last frame received (any kind, pongs included).
    last_rx_ns: AtomicU64,
    /// Inbound messages queued for / being handled by the control loop.
    inflight: AtomicUsize,
    inflight_done: Notify,
}

enum Next {
    Msg(Message),
    Close,
    Empty,
}

impl Client {
    fn new(id: u64) -> Self {
        Client {
            id,
            lanes: Mutex::new(Lanes::default()),
            notify: Notify::new(),
            kill: Notify::new(),
            last_rx_ns: AtomicU64::new(now_ns()),
            inflight: AtomicUsize::new(0),
            inflight_done: Notify::new(),
        }
    }

    fn lanes(&self) -> std::sync::MutexGuard<'_, Lanes> {
        self.lanes.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn put_control(&self, msg: Message) {
        let mut l = self.lanes();
        if l.closing {
            return;
        }
        if l.control.len() >= CONTROL_QUEUE_MAX {
            l.closing = true;
            l.close_now = true;
            tracing::warn!("ws client control queue overflow ({} msgs); disconnecting", l.control.len());
            drop(l);
            self.notify.notify_one();
            self.kill.notify_one();
            return;
        }
        l.control.push_back(msg);
        drop(l);
        self.notify.notify_one();
    }

    pub fn put_control_text(&self, text: &Utf8Bytes) {
        self.put_control(Message::Text(text.clone()));
    }

    pub fn put_status(&self, text: &Utf8Bytes) {
        self.lanes().status = Some(Message::Text(text.clone()));
        self.notify.notify_one();
    }

    pub fn put_stream(&self, msg: Message) {
        let mut l = self.lanes();
        if l.stream.len() >= STREAM_QUEUE_MAX {
            l.stream.pop_front();
        }
        l.stream.push_back(msg);
        drop(l);
        self.notify.notify_one();
    }

    /// The control loop finished with one inbound message from this client.
    pub fn message_done(&self) {
        self.inflight.fetch_sub(1, Ordering::AcqRel);
        self.inflight_done.notify_one();
    }

    pub fn streaming(&self) -> bool {
        !self.lanes().paused
    }

    /// Pausing drops the queued stream frames and status.
    pub fn set_streaming(&self, on: bool) {
        let mut l = self.lanes();
        l.paused = !on;
        if !on {
            l.stream.clear();
            l.status = None;
        }
    }

    fn next(&self) -> Next {
        let mut l = self.lanes();
        if l.close_now {
            return Next::Close;
        }
        if let Some(m) = l.control.pop_front() {
            return Next::Msg(m);
        }
        if let Some(m) = l.status.take() {
            return Next::Msg(m);
        }
        match l.stream.pop_front() {
            Some(m) => Next::Msg(m),
            None => Next::Empty,
        }
    }
}

#[derive(Clone)]
pub struct WsState {
    pub tx: mpsc::UnboundedSender<Event>,
    pub next_id: Arc<AtomicU64>,
}

pub fn router(state: WsState) -> axum::Router {
    axum::Router::new().fallback(ws_handler).with_state(state)
}

async fn ws_handler(ws: WebSocketUpgrade, headers: HeaderMap, State(st): State<WsState>) -> Response {
    let get = |h: header::HeaderName| headers.get(h).and_then(|v| v.to_str().ok());
    let origin = get(header::ORIGIN);
    if !origin_allowed(origin, get(header::HOST)) {
        tracing::warn!("ws upgrade refused for origin {origin:?}");
        return StatusCode::FORBIDDEN.into_response();
    }
    let id = st.next_id.fetch_add(1, Ordering::Relaxed);
    ws.max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| handle_socket(socket, st.tx, id))
}

/// No `Origin` (not a browser), a loopback page, or a page from the host the
/// server was reached at (`Host`, e.g. the UI on a Pi opened as
/// `http://pi.local:8766`).
fn origin_allowed(origin: Option<&str>, host: Option<&str>) -> bool {
    let Some(origin) = origin else { return true };
    let Some(authority) = origin.strip_prefix("http://").or_else(|| origin.strip_prefix("https://")) else {
        return false; // "null" (file://, sandboxed frames) or garbage
    };
    let o = host_part(authority);
    let loopback = o.eq_ignore_ascii_case("localhost")
        || o == "[::1]"
        || o.parse::<std::net::Ipv4Addr>().is_ok_and(|ip| ip.is_loopback());
    loopback || host.is_some_and(|h| host_part(h).eq_ignore_ascii_case(o))
}

/// `host[:port]` → host (IPv6 literals keep their brackets).
fn host_part(authority: &str) -> &str {
    match authority.strip_prefix('[') {
        Some(rest) => rest.find(']').map_or(authority, |i| &authority[..i + 2]),
        None => authority.split(':').next().unwrap_or(authority),
    }
}

async fn handle_socket(socket: WebSocket, tx: mpsc::UnboundedSender<Event>, id: u64) {
    let client = Arc::new(Client::new(id));
    if tx.send(Event::Connect(client.clone())).is_err() {
        return;
    }
    let (mut sink, mut stream) = socket.split();
    let c2 = client.clone();
    let sender = async move {
        let mut next_ping = tokio::time::Instant::now() + PING_EVERY;
        let mut last_ping_ns: Option<u64> = None;
        loop {
            if tokio::time::Instant::now() >= next_ping {
                if last_ping_ns.is_some_and(|t| c2.last_rx_ns.load(Ordering::Relaxed) < t) {
                    tracing::info!("ws client sent nothing for {} s after a ping; disconnecting", PING_EVERY.as_secs());
                    return;
                }
                last_ping_ns = Some(now_ns());
                next_ping += PING_EVERY;
                if !send_or_drop(&mut sink, &c2, Message::Ping(Default::default())).await {
                    return;
                }
            }
            match c2.next() {
                Next::Msg(m) => {
                    if !send_or_drop(&mut sink, &c2, m).await {
                        return;
                    }
                }
                Next::Close => {
                    let frame = CloseFrame { code: 1008, reason: "client too slow".into() };
                    let close = sink.send(Message::Close(Some(frame)));
                    let _ = tokio::time::timeout(Duration::from_secs(1), close).await;
                    return;
                }
                Next::Empty => {
                    tokio::select! {
                        _ = c2.notify.notified() => {}
                        _ = tokio::time::sleep_until(next_ping) => {}
                    }
                }
            }
        }
    };
    let tx2 = tx.clone();
    let c3 = client.clone();
    let receiver = async move {
        while let Some(Ok(m)) = stream.next().await {
            c3.last_rx_ns.store(now_ns(), Ordering::Relaxed);
            let text = match m {
                Message::Text(t) => t.to_string(),
                // The reference json-decodes binary frames too.
                Message::Binary(b) => String::from_utf8_lossy(&b).into_owned(),
                Message::Close(_) => break,
                _ => continue,
            };
            // Backpressure: stop reading while the control loop is behind.
            loop {
                let done = c3.inflight_done.notified();
                if c3.inflight.load(Ordering::Acquire) < INBOUND_MAX {
                    break;
                }
                done.await;
            }
            c3.inflight.fetch_add(1, Ordering::AcqRel);
            if tx2.send(Event::Message(id, text)).is_err() {
                break;
            }
        }
    };
    tokio::select! {
        _ = sender => {},
        _ = receiver => {},
    }
    let _ = tx.send(Event::Disconnect(id));
}

/// One send, bounded by `SEND_TIMEOUT` and cut short by an overflow close.
/// False = drop the connection.
async fn send_or_drop(
    sink: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    c: &Client,
    m: Message,
) -> bool {
    tokio::select! {
        r = tokio::time::timeout(SEND_TIMEOUT, sink.send(m)) => match r {
            Ok(Ok(())) => true,
            Ok(Err(_)) => false,
            Err(_) => {
                tracing::warn!("ws client stuck: one send took > {} s; disconnecting", SEND_TIMEOUT.as_secs());
                false
            }
        },
        _ = c.kill.notified() => false,
    }
}

/// `[type=1:u8][reserved:u8][n_bins:u16 LE][f32 LE × n_bins]`; `gain` is
/// applied as f32 (skipped when 1.0, i.e. for raw dB).
pub fn encode_fft_binary(frame: &[f32], gain: f64) -> Vec<u8> {
    let n = frame.len();
    let mut out = Vec::with_capacity(4 + 4 * n);
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(n as u16).to_le_bytes());
    let g = gain as f32;
    for &v in frame {
        let v = if gain != 1.0 { v * g } else { v };
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_priority_and_drop_oldest() {
        let c = Client::new(1);
        for i in 0..6 {
            c.put_stream(Message::Text(format!("s{i}").into()));
        }
        c.put_status(&"st".into());
        c.put_control(Message::Text("c".into()));
        let mut got = vec![];
        while let Next::Msg(Message::Text(t)) = c.next() {
            got.push(t.to_string());
        }
        assert_eq!(got, ["c", "st", "s2", "s3", "s4", "s5"]);
    }

    #[test]
    fn control_overflow_closes() {
        let c = Client::new(1);
        for _ in 0..CONTROL_QUEUE_MAX + 1 {
            c.put_control(Message::Text("m".into()));
        }
        assert!(matches!(c.next(), Next::Close));
    }

    #[test]
    fn origin_check() {
        let ok = |o, h| origin_allowed(o, h);
        assert!(ok(None, Some("127.0.0.1:8765")), "non-browser client");
        assert!(ok(Some("http://127.0.0.1:8766"), Some("127.0.0.1:8765")));
        assert!(ok(Some("http://localhost:8766"), Some("127.0.0.1:8765")));
        assert!(ok(Some("http://[::1]:8766"), Some("[::1]:8765")));
        assert!(ok(Some("http://pi.local:8766"), Some("pi.local:8765")), "UI served by a remote host");
        assert!(!ok(Some("https://evil.example"), Some("127.0.0.1:8765")));
        assert!(!ok(Some("http://pi.local.evil.example"), Some("pi.local:8765")));
        assert!(!ok(Some("null"), Some("127.0.0.1:8765")));
        assert!(!ok(Some("http://192.168.1.5:8766"), None));
    }

    #[test]
    fn fft_binary_header() {
        let b = encode_fft_binary(&[0.5, 0.25], 2.0);
        assert_eq!(&b[..4], &[1, 0, 2, 0]);
        assert_eq!(f32::from_le_bytes(b[4..8].try_into().unwrap()), 1.0);
    }
}
