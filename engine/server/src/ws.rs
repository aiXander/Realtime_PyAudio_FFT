//! WebSocket transport (`server/io/ws_server.py`): per-client outbound
//! lanes and the axum connection tasks. All state decisions (greeting,
//! broadcast, dispatch) happen in the control loop (`app.rs`); a connection
//! task only shovels frames between its socket and the loop.
//!
//! Three outbound lanes per client, drained in this priority:
//!   1. control — FIFO, never dropped; > 1024 queued closes the client (1008);
//!   2. status  — latest-only `server_status`;
//!   3. stream  — snapshots + binary FFT, drop-oldest, max 4.

use axum::extract::ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{Notify, mpsc};

pub const STREAM_QUEUE_MAX: usize = 4;
pub const CONTROL_QUEUE_MAX: usize = 1024;

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
}

enum Next {
    Msg(Message),
    Close,
    Empty,
}

impl Client {
    fn new(id: u64) -> Self {
        Client { id, lanes: Mutex::new(Lanes::default()), notify: Notify::new() }
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

async fn ws_handler(ws: WebSocketUpgrade, State(st): State<WsState>) -> Response {
    let id = st.next_id.fetch_add(1, Ordering::Relaxed);
    ws.on_upgrade(move |socket| handle_socket(socket, st.tx, id))
}

async fn handle_socket(socket: WebSocket, tx: mpsc::UnboundedSender<Event>, id: u64) {
    let client = Arc::new(Client::new(id));
    if tx.send(Event::Connect(client.clone())).is_err() {
        return;
    }
    let (mut sink, mut stream) = socket.split();
    let c2 = client.clone();
    let sender = async move {
        loop {
            match c2.next() {
                Next::Msg(m) => {
                    if sink.send(m).await.is_err() {
                        return;
                    }
                }
                Next::Close => {
                    let frame = CloseFrame { code: 1008, reason: "client too slow".into() };
                    let _ = sink.send(Message::Close(Some(frame))).await;
                    return;
                }
                Next::Empty => c2.notify.notified().await,
            }
        }
    };
    let tx2 = tx.clone();
    let receiver = async move {
        while let Some(Ok(m)) = stream.next().await {
            let text = match m {
                Message::Text(t) => t.to_string(),
                // The reference json-decodes binary frames too.
                Message::Binary(b) => String::from_utf8_lossy(&b).into_owned(),
                Message::Close(_) => break,
                _ => continue,
            };
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
    fn fft_binary_header() {
        let b = encode_fft_binary(&[0.5, 0.25], 2.0);
        assert_eq!(&b[..4], &[1, 0, 2, 0]);
        assert_eq!(f32::from_le_bytes(b[4..8].try_into().unwrap()), 1.0);
    }
}
