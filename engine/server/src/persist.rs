//! Drag-aware debounced persister for `configs/main.yaml` (`config.Persister`).
//!
//! `commit = false` (a slider drag) waits 1 s, `commit = true` 50 ms; both are
//! capped at 250 ms after the first unsaved change. The control loop polls
//! `deadline()` and calls `fire()`; writes are atomic (`config::write_yaml_atomic`).

use crate::config::{Config, write_yaml_atomic};
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub const DRAG_DELAY: Duration = Duration::from_secs(1);
pub const COMMIT_DELAY: Duration = Duration::from_millis(50);
pub const MAX_DELAY: Duration = Duration::from_millis(250);

pub struct Persister {
    path: PathBuf,
    attached: bool,
    deadline: Option<Instant>,
    earliest: Option<Instant>,
    dirty: bool,
}

impl Persister {
    pub fn new(path: PathBuf) -> Self {
        Persister { path, attached: false, deadline: None, earliest: None, dirty: false }
    }

    /// Requests before `attach` are ignored (the reference has no loop yet).
    pub fn attach(&mut self) {
        self.attached = true;
    }

    pub fn request(&mut self, commit: bool) {
        if !self.attached {
            return;
        }
        self.dirty = true;
        let now = Instant::now();
        let mut target = now + if commit { COMMIT_DELAY } else { DRAG_DELAY };
        let earliest = *self.earliest.get_or_insert(now + MAX_DELAY);
        if earliest < target {
            target = earliest;
        }
        self.deadline = Some(target);
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn fire(&mut self, cfg: &Config) {
        self.deadline = None;
        self.earliest = None;
        if !std::mem::take(&mut self.dirty) {
            return;
        }
        if let Err(e) = write_yaml_atomic(&self.path, cfg) {
            tracing::warn!("persist failed: {e}");
        }
    }

    /// Final synchronous flush on shutdown.
    pub fn flush_now(&mut self, cfg: &Config) {
        self.deadline = None;
        if !std::mem::take(&mut self.dirty) {
            return;
        }
        if let Err(e) = write_yaml_atomic(&self.path, cfg) {
            tracing::warn!("final persist failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debounce_caps() {
        let mut p = Persister::new(PathBuf::from("/nonexistent/x.yaml"));
        p.request(true);
        assert!(p.deadline().is_none(), "ignored before attach");
        p.attach();
        let t0 = Instant::now();
        p.request(false);
        let d = p.deadline().unwrap() - t0;
        assert!(d <= MAX_DELAY + Duration::from_millis(5), "drag capped at 250 ms: {d:?}");
        p.fire_without_write();
        p.request(true);
        let d = p.deadline().unwrap() - Instant::now();
        assert!(d <= COMMIT_DELAY);
    }

    impl Persister {
        fn fire_without_write(&mut self) {
            self.deadline = None;
            self.earliest = None;
            self.dirty = false;
        }
    }
}
