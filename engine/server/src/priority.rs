//! Scheduling priority for the server's threads.
//!
//! `boost_current_thread` is the reference policy (`server/priority.py`):
//! macOS QoS `USER_INITIATED` (no privileges; biases onto P-cores), Linux
//! absolute niceness −5, never lowering a thread already at/above it. The
//! main thread uses it.
//!
//! `realtime_current_thread` is for the DSP / FFT workers and goes one step
//! further (deviation from the reference): 10-minute soaks under the
//! reference policy saw the workers go unscheduled for 30–70 ms a few times
//! per run while the CoreAudio callback stayed perfectly regular, so the
//! backlog policy skipped blocks; a real-time class fixed it
//! (`docs/reference/rust_server.md` → Worker scheduling).
//! - macOS: the Mach time-constraint policy, `period` = the worker's wake
//!   cadence (block period for DSP, hop period for FFT), `computation` = a
//!   quarter of it, `constraint` = the period, preemptible (the CoreAudio IO
//!   thread can always preempt a worker). No privileges needed.
//! - Linux: `SCHED_FIFO` at a low real-time priority when permitted
//!   (`RLIMIT_RTPRIO` / `CAP_SYS_NICE`).
//!
//! Every step is best-effort: a refused call leaves the previous one in
//! place (real-time → the reference policy → unchanged) and never fails.

#[cfg(target_os = "macos")]
pub fn boost_current_thread(label: &str) -> bool {
    // SAFETY: plain libc call on the current thread.
    let rc = unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INITIATED, 0)
    };
    if rc == 0 {
        tracing::info!("priority: {label} -> QOS_CLASS_USER_INITIATED");
        true
    } else {
        tracing::debug!("priority: pthread_set_qos rc={rc} for {label}");
        false
    }
}

#[cfg(target_os = "linux")]
pub fn boost_current_thread(label: &str) -> bool {
    const TARGET_NICE: libc::c_int = -5;
    // SAFETY: plain libc calls; who = 0 is the calling thread on Linux.
    unsafe {
        *libc::__errno_location() = 0;
        let cur = libc::getpriority(libc::PRIO_PROCESS, 0);
        if *libc::__errno_location() != 0 {
            return false;
        }
        if cur <= TARGET_NICE {
            return true;
        }
        if libc::setpriority(libc::PRIO_PROCESS, 0, TARGET_NICE) == 0 {
            tracing::info!("priority: {label} -> nice {TARGET_NICE}");
            return true;
        }
    }
    false
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn boost_current_thread(_label: &str) -> bool {
    false
}

/// Real-time class for a worker thread waking every `period_s`; falls back
/// to `boost_current_thread`. Returns true when the real-time class is on.
/// Called once at worker start (the period is a scheduling hint, so it isn't
/// re-applied when a later reconfigure changes the cadence).
#[cfg(target_os = "macos")]
pub fn realtime_current_thread(label: &str, period_s: f64) -> bool {
    boost_current_thread(label);
    match time_constraint(period_s) {
        Ok(()) => {
            tracing::info!(
                "priority: {label} -> real-time (time-constraint, period {:.2} ms)",
                period_s * 1e3
            );
            true
        }
        Err(kr) => {
            tracing::warn!(
                "priority: time-constraint policy refused for {label} (kern_return {kr}); \
                 staying at QOS_CLASS_USER_INITIATED"
            );
            false
        }
    }
}

// libc deprecates its Mach bindings in favour of the `mach2` crate; not
// worth a dependency for one timebase call.
#[allow(deprecated)]
#[cfg(target_os = "macos")]
fn time_constraint(period_s: f64) -> Result<(), libc::kern_return_t> {
    let period_ns = (period_s * 1e9).clamp(1e6, 1e8) as u64; // 1–100 ms
    // SAFETY: mach_timebase_info / thread_policy_set on the calling thread,
    // with a correctly sized policy struct.
    unsafe {
        let mut tb = libc::mach_timebase_info { numer: 0, denom: 0 };
        if libc::mach_timebase_info(&mut tb) != libc::KERN_SUCCESS || tb.numer == 0 {
            return Err(-1);
        }
        let to_abs = |ns: u64| (ns as u128 * tb.denom as u128 / tb.numer as u128) as u32;
        let mut pol = libc::thread_time_constraint_policy {
            period: to_abs(period_ns),
            computation: to_abs(period_ns / 4),
            constraint: to_abs(period_ns),
            preemptible: 1,
        };
        let kr = libc::thread_policy_set(
            libc::pthread_mach_thread_np(libc::pthread_self()),
            libc::THREAD_TIME_CONSTRAINT_POLICY as libc::thread_policy_flavor_t,
            (&mut pol as *mut libc::thread_time_constraint_policy).cast(),
            libc::THREAD_TIME_CONSTRAINT_POLICY_COUNT,
        );
        if kr == libc::KERN_SUCCESS { Ok(()) } else { Err(kr) }
    }
}

#[cfg(target_os = "linux")]
pub fn realtime_current_thread(label: &str, _period_s: f64) -> bool {
    // Low end of the 1–99 range: above every normal thread, below the
    // kernel's and audio servers' real-time threads.
    const FIFO_PRIORITY: libc::c_int = 10;
    let param = libc::sched_param { sched_priority: FIFO_PRIORITY };
    // SAFETY: plain libc call on the calling thread.
    let rc = unsafe { libc::pthread_setschedparam(libc::pthread_self(), libc::SCHED_FIFO, &param) };
    if rc == 0 {
        tracing::info!("priority: {label} -> SCHED_FIFO {FIFO_PRIORITY}");
        return true;
    }
    tracing::info!("priority: SCHED_FIFO not permitted for {label} (errno {rc}); using nice");
    boost_current_thread(label);
    false
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn realtime_current_thread(_label: &str, _period_s: f64) -> bool {
    false
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    #[test]
    fn time_constraint_accepted_for_worker_periods() {
        // Block periods 128–2048 @ 16–96 kHz and hop periods up to the
        // clamp must all be accepted by the kernel (on a scratch thread).
        for period_s in [128.0 / 96_000.0, 256.0 / 48_000.0, 512.0 / 48_000.0, 4096.0 / 16_000.0] {
            let ok = std::thread::spawn(move || super::time_constraint(period_s)).join().unwrap();
            assert_eq!(ok, Ok(()), "period {period_s}");
        }
    }
}
