//! Best-effort scheduling-priority elevation (`server/priority.py`).
//!
//! macOS: `pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED)` per
//! thread (no privileges; biases onto P-cores). Linux: absolute niceness −5
//! for the calling thread, never lowering one already at/above it. Other
//! platforms: no-op. Never fails loudly.

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
