//! System audio capture on macOS 14.2+ via a Core Audio process tap
//! (`server/audio/system_tap.py`; read its docstring for the why).
//!
//! A tap on the output device (all processes, unmuted) inside a private
//! aggregate device that has the output device as clock sub-device and the
//! tap not auto-started, so silence arrives as zero blocks instead of no
//! blocks. cpal's own loopback recording builds the aggregate without that
//! clock, hence this module. Raw FFI (CoreAudio, CoreFoundation, libobjc
//! are already linked through cpal), no extra crates.

pub const SYSTEM_AUDIO_NAME: &str = "System audio";
pub const TAP_DEVICE_PREFIX: &str = "audio-server system tap";
/// UID prefix of the same devices: still readable while one is being torn
/// down and its name already isn't.
pub const TAP_UID_PREFIX: &str = "com.audio-server.system-tap.";

#[cfg(target_os = "macos")]
mod imp {
    use super::TAP_DEVICE_PREFIX;
    use std::ffi::{CStr, c_char, c_long, c_void};
    use std::sync::atomic::{AtomicU32, Ordering};

    type Id = *mut c_void;
    type Sel = *const c_void;
    type Cf = *const c_void;

    #[repr(C)]
    struct Addr {
        selector: u32,
        scope: u32,
        element: u32,
    }
    #[repr(C)]
    struct Opaque {
        _p: [u8; 0],
    }

    const fn fourcc(s: &[u8; 4]) -> u32 {
        u32::from_be_bytes(*s)
    }
    const SYSTEM_OBJECT: u32 = 1;
    const GLOBAL: u32 = fourcc(b"glob");
    const DEFAULT_OUTPUT: u32 = fourcc(b"dOut");
    const DEVICE_UID: u32 = fourcc(b"uid ");
    const PROCESS_LIST: u32 = fourcc(b"prs#");
    const PROCESS_PID: u32 = fourcc(b"ppid");
    const PROCESS_RUNNING_OUTPUT: u32 = fourcc(b"piro");
    const UTF8: u32 = 0x0800_0100;

    #[link(name = "CoreAudio", kind = "framework")]
    unsafe extern "C" {
        fn AudioObjectGetPropertyDataSize(obj: u32, a: *const Addr, qs: u32, q: *const c_void, size: *mut u32) -> i32;
        fn AudioObjectGetPropertyData(
            obj: u32,
            a: *const Addr,
            qs: u32,
            q: *const c_void,
            size: *mut u32,
            data: *mut c_void,
        ) -> i32;
        fn AudioHardwareCreateProcessTap(desc: Id, out: *mut u32) -> i32;
        fn AudioHardwareDestroyProcessTap(id: u32) -> i32;
        fn AudioHardwareCreateAggregateDevice(desc: Cf, out: *mut u32) -> i32;
        fn AudioHardwareDestroyAggregateDevice(id: u32) -> i32;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        static kCFTypeDictionaryKeyCallBacks: Opaque;
        static kCFTypeDictionaryValueCallBacks: Opaque;
        static kCFTypeArrayCallBacks: Opaque;
        static kCFBooleanTrue: Cf;
        fn CFStringCreateWithCString(alloc: Cf, s: *const c_char, enc: u32) -> Cf;
        fn CFStringGetCString(s: Cf, buf: *mut c_char, len: c_long, enc: u32) -> bool;
        fn CFDictionaryCreateMutable(alloc: Cf, cap: c_long, k: *const Opaque, v: *const Opaque) -> *mut c_void;
        fn CFDictionarySetValue(d: *mut c_void, k: Cf, v: Cf);
        fn CFArrayCreate(alloc: Cf, vals: *const Cf, n: c_long, cb: *const Opaque) -> Cf;
        fn CFRelease(x: Cf);
    }

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(pool: *mut c_void);
    }

    /// `objc_msgSend` cast to the exact prototype (arm64 is not variadic-safe).
    macro_rules! msg {
        ($ret:ty; $obj:expr, $sel:expr $(, $t:ty = $a:expr)*) => {{
            let f: unsafe extern "C" fn(Id, Sel $(, $t)*) -> $ret =
                unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
            unsafe { f($obj, sel_registerName($sel.as_ptr()) $(, $a)*) }
        }};
    }

    pub fn available() -> bool {
        !unsafe { objc_getClass(c"CATapDescription".as_ptr()) }.is_null()
    }

    fn prop<T: Default>(obj: u32, selector: u32) -> Option<T> {
        let mut v = T::default();
        let mut size = size_of::<T>() as u32;
        let a = Addr { selector, scope: GLOBAL, element: 0 };
        let st = unsafe { AudioObjectGetPropertyData(obj, &a, 0, std::ptr::null(), &mut size, (&raw mut v).cast()) };
        (st == 0).then_some(v)
    }

    fn prop_string(obj: u32, selector: u32) -> Option<String> {
        let r = prop::<usize>(obj, selector).filter(|&p| p != 0)? as Cf;
        let mut buf = [0 as c_char; 1024];
        let ok = unsafe { CFStringGetCString(r, buf.as_mut_ptr(), buf.len() as c_long, UTF8) };
        unsafe { CFRelease(r) };
        ok.then(|| unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
    }

    /// UID of the current default output device (what a new tap records).
    pub fn default_output_uid() -> Option<String> {
        let dev = prop::<u32>(SYSTEM_OBJECT, DEFAULT_OUTPUT).filter(|&d| d != 0)?;
        prop_string(dev, DEVICE_UID)
    }

    /// Whether another process is playing audio right now (the probe's answer
    /// for "System audio", same as the Python engine).
    pub fn output_playing_elsewhere() -> bool {
        let a = Addr { selector: PROCESS_LIST, scope: GLOBAL, element: 0 };
        let mut size = 0u32;
        if unsafe { AudioObjectGetPropertyDataSize(SYSTEM_OBJECT, &a, 0, std::ptr::null(), &mut size) } != 0 {
            return false;
        }
        let mut procs = vec![0u32; size as usize / 4];
        let st = unsafe {
            AudioObjectGetPropertyData(SYSTEM_OBJECT, &a, 0, std::ptr::null(), &mut size, procs.as_mut_ptr().cast())
        };
        let me = std::process::id() as i32;
        st == 0
            && procs.iter().any(|&p| {
                prop::<i32>(p, PROCESS_PID) != Some(me) && prop::<u32>(p, PROCESS_RUNNING_OUTPUT).unwrap_or(0) != 0
            })
    }

    /// A process tap on one output device + the private aggregate device
    /// wrapping it (named `name`; open it as an input from this process).
    /// Dropping it destroys both.
    pub struct SystemTap {
        pub name: String,
        pub output_uid: String,
        tap_id: u32,
        agg_id: u32,
    }

    static INSTANCES: AtomicU32 = AtomicU32::new(0);

    /// +1 CF/ObjC references, released when the builder is dropped.
    struct Owned(Vec<Cf>);
    impl Owned {
        fn own(&mut self, r: Cf) -> Cf {
            self.0.push(r);
            r
        }
        fn str(&mut self, s: &str) -> Cf {
            let c = std::ffi::CString::new(s).unwrap_or_default();
            self.own(unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) })
        }
        fn array(&mut self, vals: &[Cf]) -> Cf {
            let a = unsafe { CFArrayCreate(std::ptr::null(), vals.as_ptr(), vals.len() as c_long, &kCFTypeArrayCallBacks) };
            self.own(a)
        }
        fn dict(&mut self, items: &[(&str, Cf)]) -> Cf {
            let d = unsafe {
                CFDictionaryCreateMutable(
                    std::ptr::null(),
                    items.len() as c_long,
                    &kCFTypeDictionaryKeyCallBacks,
                    &kCFTypeDictionaryValueCallBacks,
                )
            };
            self.own(d);
            for (k, v) in items {
                let k = self.str(k);
                unsafe { CFDictionarySetValue(d, k, *v) };
            }
            d
        }
    }
    impl Drop for Owned {
        fn drop(&mut self) {
            for &r in &self.0 {
                unsafe { CFRelease(r) };
            }
        }
    }

    impl SystemTap {
        pub fn new(output_uid: &str) -> Result<Self, String> {
            if !available() {
                return Err("system audio capture needs macOS 14.2+".into());
            }
            let tag = format!("{}.{}", std::process::id(), INSTANCES.fetch_add(1, Ordering::Relaxed) + 1);
            let mut me =
                SystemTap { name: format!("{TAP_DEVICE_PREFIX} {tag}"), output_uid: output_uid.into(), tap_id: 0, agg_id: 0 };
            let pool = unsafe { objc_autoreleasePoolPush() };
            let res = me.build(&tag);
            unsafe { objc_autoreleasePoolPop(pool) };
            res.map(|()| me) // on error, dropping `me` destroys what was made
        }

        fn build(&mut self, tag: &str) -> Result<(), String> {
            let mut o = Owned(Vec::new());
            let out_uid = o.str(&self.output_uid);
            let empty = o.array(&[]);
            // [[CATapDescription alloc] initWithProcesses:@[] andDeviceUID:uid withStream:0];
            // an empty process list + exclusive = every process.
            let class = unsafe { objc_getClass(c"CATapDescription".as_ptr()) };
            let desc = msg!(Id; class, c"alloc");
            let desc = msg!(Id; desc, c"initWithProcesses:andDeviceUID:withStream:", Cf = empty, Cf = out_uid, c_long = 0);
            if desc.is_null() {
                return Err("CATapDescription init failed".into());
            }
            o.own(desc);
            msg!((); desc, c"setMuteBehavior:", c_long = 0); // CATapUnmuted
            msg!((); desc, c"setPrivate:", bool = true);
            msg!((); desc, c"setExclusive:", bool = true);
            let name = o.str(&self.name);
            msg!((); desc, c"setName:", Cf = name);
            let st = unsafe { AudioHardwareCreateProcessTap(desc, &mut self.tap_id) };
            if st != 0 {
                return Err(format!("AudioHardwareCreateProcessTap failed (OSStatus {st})"));
            }
            let uuid = msg!(Id; desc, c"UUID");
            let tap_uid = msg!(Cf; uuid, c"UUIDString");

            let t = unsafe { kCFBooleanTrue };
            let agg_uid = o.str(&format!("{}{tag}", super::TAP_UID_PREFIX));
            let clock = o.dict(&[("uid", out_uid)]);
            let subdevices = o.array(&[clock]);
            let sub_tap = o.dict(&[("uid", tap_uid), ("drift", t)]);
            let taps = o.array(&[sub_tap]);
            let props = o.dict(&[
                ("name", name),
                ("uid", agg_uid),
                ("master", out_uid),
                ("subdevices", subdevices),
                ("taps", taps),
                ("private", t),
            ]);
            let st = unsafe { AudioHardwareCreateAggregateDevice(props, &mut self.agg_id) };
            if st != 0 {
                return Err(format!("AudioHardwareCreateAggregateDevice failed (OSStatus {st})"));
            }
            Ok(())
        }
    }

    impl Drop for SystemTap {
        fn drop(&mut self) {
            if self.agg_id != 0 {
                unsafe { AudioHardwareDestroyAggregateDevice(self.agg_id) };
            }
            if self.tap_id != 0 {
                unsafe { AudioHardwareDestroyProcessTap(self.tap_id) };
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub fn available() -> bool {
        false
    }
    pub fn default_output_uid() -> Option<String> {
        None
    }
    pub fn output_playing_elsewhere() -> bool {
        false
    }
    pub struct SystemTap {
        pub name: String,
        pub output_uid: String,
    }
    impl SystemTap {
        pub fn new(_output_uid: &str) -> Result<Self, String> {
            Err("system audio capture needs macOS 14.2+".into())
        }
    }
}

pub use imp::*;
