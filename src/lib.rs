//! This crate does low latency streaming automatic speech recognition and tagging for realtime-esque
//! applications.
#![cfg_attr(docsrs, feature(doc_cfg))]
use std::ffi::CString;

/// Automatic speech recognition
pub mod asr;

/// Audio tagging
pub mod tag;

struct DropCString(*mut i8);

impl Drop for DropCString {
    fn drop(&mut self) {
        unsafe { drop(CString::from_raw(self.0)) }
    }
}

fn track_cstr(dcs: &mut Vec<DropCString>, s: &str) -> *const i8 {
    let p = CString::new(s).unwrap().into_raw();
    dcs.push(DropCString(p));
    p
}

fn default_num_threads() -> usize {
    if cfg!(feature = "cuda") || cfg!(feature = "directml") {
        1
    } else {
        num_cpus::get_physical().min(4)
    }
}

fn default_compute_provider() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "directml") {
        "directml"
    } else {
        "cpu"
    }
}
