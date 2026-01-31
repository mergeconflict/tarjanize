//! Test: Reference to thread-local static creates edge.
//!
//! This tests both the `thread_local!` macro style and raw `#[thread_local]`
//! attribute style.

#![feature(thread_local)]

use std::cell::Cell;

// Standard thread_local! macro style - creates a LocalKey<T> wrapper.
thread_local! {
    pub static TLS_MACRO: Cell<i32> = const { Cell::new(42) };
}

// Raw #[thread_local] attribute - this is what ThreadLocalRef in THIR captures.
#[thread_local]
pub static TLS_RAW: Cell<i32> = Cell::new(0);

pub fn uses_macro_tls() -> i32 {
    TLS_MACRO.with(|v| v.get())
}

pub fn uses_raw_tls() -> i32 {
    TLS_RAW.get()
}
