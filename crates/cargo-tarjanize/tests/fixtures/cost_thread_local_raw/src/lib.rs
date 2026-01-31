//! Test: Raw `#[thread_local]` static extraction.
#![feature(thread_local)]

#[thread_local]
pub static COUNTER: std::cell::Cell<i32> = std::cell::Cell::new(0);

pub fn increment() {
    COUNTER.set(COUNTER.get() + 1);
}
