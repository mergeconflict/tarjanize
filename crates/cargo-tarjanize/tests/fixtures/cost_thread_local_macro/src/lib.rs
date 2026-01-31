//! Test: `thread_local!` macro-generated statics.

use std::cell::RefCell;

thread_local! {
    pub static COUNTER: RefCell<i32> = const { RefCell::new(0) };
}

pub fn increment() {
    COUNTER.with(|c| *c.borrow_mut() += 1);
}
