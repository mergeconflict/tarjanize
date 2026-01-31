//! Test: Macro invocation creates edge to called function.

pub fn target_fn() {}

macro_rules! call_target {
    () => {
        target_fn()
    };
}

pub fn caller() {
    call_target!();
}
