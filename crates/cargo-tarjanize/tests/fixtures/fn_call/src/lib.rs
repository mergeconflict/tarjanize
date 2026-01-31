//! Test: Function body calling another function creates an edge.

pub fn target_fn() {}

pub fn caller_fn() {
    target_fn();
}
