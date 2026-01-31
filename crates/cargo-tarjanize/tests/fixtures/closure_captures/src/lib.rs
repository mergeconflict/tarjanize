//! Test: Closure captures create edges.

pub fn helper() -> i32 {
    42
}

pub fn caller() {
    let _closure = || helper();
}
