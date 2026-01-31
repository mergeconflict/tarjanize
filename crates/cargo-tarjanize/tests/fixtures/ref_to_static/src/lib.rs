//! Test: Reference to static item creates edge.

pub static S: i32 = 0;

pub fn caller() -> i32 {
    S
}
