//! Test: Reference to const item creates edge.

pub const C: i32 = 0;

pub fn caller() -> i32 {
    C
}
