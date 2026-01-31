//! Test: Array repeat expression creates edge.

pub fn helper() -> i32 {
    42
}

pub fn caller() -> [i32; 3] {
    [helper(); 3]
}
