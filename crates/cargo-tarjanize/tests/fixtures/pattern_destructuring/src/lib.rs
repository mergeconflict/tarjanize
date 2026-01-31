//! Test: Struct destructuring pattern creates edge.

pub struct Target {
    pub x: i32,
    pub y: i32,
}

pub fn caller(t: Target) -> i32 {
    let Target { x, y } = t;
    x + y
}
