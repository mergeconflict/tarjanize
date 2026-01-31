//! Test: Multiple closures aggregate to same parent.

pub fn uses_closures() -> i32 {
    let add = |x: i32, y: i32| x + y;
    let mul = |x: i32, y: i32| x * y;
    add(2, 3) + mul(4, 5)
}
