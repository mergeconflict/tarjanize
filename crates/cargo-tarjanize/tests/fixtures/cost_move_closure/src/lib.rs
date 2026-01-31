//! Test: Move closure cost aggregates correctly.

pub fn uses_move_closure() -> i32 {
    let value = 42;
    let closure = move || value;
    closure()
}
