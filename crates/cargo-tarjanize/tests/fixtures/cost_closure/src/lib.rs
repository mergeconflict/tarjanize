//! Test: Closure cost aggregates to containing function.

pub fn uses_closure() -> i32 {
    let add_one = |x: i32| x + 1;
    add_one(41)
}
