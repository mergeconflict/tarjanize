//! Test: Nested function cost aggregates to outer function.

pub fn outer_fn() -> i32 {
    fn nested_fn() -> i32 {
        42
    }
    nested_fn()
}
