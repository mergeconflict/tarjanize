//! Test: Closure inside closure aggregates to outermost function.

pub fn nested_closure_user() -> i32 {
    let outer = |x: i32| {
        let inner = |y: i32| y * 2;
        inner(x) + 1
    };
    outer(10)
}
