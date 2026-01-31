//! Test: Deeply nested functions all aggregate to outermost function.

pub fn outer_fn() -> i32 {
    fn level1_fn() -> i32 {
        fn level2_fn() -> i32 {
            42
        }
        level2_fn()
    }
    level1_fn()
}
