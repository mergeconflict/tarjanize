//! Test: Multiple nested functions in one parent.

pub fn outer_fn() -> i32 {
    fn helper1() -> i32 {
        1
    }
    fn helper2() -> i32 {
        2
    }
    helper1() + helper2()
}
