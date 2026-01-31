//! Test: Nested items inside functions are collapsed to the parent function.
//!
//! Items defined inside function bodies (nested functions, statics, consts) can't
//! be split independently. They should not appear as separate symbols, and any
//! dependency on them should be collapsed to the containing function.

pub struct Helper;

impl Helper {
    pub fn do_work(&self) -> i32 {
        42
    }
}

/// Outer function containing a nested function.
pub fn outer() -> i32 {
    // This nested function should be collapsed to `outer`.
    fn inner() -> i32 {
        let h = Helper;
        h.do_work()
    }

    inner()
}

/// Function that calls the outer function.
/// This should have a dependency on `outer`, not on `inner`.
pub fn caller() -> i32 {
    outer()
}
