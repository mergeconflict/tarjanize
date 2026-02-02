//! Test: Methods within an impl calling each other.
//!
//! This tests that when method A calls method B via `self.b()`,
//! the dependency is normalized to the impl block, not recorded
//! as a dependency to `{{impl}}::b`.

pub struct Task {
    pub value: i32,
}

impl Task {
    pub fn run(&self) {
        // Call another method on self
        self.helper();
    }

    pub fn helper(&self) -> i32 {
        self.value
    }
}
