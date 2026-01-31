//! Test: Self keyword in impl resolves to self type.

pub struct T;

impl T {
    pub fn new() -> Self {
        Self
    }
}
