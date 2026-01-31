//! Test: Impl block matched via `{{impl}}` key for profile cost.

pub struct MyType {
    value: i32,
}

impl MyType {
    pub fn new(value: i32) -> Self {
        Self { value }
    }

    pub fn compute(&self) -> i32 {
        let mut result = self.value;
        for _ in 0..10 {
            result = result.wrapping_mul(2);
        }
        result
    }
}
