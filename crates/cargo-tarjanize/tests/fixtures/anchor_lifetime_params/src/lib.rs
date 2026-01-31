//! Test: Lifetime params in impl do not affect anchors.

pub struct MyType<'a>(&'a i32);

impl<'a> MyType<'a> {
    pub fn method(&self) -> &'a i32 {
        self.0
    }
}
