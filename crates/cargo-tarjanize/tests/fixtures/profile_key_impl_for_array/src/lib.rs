//! Test: `impl Trait for [T; N]` profile key.

pub struct MyType;

pub trait MyTrait {
    fn method(&self);
}

impl MyTrait for [MyType; 3] {
    fn method(&self) {}
}
