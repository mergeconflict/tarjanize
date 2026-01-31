//! Test: `impl Trait for (A, B)` profile key.

pub struct A;
pub struct B;

pub trait MyTrait {
    fn method(&self);
}

impl MyTrait for (A, B) {
    fn method(&self) {}
}
