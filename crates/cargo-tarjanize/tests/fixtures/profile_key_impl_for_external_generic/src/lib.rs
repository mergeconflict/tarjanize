//! Test: `impl Trait for Box<T>` profile key (impl for external generic type).

pub struct MyType;

pub trait MyTrait {
    fn method(&self);
}

impl MyTrait for Box<MyType> {
    fn method(&self) {}
}
