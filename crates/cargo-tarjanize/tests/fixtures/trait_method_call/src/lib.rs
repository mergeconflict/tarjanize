//! Test: Trait method call normalizes to trait.

pub trait MyTrait {
    fn method(&self);
}

pub struct MyType;

impl MyTrait for MyType {
    fn method(&self) {}
}

pub fn caller(t: &dyn MyTrait) {
    t.method();
}
