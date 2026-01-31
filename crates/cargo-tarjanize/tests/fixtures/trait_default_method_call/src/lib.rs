//! Test: Trait default method call normalizes to trait.

pub trait MyTrait {
    fn default_method(&self) -> i32 {
        42
    }
}

pub struct MyType;
impl MyTrait for MyType {}

pub fn caller(t: &MyType) -> i32 {
    t.default_method()
}
