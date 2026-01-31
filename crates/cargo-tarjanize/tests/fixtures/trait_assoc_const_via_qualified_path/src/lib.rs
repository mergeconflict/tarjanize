//! Test: <T as Trait>::CONST normalizes to trait.

pub trait MyTrait {
    const VALUE: i32;
}

pub struct MyType;

impl MyTrait for MyType {
    const VALUE: i32 = 42;
}

pub fn caller() -> i32 {
    <MyType as MyTrait>::VALUE
}
