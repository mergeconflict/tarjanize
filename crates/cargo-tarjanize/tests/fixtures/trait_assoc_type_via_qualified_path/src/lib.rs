//! Test: <T as Trait>::Type normalizes to trait.

pub trait MyTrait {
    type Output;
}

pub struct MyType;
pub struct OutputType;

impl MyTrait for MyType {
    type Output = OutputType;
}

pub fn caller() -> <MyType as MyTrait>::Output {
    OutputType
}
