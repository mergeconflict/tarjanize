//! Test: Local self type with trait type param.

pub trait MyTrait<T> {}
pub struct MyType;

impl<T> MyTrait<T> for MyType {}
