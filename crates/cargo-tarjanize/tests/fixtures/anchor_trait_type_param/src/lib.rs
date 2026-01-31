//! Test: Trait type param is an anchor.

pub trait MyTrait<T> {}
pub struct MyType;
pub struct ParamType;

impl MyTrait<ParamType> for MyType {}
