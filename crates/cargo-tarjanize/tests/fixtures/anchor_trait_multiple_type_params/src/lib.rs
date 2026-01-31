//! Test: Multiple trait type params.

pub trait MyTrait<A, B> {}
pub struct MyType;
pub struct ParamA;
pub struct ParamB;

impl MyTrait<ParamA, ParamB> for MyType {}
