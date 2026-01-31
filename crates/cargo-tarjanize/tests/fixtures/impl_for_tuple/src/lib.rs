//! Test: Impl for tuple type creates edges to element types and trait.

pub trait MyTrait {}
pub struct A;
pub struct B;

impl MyTrait for (A, B) {}
