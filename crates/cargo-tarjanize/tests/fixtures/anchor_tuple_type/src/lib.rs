//! Test: Impl for tuple type - tuple is fundamental.

pub trait MyTrait {}
pub struct A;
pub struct B;

impl MyTrait for (A, B) {}
