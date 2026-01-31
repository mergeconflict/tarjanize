//! Test: Impl for array type creates edges to element type and trait.

pub trait MyTrait {}
pub struct Foo;

impl MyTrait for [Foo; 3] {}
