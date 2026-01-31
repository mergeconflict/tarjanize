//! Test: Impl for external generic with local type arg creates edges.

pub trait MyTrait {}
pub struct Foo;

impl MyTrait for Box<Foo> {}
