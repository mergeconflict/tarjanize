//! Test: impl Trait for &T - reference is fundamental, anchor is T.

pub trait MyTrait {}
pub struct MyType;

impl MyTrait for &MyType {}
