//! Test: crate:: prefixed self type.

pub trait MyTrait {}
pub struct MyType;

impl MyTrait for crate::MyType {}
