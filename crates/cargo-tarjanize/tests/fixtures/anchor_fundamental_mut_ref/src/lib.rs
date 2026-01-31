//! Test: &mut T is fundamental - anchor is T.

pub trait MyTrait {}
pub struct MyType;

impl MyTrait for &mut MyType {}
