//! Test: Generic trait impl.

pub trait MyTrait<T> {}
pub struct MyType;

impl<T> MyTrait<T> for MyType {}
