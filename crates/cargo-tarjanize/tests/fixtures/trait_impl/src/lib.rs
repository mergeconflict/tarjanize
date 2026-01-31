//! Test: Trait impl creates edges to both the trait and the self type.

pub trait MyTrait {}
pub struct MyType;
impl MyTrait for MyType {}
