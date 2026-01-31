//! Test: Impl for dyn trait creates edges to both traits.

pub trait MyTrait {}
pub trait OtherTrait {}

impl MyTrait for dyn OtherTrait {}
