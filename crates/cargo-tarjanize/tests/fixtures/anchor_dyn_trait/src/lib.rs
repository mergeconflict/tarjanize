//! Test: Impl for dyn Trait.

pub trait MyTrait {}
pub trait OtherTrait {}

impl OtherTrait for dyn MyTrait {}
