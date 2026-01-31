//! Test: `impl Trait for dyn OtherTrait` profile key.

pub trait BaseTrait {}

pub trait ExtendedTrait {
    fn method(&self);
}

impl ExtendedTrait for dyn BaseTrait {
    fn method(&self) {}
}
