//! Test: dyn Trait creates edge to the trait.

pub trait Tr {}

pub fn takes_dyn(_: &dyn Tr) {}
