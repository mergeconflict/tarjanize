//! Test: `impl Trait` argument creates edge to the trait.

pub trait Tr {}

pub fn takes_impl(_: impl Tr) {}
