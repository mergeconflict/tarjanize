//! Test: Function generic bound creates edge.

pub trait MyTrait {}

pub fn generic_fn<T: MyTrait>(_x: T) {}
