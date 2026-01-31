//! Test: Multiple trait bounds with + create edges to all traits.

pub trait Tr1 {}
pub trait Tr2 {}
pub trait Tr3 {}

pub fn multi_bound<T: Tr1 + Tr2 + Tr3>(_: T) {}
