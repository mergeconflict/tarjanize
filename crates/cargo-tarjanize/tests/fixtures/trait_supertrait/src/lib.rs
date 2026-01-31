//! Test: Trait supertrait creates an edge.

pub trait Supertrait {}

pub trait Subtrait: Supertrait {}
