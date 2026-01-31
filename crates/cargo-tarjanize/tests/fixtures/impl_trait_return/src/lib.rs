//! Test: `impl Trait` return type creates edge to the trait.

pub trait Tr {}
pub struct S;

impl Tr for S {}

pub fn returns_impl() -> impl Tr {
    S
}
