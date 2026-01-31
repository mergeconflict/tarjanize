//! Test: Impl associated type definition creates edge.

pub trait TraitWithAssoc {
    type Output;
}

pub struct ImplType;
pub struct OutputType;

impl TraitWithAssoc for ImplType {
    type Output = OutputType;
}
