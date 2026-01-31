//! Test: Trait associated type bound creates edge.

pub trait BoundTrait {}

pub trait TraitWithAssocBound {
    type Item: BoundTrait;
}
