//! Test: Struct generic bound creates edge.

pub trait BoundTrait {}

pub struct StructWithBound<T: BoundTrait> {
    pub value: T,
}
