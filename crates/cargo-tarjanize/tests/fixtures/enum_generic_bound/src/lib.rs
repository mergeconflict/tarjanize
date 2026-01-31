//! Test: Enum generic bound creates edge.

pub trait BoundTrait {}

pub enum EnumWithBound<T: BoundTrait> {
    Variant(T),
}
