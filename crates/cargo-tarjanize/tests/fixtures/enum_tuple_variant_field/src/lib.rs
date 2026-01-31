//! Test: Enum tuple variant field creates edge.

pub struct TupleType;

pub enum EnumWithTuple {
    Variant(TupleType),
}
