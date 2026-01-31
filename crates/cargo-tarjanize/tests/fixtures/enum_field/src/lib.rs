//! Test: Enum variant field type creates an edge.

pub struct FieldType;

pub enum MyEnum {
    Variant(FieldType),
}
