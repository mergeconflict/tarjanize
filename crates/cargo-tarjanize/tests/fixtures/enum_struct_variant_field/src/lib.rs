//! Test: Enum struct variant field creates edge.

pub struct FieldType;

pub enum EnumWithStruct {
    Variant { field: FieldType },
}
