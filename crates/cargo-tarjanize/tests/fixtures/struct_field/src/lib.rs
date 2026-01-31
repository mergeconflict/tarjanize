//! Test: Struct field type creates an edge.

pub struct TargetType;

pub struct ContainerType {
    pub field: TargetType,
}
