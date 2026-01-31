//! Test: Union field type creates an edge.

#[derive(Copy, Clone)]
pub struct FieldType;

pub union MyUnion {
    pub field: FieldType,
}
