//! Test: Proc macro attribute creates edges.
//!
//! Uses derive(Debug) which generates code referencing the struct fields.

#[derive(Debug)]
pub struct FieldType;

#[derive(Debug)]
pub struct Container {
    pub field: FieldType,
}
