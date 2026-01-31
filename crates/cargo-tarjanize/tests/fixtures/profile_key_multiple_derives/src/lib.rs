//! Test: Multiple derives are disambiguated correctly.

#[derive(Debug, Clone, Default)]
pub struct MyType {
    pub field: i32,
}
