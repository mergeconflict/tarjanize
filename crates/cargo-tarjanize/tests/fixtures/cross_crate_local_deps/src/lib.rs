//! Test: Only local crate dependencies are captured.
//!
//! This test uses std types but should not have edges to std.

pub struct LocalType;

pub fn caller() -> Vec<LocalType> {
    vec![LocalType]
}
