//! Test: Cross-module path creates edge to item, not module.

pub mod inner {
    pub struct S;
}

pub fn caller() -> inner::S {
    inner::S
}
