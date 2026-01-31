//! Test: ByUse expression (`.use` postfix syntax) creates edges.
//!
//! The `.use` postfix syntax is an experimental feature for ergonomic clones.
//! When used, THIR represents it as `ExprKind::ByUse { expr, .. }`.

#![feature(ergonomic_clones)]

use std::clone::UseCloned;

#[derive(Clone)]
pub struct Data {
    pub value: i32,
}

impl UseCloned for Data {}

pub fn takes_ownership(data: Data) -> i32 {
    data.value
}

pub fn uses_by_use(data: &Data) -> i32 {
    // The `.use` syntax clones data and passes ownership.
    // This should create an edge to `takes_ownership`.
    takes_ownership(data.use)
}
