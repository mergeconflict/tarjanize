//! Test: 'static lifetime is not a dependency target.

pub struct S;

pub fn caller(_: &'static S) {}
