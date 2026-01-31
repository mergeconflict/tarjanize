//! Test: crate:: keyword is not a dependency target.

pub struct S;

pub fn caller() -> crate::S {
    crate::S
}
