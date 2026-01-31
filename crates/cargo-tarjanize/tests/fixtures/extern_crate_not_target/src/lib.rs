//! Test: extern crate is not a dependency target.

extern crate std;

pub struct S;

pub fn caller() -> S {
    S
}
