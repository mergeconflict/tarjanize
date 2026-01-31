//! Test: Built-in attributes are not dependency targets.

pub struct S;

#[inline]
#[must_use]
pub fn caller() -> S {
    S
}
