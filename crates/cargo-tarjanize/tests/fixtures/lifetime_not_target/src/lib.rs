//! Test: Lifetimes are not dependency targets.

pub struct S;

pub fn caller<'a>(_: &'a S) -> &'a S {
    &S
}
