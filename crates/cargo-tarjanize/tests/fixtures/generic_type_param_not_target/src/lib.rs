//! Test: Generic type parameters are not dependency targets.

pub struct S;

pub fn generic<T>(_: T) -> S {
    S
}
