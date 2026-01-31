//! Test: Turbofish syntax creates edges to type and function.

pub struct S;

pub fn generic<T>() {}

pub fn caller() {
    generic::<S>();
}
