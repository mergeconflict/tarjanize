//! Test: Function pointer type creates edges to param and return types.

pub struct A;
pub struct B;

pub fn takes_fn_ptr(_: fn(A) -> B) {}
