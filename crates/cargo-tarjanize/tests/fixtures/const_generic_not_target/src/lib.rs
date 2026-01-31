//! Test: Const generic parameters are not dependency targets.

pub struct S;

pub fn generic<const N: usize>() -> S {
    S
}
