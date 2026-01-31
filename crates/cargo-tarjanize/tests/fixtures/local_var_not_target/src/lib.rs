//! Test: Local variables are not edge targets.

pub struct S;

pub fn caller() {
    let x = S;
    let _ = x;
}
