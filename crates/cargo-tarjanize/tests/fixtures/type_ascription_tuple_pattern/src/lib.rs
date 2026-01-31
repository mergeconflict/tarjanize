//! Test: Type ascription in tuple pattern creates edges.

pub struct A;
pub struct B;

pub fn caller() {
    let (_x, _y): (A, B) = (A, B);
}
