//! Test: Type ascription in struct pattern creates edge.

pub struct S {
    pub x: i32,
}

pub fn caller() {
    let S { x: _ }: S = S { x: 1 };
}
