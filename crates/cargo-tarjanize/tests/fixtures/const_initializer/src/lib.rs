//! Test: Const initializer function call creates an edge.

const fn helper() -> i32 { 42 }

pub const MY_CONST: i32 = helper();
