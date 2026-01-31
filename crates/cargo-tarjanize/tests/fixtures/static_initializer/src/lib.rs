//! Test: Static initializer expression creates edge.

pub const fn helper() -> i32 {
    42
}

pub static MY_STATIC: i32 = helper();
