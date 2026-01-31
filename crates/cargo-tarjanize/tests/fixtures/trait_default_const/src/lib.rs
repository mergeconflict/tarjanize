//! Test: Trait default const value creates edge.

pub const fn helper() -> i32 {
    42
}

pub trait MyTrait {
    const VALUE: i32 = helper();
}
