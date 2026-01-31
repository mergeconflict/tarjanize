//! Test: Const in pattern matching creates edge.

pub const MAGIC: i32 = 42;

pub fn caller(x: i32) -> bool {
    matches!(x, MAGIC)
}
