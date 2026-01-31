//! Test: `const { ... }` cost aggregates to containing function.

pub fn with_const_block() -> usize {
    const { std::mem::size_of::<i64>() }
}
