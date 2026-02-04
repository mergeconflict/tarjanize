//! Crate A - depends on crate-b (hyphenated name).

pub fn uses_b() -> i32 {
    crate_b::helper() + 1
}
