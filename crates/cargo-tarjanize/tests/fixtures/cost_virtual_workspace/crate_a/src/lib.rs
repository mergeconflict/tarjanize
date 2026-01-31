//! Crate A in virtual workspace - depends on crate_b.

pub fn uses_b() -> i32 {
    crate_b::helper() + 1
}
