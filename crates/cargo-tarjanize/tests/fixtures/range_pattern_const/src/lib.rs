//! Test: Range patterns with const bounds create edges to those consts.
//!
//! Pattern matching like `MIN..=MAX` should capture dependencies on the
//! constants used as range bounds.

pub const MIN: i32 = 0;
pub const MAX: i32 = 100;

pub fn in_range(x: i32) -> bool {
    // Using direct match instead of matches! macro
    match x {
        MIN..=MAX => true,
        _ => false,
    }
}
