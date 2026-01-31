//! Test: Const in range patterns creates dependency.

pub const MIN_VALUE: i32 = 0;
pub const MAX_VALUE: i32 = 100;

pub fn classify(value: i32) -> &'static str {
    match value {
        MIN_VALUE..=MAX_VALUE => "in range",
        _ => "out of range",
    }
}
