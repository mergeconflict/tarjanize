//! Test: Symbols with special characters (underscores, numbers).

pub fn _underscore_fn() -> i32 {
    1
}

pub fn fn_with_123() -> i32 {
    123
}

pub fn __double_underscore() -> i32 {
    2
}

pub const _CONST_123: i32 = 456;
