//! Test: Dependencies inside const blocks are captured.
//!
//! Const blocks (`const { ... }`) can contain references to other items.
//! These dependencies must be captured.

pub const BASE_VALUE: i32 = 10;

pub const fn helper() -> i32 {
    5
}

pub fn uses_const_block() -> i32 {
    // This const block references BASE_VALUE and helper().
    const { BASE_VALUE + helper() }
}
