//! Test: Function with internal statics (simulating tracing callsite pattern).
//!
//! Note: We don't use actual tracing crate to keep fixtures minimal.
//! This tests the general pattern of macro-generated statics.

pub fn traced_function() -> i32 {
    // Simulating what tracing macros do: create a static for the callsite
    static CALLSITE: &str = "traced_function";
    let _ = CALLSITE;
    42
}
