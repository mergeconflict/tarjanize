//! Test case for structs defined inside closures.
//!
//! This mimics the pattern from `tokio::select!` which generates internal
//! structs like `__tokio_select_util::Mask` inside closures.

/// Function containing a closure with an internal struct.
pub fn run() {
    let _closure = || {
        // Structs can be defined inside closure bodies.
        // This creates a DefPath like: crate::run::{{closure}}::InternalStruct
        mod __internal_util {
            pub struct Mask;
        }
        __internal_util::Mask
    };
}
