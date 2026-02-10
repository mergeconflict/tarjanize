//! Test: `pub use` re-exports are extracted as symbols.
//!
//! Re-exports have real compilation cost (path resolution, visibility checking)
//! and create dependency edges. Facade crates that consist entirely of re-exports
//! should still have symbols.

mod inner {
    pub struct Original;

    pub fn helper() {}
}

/// Re-export a struct from a submodule.
pub use inner::Original;

/// Re-export a function from a submodule.
pub use inner::helper;
