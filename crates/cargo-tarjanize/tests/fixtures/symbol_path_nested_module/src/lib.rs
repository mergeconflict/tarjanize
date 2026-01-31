//! Test: Nested module path extraction.

pub mod outer {
    pub mod inner {
        pub fn nested_fn() {}
    }
}
