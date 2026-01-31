//! Test: Impl in submodule has correct path prefix.

pub mod submod {
    pub struct MyType;

    impl MyType {
        pub fn method(&self) {}
    }
}
