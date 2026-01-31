//! Test: Generic bounds in impl - the bound trait itself is not an anchor.
//!
//! The anchor here is MyType (the self type), not LocalTrait (the bound).

pub trait LocalTrait {}
pub struct MyType<T>(T);

impl<T: LocalTrait> MyType<T> {
    pub fn constrained() {}
}
