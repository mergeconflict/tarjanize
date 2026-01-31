//! Test: Where clause in impl - the bound type itself is not an anchor.
//!
//! The anchor here is MyType (the self type).

pub trait LocalTrait {}
pub struct MyType<T>(T);

impl<T> MyType<T>
where
    T: LocalTrait,
{
    pub fn constrained() {}
}
