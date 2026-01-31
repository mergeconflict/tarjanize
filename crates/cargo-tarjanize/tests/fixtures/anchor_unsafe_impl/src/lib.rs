//! Test: Unsafe impl has same anchor rules.

pub unsafe trait UnsafeTrait {}
pub struct MyType;

unsafe impl UnsafeTrait for MyType {}
