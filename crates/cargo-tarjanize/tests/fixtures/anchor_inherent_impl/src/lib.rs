//! Test: Inherent impl anchor is self type.

pub struct MyType;

impl MyType {
    pub fn method(&self) {}
}
