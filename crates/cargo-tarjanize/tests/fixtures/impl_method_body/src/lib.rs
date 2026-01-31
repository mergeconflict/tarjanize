//! Test: Impl method body creates edge.

pub fn helper() {}

pub struct MyType;

impl MyType {
    pub fn method(&self) {
        helper();
    }
}
