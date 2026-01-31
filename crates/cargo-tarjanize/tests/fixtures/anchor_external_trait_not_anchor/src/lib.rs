//! Test: External trait is not an anchor.

pub struct MyType;

// Clone is external, so it's not an anchor
impl Clone for MyType {
    fn clone(&self) -> Self {
        MyType
    }
}
