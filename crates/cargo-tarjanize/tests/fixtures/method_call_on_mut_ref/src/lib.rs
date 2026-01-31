//! Test: Method call on &mut T normalizes to impl.

pub struct MyType;

impl MyType {
    pub fn method(&mut self) {}
}

pub fn caller(t: &mut MyType) {
    t.method();
}
