//! Test: Method call on &T normalizes to impl.

pub struct MyType;

impl MyType {
    pub fn method(&self) {}
}

pub fn caller(t: &MyType) {
    t.method();
}
