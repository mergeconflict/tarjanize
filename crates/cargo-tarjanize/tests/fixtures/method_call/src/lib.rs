//! Test: Method call creates an edge to the impl.

pub struct MyType {
    pub value: i32,
}

impl MyType {
    pub fn method(&self) -> i32 {
        self.value
    }
}

pub fn caller(t: MyType) -> i32 {
    t.method()
}
