//! Test: Associated function via path normalizes to the impl.

pub struct MyType;

impl MyType {
    pub fn assoc_fn() {}
}

pub fn caller() { MyType::assoc_fn(); }
