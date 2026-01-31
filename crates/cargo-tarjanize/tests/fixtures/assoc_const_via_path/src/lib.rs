//! Test: Type::CONST via path normalizes to impl.

pub struct MyType;

impl MyType {
    pub const VALUE: i32 = 42;
}

pub fn caller() -> i32 {
    MyType::VALUE
}
