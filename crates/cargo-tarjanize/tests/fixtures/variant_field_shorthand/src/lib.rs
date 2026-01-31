//! Test: Variant field shorthand normalizes to Enum.

pub enum MyEnum {
    Variant { value: i32 },
}

pub fn caller(e: MyEnum) -> i32 {
    match e {
        MyEnum::Variant { value } => value,
    }
}
