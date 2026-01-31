//! Test: Pattern match on enum variant normalizes to Enum.

pub enum MyEnum {
    Variant(i32),
}

pub fn caller(e: MyEnum) -> i32 {
    match e {
        MyEnum::Variant(x) => x,
    }
}
