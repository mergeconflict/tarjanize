//! Test: Enum variant via path normalizes to the enum.

pub enum MyEnum { Variant }

pub fn uses_variant() -> MyEnum { MyEnum::Variant }
