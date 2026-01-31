//! Test: Multiple impls on same type use `{{impl}}[N]` disambiguation.

pub struct MyType;

pub trait TraitA {
    fn method_a(&self);
}

pub trait TraitB {
    fn method_b(&self);
}

impl TraitA for MyType {
    fn method_a(&self) {}
}

impl TraitB for MyType {
    fn method_b(&self) {}
}

impl MyType {
    pub fn inherent_method(&self) {}
}
