//! Test: Trait impl uses `{{impl}}` notation in profile key.

pub struct MyType;

pub trait MyTrait {
    fn trait_method(&self);
}

impl MyTrait for MyType {
    fn trait_method(&self) {}
}
