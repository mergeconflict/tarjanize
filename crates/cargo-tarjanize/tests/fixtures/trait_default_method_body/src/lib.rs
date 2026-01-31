//! Test: Trait default method body creates edge.

pub fn helper() {}

pub trait MyTrait {
    fn default_method(&self) {
        helper();
    }
}
