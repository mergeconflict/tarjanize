//! Test: Box<LocalType> - LocalType is the anchor.

pub trait MyTrait {}
pub struct LocalType;

impl MyTrait for Box<LocalType> {}
