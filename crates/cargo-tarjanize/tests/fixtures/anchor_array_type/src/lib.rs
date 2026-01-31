//! Test: Impl for array type - array is fundamental.

pub trait MyTrait {}
pub struct Element;

impl MyTrait for [Element; 3] {}
