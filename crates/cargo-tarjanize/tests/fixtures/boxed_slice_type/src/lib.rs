//! Test: Box<[T]> creates edge to element type.

pub struct Element;

pub fn takes_boxed_slice(_: Box<[Element]>) {}
