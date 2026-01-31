//! Test: Slice types create edges to element type.

pub struct Element;

pub fn takes_slice(_: &[Element]) {}
pub fn takes_mut_slice(_: &mut [Element]) {}
