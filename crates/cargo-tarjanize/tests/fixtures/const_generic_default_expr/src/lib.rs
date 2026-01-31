//! Test: Const generic default expression creates edge.

pub const DEFAULT_SIZE: usize = 10;

pub struct Buffer<const N: usize = DEFAULT_SIZE>;
