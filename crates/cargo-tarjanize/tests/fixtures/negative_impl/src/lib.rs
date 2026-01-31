//! Test: Negative impl creates edges.

#![feature(auto_traits)]
#![feature(negative_impls)]

pub auto trait MyAuto {}
pub struct NoAuto;

impl !MyAuto for NoAuto {}
