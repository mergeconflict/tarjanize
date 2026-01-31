//! Test: impl<T> Trait for T - blanket impl has no local anchors (unless T is bounded).

pub trait MyTrait {}

impl<T> MyTrait for T {}
