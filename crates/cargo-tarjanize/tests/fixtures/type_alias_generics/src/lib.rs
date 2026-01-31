//! Test: Type alias with generics creates edges to all types.

pub struct Inner;
pub struct Wrapper<T>(T);

pub type MyAlias = Wrapper<Inner>;
