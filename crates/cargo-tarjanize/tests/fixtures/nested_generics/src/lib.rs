//! Test: Nested generic types create edges to all types.

pub struct Outer<T>(T);
pub struct Inner;

pub fn nested() -> Outer<Outer<Inner>> {
    todo!()
}
