//! Test: Multiple type params in impl.

pub struct Container<A, B>(A, B);

impl<A, B> Container<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Container(a, b)
    }
}
