//! Test: impl<T> Type<T> - generic inherent impl.

pub struct Container<T>(T);

impl<T> Container<T> {
    pub fn new(value: T) -> Self {
        Container(value)
    }
}
