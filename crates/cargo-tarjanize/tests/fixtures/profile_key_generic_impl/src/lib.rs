//! Test: Generic impl has correct profile key (generics not in path).

pub struct Container<T>(T);

impl<T> Container<T> {
    pub fn get(&self) -> &T {
        &self.0
    }
}
