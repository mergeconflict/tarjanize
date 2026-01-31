//! Test: Code using only std has no local dependencies.

pub fn caller() -> Vec<i32> {
    vec![1, 2, 3]
}
