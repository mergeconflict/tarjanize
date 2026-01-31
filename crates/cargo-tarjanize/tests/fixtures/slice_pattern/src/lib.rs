//! Test: Slice pattern creates edge.

pub struct Element;

pub fn caller(slice: &[Element]) {
    if let [first, ..] = slice {
        let _ = first;
    }
}
