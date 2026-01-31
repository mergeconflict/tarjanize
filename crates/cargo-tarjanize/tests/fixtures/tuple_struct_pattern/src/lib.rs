//! Test: Tuple struct pattern creates edge.

pub struct Wrapper(pub i32);

pub fn caller(w: Wrapper) -> i32 {
    let Wrapper(x) = w;
    x
}
