//! Test: Tuple field access (.0, .1) is not a dependency target.

pub struct S(pub i32);

pub fn caller(s: S) -> i32 {
    s.0
}
