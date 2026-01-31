//! Test: External trait with local type param as anchor.

pub struct LocalType;

// From is external, but LocalType is local and becomes an anchor
impl From<LocalType> for i32 {
    fn from(_: LocalType) -> i32 {
        0
    }
}
