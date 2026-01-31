//! Test: Vec<T> is NOT fundamental - no local anchor for Vec<LocalType>.

pub trait MyTrait {}
pub struct LocalType;

// This impl has no local anchor because Vec is not fundamental and is external
// The orphan rule would reject this, but we still need to extract anchors correctly
impl MyTrait for Vec<LocalType> {}
