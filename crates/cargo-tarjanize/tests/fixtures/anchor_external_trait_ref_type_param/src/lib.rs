//! Test: External trait with reference to local type param.

pub struct LocalType;

// AsRef is external, LocalType is local and becomes an anchor through &LocalType
impl AsRef<LocalType> for i32 {
    fn as_ref(&self) -> &LocalType {
        &LocalType
    }
}
