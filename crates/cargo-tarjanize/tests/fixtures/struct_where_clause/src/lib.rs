//! Test: Struct where clause creates edge.

pub trait WhereTrait {}

pub struct StructWithWhere<T>
where
    T: WhereTrait,
{
    pub value: T,
}
