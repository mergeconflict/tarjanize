//! Test: Function where clause creates edge.

pub trait WhereTrait {}

pub fn fn_with_where<T>(_x: T)
where
    T: WhereTrait,
{
}
