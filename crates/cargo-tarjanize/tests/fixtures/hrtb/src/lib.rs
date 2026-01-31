//! Test: Higher-ranked trait bounds create edge.

pub trait Tr<'a> {}

pub fn hrtb<T>(_: T)
where
    T: for<'a> Tr<'a>,
{
}
