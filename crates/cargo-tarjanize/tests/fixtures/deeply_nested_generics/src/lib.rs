//! Test: Deeply nested generics create edges to all types.

pub struct A;
pub struct B;
pub struct C;
pub struct Wrapper<T>(T);

pub fn nested() -> Wrapper<Wrapper<(A, Wrapper<B>, C)>> {
    todo!()
}
