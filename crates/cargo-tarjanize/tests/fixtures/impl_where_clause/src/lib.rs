//! Test: Impl where clause creates edge.

pub trait WhereTrait {}
pub trait ImplTrait {}
pub struct ImplType<T>(T);

impl<T> ImplTrait for ImplType<T> where T: WhereTrait {}
