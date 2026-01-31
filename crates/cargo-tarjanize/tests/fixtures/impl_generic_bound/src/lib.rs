//! Test: Impl generic bound creates edge.

pub trait BoundTrait {}
pub trait ImplTrait {}
pub struct ImplType<T>(T);

impl<T: BoundTrait> ImplTrait for ImplType<T> {}
