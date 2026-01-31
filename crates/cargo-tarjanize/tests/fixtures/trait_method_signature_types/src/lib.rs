//! Test: Types in trait method signatures create edges to trait.

pub struct ParamType;
pub struct ReturnType;

pub trait MyTrait {
    fn method(&self, param: ParamType) -> ReturnType;
}
