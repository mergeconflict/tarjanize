//! Test: Function body type annotation creates edge.

pub struct BodyType;

pub fn fn_uses_body() {
    let _x: BodyType = BodyType;
}
