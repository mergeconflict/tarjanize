//! Test: Struct init shorthand creates edge.

pub struct Target;

pub struct Container {
    pub field: Target,
}

pub fn caller() -> Container {
    let field = Target;
    Container { field }
}
