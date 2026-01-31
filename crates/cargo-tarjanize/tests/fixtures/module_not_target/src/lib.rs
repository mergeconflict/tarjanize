//! Test: Modules are not edge targets; edges go to items within modules.

pub mod m {
    pub struct S;
}

pub fn caller() -> m::S { m::S }
