//! Test: dyn Trait + auto-trait creates edge to the main trait.
//!
//! Note: Rust only allows one non-auto trait in a dyn type. We test with
//! `dyn Tr + Send`, which is valid (Send is an auto-trait).

pub trait Tr {}

pub fn takes_dyn(_: &(dyn Tr + Send)) {}
