//! Test: Raw pointer types create edges to pointee type.

pub struct Pointee;

pub fn takes_const_ptr(_: *const Pointee) {}
pub fn takes_mut_ptr(_: *mut Pointee) {}
