//! Test: Unsafe binder expressions create edges.
//!
//! Unsafe binders (`unsafe<'a> T`) are an experimental feature.
//! THIR uses `ExprKind::WrapUnsafeBinder`, `ExprKind::PlaceUnwrapUnsafeBinder`,
//! and `ExprKind::ValueUnwrapUnsafeBinder` for these operations.

#![feature(unsafe_binders)]
#![expect(incomplete_features)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

pub struct Container<'a> {
    pub data: &'a i32,
}

pub fn get_value(c: &Container<'_>) -> i32 {
    *c.data
}

pub fn wrap_container(data: &i32) -> unsafe<'a> Container<'a> {
    // wrap_binder! creates an unsafe binder, represented as WrapUnsafeBinder in THIR.
    unsafe { wrap_binder!(Container { data }) }
}

pub fn unwrap_and_use(bound: unsafe<'a> Container<'a>) -> i32 {
    // unwrap_binder! unwraps the unsafe binder.
    // This is represented as PlaceUnwrapUnsafeBinder or ValueUnwrapUnsafeBinder in THIR.
    unsafe {
        let container = unwrap_binder!(bound);
        get_value(&container)
    }
}
