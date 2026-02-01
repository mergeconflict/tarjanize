//! Test: Visibility extraction.

#![expect(dead_code)]

pub fn public_fn() {}
fn private_fn() {}
pub(crate) fn crate_fn() {}
