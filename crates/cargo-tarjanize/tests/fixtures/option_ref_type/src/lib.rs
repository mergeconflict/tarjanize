//! Test: Option<&T> creates edge to inner type.

pub struct Inner;

pub fn takes_option(_: Option<&Inner>) {}
