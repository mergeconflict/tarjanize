//! Test: Symbols with unicode in names.

#![allow(non_snake_case)]

pub fn 日本語() -> &'static str {
    "Japanese"
}

pub struct Ελληνικά;

pub const UNICODE_CONST: &str = "crab";
