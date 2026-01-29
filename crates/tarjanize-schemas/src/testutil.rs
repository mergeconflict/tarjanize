//! Shared proptest strategies for schema tests.

use proptest::prelude::*;

/// Strategy for generating arbitrary identifier-like names.
pub fn arb_name() -> impl Strategy<Value = String> {
    "[a-z_][a-z0-9_]{0,19}"
}

/// Strategy for generating arbitrary symbol paths (e.g., `crate::module::Symbol`).
pub fn arb_path() -> impl Strategy<Value = String> {
    (arb_name(), arb_name())
        .prop_map(|(crate_name, symbol)| format!("{crate_name}::{symbol}"))
}
