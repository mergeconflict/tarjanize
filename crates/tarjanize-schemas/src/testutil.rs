//! Shared proptest strategies for schema tests.
//!
//! Why: keeps test-only generators centralized so production types stay clean
//! while tests reuse consistent data shapes.

use proptest::prelude::*;

/// Strategy for generating arbitrary identifier-like names.
///
/// Why: ensures deterministic, bounded names for serde roundtrip tests.
pub fn arb_name() -> impl Strategy<Value = String> {
    "[a-z_][a-z0-9_]{0,19}"
}

/// Strategy for generating arbitrary symbol paths (e.g., `crate::module::Symbol`).
///
/// Why: tests need realistic-looking paths to validate dependency handling.
pub fn arb_path() -> impl Strategy<Value = String> {
    (arb_name(), arb_name())
        .prop_map(|(crate_name, symbol)| format!("{crate_name}::{symbol}"))
}
