//! Cost model schema for predicting synthetic crate compilation times.
//!
//! The `CostModel` stores fitted regression coefficients from MAGSAC++
//! and is serialized to JSON for consumption by downstream tools
//! (condense, viz). Lives in `tarjanize-schemas` because it's a shared
//! serialized data format, not tied to any single phase.
//!
//! Why: keeping the schema in this crate ensures all phases share the same
//! serialization contract without re-defining coefficients.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Serializable cost model for predicting synthetic crate wall times.
///
/// Stores the coefficients from the fitted 3-variable no-intercept model:
/// ```text
/// wall = coeff_attr * attr + coeff_meta * meta + coeff_other * other
/// ```
///
/// For synthetic crates (after condensation), the caller computes:
/// - `attr`: sum of per-symbol event times across the partition
/// - `meta`: max of constituent targets' `metadata_decode_*` event sums
/// - `other`: max of constituent targets' non-metadata event sums
///
/// The max-constituent heuristic works because a synthetic crate inherits
/// at least as much metadata/other overhead as its largest constituent.
///
/// Named fields ensure schema mismatches fail at deserialization time.
///
/// Why: condense and schedule need a stable, portable model definition that
/// can be persisted and reused across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Coefficient for symbol-attributed cost.
    pub coeff_attr: f64,
    /// Coefficient for metadata decode cost (`metadata_decode_*` events).
    pub coeff_meta: f64,
    /// Coefficient for remaining unattributed cost (non-metadata events).
    pub coeff_other: f64,
    /// R-squared of the model fit.
    pub r_squared: f64,
    /// MAGSAC++ inlier threshold for classifying outliers.
    pub inlier_threshold: f64,
}

impl CostModel {
    /// Predicts wall time for a synthetic crate.
    ///
    /// The three inputs mirror the original model's predictors:
    /// - `attr`: sum of symbol-attributed event times
    /// - `meta`: metadata decode cost (max of constituent targets)
    /// - `other`: non-metadata unattributed cost (max of constituent targets)
    ///
    /// Why: this encapsulates the fitted regression so callers don't reimplement
    /// the formula and drift from the stored coefficients.
    pub fn predict(&self, attr: f64, meta: f64, other: f64) -> f64 {
        self.coeff_attr * attr
            + self.coeff_meta * meta
            + self.coeff_other * other
    }
}

/// Loads a `CostModel` from a JSON file.
///
/// Why: downstream tools need a simple, standard way to rehydrate the model
/// from the persisted schema without custom parsing logic.
pub fn load_cost_model(path: &Path) -> std::io::Result<CostModel> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    serde_json::from_reader(reader)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
