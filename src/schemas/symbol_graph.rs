use std::collections::HashSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Root structure representing the entire symbol graph of a workspace.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SymbolGraph {
    /// Name of the workspace
    pub workspace_name: String,

    /// All crates in the workspace. Crates are represented as their
    /// respective root modules
    pub crates: Vec<Module>,

    /// Dependency edges between symbols
    pub edges: HashSet<Edge>,
}

/// A module (or crate root) containing symbols and submodules.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Module {
    /// Module name
    pub name: String,

    /// Symbols defined in this module. A symbol is either a module-level
    /// definition (ra_ap_hir::ModuleDef) or an impl block (ra_ap_hir::Impl)
    pub symbols: Vec<Symbol>,

    /// Child modules, omit if empty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submodules: Option<Vec<Module>>,
}

/// A symbol in the crate - either a module-level definition or an impl block.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Symbol {
    /// Symbol name. Impl names are as they appear in the source,
    /// e.g. 'impl Trait for Type'
    pub name: String,

    /// Path to symbol's file relative to crate root
    pub file: String,

    /// Approximate cost (unitless) of compiling this symbol. Given two
    /// symbols A and B, the estimated cost of compiling A and B in series
    /// is A.cost + B.cost, and the estimated cost of compiling A and B in
    /// parallel is max(A.cost, B.cost)
    #[schemars(range(min = 0.0))]
    pub cost: f64,

    /// Symbol-specific fields depending on whether this is a module
    /// definition or impl block
    #[serde(flatten)]
    pub kind: SymbolKind,
}

/// Discriminated union for symbol-specific fields.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    /// Module-level definition, analogous to ra_ap_hir::ModuleDef
    ModuleDef {
        /// Symbol kind (function, struct, enum, trait, etc.)
        kind: String,

        /// Visibility specifier (pub, pub(crate), etc.). Private if absent
        #[serde(skip_serializing_if = "Option::is_none")]
        visibility: Option<String>,
    },

    /// Impl block, analogous to ra_ap_hir::Impl. At least one of self_type or
    /// trait must be present to satisfy the orphan rule.
    Impl {
        /// Fully qualified path to self type. Absent if the self type is in
        /// another crate, or for blanket impl.
        #[serde(skip_serializing_if = "Option::is_none")]
        self_type: Option<String>,

        /// Fully qualified path to trait. Absent if the trait is in another
        /// crate, or for inherent impl.
        #[serde(rename = "trait", skip_serializing_if = "Option::is_none")]
        trait_: Option<String>,
    },
}

/// A dependency edge between two symbols.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct Edge {
    /// Fully qualified path of the dependent symbol
    pub from: String,

    /// Fully qualified path of the dependency symbol
    pub to: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};

    /// Verifies that the JSON Schema generated from Rust types matches the
    /// checked-in schema file.
    ///
    /// Why: The Rust structs are the source of truth for the schema, but we
    /// also want the JSON Schema file in the repository for documentation
    /// and for use by external tools (validators, code generators, etc.).
    /// This test ensures they stay in sync - if you change the Rust types,
    /// the test will fail until you regenerate the schema file.
    ///
    /// To update the golden file after changing the Rust types:
    ///   GENERATE_GOLDEN=1 cargo nextest run schema_matches_golden_file
    #[test]
    fn schema_matches_golden_file() -> Result<()> {
        let schema = schemars::schema_for!(SymbolGraph);

        let golden_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/schemas/symbol_graph.schema.json"
        );

        // When GENERATE_GOLDEN is set, overwrite the golden file instead of
        // comparing. This is the mechanism for updating the schema after
        // intentional changes to the Rust types.
        if std::env::var("GENERATE_GOLDEN").is_ok() {
            let json = serde_json::to_string_pretty(&schema)?;
            std::fs::write(golden_path, &json)?;
            println!("Updated golden file: {}", golden_path);
            return Ok(());
        }

        let golden_json = std::fs::read_to_string(golden_path)
            .context("Golden file not found. Run with GENERATE_GOLDEN=1 to create it.")?;

        // Compare Schema objects rather than JSON strings. This makes the test
        // insensitive to formatting differences (key ordering, whitespace).
        let expected: schemars::Schema = serde_json::from_str(&golden_json)
            .context("Failed to parse golden file as JSON Schema")?;

        assert_eq!(
            schema, expected,
            "Schema doesn't match golden file. Run with GENERATE_GOLDEN=1 to update."
        );

        Ok(())
    }
}
