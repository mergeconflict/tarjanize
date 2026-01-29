//! Module traversal and symbol extraction for tarjanize.
//!
//! This module orchestrates the extraction of symbols from a module hierarchy.
//! It recursively walks through modules, delegating the actual symbol extraction
//! to specialized modules:
//! - `module_defs`: Extracts ModuleDef items (functions, structs, etc.)
//! - `impls`: Extracts impl blocks
//!
//! The extracted symbols are organized into a tree structure matching the
//! module hierarchy of the source code.

use std::collections::HashMap;

use ra_ap_base_db::VfsPath;
use ra_ap_hir::{Module as HirModule, ModuleDef, Semantics};
use ra_ap_ide_db::RootDatabase;
use tarjanize_schemas::{Module, Symbol};
use tracing::debug_span;

use crate::impls::extract_impl;
use crate::module_defs::extract_module_def;

/// Extract a module and its contents as a Module.
///
/// Returns a tuple of (module_name, module). The name is stored separately
/// because the schema uses HashMaps keyed by name rather than storing the
/// name inside the Module struct.
///
/// This function recursively processes a module, extracting:
/// - All symbols (ModuleDefs and impl blocks) with their dependencies
/// - Child submodules
///
/// The crate_name parameter is used to build fully-qualified paths for
/// dependencies. The crate_root is the directory containing the crate's
/// lib.rs/main.rs, used to compute relative file paths for symbols.
pub(crate) fn extract_module(
    sema: &Semantics<'_, RootDatabase>,
    crate_root: &VfsPath,
    module: &HirModule,
    crate_name: &str,
) -> (String, Module) {
    let db = sema.db;

    // Module name: use crate name for root module, otherwise use module name.
    let module_name = module
        .name(db)
        .map(|n| n.as_str().to_owned())
        .unwrap_or_else(|| crate_name.to_owned());

    let _span = debug_span!("extract_module", %module_name).entered();

    // Extract symbols defined directly in this module.
    let symbols = extract_module_symbols(sema, crate_root, module);

    // Recursively extract child modules into a HashMap keyed by name.
    let submodules: HashMap<String, Module> = module
        .children(db)
        .map(|child| extract_module(sema, crate_root, &child, crate_name))
        .collect();

    (
        module_name,
        Module {
            symbols,
            submodules,
        },
    )
}

/// Extract all symbols from a module (ModuleDefs and impl blocks).
///
/// Returns a HashMap keyed by symbol name. Multiple impl blocks with the same
/// signature (e.g., two `impl Foo` blocks) are merged into a single Symbol
/// with combined cost and merged dependencies.
//
// TODO: Experiment with parallelization using rayon. Currently crates are
// processed in parallel, but symbol extraction within modules is sequential.
// Considerations:
// - Semantics must be Send + Sync for par_iter()
// - May cause lock contention on the database
// - Overhead may outweigh benefit for small modules.

fn extract_module_symbols(
    sema: &Semantics<'_, RootDatabase>,
    crate_root: &VfsPath,
    module: &HirModule,
) -> HashMap<String, Symbol> {
    let db = sema.db;
    let mut symbols = HashMap::new();

    // Extract module-level declarations (functions, structs, consts, etc.).
    symbols.extend(
        module
            .declarations(db)
            .into_iter()
            .filter_map(|def| extract_module_def(sema, crate_root, def)),
    );

    // Extract macro_rules! macros separately - they're "legacy macros" in
    // rust-analyzer terminology and aren't included in declarations().
    symbols.extend(module.legacy_macros(db).into_iter().filter_map(|mac| {
        extract_module_def(sema, crate_root, ModuleDef::Macro(mac))
    }));

    // Extract impl blocks separately - they're not ModuleDefs but are still
    // important compilation units with their own dependencies.
    module
        .impl_defs(db)
        .into_iter()
        .filter_map(|impl_| extract_impl(sema, crate_root, impl_))
        .for_each(|(name, symbol)| {
            // There can be multiple impl blocks with the same name/signature.
            // We merge them by summing costs and combining dependencies.
            symbols
                .entry(name)
                .and_modify(|existing| {
                    existing.cost += symbol.cost;
                    existing.dependencies.extend(symbol.dependencies.clone());
                })
                .or_insert(symbol);
        });

    symbols
}

#[cfg(test)]
mod tests {
    use ra_ap_hir::{Crate, Semantics, attach_db};
    use ra_ap_ide_db::RootDatabase;
    use ra_ap_test_fixture::WithFixture;

    use super::extract_module;
    use crate::file_path;

    /// Multiple impl blocks with the same signature should be merged,
    /// combining their costs and dependencies.
    #[test]
    fn test_impl_merging() {
        let db = RootDatabase::with_files(
            r#"
//- /lib.rs crate:test_crate
pub struct Foo;
pub struct DepA;
pub struct DepB;

impl Foo {
    pub fn method_a(&self) -> DepA { DepA }
}

impl Foo {
    pub fn method_b(&self) -> DepB { DepB }
}
"#,
        );

        attach_db(&db, || {
            let krate = Crate::all(&db)
                .into_iter()
                .find(|k| {
                    k.display_name(&db)
                        .is_some_and(|n| n.to_string() == "test_crate")
                })
                .expect("test_crate not found");

            let crate_root = file_path(&db, krate.root_file(&db))
                .expect("crate root file")
                .parent()
                .expect("crate root parent");

            let sema = Semantics::new(&db);
            let root_module = krate.root_module(&db);
            let (_, module) =
                extract_module(&sema, &crate_root, &root_module, "test_crate");

            // Should have exactly one "impl Foo" symbol (merged)
            let symbol = module
                .symbols
                .get("impl Foo")
                .expect("Should have merged 'impl Foo' symbol");

            // Dependencies should include both DepA and DepB
            assert!(
                symbol.dependencies.contains("test_crate::DepA"),
                "Should depend on DepA"
            );
            assert!(
                symbol.dependencies.contains("test_crate::DepB"),
                "Should depend on DepB"
            );
        });
    }
}
