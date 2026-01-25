use std::collections::HashSet;

use ra_ap_hir::{Impl, Module, ModuleDef, Semantics, symbols::SymbolCollector};
use ra_ap_ide_db::RootDatabase;

use crate::dependencies::{Dependency, find_dependencies, find_impl_dependencies, is_local, is_local_dep};
use crate::schemas::{Edge, Module as SchemaModule, Symbol, SymbolKind};

/// Extract a module and its contents as a SchemaModule.
///
/// This function recursively processes a module, extracting:
/// - All symbols (ModuleDefs and impl blocks)
/// - Child submodules
/// - Dependency edges between symbols
///
/// The crate_name parameter is used to build fully-qualified paths for edges.
pub fn extract_module(
    sema: &Semantics<'_, RootDatabase>,
    module: &Module,
    crate_name: &str,
    edges: &mut HashSet<Edge>,
) -> SchemaModule {
    let db = sema.db;

    // Module name: use crate name for root module, otherwise use module name.
    let module_name = module
        .name(db)
        .map(|n| n.as_str().to_owned())
        .unwrap_or_else(|| crate_name.to_owned());

    // Build the fully-qualified path to this module for edge construction.
    let module_path = build_module_path(db, module, crate_name);

    // Extract symbols defined directly in this module.
    let symbols = extract_symbols(sema, module, &module_path, edges);

    // Recursively extract child modules.
    let children: Vec<_> = module
        .children(db)
        .into_iter()
        .map(|child| extract_module(sema, &child, crate_name, edges))
        .collect();

    let submodules = if children.is_empty() {
        None
    } else {
        Some(children)
    };

    SchemaModule {
        name: module_name,
        symbols,
        submodules,
    }
}

/// Build the fully-qualified path to a module (e.g., "mycrate::foo::bar").
///
/// We build paths manually rather than using rust-analyzer's display methods
/// because we need stable, consistent identifiers for the dependency graph.
fn build_module_path(db: &RootDatabase, module: &Module, crate_name: &str) -> String {
    // Collect module names from root to this module by walking up the tree.
    let mut parts = Vec::new();
    let mut current = Some(*module);

    while let Some(m) = current {
        if let Some(name) = m.name(db) {
            parts.push(name.as_str().to_owned());
        }
        current = m.parent(db);
    }

    parts.reverse();

    if parts.is_empty() {
        crate_name.to_string()
    } else {
        format!("{}::{}", crate_name, parts.join("::"))
    }
}

/// Extract all symbols from a module (ModuleDefs and impl blocks).
fn extract_symbols(
    sema: &Semantics<'_, RootDatabase>,
    module: &Module,
    module_path: &str,
    edges: &mut HashSet<Edge>,
) -> Vec<Symbol> {
    let db = sema.db;
    let mut symbols = Vec::new();

    // Extract ModuleDef symbols using SymbolCollector.
    // SymbolCollector gathers all named items defined in this module.
    let collected = SymbolCollector::collect_module(db, *module);

    for sym in collected {
        if let Some(symbol) = extract_module_def(sema, &sym, module_path, edges) {
            symbols.push(symbol);
        }
    }

    // Extract impl blocks separately - they're not ModuleDefs but are still
    // important compilation units with their own dependencies.
    for impl_ in module.impl_defs(db) {
        if let Some(symbol) = extract_impl(sema, impl_, module_path, edges) {
            symbols.push(symbol);
        }
    }

    symbols
}

/// Extract a single ModuleDef as a Symbol.
///
/// Returns None for items we skip (e.g., built-in types).
fn extract_module_def(
    sema: &Semantics<'_, RootDatabase>,
    sym: &ra_ap_hir::symbols::FileSymbol,
    module_path: &str,
    edges: &mut HashSet<Edge>,
) -> Option<Symbol> {
    let db = sema.db;

    let kind = module_def_kind_str(&sym.def);

    // Skip built-in types - they're language primitives, not workspace items.
    if kind == "builtin" {
        return None;
    }

    let name = sym.name.as_str().to_owned();
    let symbol_path = format!("{}::{}", module_path, name);

    // TODO: Implement proper file path calculation relative to crate root.
    let file = String::new();

    // TODO: Implement cost calculation based on syntax size.
    let cost = 0.0;

    // Collect dependencies for this symbol, filtered to local workspace items.
    let deps = find_dependencies(sema, sym.def);
    for dep in deps {
        if is_local_dep(db, &dep) {
            if let Some(dep_path) = dependency_path(db, &dep) {
                edges.insert(Edge {
                    from: symbol_path.clone(),
                    to: dep_path,
                });
            }
        }
    }

    Some(Symbol {
        name,
        file,
        cost,
        kind: SymbolKind::ModuleDef {
            kind: kind.to_string(),
            visibility: None, // TODO: Extract visibility
        },
    })
}

/// Extract an impl block as a Symbol.
fn extract_impl(
    sema: &Semantics<'_, RootDatabase>,
    impl_: Impl,
    module_path: &str,
    edges: &mut HashSet<Edge>,
) -> Option<Symbol> {
    let db = sema.db;

    // Build the impl name (e.g., "impl Trait for Type" or "impl Type").
    // We extract the ADT name directly rather than using display(), which
    // requires an Edition parameter. For complex types (generics, references),
    // we fall back to a placeholder.
    let self_ty = impl_.self_ty(db);
    let self_ty_name = self_ty
        .as_adt()
        .and_then(|adt| match adt {
            ra_ap_hir::Adt::Struct(s) => Some(s.name(db).as_str().to_owned()),
            ra_ap_hir::Adt::Enum(e) => Some(e.name(db).as_str().to_owned()),
            ra_ap_hir::Adt::Union(u) => Some(u.name(db).as_str().to_owned()),
        })
        .unwrap_or_else(|| "(complex type)".to_string());

    let name = if let Some(trait_) = impl_.trait_(db) {
        format!("impl {} for {}", trait_.name(db).as_str(), self_ty_name)
    } else {
        format!("impl {}", self_ty_name)
    };

    let symbol_path = format!("{}::{}", module_path, name);

    // TODO: Implement proper file path calculation.
    let file = String::new();

    // TODO: Implement cost calculation.
    let cost = 0.0;

    // Extract self_type and trait paths for the schema.
    // These are used to enforce the orphan rule during analysis.
    let self_type_path = self_ty
        .as_adt()
        .and_then(|adt| module_def_path(db, &adt.into()));

    let trait_path = impl_
        .trait_(db)
        .and_then(|t| module_def_path(db, &t.into()));

    // Collect dependencies from the impl declaration (self type, trait).
    let impl_deps = find_impl_dependencies(db, impl_);
    for dep in impl_deps.deps {
        if is_local(db, &dep) {
            if let Some(dep_path) = module_def_path(db, &dep) {
                edges.insert(Edge {
                    from: symbol_path.clone(),
                    to: dep_path,
                });
            }
        }
    }

    // Collect dependencies from associated items (methods, consts, type aliases).
    // Per PLAN.md, we collapse these to the impl block - the impl depends on
    // whatever its methods depend on, but the methods aren't separate symbols.
    for item in impl_deps.items {
        let item_def: ModuleDef = match item {
            ra_ap_hir::AssocItem::Function(f) => f.into(),
            ra_ap_hir::AssocItem::Const(c) => c.into(),
            ra_ap_hir::AssocItem::TypeAlias(t) => t.into(),
        };
        let item_deps = find_dependencies(sema, item_def);
        for dep in item_deps {
            if is_local_dep(db, &dep) {
                if let Some(dep_path) = dependency_path(db, &dep) {
                    edges.insert(Edge {
                        from: symbol_path.clone(),
                        to: dep_path,
                    });
                }
            }
        }
    }

    Some(Symbol {
        name,
        file,
        cost,
        kind: SymbolKind::Impl {
            self_type: self_type_path,
            trait_: trait_path,
        },
    })
}

/// Returns the kind string for a ModuleDef.
fn module_def_kind_str(def: &ModuleDef) -> &'static str {
    match def {
        ModuleDef::Module(_) => "module",
        ModuleDef::Function(_) => "function",
        ModuleDef::Adt(adt) => match adt {
            ra_ap_hir::Adt::Struct(_) => "struct",
            ra_ap_hir::Adt::Enum(_) => "enum",
            ra_ap_hir::Adt::Union(_) => "union",
        },
        ModuleDef::Variant(_) => "variant",
        ModuleDef::Const(_) => "const",
        ModuleDef::Static(_) => "static",
        ModuleDef::Trait(_) => "trait",
        ModuleDef::TraitAlias(_) => "trait_alias",
        ModuleDef::TypeAlias(_) => "type_alias",
        ModuleDef::BuiltinType(_) => "builtin",
        ModuleDef::Macro(_) => "macro",
    }
}

/// Returns the fully-qualified path for a ModuleDef.
///
/// Returns None for items without clear paths (e.g., built-in types).
fn module_def_path(db: &RootDatabase, def: &ModuleDef) -> Option<String> {
    // Get the module containing this def.
    let module = match def {
        ModuleDef::Module(m) => Some(*m),
        ModuleDef::Function(f) => Some(f.module(db)),
        ModuleDef::Adt(adt) => Some(adt.module(db)),
        ModuleDef::Variant(v) => Some(v.module(db)),
        ModuleDef::Const(c) => Some(c.module(db)),
        ModuleDef::Static(s) => Some(s.module(db)),
        ModuleDef::Trait(t) => Some(t.module(db)),
        ModuleDef::TraitAlias(t) => Some(t.module(db)),
        ModuleDef::TypeAlias(t) => Some(t.module(db)),
        ModuleDef::BuiltinType(_) => None,
        ModuleDef::Macro(m) => Some(m.module(db)),
    }?;

    // Get the crate name.
    let krate = module.krate();
    let crate_name = krate
        .display_name(db)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(unnamed)".to_string());

    // Build the module path.
    let module_path = build_module_path(db, &module, &crate_name);

    // Get the item name.
    let name = match def {
        ModuleDef::Module(m) => m.name(db).map(|n| n.as_str().to_owned()),
        ModuleDef::Function(f) => Some(f.name(db).as_str().to_owned()),
        ModuleDef::Adt(adt) => match adt {
            ra_ap_hir::Adt::Struct(s) => Some(s.name(db).as_str().to_owned()),
            ra_ap_hir::Adt::Enum(e) => Some(e.name(db).as_str().to_owned()),
            ra_ap_hir::Adt::Union(u) => Some(u.name(db).as_str().to_owned()),
        },
        ModuleDef::Variant(v) => Some(v.name(db).as_str().to_owned()),
        ModuleDef::Const(c) => c.name(db).map(|n| n.as_str().to_owned()),
        ModuleDef::Static(s) => Some(s.name(db).as_str().to_owned()),
        ModuleDef::Trait(t) => Some(t.name(db).as_str().to_owned()),
        ModuleDef::TraitAlias(t) => Some(t.name(db).as_str().to_owned()),
        ModuleDef::TypeAlias(t) => Some(t.name(db).as_str().to_owned()),
        ModuleDef::BuiltinType(_) => None,
        ModuleDef::Macro(m) => Some(m.name(db).as_str().to_owned()),
    }?;

    Some(format!("{}::{}", module_path, name))
}

/// Returns the fully-qualified path for a Dependency.
///
/// This handles both ModuleDefs and Impls.
fn dependency_path(db: &RootDatabase, dep: &Dependency) -> Option<String> {
    match dep {
        Dependency::ModuleDef(def) => module_def_path(db, def),
        Dependency::Impl(impl_) => impl_path(db, impl_),
    }
}

/// Returns the fully-qualified path for an Impl.
///
/// Impl paths use the format "crate::module::impl Trait for Type" or
/// "crate::module::impl Type" for inherent impls.
fn impl_path(db: &RootDatabase, impl_: &ra_ap_hir::Impl) -> Option<String> {
    let module = impl_.module(db);
    let krate = module.krate();
    let crate_name = krate
        .display_name(db)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(unnamed)".to_string());

    let module_path = build_module_path(db, &module, &crate_name);

    // Build the impl name (e.g., "impl Trait for Type" or "impl Type").
    let self_ty = impl_.self_ty(db);
    let self_ty_name = self_ty
        .as_adt()
        .and_then(|adt| match adt {
            ra_ap_hir::Adt::Struct(s) => Some(s.name(db).as_str().to_owned()),
            ra_ap_hir::Adt::Enum(e) => Some(e.name(db).as_str().to_owned()),
            ra_ap_hir::Adt::Union(u) => Some(u.name(db).as_str().to_owned()),
        })
        .unwrap_or_else(|| "(complex type)".to_string());

    let impl_name = if let Some(trait_) = impl_.trait_(db) {
        format!("impl {} for {}", trait_.name(db).as_str(), self_ty_name)
    } else {
        format!("impl {}", self_ty_name)
    };

    Some(format!("{}::{}", module_path, impl_name))
}
