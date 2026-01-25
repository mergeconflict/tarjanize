//! Module and symbol extraction for tarjanize.
//!
//! This module contains the core extraction logic that transforms rust-analyzer's
//! HIR representation into our schema format. It recursively processes modules,
//! extracting both `ModuleDef` items (functions, structs, etc.) and impl blocks.
//!
//! Key responsibilities:
//! - Building fully-qualified paths for symbols
//! - Converting HIR types to schema types
//! - Computing file paths relative to crate root
//! - Collecting dependency edges
//!
//! This module works closely with `dependencies.rs` which handles the actual
//! dependency analysis for individual symbols.

use std::collections::HashSet;

use ra_ap_hir::{
    DisplayTarget, HasSource, HasVisibility, HirDisplay, HirFileId, Impl,
    Module, ModuleDef, Semantics, symbols::SymbolCollector,
};
use ra_ap_ide_db::RootDatabase;
use ra_ap_paths::AbsPath;
use ra_ap_syntax::AstNode;
use tracing::warn;

use crate::dependencies::{
    Dependency, find_dependencies, find_impl_dependencies, is_local,
    is_local_dep,
};
use crate::extract::FilePathResolver;
use crate::schemas::{Edge, Module as SchemaModule, Symbol, SymbolKind};

/// Extract a module and its contents as a SchemaModule.
///
/// This function recursively processes a module, extracting:
/// - All symbols (ModuleDefs and impl blocks)
/// - Child submodules
/// - Dependency edges between symbols
///
/// The crate_name parameter is used to build fully-qualified paths for edges.
/// The crate_root is the directory containing the crate's lib.rs/main.rs,
/// used to compute relative file paths for symbols.
pub fn extract_module<F: FilePathResolver>(
    sema: &Semantics<'_, RootDatabase>,
    file_resolver: &F,
    crate_root: Option<&AbsPath>,
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
    let symbols = extract_symbols(
        sema,
        file_resolver,
        crate_root,
        module,
        &module_path,
        edges,
    );

    // Recursively extract child modules.
    let children: Vec<_> = module
        .children(db)
        .map(|child| {
            extract_module(
                sema,
                file_resolver,
                crate_root,
                &child,
                crate_name,
                edges,
            )
        })
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
fn build_module_path(
    db: &RootDatabase,
    module: &Module,
    crate_name: &str,
) -> String {
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
fn extract_symbols<F: FilePathResolver>(
    sema: &Semantics<'_, RootDatabase>,
    file_resolver: &F,
    crate_root: Option<&AbsPath>,
    module: &Module,
    module_path: &str,
    edges: &mut HashSet<Edge>,
) -> Vec<Symbol> {
    let db = sema.db;
    let mut symbols = Vec::new();

    // Extract ModuleDef symbols using SymbolCollector.
    // SymbolCollector gathers all named items defined in this module.
    // The second argument (false) means collect all items, not just pub items.
    let collected = SymbolCollector::new_module(db, *module, false);

    for sym in collected {
        if let Some(symbol) = extract_module_def(
            sema,
            file_resolver,
            crate_root,
            &sym,
            module_path,
            edges,
        ) {
            symbols.push(symbol);
        }
    }

    // Extract impl blocks separately - they're not ModuleDefs but are still
    // important compilation units with their own dependencies.
    for impl_ in module.impl_defs(db) {
        if let Some(symbol) = extract_impl(
            sema,
            file_resolver,
            crate_root,
            impl_,
            module_path,
            edges,
        ) {
            symbols.push(symbol);
        }
    }

    symbols
}

/// Extract a single ModuleDef as a Symbol.
///
/// Returns None for items we skip (e.g., built-in types).
fn extract_module_def<F: FilePathResolver>(
    sema: &Semantics<'_, RootDatabase>,
    file_resolver: &F,
    crate_root: Option<&AbsPath>,
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

    // Compute file path relative to crate root.
    // We use the HirFileId from the symbol's location, convert to FileId,
    // then look up the path in the VFS and make it relative to crate root.
    let file = compute_relative_file_path(
        db,
        file_resolver,
        crate_root,
        sym.loc.hir_file_id,
    );

    // Compute cost as the byte size of the symbol's syntax node.
    // This is a rough proxy for compile-time complexity.
    let cost: f64 = u32::from(sym.loc.ptr.text_range().len()).into();

    // Collect dependencies for this symbol, filtered to local workspace items.
    let deps = find_dependencies(sema, sym.def);
    for dep in deps {
        if is_local_dep(db, &dep)
            && let Some(dep_path) = dependency_path(db, &dep)
        {
            edges.insert(Edge {
                from: symbol_path.clone(),
                to: dep_path,
            });
        }
    }

    // Extract visibility from the ModuleDef.
    let visibility = extract_visibility(db, &sym.def);

    Some(Symbol {
        name,
        file,
        cost,
        kind: SymbolKind::ModuleDef {
            kind: kind.to_string(),
            visibility,
        },
    })
}

/// Extract an impl block as a Symbol.
fn extract_impl<F: FilePathResolver>(
    sema: &Semantics<'_, RootDatabase>,
    file_resolver: &F,
    crate_root: Option<&AbsPath>,
    impl_: Impl,
    module_path: &str,
    edges: &mut HashSet<Edge>,
) -> Option<Symbol> {
    let db = sema.db;

    // Build the impl name (e.g., "impl Trait for Type" or "impl Type").
    // We use HirDisplay to get proper type names including generics, references,
    // slices, trait objects, etc. This ensures unique names for impls like
    // `impl<T> Foo for &T` vs `impl<T> Foo for Box<T>`.
    let self_ty = impl_.self_ty(db);
    let display_target =
        DisplayTarget::from_crate(db, impl_.module(db).krate(db).into());
    let self_ty_name = self_ty.display(db, display_target).to_string();

    let name = if let Some(trait_) = impl_.trait_(db) {
        format!("impl {} for {}", trait_.name(db).as_str(), self_ty_name)
    } else {
        format!("impl {}", self_ty_name)
    };

    let symbol_path = format!("{}::{}", module_path, name);

    // Compute file path and cost from the impl's source.
    let (file, cost) = impl_
        .source(db)
        .map(|source| {
            let file = compute_relative_file_path(
                db,
                file_resolver,
                crate_root,
                source.file_id,
            );
            let cost: f64 =
                u32::from(source.value.syntax().text_range().len()).into();
            (file, cost)
        })
        .unwrap_or_else(|| {
            warn!(impl_name = %name, "extract_impl: no source for impl block");
            (String::new(), 0.0)
        });

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
        if is_local(db, &dep)
            && let Some(dep_path) = module_def_path(db, &dep)
        {
            edges.insert(Edge {
                from: symbol_path.clone(),
                to: dep_path,
            });
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
            if is_local_dep(db, &dep)
                && let Some(dep_path) = dependency_path(db, &dep)
            {
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
        ModuleDef::TypeAlias(t) => Some(t.module(db)),
        ModuleDef::BuiltinType(_) => None,
        ModuleDef::Macro(m) => Some(m.module(db)),
    }?;

    // Get the crate name.
    let krate = module.krate(db);
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
    let krate = module.krate(db);
    let crate_name = krate
        .display_name(db)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(unnamed)".to_string());

    let module_path = build_module_path(db, &module, &crate_name);

    // Build the impl name (e.g., "impl Trait for Type" or "impl Type").
    // Use HirDisplay for proper type names (same as extract_impl).
    let self_ty = impl_.self_ty(db);
    let display_target = DisplayTarget::from_crate(db, krate.into());
    let self_ty_name = self_ty.display(db, display_target).to_string();

    let impl_name = if let Some(trait_) = impl_.trait_(db) {
        format!("impl {} for {}", trait_.name(db).as_str(), self_ty_name)
    } else {
        format!("impl {}", self_ty_name)
    };

    Some(format!("{}::{}", module_path, impl_name))
}

/// Compute the file path relative to the crate root.
///
/// This takes a HirFileId (which may be a real file or a macro expansion),
/// converts it to an absolute file path, then makes it relative to the
/// crate root directory.
///
/// Returns an empty string if the file is a macro expansion or if the path
/// cannot be computed.
fn compute_relative_file_path<F: FilePathResolver>(
    db: &RootDatabase,
    file_resolver: &F,
    crate_root: Option<&AbsPath>,
    hir_file_id: HirFileId,
) -> String {
    // Convert HirFileId to FileId. If the HirFileId is a macro expansion,
    // original_file() traces back to the real source file.
    let file_id = hir_file_id.original_file(db).file_id(db);

    // Look up the path using the file resolver.
    let vfs_path = file_resolver.file_path(file_id);

    // Convert to a filesystem path. VfsPath can be either a real path or
    // a virtual path; we only care about real paths.
    let Some(abs_path) = vfs_path.as_path() else {
        return String::new();
    };

    // Make the path relative to the crate root.
    match crate_root {
        Some(root) => abs_path
            .strip_prefix(root)
            .map(|p| p.as_str().to_owned())
            .unwrap_or_else(|| abs_path.as_str().to_owned()),
        None => abs_path.as_str().to_owned(),
    }
}

/// Extract visibility from a ModuleDef.
///
/// Returns None for private items (the schema uses absence to indicate private),
/// or Some("pub"), Some("pub(crate)"), etc. for visible items.
fn extract_visibility(db: &RootDatabase, def: &ModuleDef) -> Option<String> {
    use ra_ap_hir::Visibility;

    // ModuleDef implements HasVisibility, so we can use the trait method directly.
    let visibility = def.visibility(db);

    // Convert to string representation. Private items return None.
    match visibility {
        Visibility::Public => Some("pub".to_string()),
        Visibility::PubCrate(_) => Some("pub(crate)".to_string()),
        Visibility::Module(_, vis_explicitness) => {
            // Module visibility includes pub(super), pub(in path).
            // The exact path is encoded in the module ID, but for our purposes
            // we just indicate it's restricted visibility.
            // vis_explicitness tells us if the visibility was explicit (pub(super))
            // or implicit (private). Implicit means private.
            if vis_explicitness.is_explicit() {
                Some("pub(restricted)".to_string())
            } else {
                None // private
            }
        }
    }
}
