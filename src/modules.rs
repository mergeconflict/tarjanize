use ra_ap_hir::{AssocItem, Impl, Module, ModuleDef, Semantics, db::HirDatabase, symbols::SymbolCollector};
use ra_ap_ide_db::RootDatabase;

use crate::dependencies::{find_dependencies, find_impl_dependencies, is_local};

pub fn visit_module(sema: &Semantics<'_, RootDatabase>, module: &Module) {
    let db = sema.db;
    let module_name = module
        .name(db)
        .map(|n| n.as_str().to_owned())
        .unwrap_or_else(|| "(crate root)".to_string());

    let symbols = SymbolCollector::collect_module(db, *module);

    if symbols.is_empty() {
        return;
    }

    println!("  mod {}:", module_name);
    for symbol in &symbols {
        let kind = module_def_kind(&symbol.def);
        let container = symbol
            .container_name
            .as_ref()
            .map(|c| format!(" (in {})", c))
            .unwrap_or_default();

        // Find dependencies for this symbol, filtered to local (workspace) items
        let deps = find_dependencies(sema, symbol.def);
        let dep_names: Vec<_> = deps
            .iter()
            .filter(|d| is_local(db, d))
            .filter_map(|d| def_name(db, d))
            .collect();

        if dep_names.is_empty() {
            println!("    {} {}{}", kind, symbol.name, container);
        } else {
            println!(
                "    {} {}{} -> [{}]",
                kind,
                symbol.name,
                container,
                dep_names.join(", ")
            );
        }
    }

    // Process impl blocks in this module.
    // Impls are not ModuleDefs, so they're not in SymbolCollector results.
    // We need to handle them separately.
    let impls = module.impl_defs(db);
    for impl_ in impls {
        visit_impl(db, impl_);
    }
}

/// Print an impl block and its dependencies.
fn visit_impl(db: &dyn HirDatabase, impl_: Impl) {
    let impl_deps = find_impl_dependencies(db, impl_);

    // Build a description of the impl.
    // We use the ADT name when available, falling back to a placeholder.
    let self_ty = impl_.self_ty(db);
    let self_ty_name = self_ty
        .as_adt()
        .and_then(|adt| match adt {
            ra_ap_hir::Adt::Struct(s) => Some(s.name(db).as_str().to_owned()),
            ra_ap_hir::Adt::Enum(e) => Some(e.name(db).as_str().to_owned()),
            ra_ap_hir::Adt::Union(u) => Some(u.name(db).as_str().to_owned()),
        })
        .unwrap_or_else(|| "(complex type)".to_string());

    let impl_desc = if let Some(trait_) = impl_.trait_(db) {
        format!("impl {} for {}", trait_.name(db).as_str(), self_ty_name)
    } else {
        format!("impl {}", self_ty_name)
    };

    // Get dependency names (filtered to local items)
    let dep_names: Vec<_> = impl_deps
        .deps
        .iter()
        .filter(|d| is_local(db, d))
        .filter_map(|d| def_name(db, d))
        .collect();

    // Print the impl with its dependencies
    if dep_names.is_empty() {
        println!("    {}", impl_desc);
    } else {
        println!("    {} -> [{}]", impl_desc, dep_names.join(", "));
    }

    // Print associated items in the impl
    for item in impl_deps.items {
        let item_name = assoc_item_name(db, &item);
        let item_kind = assoc_item_kind(&item);
        println!("      {} {} (in impl)", item_kind, item_name);
    }
}

fn assoc_item_name(db: &dyn HirDatabase, item: &AssocItem) -> String {
    match item {
        AssocItem::Function(f) => f.name(db).as_str().to_owned(),
        AssocItem::Const(c) => c
            .name(db)
            .map(|n| n.as_str().to_owned())
            .unwrap_or_else(|| "(unnamed)".to_string()),
        AssocItem::TypeAlias(t) => t.name(db).as_str().to_owned(),
    }
}

fn assoc_item_kind(item: &AssocItem) -> &'static str {
    match item {
        AssocItem::Function(_) => "fn",
        AssocItem::Const(_) => "const",
        AssocItem::TypeAlias(_) => "type",
    }
}

fn module_def_kind(def: &ModuleDef) -> &'static str {
    match def {
        ModuleDef::Module(_) => "mod",
        ModuleDef::Function(_) => "fn",
        ModuleDef::Adt(adt) => match adt {
            ra_ap_hir::Adt::Struct(_) => "struct",
            ra_ap_hir::Adt::Enum(_) => "enum",
            ra_ap_hir::Adt::Union(_) => "union",
        },
        ModuleDef::Variant(_) => "variant",
        ModuleDef::Const(_) => "const",
        ModuleDef::Static(_) => "static",
        ModuleDef::Trait(_) => "trait",
        ModuleDef::TraitAlias(_) => "trait alias",
        ModuleDef::TypeAlias(_) => "type",
        ModuleDef::BuiltinType(_) => "builtin",
        ModuleDef::Macro(_) => "macro",
    }
}

fn def_name(db: &dyn HirDatabase, def: &ModuleDef) -> Option<String> {
    match def {
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
        ModuleDef::BuiltinType(b) => Some(b.name().as_str().to_owned()),
        ModuleDef::Macro(m) => Some(m.name(db).as_str().to_owned()),
    }
}
