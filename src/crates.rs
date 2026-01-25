use ra_ap_hir::{Crate, Semantics};
use ra_ap_ide_db::RootDatabase;

use crate::modules::visit_module;

/// Visit a crate and print all its symbols with dependencies.
pub fn visit_crate(sema: &Semantics<'_, RootDatabase>, krate: Crate) {
    let db = sema.db;
    let name = krate
        .display_name(db)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(unnamed)".to_string());

    println!("Crate: {}", name);

    let modules = krate.modules(db);
    println!("  {} modules", modules.len());

    for module in modules {
        visit_module(sema, &module);
    }
    println!();
}
