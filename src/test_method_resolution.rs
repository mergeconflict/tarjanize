//! Test to understand resolve_method_call behavior.
//!
//! Run with: cargo test test_method_resolution -- --nocapture

#[cfg(test)]
mod tests {
    use crate::workspaces::load_workspace;
    use ra_ap_base_db::SourceDatabase;
    use ra_ap_hir::{AsAssocItem, HasSource, Semantics};
    use ra_ap_syntax::{AstNode, ast};

    #[test]
    fn test_method_resolution() {
        let db = load_workspace("tests/fixtures/method_resolution")
            .expect("Failed to load fixture");
        let sema = Semantics::new(&db);

        // Find the caller function and analyze its method calls
        let crate_graph = db.crate_graph();

        for krate_id in crate_graph.iter() {
            if !crate_graph[krate_id].origin.is_local() {
                continue;
            }

            let krate = ra_ap_hir::Crate::from(krate_id);
            let name = krate
                .display_name(&db)
                .map(|n| n.to_string())
                .unwrap_or_default();

            if name != "method_resolution" {
                continue;
            }

            println!("\n=== Analyzing crate: {} ===\n", name);

            for module in krate.modules(&db) {
                for def in module.declarations(&db) {
                    if let ra_ap_hir::ModuleDef::Function(func) = def {
                        let func_name = func.name(&db);
                        if func_name.as_str() != "caller" {
                            continue;
                        }

                        println!("Found function: {}", func_name.as_str());

                        // Get the source and find method calls
                        if let Some(source) = func.source(&db) {
                            let file_id = source.file_id;
                            let root = sema.parse_or_expand(file_id);

                            // Find the function node in the parsed tree
                            let func_syntax = source.value.syntax();
                            if let Some(func_node) = root
                                .descendants()
                                .find(|n| n.text_range() == func_syntax.text_range())
                            {
                                // Find all method call expressions
                                for method_call in func_node
                                    .descendants()
                                    .filter_map(ast::MethodCallExpr::cast)
                                {
                                    analyze_method_call(&sema, &method_call);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn analyze_method_call(
        sema: &Semantics<'_, ra_ap_ide_db::RootDatabase>,
        call: &ast::MethodCallExpr,
    ) {
        let method_name = call
            .name_ref()
            .map(|n| n.text().to_string())
            .unwrap_or_else(|| "???".to_string());

        let receiver = call
            .receiver()
            .map(|r| r.syntax().text().to_string())
            .unwrap_or_else(|| "???".to_string());

        println!("\nMethod call: {}.{}()", receiver, method_name);

        // Try to resolve the method call
        match sema.resolve_method_call(call) {
            Some(func) => {
                let func_name = func.name(sema.db);
                println!("  Resolved to function: {}", func_name.as_str());

                // Check if it's an associated item
                match func.as_assoc_item(sema.db) {
                    Some(assoc_item) => {
                        println!("  Is associated item: yes");

                        // Get the container - trait or impl?
                        use ra_ap_hir::AssocItemContainer;
                        match assoc_item.container(sema.db) {
                            AssocItemContainer::Trait(trait_) => {
                                println!(
                                    "  Container: TRAIT {}",
                                    trait_.name(sema.db).as_str()
                                );
                                println!("  -> Would collapse to: TRAIT");
                            }
                            AssocItemContainer::Impl(impl_) => {
                                // We found the impl!
                                let self_ty = impl_.self_ty(sema.db);
                                let trait_ = impl_.trait_(sema.db);

                                println!("  Container: IMPL");
                                println!("    Self type: {:?}", self_ty.as_adt().map(|adt| {
                                    match adt {
                                        ra_ap_hir::Adt::Struct(s) => s.name(sema.db).as_str().to_owned(),
                                        ra_ap_hir::Adt::Enum(e) => e.name(sema.db).as_str().to_owned(),
                                        ra_ap_hir::Adt::Union(u) => u.name(sema.db).as_str().to_owned(),
                                    }
                                }));
                                if let Some(t) = trait_ {
                                    println!("    Trait: {}", t.name(sema.db).as_str());
                                } else {
                                    println!("    Trait: (inherent impl)");
                                }
                                println!("  -> Would collapse to: IMPL");
                            }
                        }
                    }
                    None => {
                        println!("  Is associated item: no (free function?)");
                    }
                }
            }
            None => {
                println!("  Resolution: FAILED");
            }
        }
    }
}
