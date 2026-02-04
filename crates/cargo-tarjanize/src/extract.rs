//! Symbol extraction from rustc's HIR and THIR.
//!
//! This module walks the compiler's internal representations to extract:
//! - Module structure and symbol definitions
//! - Dependencies between symbols (from types and THIR bodies)
//! - Impl block anchors for orphan rule compliance
//!
//! ## Why THIR?
//!
//! THIR (Typed High-level IR) preserves source-level information that MIR loses:
//! - Static references: `ExprKind::StaticRef { def_id }`
//! - Named consts: `ExprKind::NamedConst { def_id }`
//! - Const patterns: `Pat.extra.expanded_const`
//! - Function items: `ExprKind::ZstLiteral` with `FnDef` type
//!
//! We use `-Zno-steal-thir` to preserve THIR after MIR is built.

use std::collections::{HashMap, HashSet};

use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_middle::thir::{self, ExprKind, PatKind};
use rustc_middle::ty::{self, Ty, TyCtxt, Visibility as TyVisibility};
use tarjanize_schemas::{Module, Symbol, SymbolKind, Visibility};

/// Result of symbol extraction: the module tree.
///
/// Symbol keys use rustc's `DefPath` format (e.g., `crate::module::{{impl}}[1]`),
/// which matches the format used by `-Zself-profile`. This allows direct lookup
/// of profile timing data without any mapping.
pub struct ExtractionResult {
    /// The extracted module hierarchy.
    pub module: Module,
}

/// Extract all symbols from a crate into a Module hierarchy.
///
/// `workspace_crates` is the list of crate names that are part of the workspace.
/// Dependencies on these crates will be captured; dependencies on external crates
/// (like std) are filtered out.
///
/// Returns the module tree and a map from profile keys to symbol paths.
pub fn extract_crate(
    tcx: TyCtxt<'_>,
    crate_name: &str,
    workspace_crates: &[String],
) -> ExtractionResult {
    let mut extractor = Extractor::new(tcx, crate_name, workspace_crates);
    extractor.extract_all_items();
    extractor.into_result()
}

/// State for symbol extraction.
struct Extractor<'tcx> {
    tcx: TyCtxt<'tcx>,
    crate_name: String,
    /// Workspace crate names for dependency filtering.
    /// Dependencies on these crates are captured; external deps are filtered out.
    workspace_crates: HashSet<String>,
    /// Flat map of all extracted symbols, keyed by their `DefPath`.
    /// We build the module tree from this at the end.
    ///
    /// Keys use rustc's `DefPath` format (e.g., `crate::module::{{impl}}[1]`)
    /// which matches `-Zself-profile` output for direct cost lookup.
    symbols: HashMap<String, Symbol>,
}

impl<'tcx> Extractor<'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        crate_name: &str,
        workspace_crates: &[String],
    ) -> Self {
        // Normalize package names (hyphens) to crate names (underscores).
        // Cargo uses hyphens in package names but rustc uses underscores.
        let workspace_crates: HashSet<String> = workspace_crates
            .iter()
            .map(|name| name.replace('-', "_"))
            .collect();

        Self {
            tcx,
            crate_name: crate_name.to_string(),
            workspace_crates,
            symbols: HashMap::new(),
        }
    }

    /// Extract all top-level items in the crate.
    fn extract_all_items(&mut self) {
        // Iterate all items in the crate.
        let items = self.tcx.hir_crate_items(());
        for item_id in items.free_items() {
            let def_id = item_id.owner_id.def_id;
            let path = self.def_path_str(def_id.to_def_id());
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.extract_item(def_id);
                }));
            if result.is_err() {
                tracing::warn!(path, "panic during item extraction");
            }
        }
    }

    /// Extract a single item and its nested items.
    fn extract_item(&mut self, local_def_id: LocalDefId) {
        let def_id = local_def_id.to_def_id();

        // Skip items nested inside functions/consts.
        // These can't be split independently and their dependencies are captured
        // when analyzing the parent's body. Examples include:
        // - Nested functions inside functions
        // - Statics created by macro expansion (e.g., tracing's __CALLSITE)
        if self.is_nested_in_body(def_id) {
            return;
        }

        let def_kind = self.tcx.def_kind(def_id);

        match def_kind {
            // Regular module-level definitions.
            DefKind::Fn | DefKind::AssocFn => {
                self.extract_function(local_def_id);
            }
            DefKind::Struct | DefKind::Enum | DefKind::Union => {
                self.extract_adt(local_def_id);
            }
            DefKind::Trait | DefKind::TraitAlias => {
                self.extract_trait(local_def_id);
            }
            DefKind::TyAlias => {
                self.extract_type_alias(local_def_id);
            }
            DefKind::Const | DefKind::Static { .. } => {
                self.extract_const_or_static(local_def_id);
            }
            DefKind::Macro(_) => {
                self.extract_macro(local_def_id);
            }

            // Impl blocks get special handling for anchors.
            DefKind::Impl { .. } => {
                self.extract_impl(local_def_id);
            }

            // Items we skip:
            // - Mod: module structure is built from symbol paths
            // - Foreign items (extern "C" blocks)
            // - Use declarations (re-exports)
            // - Extern crate declarations
            // - Global asm
            // - Closure, coroutine (not top-level)
            // - Associated items are handled when extracting their parent
            DefKind::Mod
            | DefKind::ForeignMod
            | DefKind::ForeignTy
            | DefKind::Use
            | DefKind::ExternCrate
            | DefKind::GlobalAsm
            | DefKind::Closure
            | DefKind::AssocTy
            | DefKind::AssocConst
            | DefKind::OpaqueTy
            | DefKind::Field
            | DefKind::LifetimeParam
            | DefKind::ConstParam
            | DefKind::TyParam
            | DefKind::Variant
            | DefKind::Ctor(..)
            | DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::SyntheticCoroutineBody => {}
        }
    }

    /// Extract a function or method.
    fn extract_function(&mut self, local_def_id: LocalDefId) {
        let def_id = local_def_id.to_def_id();
        // Use raw DefPath as key - matches profile data format directly.
        let key = self.raw_def_path(def_id);

        // Get dependencies from signature and body.
        // Use catch_unwind to prevent panics from stopping extraction.
        let mut deps = HashSet::new();

        // Collect deps from resolved types in signature (catches most deps).
        let sig_result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.collect_signature_deps(def_id, &mut deps);
            }));
        if sig_result.is_err() {
            tracing::warn!(key, "panic during signature dependency collection");
        }

        // Collect deps from HIR signature (catches type aliases that get resolved away).
        let hir_result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.collect_hir_signature_deps(local_def_id, &mut deps);
            }));
        if hir_result.is_err() {
            tracing::warn!(
                key,
                "panic during HIR signature dependency collection"
            );
        }

        // Collect body deps from THIR (covers all cases including statics, const patterns).
        self.collect_thir_body_deps(local_def_id, &mut deps);

        // Collect deps from nested items (nested functions, statics created by macros, etc.).
        // These items are collapsed to this function, so their deps become our deps.
        self.collect_nested_item_deps(local_def_id, &mut deps);

        // Collect deps from HIR body (catches range pattern const bounds that THIR loses).
        let hir_body_result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.collect_hir_body_deps(local_def_id, &mut deps);
            }));
        if hir_body_result.is_err() {
            tracing::warn!(key, "panic during HIR body dependency collection");
        }

        let symbol = Symbol {
            file: self.source_file(local_def_id),
            frontend_cost_ms: 0.0,
            backend_cost_ms: 0.0,
            dependencies: deps,
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: self.extract_visibility(def_id),
            },
        };

        self.symbols.insert(key, symbol);
    }

    /// Extract an ADT (struct, enum, or union).
    fn extract_adt(&mut self, local_def_id: LocalDefId) {
        let def_id = local_def_id.to_def_id();
        let key = self.raw_def_path(def_id);
        let def_kind = self.tcx.def_kind(def_id);

        let kind_name = match def_kind {
            DefKind::Struct => "Struct",
            DefKind::Enum => "Enum",
            DefKind::Union => "Union",
            _ => unreachable!(),
        };

        // Get dependencies from field types and bounds.
        let mut deps = HashSet::new();
        self.collect_adt_deps(def_id, &mut deps);

        let symbol = Symbol {
            file: self.source_file(local_def_id),
            frontend_cost_ms: 0.0,
            backend_cost_ms: 0.0,
            dependencies: deps,
            kind: SymbolKind::ModuleDef {
                kind: kind_name.to_string(),
                visibility: self.extract_visibility(def_id),
            },
        };

        self.symbols.insert(key, symbol);
    }

    /// Extract a trait definition.
    fn extract_trait(&mut self, local_def_id: LocalDefId) {
        let def_id = local_def_id.to_def_id();
        let key = self.raw_def_path(def_id);

        // Get dependencies from supertraits and associated items.
        let mut deps = HashSet::new();
        self.collect_trait_deps(def_id, &mut deps);

        let symbol = Symbol {
            file: self.source_file(local_def_id),
            frontend_cost_ms: 0.0,
            backend_cost_ms: 0.0,
            dependencies: deps,
            kind: SymbolKind::ModuleDef {
                kind: "Trait".to_string(),
                visibility: self.extract_visibility(def_id),
            },
        };

        self.symbols.insert(key, symbol);
    }

    /// Extract a type alias.
    fn extract_type_alias(&mut self, local_def_id: LocalDefId) {
        let def_id = local_def_id.to_def_id();
        let key = self.raw_def_path(def_id);

        let mut deps = HashSet::new();
        // Type aliases reference their aliased type.
        if let Some(ty) = self.tcx.type_of(def_id).no_bound_vars() {
            self.collect_type_deps(ty, &mut deps);
        }

        let symbol = Symbol {
            file: self.source_file(local_def_id),
            frontend_cost_ms: 0.0,
            backend_cost_ms: 0.0,
            dependencies: deps,
            kind: SymbolKind::ModuleDef {
                kind: "TypeAlias".to_string(),
                visibility: self.extract_visibility(def_id),
            },
        };

        self.symbols.insert(key, symbol);
    }

    /// Extract a const or static item.
    fn extract_const_or_static(&mut self, local_def_id: LocalDefId) {
        let def_id = local_def_id.to_def_id();
        let key = self.raw_def_path(def_id);
        let def_kind = self.tcx.def_kind(def_id);

        let kind_name = match def_kind {
            DefKind::Const => "Const",
            DefKind::Static { .. } => "Static",
            _ => unreachable!(),
        };

        let mut deps = HashSet::new();

        // Consts/statics don't have fn_sig - get their type directly.
        let ty = self.tcx.type_of(def_id).skip_binder();
        self.collect_type_deps(ty, &mut deps);

        // Collect bounds from generics.
        self.collect_generic_bounds_deps(def_id, &mut deps);

        // Collect body deps from THIR.
        self.collect_thir_body_deps(local_def_id, &mut deps);

        let symbol = Symbol {
            file: self.source_file(local_def_id),
            frontend_cost_ms: 0.0,
            backend_cost_ms: 0.0,
            dependencies: deps,
            kind: SymbolKind::ModuleDef {
                kind: kind_name.to_string(),
                visibility: self.extract_visibility(def_id),
            },
        };

        self.symbols.insert(key, symbol);
    }

    /// Extract a macro definition.
    fn extract_macro(&mut self, local_def_id: LocalDefId) {
        let def_id = local_def_id.to_def_id();
        let key = self.raw_def_path(def_id);

        // Macros don't have type-level dependencies we can easily extract.
        let symbol = Symbol {
            file: self.source_file(local_def_id),
            frontend_cost_ms: 0.0,
            backend_cost_ms: 0.0,
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Macro".to_string(),
                visibility: self.extract_visibility(def_id),
            },
        };

        self.symbols.insert(key, symbol);
    }

    /// Extract an impl block with its anchors.
    ///
    /// Uses `DefPath` format as the symbol key (e.g., `crate::module::{{impl}}[1]`)
    /// to match `-Zself-profile` output. The human-readable name (e.g.,
    /// `impl Trait for Type`) is stored in `SymbolKind::Impl.name`.
    fn extract_impl(&mut self, local_def_id: LocalDefId) {
        let def_id = local_def_id.to_def_id();

        // Use raw DefPath as symbol key - matches profile data format.
        let key = self.raw_def_path(def_id);

        // Build human-readable impl name for display.
        let impl_name = self.impl_name(def_id);

        // Collect dependencies from all associated items.
        let mut deps = HashSet::new();
        self.collect_impl_deps(def_id, &mut deps);

        // Extract anchors for orphan rule.
        let anchors = self.extract_impl_anchors(def_id);

        let symbol = Symbol {
            file: self.source_file(local_def_id),
            frontend_cost_ms: 0.0,
            backend_cost_ms: 0.0,
            dependencies: deps,
            kind: SymbolKind::Impl {
                name: impl_name,
                anchors,
            },
        };

        self.symbols.insert(key, symbol);
    }

    // -------------------------------------------------------------------------
    // Dependency collection
    // -------------------------------------------------------------------------

    /// Collect dependencies from a function's signature (params, return type, bounds).
    fn collect_signature_deps(
        &self,
        def_id: DefId,
        deps: &mut HashSet<String>,
    ) {
        // Get the function signature.
        let sig = self.tcx.fn_sig(def_id).skip_binder();

        // Input types.
        for input in sig.inputs().skip_binder() {
            self.collect_type_deps(*input, deps);
        }

        // Output type.
        self.collect_type_deps(sig.output().skip_binder(), deps);

        // Collect bounds from generics.
        self.collect_generic_bounds_deps(def_id, deps);
    }

    /// Collect dependencies from HIR signature types.
    ///
    /// This supplements `collect_signature_deps` by walking the HIR syntax tree
    /// to capture type alias references that get resolved away in semantic queries.
    /// For example, `fn foo(_: MyAlias)` where `type MyAlias = i32` - the semantic
    /// signature shows `i32`, but we want to capture the dep on `MyAlias`.
    fn collect_hir_signature_deps(
        &self,
        local_def_id: LocalDefId,
        deps: &mut HashSet<String>,
    ) {
        use rustc_hir::{HirId, OwnerId};

        // Convert LocalDefId to HirId via OwnerId.
        let owner_id = OwnerId {
            def_id: local_def_id,
        };
        let hir_id: HirId = owner_id.into();

        // Get the HIR function declaration for this item.
        let Some(fn_decl) = self.tcx.hir_fn_decl_by_hir_id(hir_id) else {
            return;
        };

        // Walk parameter types.
        for ty in fn_decl.inputs {
            self.collect_hir_ty_deps(ty, deps);
        }

        // Walk return type.
        if let rustc_hir::FnRetTy::Return(ret_ty) = fn_decl.output {
            self.collect_hir_ty_deps(ret_ty, deps);
        }
    }

    /// Collect dependencies from a HIR type.
    ///
    /// This recursively walks the HIR type tree to find path types that resolve
    /// to type aliases or other definitions we care about.
    fn collect_hir_ty_deps(
        &self,
        ty: &rustc_hir::Ty<'_>,
        deps: &mut HashSet<String>,
    ) {
        use rustc_hir::TyKind;
        use rustc_hir::def::Res;

        match &ty.kind {
            TyKind::Path(qpath) => {
                // Extract the resolution from the path.
                let res = match qpath {
                    rustc_hir::QPath::Resolved(_, path) => Some(path.res),
                    rustc_hir::QPath::TypeRelative(_, _) => None, // Would need typeck results
                };

                if let Some(Res::Def(def_kind, def_id)) = res {
                    // We're interested in type aliases specifically.
                    if matches!(def_kind, DefKind::TyAlias) {
                        self.maybe_add_dep(def_id, deps);
                    }
                }
            }
            TyKind::Slice(inner) | TyKind::Array(inner, _) => {
                self.collect_hir_ty_deps(inner, deps);
            }
            TyKind::Ptr(mut_ty) | TyKind::Ref(_, mut_ty) => {
                self.collect_hir_ty_deps(mut_ty.ty, deps);
            }
            TyKind::Tup(tys) => {
                for inner in *tys {
                    self.collect_hir_ty_deps(inner, deps);
                }
            }
            // Other HIR type kinds that don't contribute type alias deps.
            // We list them explicitly to catch new variants.
            TyKind::Never
            | TyKind::TraitObject(_, _)
            | TyKind::FnPtr(_)
            | TyKind::OpaqueDef(_)
            | TyKind::Infer(())
            | TyKind::InferDelegation(_, _)
            | TyKind::UnsafeBinder(_)
            | TyKind::Pat(_, _)
            | TyKind::TraitAscription(_)
            | TyKind::Err(_) => {}
        }
    }

    /// Collect dependencies from nested items (nested functions, statics, consts).
    ///
    /// Items defined inside function bodies can't be split independently and their
    /// compilation costs are aggregated to the parent via profile normalization.
    /// This function collects their dependencies so they become part of the parent's
    /// dependency set.
    fn collect_nested_item_deps(
        &self,
        local_def_id: LocalDefId,
        deps: &mut HashSet<String>,
    ) {
        let def_id = local_def_id.to_def_id();

        // Iterate all items in the crate looking for those nested in this function.
        let items = self.tcx.hir_crate_items(());
        for item_id in items.free_items() {
            let child_local = item_id.owner_id.def_id;
            let child_def_id = child_local.to_def_id();

            // Only process items whose direct parent is this function.
            if self.tcx.parent(child_def_id) != def_id {
                continue;
            }

            let child_kind = self.tcx.def_kind(child_def_id);
            match child_kind {
                // Nested functions - collect deps recursively.
                DefKind::Fn => {
                    let result = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| {
                            self.collect_thir_body_deps(child_local, deps);
                            // Recursively collect deps from items nested in the nested fn.
                            self.collect_nested_item_deps(child_local, deps);
                        }),
                    );
                    if result.is_err() {
                        tracing::warn!(
                            parent = %self.def_path_str(def_id),
                            nested = %self.def_path_str(child_def_id),
                            "panic during nested function dependency collection"
                        );
                    }
                }
                // Statics and consts - collect their body deps.
                DefKind::Static { .. } | DefKind::Const => {
                    let result = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| {
                            self.collect_thir_body_deps(child_local, deps);
                        }),
                    );
                    if result.is_err() {
                        tracing::warn!(
                            parent = %self.def_path_str(def_id),
                            nested = %self.def_path_str(child_def_id),
                            "panic during nested static/const dependency collection"
                        );
                    }
                }
                // Other nested items are skipped.
                _ => {}
            }
        }
    }

    /// Collect dependencies from HIR body patterns.
    ///
    /// This supplements THIR body analysis by walking HIR to capture const references
    /// in range pattern bounds. THIR evaluates these bounds at lowering time, losing
    /// the original const `DefId`s. HIR preserves them as `PatExpr` nodes.
    fn collect_hir_body_deps(
        &self,
        local_def_id: LocalDefId,
        deps: &mut HashSet<String>,
    ) {
        use rustc_hir::def::Res;
        use rustc_hir::intravisit::{self, Visitor};
        use rustc_hir::{Pat, PatExprKind, PatKind};

        // Visitor to find range patterns and extract const refs from bounds.
        struct RangePatternVisitor<'a, 'tcx> {
            extractor: &'a Extractor<'tcx>,
            deps: &'a mut HashSet<String>,
        }

        impl<'tcx> Visitor<'tcx> for RangePatternVisitor<'_, 'tcx> {
            fn visit_pat(&mut self, p: &'tcx Pat<'tcx>) {
                if let PatKind::Range(lo, hi, _) = &p.kind {
                    // Extract const refs from the lower bound.
                    if let Some(lo_expr) = lo {
                        self.extract_const_from_pat_expr(lo_expr);
                    }
                    // Extract const refs from the upper bound.
                    if let Some(hi_expr) = hi {
                        self.extract_const_from_pat_expr(hi_expr);
                    }
                }
                // Continue visiting nested patterns.
                intravisit::walk_pat(self, p);
            }
        }

        impl<'tcx> RangePatternVisitor<'_, 'tcx> {
            /// Extract a const reference from a range bound pattern expression.
            fn extract_const_from_pat_expr(
                &mut self,
                pat_expr: &rustc_hir::PatExpr<'tcx>,
            ) {
                // Range bounds that reference named consts use PatExprKind::Path.
                if let PatExprKind::Path(qpath) = &pat_expr.kind {
                    let res = match qpath {
                        rustc_hir::QPath::Resolved(_, path) => Some(path.res),
                        rustc_hir::QPath::TypeRelative(_, _) => None,
                    };

                    if let Some(Res::Def(DefKind::Const, def_id)) = res {
                        self.extractor.maybe_add_dep(def_id, self.deps);
                    }
                }
            }
        }

        // Get the HIR body for this item.
        let Some(body) = self.tcx.hir_maybe_body_owned_by(local_def_id) else {
            return;
        };

        let mut visitor = RangePatternVisitor {
            extractor: self,
            deps,
        };
        visitor.visit_body(body);
    }

    /// Collect dependencies from an ADT's fields and bounds.
    fn collect_adt_deps(&self, def_id: DefId, deps: &mut HashSet<String>) {
        let adt_def = self.tcx.adt_def(def_id);

        // Field types for all variants.
        for variant in adt_def.variants() {
            for field in &variant.fields {
                let field_ty = self.tcx.type_of(field.did).skip_binder();
                self.collect_type_deps(field_ty, deps);
            }
        }

        // Generic bounds.
        self.collect_generic_bounds_deps(def_id, deps);

        // Default type parameters (e.g., `T = DefaultType`).
        self.collect_default_type_params(def_id, deps);
    }

    /// Collect dependencies from default type and const parameters.
    ///
    /// Handles `struct Foo<T = DefaultType>(T)` where `DefaultType` is a dependency,
    /// and `struct Buffer<const N: usize = DEFAULT_SIZE>` where `DEFAULT_SIZE` is a dependency.
    fn collect_default_type_params(
        &self,
        def_id: DefId,
        deps: &mut HashSet<String>,
    ) {
        let generics = self.tcx.generics_of(def_id);
        for param in &generics.own_params {
            match param.kind {
                ty::GenericParamDefKind::Type {
                    has_default: true, ..
                } => {
                    let default_ty =
                        self.tcx.type_of(param.def_id).skip_binder();
                    self.collect_type_deps(default_ty, deps);
                }
                ty::GenericParamDefKind::Const {
                    has_default: true, ..
                } => {
                    // For const generics with defaults, get the default value's ty::Const.
                    // If it's an unevaluated reference to a named const, extract the DefId.
                    let default_const = self
                        .tcx
                        .const_param_default(param.def_id)
                        .skip_binder();
                    if let ty::ConstKind::Unevaluated(uneval) =
                        default_const.kind()
                    {
                        // The default references a named const or anonymous const body.
                        // Check if the DefId is an AnonConst (in which case we need to walk its body)
                        // or a named const (which we can add directly).
                        let def_kind = self.tcx.def_kind(uneval.def);
                        if matches!(def_kind, DefKind::AnonConst) {
                            // Anonymous const - walk its THIR body to find const references.
                            if let Some(local_def_id) = uneval.def.as_local() {
                                self.collect_thir_body_deps(local_def_id, deps);
                            }
                        } else {
                            // Named const - add it directly.
                            self.maybe_add_dep(uneval.def, deps);
                        }
                    }
                }
                // Type and const parameters without defaults don't contribute dependencies.
                ty::GenericParamDefKind::Type {
                    has_default: false, ..
                }
                | ty::GenericParamDefKind::Const {
                    has_default: false, ..
                }
                | ty::GenericParamDefKind::Lifetime => {}
            }
        }
    }

    /// Collect dependencies from a trait's supertraits and associated items.
    fn collect_trait_deps(&self, def_id: DefId, deps: &mut HashSet<String>) {
        // Supertraits via predicates.
        for (pred, _span) in self.tcx.predicates_of(def_id).predicates {
            if let ty::ClauseKind::Trait(trait_pred) = pred.kind().skip_binder()
            {
                self.maybe_add_dep(trait_pred.def_id(), deps);
            }
        }

        // Associated items.
        for item in self.tcx.associated_items(def_id).in_definition_order() {
            match item.kind {
                ty::AssocKind::Type { .. } => {
                    // Associated type bounds.
                    for pred in
                        self.tcx.explicit_item_bounds(item.def_id).skip_binder()
                    {
                        self.collect_predicate_deps(pred.0, deps);
                    }
                }
                ty::AssocKind::Fn { .. } => {
                    // Method signature.
                    self.collect_signature_deps(item.def_id, deps);
                    // Default method body (if present).
                    if let Some(local_def_id) = item.def_id.as_local() {
                        self.collect_thir_body_deps(local_def_id, deps);
                    }
                }
                ty::AssocKind::Const { .. } => {
                    // Associated const type.
                    let ty = self.tcx.type_of(item.def_id).skip_binder();
                    self.collect_type_deps(ty, deps);
                    // Default const value (if present).
                    if let Some(local_def_id) = item.def_id.as_local() {
                        self.collect_thir_body_deps(local_def_id, deps);
                    }
                }
            }
        }
    }

    /// Collect dependencies from an impl block.
    fn collect_impl_deps(&self, def_id: DefId, deps: &mut HashSet<String>) {
        // Self type.
        let self_ty = self.tcx.type_of(def_id).skip_binder();
        self.collect_type_deps(self_ty, deps);

        // Trait (if trait impl).
        if matches!(self.tcx.def_kind(def_id), DefKind::Impl { of_trait: true })
        {
            let trait_ref = self.tcx.impl_trait_ref(def_id).skip_binder();
            self.maybe_add_dep(trait_ref.def_id, deps);
            for arg in trait_ref.args {
                if let Some(ty) = arg.as_type() {
                    self.collect_type_deps(ty, deps);
                }
            }
        }

        // Generic bounds (where clauses and inline bounds like `T: Bound`).
        self.collect_generic_bounds_deps(def_id, deps);

        // Associated items in the impl.
        for item in self.tcx.associated_items(def_id).in_definition_order() {
            if let Some(local_def_id) = item.def_id.as_local() {
                match item.kind {
                    ty::AssocKind::Fn { .. } => {
                        self.collect_signature_deps(item.def_id, deps);
                        self.collect_thir_body_deps(local_def_id, deps);
                    }
                    ty::AssocKind::Type { .. } => {
                        let ty = self.tcx.type_of(item.def_id).skip_binder();
                        self.collect_type_deps(ty, deps);
                    }
                    ty::AssocKind::Const { .. } => {
                        let ty = self.tcx.type_of(item.def_id).skip_binder();
                        self.collect_type_deps(ty, deps);
                        self.collect_thir_body_deps(local_def_id, deps);
                    }
                }
            }
        }
    }

    /// Collect dependencies from generic bounds (where clauses).
    fn collect_generic_bounds_deps(
        &self,
        def_id: DefId,
        deps: &mut HashSet<String>,
    ) {
        for (pred, _span) in self.tcx.predicates_of(def_id).predicates {
            self.collect_predicate_deps(*pred, deps);
        }
    }

    /// Collect dependencies from a predicate (trait bound, etc.).
    fn collect_predicate_deps(
        &self,
        pred: ty::Clause<'tcx>,
        deps: &mut HashSet<String>,
    ) {
        match pred.kind().skip_binder() {
            ty::ClauseKind::Trait(trait_pred) => {
                self.maybe_add_dep(trait_pred.def_id(), deps);
                // Also collect type args to the trait.
                for arg in trait_pred.trait_ref.args {
                    if let Some(ty) = arg.as_type() {
                        self.collect_type_deps(ty, deps);
                    }
                }
            }
            ty::ClauseKind::Projection(proj) => {
                self.maybe_add_dep(proj.projection_term.def_id, deps);
                self.collect_type_deps(proj.term.expect_type(), deps);
            }
            ty::ClauseKind::TypeOutlives(_)
            | ty::ClauseKind::RegionOutlives(_)
            | ty::ClauseKind::ConstArgHasType(..)
            | ty::ClauseKind::WellFormed(_)
            | ty::ClauseKind::ConstEvaluatable(_)
            | ty::ClauseKind::HostEffect(_)
            | ty::ClauseKind::UnstableFeature(_) => {}
        }
    }

    /// Collect dependencies from a type.
    fn collect_type_deps(&self, ty: Ty<'tcx>, deps: &mut HashSet<String>) {
        // Walk all types in the type tree.
        for inner in ty.walk() {
            // Handle type arguments.
            if let Some(ty) = inner.as_type() {
                match ty.kind() {
                    ty::TyKind::Adt(adt_def, args) => {
                        self.maybe_add_dep(adt_def.did(), deps);
                        // Recurse into type arguments.
                        for arg in *args {
                            if let Some(ty) = arg.as_type() {
                                self.collect_type_deps(ty, deps);
                            }
                        }
                    }
                    ty::TyKind::FnDef(def_id, _) => {
                        self.maybe_add_dep(*def_id, deps);
                    }
                    ty::TyKind::Alias(ty::AliasTyKind::Projection, alias) => {
                        self.maybe_add_dep(alias.def_id, deps);
                    }
                    ty::TyKind::Alias(ty::AliasTyKind::Opaque, alias) => {
                        // Opaque types (impl Trait) - extract trait bounds.
                        // Only extract the trait itself, not recursively walking predicates
                        // to avoid infinite recursion (bounds can reference the opaque type).
                        for (pred, _) in self
                            .tcx
                            .explicit_item_bounds(alias.def_id)
                            .skip_binder()
                        {
                            if let ty::ClauseKind::Trait(trait_pred) =
                                pred.kind().skip_binder()
                            {
                                self.maybe_add_dep(trait_pred.def_id(), deps);
                            }
                        }
                    }
                    ty::TyKind::Dynamic(preds, _) => {
                        for pred in *preds {
                            if let ty::ExistentialPredicate::Trait(trait_ref) =
                                pred.skip_binder()
                            {
                                self.maybe_add_dep(trait_ref.def_id, deps);
                            }
                        }
                    }
                    // Primitive and structural types that don't introduce dependencies.
                    // We list them explicitly to catch any new TyKind variants added
                    // in future rustc versions.
                    ty::TyKind::Bool
                    | ty::TyKind::Char
                    | ty::TyKind::Int(_)
                    | ty::TyKind::Uint(_)
                    | ty::TyKind::Float(_)
                    | ty::TyKind::Str
                    | ty::TyKind::Array(_, _)
                    | ty::TyKind::Slice(_)
                    | ty::TyKind::RawPtr(_, _)
                    | ty::TyKind::Ref(_, _, _)
                    | ty::TyKind::Tuple(_)
                    | ty::TyKind::Never
                    | ty::TyKind::Param(_)
                    | ty::TyKind::Bound(_, _)
                    | ty::TyKind::Placeholder(_)
                    | ty::TyKind::Infer(_)
                    | ty::TyKind::Error(_)
                    | ty::TyKind::Foreign(_)
                    | ty::TyKind::FnPtr(_, _)
                    | ty::TyKind::Closure(_, _)
                    | ty::TyKind::CoroutineClosure(_, _)
                    | ty::TyKind::Coroutine(_, _)
                    | ty::TyKind::CoroutineWitness(_, _)
                    | ty::TyKind::Pat(_, _)
                    | ty::TyKind::UnsafeBinder(_)
                    // Alias types other than Projection and Opaque are handled
                    // by walking their args (which we do implicitly via ty.walk()).
                    | ty::TyKind::Alias(
                        ty::AliasTyKind::Inherent | ty::AliasTyKind::Free,
                        _,
                    ) => {}
                }
            }

            // Handle const arguments (e.g., array lengths like [T; SIZE]).
            if let Some(ct) = inner.as_const()
                && let ty::ConstKind::Unevaluated(uneval) = ct.kind()
            {
                // For anonymous consts like `{constant#0}`, walk THIR to find
                // the actual const reference (e.g., SIZE).
                if let Some(local_def_id) = uneval.def.as_local() {
                    self.collect_thir_body_deps(local_def_id, deps);
                }
                // Also add the original def (may be useful for named consts).
                self.maybe_add_dep(uneval.def, deps);
            }
        }
    }

    // -------------------------------------------------------------------------
    // THIR-based body analysis
    // -------------------------------------------------------------------------

    /// Collect dependencies from a function/const body using THIR.
    ///
    /// THIR (Typed High-level IR) preserves source-level references:
    /// - `StaticRef { def_id }` for static variable access
    /// - `NamedConst { def_id }` for named const references
    /// - `Pat.extra.expanded_const` for const patterns
    /// - `ZstLiteral` with `FnDef` type for function references
    fn collect_thir_body_deps(
        &self,
        local_def_id: LocalDefId,
        deps: &mut HashSet<String>,
    ) {
        let def_id = local_def_id.to_def_id();
        let path = self.def_path_str(def_id);

        // Check if this item has a THIR body.
        // Not all items have THIR - only "body owners" (functions, consts, statics).
        // Additionally, trait methods without default implementations have no body.
        let def_kind = self.tcx.def_kind(def_id);
        let has_thir = matches!(
            def_kind,
            DefKind::Fn
                | DefKind::AssocFn
                | DefKind::Const
                | DefKind::Static { .. }
                | DefKind::AssocConst
                | DefKind::AnonConst
                | DefKind::InlineConst
                | DefKind::Closure
        );

        if !has_thir {
            tracing::trace!(path, ?def_kind, "item does not have THIR body");
            return;
        }

        // Check if MIR is available - if not, there's no body and no THIR.
        // This handles trait methods without default implementations.
        if !self.tcx.is_mir_available(def_id) {
            tracing::trace!(path, "MIR not available, skipping THIR");
            return;
        }

        tracing::debug!(
            path,
            ?def_kind,
            "attempting to collect THIR body deps"
        );

        // Access THIR via tcx.thir_body(). This returns a Result because THIR
        // construction can fail for malformed code. We also need to handle the
        // Steal wrapper by borrowing.
        let thir_result = self.tcx.thir_body(local_def_id);
        let Ok((thir, root_expr)) = thir_result else {
            tracing::debug!(path, "THIR query returned error");
            return;
        };

        // Borrow the THIR from the Steal wrapper.
        // This may panic if THIR has been "stolen" by MIR building.
        tracing::debug!(path, "borrowing THIR");
        let thir = thir.borrow();
        tracing::debug!(
            path,
            num_exprs = thir.exprs.len(),
            "got THIR, walking expressions"
        );

        // Walk the THIR starting from the root expression.
        self.walk_thir_expr(&thir, root_expr, deps);
        tracing::debug!(path, ?deps, "finished THIR walk");
    }

    /// Walk a THIR expression and collect dependencies.
    #[expect(
        clippy::too_many_lines,
        reason = "large match on ExprKind variants"
    )]
    fn walk_thir_expr(
        &self,
        thir: &thir::Thir<'tcx>,
        expr_id: thir::ExprId,
        deps: &mut HashSet<String>,
    ) {
        let expr = &thir.exprs[expr_id];

        match &expr.kind {
            // Static reference, named const reference, and thread-local static reference
            // all just capture a DefId dependency.
            ExprKind::StaticRef { def_id, .. }
            | ExprKind::NamedConst { def_id, .. }
            | ExprKind::ThreadLocalRef(def_id) => {
                self.maybe_add_dep(*def_id, deps);
            }

            // Function/method calls.
            ExprKind::Call { fun, args, .. } => {
                // Walk the function expression to get the callee.
                self.walk_thir_expr(thir, *fun, deps);
                // Walk arguments.
                for arg in args {
                    self.walk_thir_expr(thir, *arg, deps);
                }
            }

            // ADT construction (struct, enum variant).
            ExprKind::Adt(adt_expr) => {
                self.maybe_add_dep(adt_expr.adt_def.did(), deps);
                // Walk field values.
                for field in &adt_expr.fields {
                    self.walk_thir_expr(thir, field.expr, deps);
                }
                // Walk base expression if present (struct update syntax).
                match &adt_expr.base {
                    thir::AdtExprBase::Base(base) => {
                        self.walk_thir_expr(thir, base.base, deps);
                    }
                    thir::AdtExprBase::DefaultFields(_)
                    | thir::AdtExprBase::None => {}
                }
            }

            // Closures - recurse into their body.
            ExprKind::Closure(closure_expr) => {
                // The closure has its own DefId and THIR body.
                // closure_id is already a LocalDefId.
                self.collect_thir_body_deps(closure_expr.closure_id, deps);
            }

            // Match expressions - walk arms and scrutinee.
            ExprKind::Match {
                scrutinee, arms, ..
            } => {
                self.walk_thir_expr(thir, *scrutinee, deps);
                for arm_id in arms {
                    let arm = &thir.arms[*arm_id];
                    // Walk the pattern - this is where const patterns appear.
                    self.walk_thir_pattern(&arm.pattern, deps);
                    // Walk guard if present.
                    if let Some(guard) = &arm.guard {
                        self.walk_thir_expr(thir, *guard, deps);
                    }
                    // Walk body.
                    self.walk_thir_expr(thir, arm.body, deps);
                }
            }

            // Let expressions (if-let, let-else).
            ExprKind::Let { expr, pat } => {
                self.walk_thir_expr(thir, *expr, deps);
                self.walk_thir_pattern(pat, deps);
            }

            // Block - walk statements and optional tail expression.
            ExprKind::Block { block } => {
                let block = &thir.blocks[*block];
                for stmt_id in &block.stmts {
                    let stmt = &thir.stmts[*stmt_id];
                    match &stmt.kind {
                        thir::StmtKind::Let {
                            pattern,
                            initializer,
                            else_block,
                            ..
                        } => {
                            self.walk_thir_pattern(pattern, deps);
                            if let Some(init) = initializer {
                                self.walk_thir_expr(thir, *init, deps);
                            }
                            // else_block is a BlockId for let-else syntax.
                            // Walk the block's statements and optional tail expr.
                            if let Some(else_block_id) = else_block {
                                let else_blk = &thir.blocks[*else_block_id];
                                for else_stmt_id in &else_blk.stmts {
                                    let else_stmt = &thir.stmts[*else_stmt_id];
                                    if let thir::StmtKind::Expr {
                                        expr, ..
                                    } = &else_stmt.kind
                                    {
                                        self.walk_thir_expr(thir, *expr, deps);
                                    }
                                }
                                if let Some(tail) = else_blk.expr {
                                    self.walk_thir_expr(thir, tail, deps);
                                }
                            }
                        }
                        thir::StmtKind::Expr { expr, .. } => {
                            self.walk_thir_expr(thir, *expr, deps);
                        }
                    }
                }
                if let Some(tail) = block.expr {
                    self.walk_thir_expr(thir, tail, deps);
                }
            }

            // Binary operations, logical ops, and assignments all have lhs/rhs.
            ExprKind::Binary { lhs, rhs, .. }
            | ExprKind::LogicalOp { lhs, rhs, .. }
            | ExprKind::Assign { lhs, rhs }
            | ExprKind::AssignOp { lhs, rhs, .. } => {
                self.walk_thir_expr(thir, *lhs, deps);
                self.walk_thir_expr(thir, *rhs, deps);
            }

            // Unary operations.
            ExprKind::Unary { arg, .. }
            | ExprKind::Cast { source: arg }
            | ExprKind::PointerCoercion { source: arg, .. }
            | ExprKind::Use { source: arg }
            | ExprKind::Borrow { arg, .. }
            | ExprKind::RawBorrow { arg, .. }
            | ExprKind::Deref { arg }
            | ExprKind::Repeat { value: arg, .. }
            | ExprKind::Field { lhs: arg, .. }
            | ExprKind::Box { value: arg }
            | ExprKind::PlaceTypeAscription { source: arg, .. }
            | ExprKind::ValueTypeAscription { source: arg, .. } => {
                self.walk_thir_expr(thir, *arg, deps);
            }

            // Index operation.
            ExprKind::Index { lhs, index } => {
                self.walk_thir_expr(thir, *lhs, deps);
                self.walk_thir_expr(thir, *index, deps);
            }

            // If expression.
            ExprKind::If {
                cond,
                then,
                else_opt,
                ..
            } => {
                self.walk_thir_expr(thir, *cond, deps);
                self.walk_thir_expr(thir, *then, deps);
                if let Some(else_expr) = else_opt {
                    self.walk_thir_expr(thir, *else_expr, deps);
                }
            }

            // Loop expressions.
            ExprKind::Loop { body } => {
                self.walk_thir_expr(thir, *body, deps);
            }

            // Return/break with optional value.
            ExprKind::Return { value } | ExprKind::Break { value, .. } => {
                if let Some(val) = value {
                    self.walk_thir_expr(thir, *val, deps);
                }
            }

            // Array/tuple expressions.
            ExprKind::Array { fields } | ExprKind::Tuple { fields } => {
                for field in fields {
                    self.walk_thir_expr(thir, *field, deps);
                }
            }

            // Inline asm outputs/inputs.
            ExprKind::InlineAsm(asm_expr) => {
                for op in &asm_expr.operands {
                    match op {
                        thir::InlineAsmOperand::In { expr, .. }
                        | thir::InlineAsmOperand::Out {
                            expr: Some(expr),
                            ..
                        }
                        | thir::InlineAsmOperand::InOut { expr, .. } => {
                            self.walk_thir_expr(thir, *expr, deps);
                        }
                        thir::InlineAsmOperand::SplitInOut {
                            in_expr,
                            out_expr,
                            ..
                        } => {
                            self.walk_thir_expr(thir, *in_expr, deps);
                            if let Some(out) = out_expr {
                                self.walk_thir_expr(thir, *out, deps);
                            }
                        }
                        // Symbol operands reference functions or statics by symbol name.
                        thir::InlineAsmOperand::SymFn { value, .. } => {
                            // SymFn contains an expression that references the function.
                            self.walk_thir_expr(thir, *value, deps);
                        }
                        thir::InlineAsmOperand::SymStatic { def_id } => {
                            // SymStatic directly contains the DefId of the static.
                            self.maybe_add_dep(*def_id, deps);
                        }
                        // Out with no expr, Const, and Label don't contain expressions
                        // or dependencies we need to extract.
                        thir::InlineAsmOperand::Out { expr: None, .. }
                        | thir::InlineAsmOperand::Const { .. }
                        | thir::InlineAsmOperand::Label { .. } => {}
                    }
                }
            }

            // Expressions with a single sub-expression to walk.
            ExprKind::Scope { value, .. }
            | ExprKind::Become { value }
            | ExprKind::Yield { value }
            | ExprKind::ConstContinue { value, .. } => {
                self.walk_thir_expr(thir, *value, deps);
            }
            ExprKind::NeverToAny { source }
            | ExprKind::PlaceUnwrapUnsafeBinder { source }
            | ExprKind::ValueUnwrapUnsafeBinder { source }
            | ExprKind::WrapUnsafeBinder { source } => {
                self.walk_thir_expr(thir, *source, deps);
            }

            // ZstLiteral is used for zero-sized type values, including function items.
            // The function DefId is in the expression's type, not the expression kind.
            // For example, `helper` has type `fn() -> i32 {helper}` which is FnDef.
            // For turbofish like `generic::<S>()`, the type is `fn() {generic::<S>}`
            // which has both the function DefId and the type arguments.
            ExprKind::ZstLiteral { .. } => {
                if let ty::TyKind::FnDef(def_id, args) = expr.ty.kind() {
                    self.maybe_add_dep(*def_id, deps);
                    // Also capture type arguments (turbofish syntax).
                    for arg in *args {
                        if let Some(ty) = arg.as_type() {
                            self.collect_type_deps(ty, deps);
                        }
                    }
                }
            }

            // Const blocks (`const { ... }`) have their own body that may contain
            // references to other items. Walk the body to capture those dependencies.
            ExprKind::ConstBlock { did, .. } => {
                // did is a DefId; convert to LocalDefId if it's local.
                if let Some(local_did) = did.as_local() {
                    self.collect_thir_body_deps(local_did, deps);
                }
            }

            // ByUse expression (`x.use`) - walk the inner expression.
            ExprKind::ByUse { expr, .. } => {
                self.walk_thir_expr(thir, *expr, deps);
            }

            // Loop-match expression (experimental #[loop_match] attribute).
            // The state is an expression that gets matched on.
            ExprKind::LoopMatch { state, .. } => {
                self.walk_thir_expr(thir, *state, deps);
            }

            // Leaf expressions - no sub-expressions to walk.
            ExprKind::VarRef { .. }
            | ExprKind::UpvarRef { .. }
            | ExprKind::Literal { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ConstParam { .. }
            | ExprKind::Continue { .. } => {}
        }
    }

    /// Walk a THIR pattern and collect dependencies.
    ///
    /// This is critical for capturing const references in patterns like
    /// `matches!(x, MAGIC)` which MIR inlines to literal comparisons.
    fn walk_thir_pattern(
        &self,
        pat: &thir::Pat<'tcx>,
        deps: &mut HashSet<String>,
    ) {
        // Check for expanded_const in the pattern's extra field.
        // When a named const (like MAGIC) is used in a pattern, rustc lowers it
        // to a PatKind::Constant but stores the original DefId in Pat.extra.expanded_const.
        if let Some(ref extra) = pat.extra
            && let Some(def_id) = extra.expanded_const
        {
            self.maybe_add_dep(def_id, deps);
        }

        match &pat.kind {
            // Variant pattern - captures enum dependency.
            PatKind::Variant {
                adt_def,
                subpatterns,
                ..
            } => {
                self.maybe_add_dep(adt_def.did(), deps);
                for field_pat in subpatterns {
                    self.walk_thir_pattern(&field_pat.pattern, deps);
                }
            }

            // Leaf pattern (single-variant ADT).
            PatKind::Leaf { subpatterns } => {
                for field_pat in subpatterns {
                    self.walk_thir_pattern(&field_pat.pattern, deps);
                }
            }

            // Deref patterns - walk the subpattern.
            PatKind::Deref { subpattern, .. }
            | PatKind::DerefPattern { subpattern, .. } => {
                self.walk_thir_pattern(subpattern, deps);
            }

            // Binding pattern with possible subpattern.
            PatKind::Binding { subpattern, .. } => {
                if let Some(subpat) = subpattern {
                    self.walk_thir_pattern(subpat, deps);
                }
            }

            // Array/slice patterns.
            PatKind::Array {
                prefix,
                slice,
                suffix,
            }
            | PatKind::Slice {
                prefix,
                slice,
                suffix,
            } => {
                for p in prefix {
                    self.walk_thir_pattern(p, deps);
                }
                if let Some(slice_pat) = slice {
                    self.walk_thir_pattern(slice_pat, deps);
                }
                for p in suffix {
                    self.walk_thir_pattern(p, deps);
                }
            }

            // Or pattern (pat1 | pat2).
            PatKind::Or { pats } => {
                for p in pats {
                    self.walk_thir_pattern(p, deps);
                }
            }

            // Patterns without sub-patterns or dependencies.
            PatKind::Constant { .. }
            | PatKind::Wild
            | PatKind::Missing
            | PatKind::Range { .. }
            | PatKind::Never
            | PatKind::Error(_) => {}
        }
    }

    // -------------------------------------------------------------------------
    // Anchor extraction for impl blocks
    // -------------------------------------------------------------------------

    /// Extract anchors for an impl block (orphan rule compliance).
    ///
    /// For `impl<P1..=Pn> Trait<T1..=Tn> for T0`, anchors include:
    /// - T0 (self type) if it's a workspace-local ADT or contains one
    /// - The trait if it's workspace-local
    /// - T1..=Tn if they contain workspace-local ADTs
    ///
    /// Note: We check `is_workspace_crate` (not just `is_local`) because the
    /// orphan rule considers all crates in the workspace as "local" for our
    /// purposes - we can reorganize code across workspace crates.
    fn extract_impl_anchors(&self, def_id: DefId) -> HashSet<String> {
        let mut anchors = HashSet::new();

        // Self type (T0).
        let self_ty = self.tcx.type_of(def_id).skip_binder();
        self.collect_anchors_from_type(self_ty, &mut anchors);

        // Trait and its type parameters (T1..=Tn).
        if matches!(self.tcx.def_kind(def_id), DefKind::Impl { of_trait: true })
        {
            let trait_ref = self.tcx.impl_trait_ref(def_id).skip_binder();

            // Trait itself.
            if self.is_workspace_crate(trait_ref.def_id) {
                anchors.insert(self.raw_def_path(trait_ref.def_id));
            }

            // Trait type parameters (skip Self which is first).
            for arg in trait_ref.args.iter().skip(1) {
                if let Some(ty) = arg.as_type() {
                    self.collect_anchors_from_type(ty, &mut anchors);
                }
            }
        }

        anchors
    }

    /// Collect workspace-local ADTs from a type for anchor purposes.
    ///
    /// Handles fundamental types (`&T`, `Box<T>`, `Pin<T>`) by unwrapping them
    /// to find the inner local type. Also recurses into type parameters of
    /// non-fundamental external types to find "uncovered" local types.
    fn collect_anchors_from_type(
        &self,
        ty: Ty<'tcx>,
        anchors: &mut HashSet<String>,
    ) {
        // Peel references - &T and &mut T are fundamental.
        let ty = ty.peel_refs();

        match ty.kind() {
            ty::TyKind::Adt(adt_def, args) => {
                if adt_def.is_fundamental() {
                    // Fundamental types (Box, Pin, etc.) - recurse into type args.
                    for arg in *args {
                        if let Some(inner_ty) = arg.as_type() {
                            self.collect_anchors_from_type(inner_ty, anchors);
                        }
                    }
                } else if self.is_workspace_crate(adt_def.did()) {
                    // Workspace-local ADT - this is an anchor.
                    anchors.insert(self.raw_def_path(adt_def.did()));
                } else {
                    // External non-fundamental ADT (like Vec, HashMap).
                    // Under orphan rules, local types in type parameters can still
                    // be anchors (uncovered types). Recurse to find them.
                    for arg in *args {
                        if let Some(inner_ty) = arg.as_type() {
                            self.collect_anchors_from_type(inner_ty, anchors);
                        }
                    }
                }
            }
            ty::TyKind::Tuple(tys) => {
                // Tuples are fundamental - recurse into elements.
                for elem_ty in *tys {
                    self.collect_anchors_from_type(elem_ty, anchors);
                }
            }
            ty::TyKind::Array(elem_ty, _) | ty::TyKind::Slice(elem_ty) => {
                // Arrays and slices - recurse into element type.
                self.collect_anchors_from_type(*elem_ty, anchors);
            }
            ty::TyKind::Dynamic(predicates, _) => {
                // dyn Trait - if the trait is workspace-local, it's an anchor.
                for pred in *predicates {
                    if let ty::ExistentialPredicate::Trait(trait_ref) =
                        pred.skip_binder()
                        && self.is_workspace_crate(trait_ref.def_id)
                    {
                        anchors.insert(self.raw_def_path(trait_ref.def_id));
                    }
                }
            }
            // Primitive and other types that can't be local anchors.
            // We list them explicitly to catch new TyKind variants.
            ty::TyKind::Bool
            | ty::TyKind::Char
            | ty::TyKind::Int(_)
            | ty::TyKind::Uint(_)
            | ty::TyKind::Float(_)
            | ty::TyKind::Str
            | ty::TyKind::Never
            | ty::TyKind::Param(_)
            | ty::TyKind::Bound(_, _)
            | ty::TyKind::Placeholder(_)
            | ty::TyKind::Infer(_)
            | ty::TyKind::Error(_)
            | ty::TyKind::Foreign(_)
            | ty::TyKind::RawPtr(_, _)
            | ty::TyKind::Ref(_, _, _)
            | ty::TyKind::FnDef(_, _)
            | ty::TyKind::FnPtr(_, _)
            | ty::TyKind::Closure(_, _)
            | ty::TyKind::CoroutineClosure(_, _)
            | ty::TyKind::Coroutine(_, _)
            | ty::TyKind::CoroutineWitness(_, _)
            | ty::TyKind::Alias(_, _)
            | ty::TyKind::Pat(_, _)
            | ty::TyKind::UnsafeBinder(_) => {}
        }
    }

    // -------------------------------------------------------------------------
    // Helper methods
    // -------------------------------------------------------------------------

    /// Add a dependency if the `def_id` belongs to a workspace crate.
    fn maybe_add_dep(&self, def_id: DefId, deps: &mut HashSet<String>) {
        if self.is_workspace_crate(def_id) {
            // Normalize the def_id (variants → enum, assoc items → container).
            let normalized = self.normalize_def_id(def_id);

            // Use raw DefPath format for consistency with symbol keys.
            let path = self.raw_def_path(normalized);

            deps.insert(path);
        }
    }

    /// Check if a `def_id` belongs to a workspace crate.
    ///
    /// This returns true for the current crate (`LOCAL_CRATE`) and for any
    /// external crate whose name is in the `workspace_crates` list.
    /// Used for dependency extraction where we want cross-crate workspace deps.
    fn is_workspace_crate(&self, def_id: DefId) -> bool {
        if def_id.krate == LOCAL_CRATE {
            return true;
        }
        // Check if this is another workspace crate.
        let crate_name = self.tcx.crate_name(def_id.krate);
        self.workspace_crates.contains(crate_name.as_str())
    }

    /// Check if a `def_id` is nested inside a function, closure, or const body.
    ///
    /// Items defined inside function/closure bodies (nested functions, statics from
    /// macro expansion like tracing's `__CALLSITE`, structs from `tokio::select!`)
    /// can't be split independently from their parent. We skip extracting them and
    /// collapse any dependencies on them to the containing function.
    fn is_nested_in_body(&self, def_id: DefId) -> bool {
        let mut current = def_id;
        loop {
            let parent = self.tcx.parent(current);
            // Reached the crate root - not nested.
            if parent == current || parent.is_crate_root() {
                return false;
            }
            // Check if parent is a function, closure, or const body.
            match self.tcx.def_kind(parent) {
                DefKind::Fn
                | DefKind::AssocFn
                | DefKind::Const
                | DefKind::AnonConst
                | DefKind::Closure => {
                    return true;
                }
                _ => {
                    // Keep walking up. Modules can be defined inside closures
                    // (e.g., `tokio::select!` generates `__tokio_select_util`),
                    // so we can't stop at DefKind::Mod boundaries.
                    current = parent;
                }
            }
        }
    }

    /// Find the containing function if this `def_id` is nested inside one.
    ///
    /// For items inside closures, returns the function containing the closure.
    fn find_containing_function(&self, def_id: DefId) -> Option<DefId> {
        let mut current = def_id;
        loop {
            let parent = self.tcx.parent(current);
            if parent == current || parent.is_crate_root() {
                return None;
            }
            match self.tcx.def_kind(parent) {
                DefKind::Fn | DefKind::AssocFn => return Some(parent),
                // Keep walking through closures and modules - they can be nested
                // inside functions (e.g., `tokio::select!` generates modules
                // inside closures inside functions).
                _ => current = parent,
            }
        }
    }

    /// Normalize a `def_id`: collapse variants to their enum, assoc items to container,
    /// and nested items to their containing function.
    fn normalize_def_id(&self, def_id: DefId) -> DefId {
        // First check if this is nested in a function body.
        if let Some(containing_fn) = self.find_containing_function(def_id) {
            return containing_fn;
        }

        match self.tcx.def_kind(def_id) {
            // Variants and constructors → parent enum/struct.
            DefKind::Variant | DefKind::Ctor(..) => self.tcx.parent(def_id),

            // Associated items → their impl/trait.
            DefKind::AssocFn | DefKind::AssocConst | DefKind::AssocTy => {
                self.tcx.parent(def_id)
            }

            // Everything else stays as-is.
            _ => def_id,
        }
    }

    /// Get the string path for a `def_id`.
    fn def_path_str(&self, def_id: DefId) -> String {
        self.tcx.def_path_str(def_id)
    }

    /// Get the raw `DefPath` string for symbol keys.
    ///
    /// Returns the path in the same format that rustc's self-profile uses,
    /// e.g., `crate::module::_[7]::{{impl}}` for derive-generated impls.
    /// This differs from `def_path_str` which returns human-readable
    /// `<Type as Trait>` format.
    ///
    /// The profile format uses:
    /// - `{{impl}}` or `{{impl}}[N]` for impl blocks
    /// - `_` or `_[N]` for anonymous items (from derive macros)
    /// - Double braces `{{...}}` for anonymous/generated items
    fn raw_def_path(&self, def_id: DefId) -> String {
        use rustc_hir::definitions::DefPathData;

        let def_path = self.tcx.def_path(def_id);
        let mut segments = Vec::new();

        // Add crate name as first segment.
        let crate_name = if def_id.is_local() {
            self.crate_name.clone()
        } else {
            self.tcx.crate_name(def_id.krate).to_string()
        };
        segments.push(crate_name);

        // Add each path segment in profile-compatible format.
        for disambiguated in &def_path.data {
            let disambiguator = disambiguated.disambiguator;
            let segment = match &disambiguated.data {
                DefPathData::CrateRoot => continue, // Skip, we added crate name
                DefPathData::TypeNs(sym)
                | DefPathData::ValueNs(sym)
                | DefPathData::MacroNs(sym)
                | DefPathData::LifetimeNs(sym) => sym.to_string(),
                DefPathData::Impl => {
                    // Format as {{impl}} or {{impl}}[N] for disambiguation.
                    if disambiguator == 0 {
                        "{{impl}}".to_string()
                    } else {
                        format!("{{{{impl}}}}[{disambiguator}]")
                    }
                }
                DefPathData::ForeignMod => "{{extern}}".to_string(),
                DefPathData::Use => "{{use}}".to_string(),
                DefPathData::GlobalAsm => "{{global_asm}}".to_string(),
                DefPathData::Closure => {
                    if disambiguator == 0 {
                        "{{closure}}".to_string()
                    } else {
                        format!("{{{{closure}}}}[{disambiguator}]")
                    }
                }
                DefPathData::Ctor => "{{constructor}}".to_string(),
                DefPathData::AnonConst | DefPathData::LateAnonConst => {
                    // Anonymous constants use _ or _[N] format in profiles.
                    if disambiguator == 0 {
                        "_".to_string()
                    } else {
                        format!("_[{disambiguator}]")
                    }
                }
                DefPathData::OpaqueTy => {
                    if disambiguator == 0 {
                        "{{opaque}}".to_string()
                    } else {
                        format!("{{{{opaque}}}}[{disambiguator}]")
                    }
                }
                DefPathData::AnonAssocTy(sym) => {
                    format!("{{{{assoc_ty:{sym}}}}}")
                }
                DefPathData::SyntheticCoroutineBody => {
                    "{{coroutine}}".to_string()
                }
                DefPathData::NestedStatic => "{{nested_static}}".to_string(),
                DefPathData::OpaqueLifetime(sym) => {
                    format!("{{{{lifetime:{sym}}}}}")
                }
                DefPathData::DesugaredAnonymousLifetime => {
                    "{{anon_lifetime}}".to_string()
                }
            };
            segments.push(segment);
        }

        segments.join("::")
    }

    /// Get a simple type/trait name without module path.
    ///
    /// For local items, returns the simple name (e.g., `MyType`).
    /// For external items, returns just the item name (e.g., `Box` instead of `std::boxed::Box`).
    /// This avoids `::` in impl names which would break `into_module()`.
    fn simple_type_name(&self, def_id: DefId) -> String {
        // Get the item name directly from the DefId.
        self.tcx.item_name(def_id).to_string()
    }

    /// Get the source file for a local `def_id`.
    fn source_file(&self, local_def_id: LocalDefId) -> String {
        let span = self.tcx.def_span(local_def_id.to_def_id());
        let source_map = self.tcx.sess.source_map();

        // span_to_location_info returns (Option<Arc<SourceFile>>, line, col, end_line, end_col).
        let (file_opt, _, _, _, _) = source_map.span_to_location_info(span);
        if let Some(file) = file_opt {
            // Try to get a relative path from the file name.
            return file
                .name
                .prefer_remapped_unconditionally()
                .to_string_lossy()
                .into_owned();
        }

        // Fallback.
        "unknown".to_string()
    }

    /// Extract visibility for a `def_id`.
    fn extract_visibility(&self, def_id: DefId) -> Visibility {
        match self.tcx.visibility(def_id) {
            TyVisibility::Public => Visibility::Public,
            TyVisibility::Restricted(_) => Visibility::NonPublic,
        }
    }

    /// Generate a name for an impl block.
    ///
    /// Uses simple names (without module paths) to avoid `::` which would
    /// break `into_module()`. Also includes trait type parameters for impls
    /// like `impl From<LocalType> for i32`.
    fn impl_name(&self, def_id: DefId) -> String {
        let self_ty = self.tcx.type_of(def_id).skip_binder();
        let self_ty_str = self.format_ty(self_ty);

        if matches!(self.tcx.def_kind(def_id), DefKind::Impl { of_trait: true })
        {
            let trait_ref = self.tcx.impl_trait_ref(def_id).skip_binder();
            // Use simple trait name to avoid `::`.
            let trait_name = self.simple_type_name(trait_ref.def_id);
            // Include trait type parameters (e.g., `From<LocalType>`).
            let trait_args: Vec<_> = trait_ref
                .args
                .iter()
                .skip(1) // Skip Self
                .filter_map(ty::GenericArg::as_type)
                .map(|t| self.format_ty(t))
                .collect();
            let trait_with_args = if trait_args.is_empty() {
                trait_name
            } else {
                format!("{}<{}>", trait_name, trait_args.join(", "))
            };
            // Check for negative impls (impl !Trait for Type).
            let polarity = self.tcx.impl_polarity(def_id);
            let neg_prefix = if polarity == ty::ImplPolarity::Negative {
                "!"
            } else {
                ""
            };
            // Check for unsafe impls (unsafe impl Trait for Type).
            // An impl is unsafe if the trait being implemented is an unsafe trait.
            let is_unsafe_trait =
                self.tcx.trait_def(trait_ref.def_id).safety.is_unsafe();
            let unsafe_prefix = if is_unsafe_trait { "unsafe " } else { "" };
            format!(
                "{unsafe_prefix}impl {neg_prefix}{trait_with_args} for {self_ty_str}"
            )
        } else {
            format!("impl {self_ty_str}")
        }
    }

    /// Format a type as a human-readable string.
    ///
    /// This produces clean names like `[Foo; 3]` instead of the debug
    /// representation `[Foo; 3_usize]` or `dyn Trait` instead of
    /// `dyn [Binder { value: Trait(...), bound_vars: [] }]`.
    fn format_ty(&self, ty: Ty<'tcx>) -> String {
        match ty.kind() {
            // ADT types: use simple name to avoid `::` in impl names.
            ty::TyKind::Adt(adt_def, args) => {
                // Use just the type name, not the full path like `std::boxed::Box`.
                // This avoids `::` in impl names which would break into_module().
                let base = self.simple_type_name(adt_def.did());
                // Only include generic args if non-empty.
                if args.is_empty() {
                    base
                } else {
                    let args_str: Vec<_> = args
                        .iter()
                        .filter_map(|arg| self.format_generic_arg(arg))
                        .collect();
                    if args_str.is_empty() {
                        base
                    } else {
                        format!("{}<{}>", base, args_str.join(", "))
                    }
                }
            }
            // Arrays: [T; N]
            ty::TyKind::Array(elem_ty, len) => {
                let elem = self.format_ty(*elem_ty);
                // Try to extract the constant value for the length.
                let len_str = len
                    .try_to_target_usize(self.tcx)
                    .map_or_else(|| format!("{len:?}"), |n| n.to_string());
                format!("[{elem}; {len_str}]")
            }
            // Slices: [T]
            ty::TyKind::Slice(elem_ty) => {
                let elem = self.format_ty(*elem_ty);
                format!("[{elem}]")
            }
            // Tuples: (T1, T2, ...)
            ty::TyKind::Tuple(tys) => {
                let elems: Vec<_> =
                    tys.iter().map(|t| self.format_ty(t)).collect();
                format!("({})", elems.join(", "))
            }
            // References: &T, &mut T
            ty::TyKind::Ref(_, inner_ty, mutbl) => {
                let inner = self.format_ty(*inner_ty);
                if mutbl.is_mut() {
                    format!("&mut {inner}")
                } else {
                    format!("&{inner}")
                }
            }
            // Raw pointers: *const T, *mut T
            ty::TyKind::RawPtr(inner_ty, mutbl) => {
                let inner = self.format_ty(*inner_ty);
                if mutbl.is_mut() {
                    format!("*mut {inner}")
                } else {
                    format!("*const {inner}")
                }
            }
            // Dynamic trait objects: dyn Trait
            ty::TyKind::Dynamic(predicates, _) => {
                let traits: Vec<_> = predicates
                    .iter()
                    .filter_map(|pred| {
                        if let ty::ExistentialPredicate::Trait(trait_ref) =
                            pred.skip_binder()
                        {
                            // Use simple name to avoid `::`.
                            Some(self.simple_type_name(trait_ref.def_id))
                        } else {
                            None
                        }
                    })
                    .collect();
                if traits.is_empty() {
                    "dyn ?".to_string()
                } else {
                    format!("dyn {}", traits.join(" + "))
                }
            }
            // Function pointers: fn(T) -> R
            ty::TyKind::FnPtr(sig_tys, _) => {
                let sig = sig_tys.skip_binder();
                let inputs: Vec<_> =
                    sig.inputs().iter().map(|t| self.format_ty(*t)).collect();
                let output = self.format_ty(sig.output());
                if output == "()" {
                    format!("fn({})", inputs.join(", "))
                } else {
                    format!("fn({}) -> {}", inputs.join(", "), output)
                }
            }
            // Type parameters: T, U, etc.
            ty::TyKind::Param(param_ty) => param_ty.name.to_string(),
            // Primitives and other types: use debug format but clean it up.
            _ => {
                // Fall back to debug format for other types.
                format!("{ty:?}")
            }
        }
    }

    /// Format a generic argument (type, lifetime, or const) as a string.
    ///
    /// Returns `None` for arguments that should be elided (e.g., late-bound or
    /// erased lifetimes). This is used when formatting generic type parameters
    /// like `MyType<'a, T>` to include all visible generic arguments.
    fn format_generic_arg(&self, arg: ty::GenericArg<'tcx>) -> Option<String> {
        // Try type argument first.
        if let Some(ty) = arg.as_type() {
            return Some(self.format_ty(ty));
        }

        // Try lifetime argument.
        if let Some(region) = arg.as_region() {
            return match region.kind() {
                // Named early-bound lifetimes: include in output.
                // The name already includes the leading `'`.
                ty::RegionKind::ReEarlyParam(early) => {
                    Some(early.name.to_string())
                }
                // Static lifetime: always include.
                ty::RegionKind::ReStatic => Some("'static".to_string()),
                // Other lifetime kinds (late-bound, erased, error, etc.): skip.
                _ => None,
            };
        }

        // Try const argument.
        if let Some(ct) = arg.as_const() {
            return ct
                .try_to_target_usize(self.tcx)
                .map(|n| n.to_string())
                .or_else(|| Some(format!("{ct:?}")));
        }

        None
    }

    /// Convert the extractor state into the final extraction result.
    ///
    /// Builds a nested Module hierarchy from the flat symbol map and returns
    /// it along with the profile mapping.
    fn into_result(self) -> ExtractionResult {
        let mut root = Module {
            symbols: HashMap::new(),
            submodules: HashMap::new(),
        };

        for (path, symbol) in self.symbols {
            tracing::debug!(path, "converting to module");

            // Split path into segments.
            let raw_segments: Vec<&str> = path.split("::").collect();

            if raw_segments.is_empty() {
                continue;
            }

            // Get the symbol name (last segment) - preserve as-is including generics.
            // For impl blocks like "impl Trait for Type<T>", the <T> is part of the name.
            let symbol_name = raw_segments.last().copied().unwrap_or("");

            // For module path segments (all except last), strip generic parameters.
            // `def_path_str` returns paths like `Foo<'a>::bar` or `Foo::<T>::baz`
            // which would incorrectly create submodules like `Foo<'a>` or `<T>`.
            let module_segments: Vec<&str> = raw_segments
                [..raw_segments.len() - 1]
                .iter()
                .map(|s| s.split('<').next().unwrap_or(s))
                .filter(|s| !s.is_empty())
                .collect();

            // Skip the crate name (first segment) only if present.
            let module_segments: &[&str] =
                if module_segments.first() == Some(&self.crate_name.as_str()) {
                    &module_segments[1..]
                } else {
                    &module_segments
                };

            // Navigate to the correct module, creating submodules as needed.
            let mut current = &mut root;
            for &segment in module_segments {
                current =
                    current.submodules.entry(segment.to_string()).or_default();
            }

            if !symbol_name.is_empty() {
                current.symbols.insert(symbol_name.to_string(), symbol);
            }
        }

        ExtractionResult { module: root }
    }
}
