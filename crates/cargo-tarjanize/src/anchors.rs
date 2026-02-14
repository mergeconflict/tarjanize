//! Impl block anchor extraction for orphan rule compliance.
//!
//! Anchors determine which workspace crates "own" an impl block under the
//! orphan rule. During the condense phase, at least one anchor must remain
//! in the same crate as its impl to keep the code valid.
//!
//! This module is split from `extract.rs` to isolate the anchor-specific
//! type traversal logic from the broader symbol extraction machinery.

use std::collections::HashSet;

use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Ty, TyCtxt};

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
///
/// `is_workspace_crate` and `raw_def_path` are callbacks into the
/// `Extractor` that owns the workspace-crate set and `DefPath` cache.
///
/// Why: anchors constrain how impls can move during condense.
pub(crate) fn extract_impl_anchors(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    is_workspace_crate: &impl Fn(DefId) -> bool,
    raw_def_path: &impl Fn(DefId) -> String,
) -> HashSet<String> {
    let mut anchors = HashSet::new();

    // Self type (T0).
    let self_ty = tcx.type_of(def_id).skip_binder();
    collect_anchors_from_type(
        tcx,
        self_ty,
        &mut anchors,
        is_workspace_crate,
        raw_def_path,
    );

    // Trait and its type parameters (T1..=Tn).
    if matches!(tcx.def_kind(def_id), DefKind::Impl { of_trait: true }) {
        let trait_ref = tcx.impl_trait_ref(def_id).skip_binder();

        // Trait itself.
        if is_workspace_crate(trait_ref.def_id) {
            anchors.insert(raw_def_path(trait_ref.def_id));
        }

        // Trait type parameters (skip Self which is first).
        for arg in trait_ref.args.iter().skip(1) {
            if let Some(ty) = arg.as_type() {
                collect_anchors_from_type(
                    tcx,
                    ty,
                    &mut anchors,
                    is_workspace_crate,
                    raw_def_path,
                );
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
///
/// Why: orphan rule anchors are derived from local type structure.
#[expect(
    clippy::only_used_in_recursion,
    reason = "tcx is threaded through recursion for the caller \
              (extract_impl_anchors) which uses it directly"
)]
pub(crate) fn collect_anchors_from_type(
    tcx: TyCtxt<'_>,
    ty: Ty<'_>,
    anchors: &mut HashSet<String>,
    is_workspace_crate: &impl Fn(DefId) -> bool,
    raw_def_path: &impl Fn(DefId) -> String,
) {
    // Peel references - &T and &mut T are fundamental.
    let ty = ty.peel_refs();

    match ty.kind() {
        ty::TyKind::Adt(adt_def, args) => {
            if adt_def.is_fundamental() {
                // Fundamental types (Box, Pin, etc.) - recurse into type args.
                for arg in *args {
                    if let Some(inner_ty) = arg.as_type() {
                        collect_anchors_from_type(
                            tcx,
                            inner_ty,
                            anchors,
                            is_workspace_crate,
                            raw_def_path,
                        );
                    }
                }
            } else if is_workspace_crate(adt_def.did()) {
                // Workspace-local ADT - this is an anchor.
                anchors.insert(raw_def_path(adt_def.did()));
            } else {
                // External non-fundamental ADT (like Vec, HashMap).
                // Under orphan rules, local types in type parameters can still
                // be anchors (uncovered types). Recurse to find them.
                for arg in *args {
                    if let Some(inner_ty) = arg.as_type() {
                        collect_anchors_from_type(
                            tcx,
                            inner_ty,
                            anchors,
                            is_workspace_crate,
                            raw_def_path,
                        );
                    }
                }
            }
        }
        ty::TyKind::Tuple(tys) => {
            // Tuples are fundamental - recurse into elements.
            for elem_ty in *tys {
                collect_anchors_from_type(
                    tcx,
                    elem_ty,
                    anchors,
                    is_workspace_crate,
                    raw_def_path,
                );
            }
        }
        ty::TyKind::Array(elem_ty, _) | ty::TyKind::Slice(elem_ty) => {
            // Arrays and slices - recurse into element type.
            collect_anchors_from_type(
                tcx,
                *elem_ty,
                anchors,
                is_workspace_crate,
                raw_def_path,
            );
        }
        ty::TyKind::Dynamic(predicates, _) => {
            // dyn Trait - if the trait is workspace-local, it's an anchor.
            for pred in *predicates {
                if let ty::ExistentialPredicate::Trait(trait_ref) =
                    pred.skip_binder()
                    && is_workspace_crate(trait_ref.def_id)
                {
                    anchors.insert(raw_def_path(trait_ref.def_id));
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
