//! Path transformation from crate-local to workspace-qualified format.
//!
//! Symbol dependencies and impl anchors use crate-local paths like
//! `crate_name::module::symbol`. This module transforms them into
//! workspace-qualified paths like `[package-name/target]::module::symbol`,
//! resolving same-package references by walking module trees to find the
//! correct target.

use std::collections::HashMap;

use tarjanize_schemas::{Package, SymbolKind};

/// Transform symbol paths from crate-name format to package/target format.
///
/// Symbol dependencies and impl anchors use paths like
/// `crate_name::module::symbol`. This function transforms them to
/// `[package-name/target]::module::symbol` using two resolution strategies:
///
/// 1. **Same-package references**: When the crate mapping points to the
///    current package, walk the module trees of all targets in that package
///    to find which target actually contains the referenced symbol.
/// 2. **Cross-package references**: Use the crate mapping directly (the
///    mapping always points to lib for cross-crate `use` resolution).
///
/// The same-package lookup is necessary because `write_crate_mapping()`
/// creates a 1:1 crate-name → target mapping, always preferring lib. When
/// a package has multiple targets sharing the same crate name (e.g.,
/// lib + test), a `crate::foo::Bar` reference in the test target might
/// point at a symbol that only exists in the test target, not lib.
///
/// Why: accurate target attribution prevents schedule and condense errors.
pub(crate) fn transform_symbol_paths(
    packages: &mut HashMap<String, Package>,
    crate_mapping: &HashMap<String, String>,
) {
    // Build a reverse mapping: "package/target" → "package-name" so
    // transform_path can detect same-package references.
    let target_to_package: HashMap<&str, &str> = crate_mapping
        .values()
        .filter_map(|target_id| {
            target_id
                .split_once('/')
                .map(|(pkg, _)| (target_id.as_str(), pkg))
        })
        .collect();

    // Process each package independently. For same-package lookups we
    // need a read-only snapshot of the module trees, so we clone the
    // root modules before mutating.
    for (pkg_name, package) in packages.iter_mut() {
        // Build snapshot: Vec<(target_key, cloned_root_module)> for
        // this package. Cloning the Module trees is cheap relative to
        // the O(deps * targets * path_depth) lookup cost.
        let target_snapshot: Vec<(String, tarjanize_schemas::Module)> = package
            .targets
            .iter()
            .map(|(key, target)| (key.clone(), target.root.clone()))
            .collect();

        // Build the borrowed slice that find_symbol_target expects.
        let target_refs: Vec<(&str, &tarjanize_schemas::Module)> =
            target_snapshot
                .iter()
                .map(|(key, root)| (key.as_str(), root))
                .collect();

        for crate_data in package.targets.values_mut() {
            transform_module_paths(
                &mut crate_data.root,
                crate_mapping,
                &target_to_package,
                pkg_name,
                &target_refs,
            );
        }
    }
}

/// Transform paths in a module and its submodules recursively.
///
/// `current_package` and `target_refs` enable same-package symbol lookup:
/// when a dependency path maps to the current package via the crate
/// mapping, `find_symbol_target` walks the module trees to find the
/// correct target.
///
/// Why: dependency rewrites must be applied consistently across the tree.
fn transform_module_paths(
    module: &mut tarjanize_schemas::Module,
    crate_mapping: &HashMap<String, String>,
    target_to_package: &HashMap<&str, &str>,
    current_package: &str,
    target_refs: &[(&str, &tarjanize_schemas::Module)],
) {
    for symbol in module.symbols.values_mut() {
        // Transform dependencies.
        symbol.dependencies = symbol
            .dependencies
            .iter()
            .map(|dep| {
                transform_path(
                    dep,
                    crate_mapping,
                    target_to_package,
                    current_package,
                    target_refs,
                )
            })
            .collect();

        // Transform impl anchors if this is an impl block.
        if let SymbolKind::Impl { anchors, .. } = &mut symbol.kind {
            *anchors = anchors
                .iter()
                .map(|anchor| {
                    transform_path(
                        anchor,
                        crate_mapping,
                        target_to_package,
                        current_package,
                        target_refs,
                    )
                })
                .collect();
        }
    }

    for submodule in module.submodules.values_mut() {
        transform_module_paths(
            submodule,
            crate_mapping,
            target_to_package,
            current_package,
            target_refs,
        );
    }
}

/// Transform a single symbol path from crate-name to package/target format.
///
/// Input:  `crate_name::module::symbol`
/// Output: `[package-name/target]::module::symbol`
///
/// For same-package references (where the crate mapping points back to the
/// current package), walks the module trees of all targets to find which
/// one actually contains the symbol. Falls back to the crate mapping
/// default if no target matches.
///
/// Cross-package references and external crates use the crate mapping
/// directly, or are returned unchanged respectively.
///
/// Why: standardizes dependency paths for later graph algorithms.
fn transform_path(
    path: &str,
    crate_mapping: &HashMap<String, String>,
    target_to_package: &HashMap<&str, &str>,
    current_package: &str,
    target_refs: &[(&str, &tarjanize_schemas::Module)],
) -> String {
    // Parse the crate name from the path (everything before the first
    // `::`)
    let Some((crate_name, rest)) = path.split_once("::") else {
        // No `::` in path — return unchanged.
        return path.to_string();
    };

    // Look up the target identifier from the crate mapping. If not
    // found, this is an external crate — return unchanged.
    let Some(default_target_id) = crate_mapping.get(crate_name) else {
        return path.to_string();
    };

    // Check whether this maps to the same package we're currently
    // processing. If so, try module-tree lookup for precise resolution.
    let mapped_package = target_to_package
        .get(default_target_id.as_str())
        .copied()
        .unwrap_or("");

    if mapped_package == current_package {
        // Same-package reference: search all targets for the symbol.
        if let Some(found_target) = find_symbol_target(target_refs, rest) {
            let target_id = format!("{current_package}/{found_target}");
            return format!("[{target_id}]::{rest}");
        }
    }

    // Cross-package or unresolvable same-package: use crate mapping
    // default.
    format!("[{default_target_id}]::{rest}")
}

/// Find which target in a package contains the symbol at `path`.
///
/// Tries three strategies in order against each target's root module:
/// 1. Exact symbol lookup — path resolves to a symbol leaf
/// 2. Impl-child truncation — strip `::method` suffix, retry as symbol
/// 3. Module lookup — path resolves as a submodule chain
///
/// Returns the target key (e.g. `"lib"`, `"test"`) of the first match,
/// or `None` if no target contains the path.
///
/// Why: same-package references need precise target resolution.
fn find_symbol_target<'a>(
    targets: &'a [(&'a str, &tarjanize_schemas::Module)],
    path: &str,
) -> Option<&'a str> {
    // Strategy 1: exact symbol match.
    for &(target_key, root) in targets {
        if module_contains_symbol(root, path) {
            return Some(target_key);
        }
    }

    // Strategy 2: impl-child — truncate to parent impl and retry.
    if let Some(parent_path) = truncate_impl_child(path) {
        for &(target_key, root) in targets {
            if module_contains_symbol(root, parent_path) {
                return Some(target_key);
            }
        }
    }

    // Strategy 3: module lookup — entire path resolves as submodules.
    for &(target_key, root) in targets {
        if module_contains_submodule(root, path) {
            return Some(target_key);
        }
    }

    None
}

/// Check whether a module tree contains a symbol at the given path.
///
/// Splits `path` on `::`, walks submodules for all but the last segment,
/// and checks the final segment against the module's `symbols` map. This
/// is O(`path_depth`) — one `HashMap` lookup per segment.
///
/// Why: used to resolve same-package references to the correct target.
fn module_contains_symbol(
    module: &tarjanize_schemas::Module,
    path: &str,
) -> bool {
    let segments: Vec<&str> = path.split("::").collect();
    let Some((leaf, parents)) = segments.split_last() else {
        return false;
    };

    // Walk submodules for the parent segments.
    let mut current = module;
    for &seg in parents {
        match current.submodules.get(seg) {
            Some(child) => current = child,
            None => return false,
        }
    }

    current.symbols.contains_key(*leaf)
}

/// Check whether a module tree contains a submodule chain matching `path`.
///
/// Splits `path` on `::` and walks the submodule tree for every segment.
/// Returns `true` if the entire chain resolves. Used for dependency paths
/// that reference a module rather than a leaf symbol.
///
/// Why: some dependencies reference modules, not leaf symbols.
fn module_contains_submodule(
    module: &tarjanize_schemas::Module,
    path: &str,
) -> bool {
    let mut current = module;
    for seg in path.split("::") {
        match current.submodules.get(seg) {
            Some(child) => current = child,
            None => return false,
        }
    }
    true
}

/// Strip an impl-child suffix from a path, returning the parent impl path.
///
/// Impl children look like `Foo::{{impl}}[0]::bar` where `::bar` is the
/// method name. This function finds the last `{{impl}}` marker, skips past
/// its `[N]` index suffix, and returns everything up to (but not including)
/// the next `::` — i.e. `Foo::{{impl}}[0]`.
///
/// Returns `None` if the path doesn't contain `{{impl}}` followed by a
/// `::` suffix (meaning it's not an impl-child path).
///
/// Why: impl methods must be attributed to their parent impl symbol.
fn truncate_impl_child(path: &str) -> Option<&str> {
    // Find the last `{{impl}}` marker in the path.
    let impl_start = path.rfind("{{impl}}")?;
    // Skip past `{{impl}}` (8 chars) to find the `[N]` index.
    let after_impl = impl_start + "{{impl}}".len();
    let rest = &path[after_impl..];

    // The index is `[N]` — find its closing bracket.
    if !rest.starts_with('[') {
        return None;
    }
    let bracket_end = rest.find(']')? + 1;
    let end_of_impl = after_impl + bracket_end;

    // Only truncate if there's a `::method` suffix after `[N]`.
    path[end_of_impl..]
        .starts_with("::")
        .then(|| &path[..end_of_impl])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a crate mapping for tests.
    ///
    /// Values are in `"package/target"` format (e.g., `"my-pkg/lib"`).
    fn make_mapping(entries: &[(&str, &str)]) -> HashMap<String, String> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// Call `transform_path` with no same-package context.
    ///
    /// Used by tests that only exercise cross-package or external-crate
    /// resolution, where module-tree lookup is irrelevant.
    fn transform_path_simple(
        path: &str,
        crate_mapping: &HashMap<String, String>,
    ) -> String {
        let empty_tp: HashMap<&str, &str> = HashMap::new();
        transform_path(path, crate_mapping, &empty_tp, "", &[])
    }

    // -- transform_path: lib target resolution --------------------------

    #[test]
    fn test_transform_path_lib_target() {
        // Lib crate name resolves to its lib target.
        let mapping = make_mapping(&[("my_crate", "my-package/lib")]);

        let result = transform_path_simple("my_crate::foo::bar::Baz", &mapping);

        assert_eq!(result, "[my-package/lib]::foo::bar::Baz");
    }

    #[test]
    fn test_transform_path_hyphenated_package_name() {
        // Package names with hyphens should be preserved in output.
        let mapping =
            make_mapping(&[("my_crate", "my-hyphenated-package/lib")]);

        let result = transform_path_simple("my_crate::Item", &mapping);

        assert_eq!(result, "[my-hyphenated-package/lib]::Item");
    }

    // -- transform_path: bin/test target resolution ---------------------

    #[test]
    fn test_transform_path_bin_target() {
        // Binary crate name resolves to its bin target via the mapping.
        let mapping =
            make_mapping(&[("ntp_admin", "omicron-ntp-admin/bin/ntp_admin")]);

        let result = transform_path_simple("ntp_admin::Args", &mapping);

        assert_eq!(result, "[omicron-ntp-admin/bin/ntp_admin]::Args");
    }

    #[test]
    fn test_transform_path_integration_test_target() {
        // Integration test crate name resolves to its test target.
        let mapping = make_mapping(&[(
            "v0_fsm_proptest_rack_coordinator",
            "bootstore/test/v0_fsm_proptest_rack_coordinator",
        )]);

        let result = transform_path_simple(
            "v0_fsm_proptest_rack_coordinator::common::foo",
            &mapping,
        );

        assert_eq!(
            result,
            "[bootstore/test/v0_fsm_proptest_rack_coordinator]\
             ::common::foo"
        );
    }

    // -- transform_path: cross-package and same-package resolution ------

    #[test]
    fn test_transform_path_cross_package_uses_lib() {
        // Cross-package dependency resolves to the dep's lib target.
        let mapping = make_mapping(&[
            ("ntp_admin", "omicron-ntp-admin/bin/ntp_admin"),
            ("other_crate", "other-package/lib"),
        ]);

        let result = transform_path_simple("other_crate::SomeStruct", &mapping);

        assert_eq!(result, "[other-package/lib]::SomeStruct");
    }

    #[test]
    fn test_transform_path_same_package_lib_from_bin() {
        // Binary references its own package's lib. The lib crate name
        // maps to the lib target, NOT the current bin target. This is
        // correct because `use omicron_ntp_admin::Foo` resolves to the
        // lib even when called from bin/ntp_admin.
        let mapping = make_mapping(&[
            ("omicron_ntp_admin", "omicron-ntp-admin/lib"),
            ("ntp_admin", "omicron-ntp-admin/bin/ntp_admin"),
        ]);

        let result = transform_path_simple(
            "omicron_ntp_admin::server::Config",
            &mapping,
        );

        assert_eq!(result, "[omicron-ntp-admin/lib]::server::Config");
    }

    // -- transform_path: external crates and edge cases -----------------

    #[test]
    fn test_transform_path_external_crate_unchanged() {
        // External crates (not in mapping) are returned unchanged.
        let mapping = make_mapping(&[("my_crate", "my-package/lib")]);

        let result = transform_path_simple("serde::Serialize", &mapping);

        assert_eq!(result, "serde::Serialize");
    }

    #[test]
    fn test_transform_path_no_colons_unchanged() {
        // Paths without `::` (just crate name) are returned unchanged.
        let mapping = make_mapping(&[("my_crate", "my-package/lib")]);

        let result = transform_path_simple("std", &mapping);

        assert_eq!(result, "std");
    }

    // -- transform_symbol_paths: same-package resolution ----------------

    /// Build a minimal symbol (`ModuleDef`) for testing module tree lookups.
    fn make_symbol(deps: &[&str]) -> tarjanize_schemas::Symbol {
        tarjanize_schemas::Symbol {
            file: "test.rs".to_string(),
            event_times_ms: HashMap::new(),
            dependencies: deps.iter().copied().map(String::from).collect(),
            kind: SymbolKind::ModuleDef {
                kind: "Struct".to_string(),
                visibility: tarjanize_schemas::Visibility::default(),
            },
        }
    }

    /// Build a minimal impl symbol with anchors for testing anchor
    /// resolution.
    fn make_impl_symbol(
        deps: &[&str],
        anchors: &[&str],
    ) -> tarjanize_schemas::Symbol {
        tarjanize_schemas::Symbol {
            file: "test.rs".to_string(),
            event_times_ms: HashMap::new(),
            dependencies: deps.iter().copied().map(String::from).collect(),
            kind: SymbolKind::Impl {
                name: "impl Test".to_string(),
                anchors: anchors.iter().copied().map(String::from).collect(),
            },
        }
    }

    /// Build a module containing the given symbol names.
    fn make_module(
        symbol_names: &[&str],
        submodules: &[(&str, tarjanize_schemas::Module)],
    ) -> tarjanize_schemas::Module {
        tarjanize_schemas::Module {
            symbols: symbol_names
                .iter()
                .map(|name| (name.to_string(), make_symbol(&[])))
                .collect(),
            submodules: submodules
                .iter()
                .map(|(name, module)| (name.to_string(), module.clone()))
                .collect(),
        }
    }

    #[test]
    fn test_transform_symbol_paths_lib_test_resolution() {
        // Symbol `TestType` exists only in the `test` target. A dep
        // referencing `my_pkg::test_mod::TestType` should resolve to
        // `[my-pkg/test]::test_mod::TestType`, not `[my-pkg/lib]`.
        let lib_root = make_module(&["LibType"], &[]);
        let test_root =
            make_module(&[], &[("test_mod", make_module(&["TestType"], &[]))]);

        // The referencing symbol lives in the test target and has a
        // same-crate dep path.
        let mut test_root_with_dep = test_root.clone();
        test_root_with_dep.symbols.insert(
            "test_fn".to_string(),
            make_symbol(&["my_pkg::test_mod::TestType"]),
        );

        let mut packages = HashMap::from([(
            "my-pkg".to_string(),
            Package {
                targets: HashMap::from([
                    (
                        "lib".to_string(),
                        tarjanize_schemas::Target {
                            root: lib_root,
                            ..Default::default()
                        },
                    ),
                    (
                        "test".to_string(),
                        tarjanize_schemas::Target {
                            root: test_root_with_dep,
                            ..Default::default()
                        },
                    ),
                ]),
            },
        )]);

        let mapping = make_mapping(&[("my_pkg", "my-pkg/lib")]);
        transform_symbol_paths(&mut packages, &mapping);

        // The dep should resolve to test target where TestType lives.
        let test_fn = packages["my-pkg"].targets["test"]
            .root
            .symbols
            .get("test_fn")
            .unwrap();
        assert!(
            test_fn
                .dependencies
                .contains("[my-pkg/test]::test_mod::TestType"),
            "expected [my-pkg/test]::test_mod::TestType, got: {:?}",
            test_fn.dependencies,
        );
    }

    #[test]
    fn test_transform_symbol_paths_impl_child_resolution() {
        // Dep path `my_pkg::Foo::{{impl}}[0]::bar` where `{{impl}}[0]`
        // exists under `Foo` in the *test* target only. The crate mapping
        // defaults to lib, so this forces impl-child truncation + lookup
        // to find the correct target.
        let lib_root = make_module(&["LibType"], &[]);
        let test_root =
            make_module(&[], &[("Foo", make_module(&["{{impl}}[0]"], &[]))]);

        let mut test_root_with_dep = test_root.clone();
        test_root_with_dep.symbols.insert(
            "test_fn".to_string(),
            make_symbol(&["my_pkg::Foo::{{impl}}[0]::bar"]),
        );

        let mut packages = HashMap::from([(
            "my-pkg".to_string(),
            Package {
                targets: HashMap::from([
                    (
                        "lib".to_string(),
                        tarjanize_schemas::Target {
                            root: lib_root,
                            ..Default::default()
                        },
                    ),
                    (
                        "test".to_string(),
                        tarjanize_schemas::Target {
                            root: test_root_with_dep,
                            ..Default::default()
                        },
                    ),
                ]),
            },
        )]);

        let mapping = make_mapping(&[("my_pkg", "my-pkg/lib")]);
        transform_symbol_paths(&mut packages, &mapping);

        let test_fn = packages["my-pkg"].targets["test"]
            .root
            .symbols
            .get("test_fn")
            .unwrap();
        assert!(
            test_fn
                .dependencies
                .contains("[my-pkg/test]::Foo::{{impl}}[0]::bar"),
            "expected [my-pkg/test]::Foo::{{impl}}[0]::bar, \
             got: {:?}",
            test_fn.dependencies,
        );
    }

    #[test]
    fn test_transform_symbol_paths_module_resolution() {
        // Dep `my_pkg::some_mod` where `some_mod` is a submodule in
        // *test* only. Crate mapping defaults to lib, so module-tree
        // lookup is needed to find the correct target.
        let lib_root = make_module(&["LibType"], &[]);
        let test_root =
            make_module(&[], &[("some_mod", make_module(&["Item"], &[]))]);

        let mut test_root_with_dep = test_root.clone();
        test_root_with_dep
            .symbols
            .insert("test_fn".to_string(), make_symbol(&["my_pkg::some_mod"]));

        let mut packages = HashMap::from([(
            "my-pkg".to_string(),
            Package {
                targets: HashMap::from([
                    (
                        "lib".to_string(),
                        tarjanize_schemas::Target {
                            root: lib_root,
                            ..Default::default()
                        },
                    ),
                    (
                        "test".to_string(),
                        tarjanize_schemas::Target {
                            root: test_root_with_dep,
                            ..Default::default()
                        },
                    ),
                ]),
            },
        )]);

        let mapping = make_mapping(&[("my_pkg", "my-pkg/lib")]);
        transform_symbol_paths(&mut packages, &mapping);

        let test_fn = packages["my-pkg"].targets["test"]
            .root
            .symbols
            .get("test_fn")
            .unwrap();
        assert!(
            test_fn.dependencies.contains("[my-pkg/test]::some_mod"),
            "expected [my-pkg/test]::some_mod, got: {:?}",
            test_fn.dependencies,
        );
    }

    #[test]
    fn test_transform_symbol_paths_cross_package_unchanged() {
        // Cross-package dep should use the crate mapping as before, not
        // do any module-tree lookup in the referencing package.
        let lib_root = make_module(&["MyType"], &[]);

        let mut other_root = make_module(&[], &[]);
        other_root
            .symbols
            .insert("caller".to_string(), make_symbol(&["my_pkg::MyType"]));

        let mut packages = HashMap::from([
            (
                "my-pkg".to_string(),
                Package {
                    targets: HashMap::from([(
                        "lib".to_string(),
                        tarjanize_schemas::Target {
                            root: lib_root,
                            ..Default::default()
                        },
                    )]),
                },
            ),
            (
                "other-pkg".to_string(),
                Package {
                    targets: HashMap::from([(
                        "lib".to_string(),
                        tarjanize_schemas::Target {
                            root: other_root,
                            ..Default::default()
                        },
                    )]),
                },
            ),
        ]);

        let mapping = make_mapping(&[
            ("my_pkg", "my-pkg/lib"),
            ("other_pkg", "other-pkg/lib"),
        ]);
        transform_symbol_paths(&mut packages, &mapping);

        let caller = packages["other-pkg"].targets["lib"]
            .root
            .symbols
            .get("caller")
            .unwrap();
        assert!(
            caller.dependencies.contains("[my-pkg/lib]::MyType"),
            "expected [my-pkg/lib]::MyType, got: {:?}",
            caller.dependencies,
        );
    }

    #[test]
    fn test_transform_symbol_paths_bin_only_package() {
        // Package with only a bin target. Same-crate ref should resolve
        // to `[my-pkg/bin/my_tool]::Args`.
        let bin_root = make_module(&["Args"], &[]);

        let mut bin_root_with_dep = bin_root.clone();
        bin_root_with_dep
            .symbols
            .insert("main_fn".to_string(), make_symbol(&["my_tool::Args"]));

        let mut packages = HashMap::from([(
            "my-pkg".to_string(),
            Package {
                targets: HashMap::from([(
                    "bin/my_tool".to_string(),
                    tarjanize_schemas::Target {
                        root: bin_root_with_dep,
                        ..Default::default()
                    },
                )]),
            },
        )]);

        // Crate mapping maps to bin (no lib exists).
        let mapping = make_mapping(&[("my_tool", "my-pkg/bin/my_tool")]);
        transform_symbol_paths(&mut packages, &mapping);

        let main_fn = packages["my-pkg"].targets["bin/my_tool"]
            .root
            .symbols
            .get("main_fn")
            .unwrap();
        assert!(
            main_fn.dependencies.contains("[my-pkg/bin/my_tool]::Args"),
            "expected [my-pkg/bin/my_tool]::Args, got: {:?}",
            main_fn.dependencies,
        );
    }

    #[test]
    fn test_transform_symbol_paths_fallback_to_crate_mapping() {
        // Unresolvable path (symbol doesn't exist in any target) should
        // fall back to the crate mapping default.
        let lib_root = make_module(&["RealType"], &[]);

        let mut test_root = make_module(&[], &[]);
        test_root.symbols.insert(
            "test_fn".to_string(),
            make_symbol(&["my_pkg::nonexistent::Ghost"]),
        );

        let mut packages = HashMap::from([(
            "my-pkg".to_string(),
            Package {
                targets: HashMap::from([
                    (
                        "lib".to_string(),
                        tarjanize_schemas::Target {
                            root: lib_root,
                            ..Default::default()
                        },
                    ),
                    (
                        "test".to_string(),
                        tarjanize_schemas::Target {
                            root: test_root,
                            ..Default::default()
                        },
                    ),
                ]),
            },
        )]);

        let mapping = make_mapping(&[("my_pkg", "my-pkg/lib")]);
        transform_symbol_paths(&mut packages, &mapping);

        // Falls back to crate mapping -> lib.
        let test_fn = packages["my-pkg"].targets["test"]
            .root
            .symbols
            .get("test_fn")
            .unwrap();
        assert!(
            test_fn
                .dependencies
                .contains("[my-pkg/lib]::nonexistent::Ghost"),
            "expected [my-pkg/lib]::nonexistent::Ghost, got: {:?}",
            test_fn.dependencies,
        );
    }

    #[test]
    fn test_transform_symbol_paths_anchor_resolution() {
        // Impl anchor `my_pkg::test_mod::TestTrait` should resolve to
        // the target where `TestTrait` exists, not the crate mapping
        // default.
        let lib_root = make_module(&["LibType"], &[]);
        let test_root =
            make_module(&[], &[("test_mod", make_module(&["TestTrait"], &[]))]);

        let mut test_root_with_impl = test_root.clone();
        test_root_with_impl.symbols.insert(
            "{{impl}}[0]".to_string(),
            make_impl_symbol(&[], &["my_pkg::test_mod::TestTrait"]),
        );

        let mut packages = HashMap::from([(
            "my-pkg".to_string(),
            Package {
                targets: HashMap::from([
                    (
                        "lib".to_string(),
                        tarjanize_schemas::Target {
                            root: lib_root,
                            ..Default::default()
                        },
                    ),
                    (
                        "test".to_string(),
                        tarjanize_schemas::Target {
                            root: test_root_with_impl,
                            ..Default::default()
                        },
                    ),
                ]),
            },
        )]);

        let mapping = make_mapping(&[("my_pkg", "my-pkg/lib")]);
        transform_symbol_paths(&mut packages, &mapping);

        let impl_sym = packages["my-pkg"].targets["test"]
            .root
            .symbols
            .get("{{impl}}[0]")
            .unwrap();
        if let SymbolKind::Impl { anchors, .. } = &impl_sym.kind {
            assert!(
                anchors.contains("[my-pkg/test]::test_mod::TestTrait"),
                "expected [my-pkg/test]::test_mod::TestTrait \
                 in anchors, got: {anchors:?}",
            );
        } else {
            panic!("expected Impl symbol kind");
        }
    }
}
