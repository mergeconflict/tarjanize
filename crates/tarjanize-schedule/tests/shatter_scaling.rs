use std::collections::{HashMap, HashSet};
use std::time::Duration;

use tarjanize_schedule::data::{ScheduleData, Summary, TargetData};
use tarjanize_schedule::recommend::shatter_target;
use tarjanize_schemas::{
    CostModel, Module, Package, Symbol, SymbolGraph, SymbolKind, Target,
    TargetTimings, Visibility,
};

/// Verifies that shattering scales metadata costs by dependency usage.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "single end-to-end scenario exercises shatter behavior holistically"
)]
fn shatter_scales_metadata_cost() {
    // Setup:
    // - Target depends on Dep1, Dep2.
    // - SymA depends on Dep1.
    // - SymB depends on nothing.
    // - Meta cost = 100.
    // - Attr cost = 10 per symbol.
    //
    // Split:
    // - GroupA (SymA): Uses Dep1. Deps=1. Ratio = 1/2 = 0.5. Meta = 50.
    // - GroupB (SymB): Uses nothing. Deps=0. Ratio = 0/2 = 0.0. Meta = 0.

    // 1. Define Dependencies (Dep1, Dep2)
    let dep1_finish = 50.0;
    let dep2_finish = 50.0; // Same finish time so they are distinct dependencies but relevant

    // 2. Define Symbols
    let mut symbols = HashMap::new();

    // SymA: Depends on Dep1
    symbols.insert(
        "SymA".to_string(),
        Symbol {
            file: "lib.rs".to_string(),
            event_times_ms: HashMap::from([("typeck".to_string(), 10.0)]),
            dependencies: HashSet::from(["[dep1/lib]::foo".to_string()]),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        },
    );

    // SymB: Independent
    symbols.insert(
        "SymB".to_string(),
        Symbol {
            file: "lib.rs".to_string(),
            event_times_ms: HashMap::from([("typeck".to_string(), 10.0)]),
            dependencies: HashSet::new(),
            kind: SymbolKind::ModuleDef {
                kind: "Function".to_string(),
                visibility: Visibility::Public,
            },
        },
    );

    let root = Module {
        symbols,
        submodules: HashMap::new(),
    };

    // 3. Define Target with High Metadata Cost
    let target = Target {
        timings: TargetTimings {
            wall_time: Duration::from_millis(200), // Arbitrary wall time
            event_times_ms: HashMap::from([
                ("metadata_decode_entry".to_string(), 100.0), // High meta cost
            ]),
        },
        dependencies: HashSet::from([
            "dep1/lib".to_string(),
            "dep2/lib".to_string(),
        ]),
        root,
    };

    // 4. Build SymbolGraph
    let mut packages = HashMap::new();
    packages.insert(
        "test-pkg".to_string(),
        Package {
            targets: HashMap::from([("lib".to_string(), target)]),
        },
    );
    // Add dummy packages for deps so they exist in graph
    packages.insert(
        "dep1".to_string(),
        Package {
            targets: HashMap::from([("lib".to_string(), Target::default())]),
        },
    );
    packages.insert(
        "dep2".to_string(),
        Package {
            targets: HashMap::from([("lib".to_string(), Target::default())]),
        },
    );

    let sg = SymbolGraph { packages };

    // 5. Build ScheduleData
    // We need Dep1 and Dep2 in the schedule with finish times.
    // And the target itself.
    let targets_data = vec![
        TargetData {
            name: "dep1/lib".to_string(),
            start: Duration::ZERO,
            finish: Duration::from_secs_f64(dep1_finish / 1000.0),
            cost: Duration::from_secs_f64(dep1_finish / 1000.0),
            slack: Duration::ZERO,
            lane: 0,
            symbol_count: 0,
            deps: vec![],
            dependents: vec![2],
            on_critical_path: true,
            forward_pred: None,
            backward_succ: None,
        },
        TargetData {
            name: "dep2/lib".to_string(),
            start: Duration::ZERO,
            finish: Duration::from_secs_f64(dep2_finish / 1000.0),
            cost: Duration::from_secs_f64(dep2_finish / 1000.0),
            slack: Duration::ZERO,
            lane: 1,
            symbol_count: 0,
            deps: vec![],
            dependents: vec![2],
            on_critical_path: false,
            forward_pred: None,
            backward_succ: None,
        },
        TargetData {
            name: "test-pkg/lib".to_string(),
            start: Duration::from_secs_f64(dep1_finish / 1000.0),
            finish: Duration::from_secs_f64((dep1_finish + 120.0) / 1000.0),
            cost: Duration::from_secs_f64(120.0 / 1000.0),
            slack: Duration::ZERO,
            lane: 0,
            symbol_count: 2,
            deps: vec![0, 1], // Depends on dep1, dep2
            dependents: vec![],
            on_critical_path: true,
            forward_pred: Some(0),
            backward_succ: None,
        },
    ];

    let schedule = ScheduleData {
        summary: Summary::default(), // Not used by shatter_target
        targets: targets_data,
        critical_path: vec![],
    };

    // 6. Define Cost Model
    // 1.0 * attr + 1.0 * meta + 0.0 * other
    let model = CostModel {
        coeff_attr: 1.0,
        coeff_meta: 1.0,
        coeff_other: 0.0,
        r_squared: 1.0,
        inlier_threshold: 1.0,
    };

    // 7. Run shatter_target
    let (new_schedule, _) =
        shatter_target(&sg, "test-pkg/lib", &schedule, Some(&model))
            .expect("shatter_target failed");

    // 8. Verify Costs
    // Groups should be:
    // - group_0: SymB (horizon 0). Deps = 0.
    // - group_1: SymA (horizon 50). Deps = 1 (dep1).

    // Find groups
    let group0 = new_schedule
        .targets
        .iter()
        .find(|t| t.name == "test-pkg/lib::group_0")
        .unwrap();
    let group1 = new_schedule
        .targets
        .iter()
        .find(|t| t.name == "test-pkg/lib::group_1")
        .unwrap();

    // Calculate expected costs
    let meta_total = 100.0;
    let attr_sym = 10.0;

    // Group 0 (SymB):
    // Deps used: 0. Total deps: 2. Ratio: 0.0.
    // Cost = 1.0 * attr (10) + 1.0 * (meta * 0.0) = 10.0
    let expected_g0 = attr_sym;

    // Group 1 (SymA):
    // Deps used: 1 (dep1). Total deps: 2. Ratio: 0.5.
    // Cost = 1.0 * attr (10) + 1.0 * (meta * 0.5) = 10.0 + 50.0 = 60.0
    let expected_g1 = attr_sym + (meta_total * 0.5);

    println!(
        "Group 0 cost: {:?} (expected {:?})",
        group0.cost,
        Duration::from_secs_f64(expected_g0 / 1000.0)
    );
    println!(
        "Group 1 cost: {:?} (expected {:?})",
        group1.cost,
        Duration::from_secs_f64(expected_g1 / 1000.0)
    );

    // Allow small float error
    let epsilon = 1e-6;

    // Check Group 0
    let g0_ms = group0.cost.as_secs_f64() * 1000.0;
    assert!(
        (g0_ms - expected_g0).abs() < epsilon,
        "Group 0 cost mismatch. Got {g0_ms}ms, expected {expected_g0}ms. (Ideally 0% meta cost)",
    );

    // Check Group 1
    let g1_ms = group1.cost.as_secs_f64() * 1000.0;
    assert!(
        (g1_ms - expected_g1).abs() < epsilon,
        "Group 1 cost mismatch. Got {g1_ms}ms, expected {expected_g1}ms. (Ideally 50% meta cost)",
    );
}
