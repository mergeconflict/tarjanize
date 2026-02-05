# Cost Model Validation Report

This document validates tarjanize's cost model against actual build times from a large Rust workspace (Omicron, ~160 crates).

## Executive Summary

**Question**: Does our profiling-based cost model predict real build times?

**Answer**: Yes. The model explains **86% of variance** for library targets (R² = 0.856) and **92% of variance** when lib+test targets are merged (R² = 0.917). It correctly identifies the most expensive crates and captures their relative costs.

**Key findings**:
- Top 3 bottlenecks correctly identified
- 4/5 top crates ranked correctly
- Relative costs accurate within 2× for 7/10 top crates
- LIB targets well-predicted; TEST/BIN targets less so (different scaling)
- **Metadata is essential** (+10% R² contribution); **linking is negligible** (+0.4%)

**Recommendation**: Use the model for **relative comparisons** (which crates are bottlenecks, how do they compare). Don't rely on absolute time predictions.

---

## 1. The Question

The fundamental question is: **Does our profiling data actually predict real build times?**

It's possible that rustc's self-profile output measures something that doesn't translate to wall-clock time, or that our formula for combining the data is wrong. We need to test this against reality.

---

## 2. Methodology

### 2.1 Data Collection

**Model predictions** were extracted using `cargo tarjanize` with:
- `CARGO_INCREMENTAL=0` — ensures complete profiling data for all CGUs
- `-Zself-profile` — captures per-query timing from rustc

**Actual build times** were captured using `cargo build --timings` with:
- Default settings (incremental compilation enabled)
- All targets: lib, bin, test, bench, examples

### 2.2 Cost Model Formula

The model predicts crate build time as:

```
crate_cost = frontend_time + backend_time + overhead

where:
  frontend_time = Σ symbol.frontend_cost_ms    (serial work)
  backend_time  = max(module.backend_cost_ms)  (parallel via CGUs)
  overhead      = metadata_ms
```

**Note:** Linking time was originally included but removed after target-level analysis showed it contributes negligibly to model accuracy (see Section A.6 and A.8).

### 2.3 Comparison Approach

For each workspace crate, we compared:
- **Model**: sum of frontend + max backend + overhead (in seconds)
- **Actual**: sum of all target build times from --timings (lib + test + bin)

We used linear regression to find the best-fit scaling factor, then evaluated correlation and residuals.

### 2.4 Test Data

- **Workspace**: Omicron (Oxide Computer Company)
- **Crate count**: 161 workspace crates
- **Symbol count**: 126,683 symbols
- **Total build time**: ~1,500s (all targets)

---

## 3. Results

### 3.1 Overall Fit

| Metric | Value |
|--------|-------|
| Pearson correlation (r) | **0.949** |
| R² (variance explained) | **0.900** |
| Linear fit | `actual = 0.34 × model - 0.88` |
| Scaling factor | Model ≈ 2.9× actual |

**What these metrics mean**:

- **Pearson correlation (r)** measures how closely two variables follow a straight-line relationship. It ranges from -1 to +1:
  - r = +1: perfect positive relationship (when one goes up, the other goes up proportionally)
  - r = 0: no relationship
  - r = -1: perfect negative relationship
  - Our r = 0.949 indicates a very strong positive relationship.

- **R²** is the correlation squared (0.949² = 0.900). It tells us what fraction of the variation in actual build times is "explained" by the model. An R² of 0.90 means 90% of the variation in build times can be predicted from the model; the remaining 10% is noise or factors we don't capture.

The model explains 90% of variance in actual build times with a strong linear relationship.

*Note: These are crate-level results (summing all targets per crate). See Section A.6 for target-level breakdown showing LIB R² = 0.856, TEST R² = 0.724.*

### 3.2 Statistical Significance

Before trusting these results, we need to rule out the possibility that the correlation happened by chance. We do this with a **hypothesis test**:

- **Null hypothesis (H₀)**: There's no real relationship — any apparent correlation is just random noise
- **Alternative hypothesis (H₁)**: The model genuinely predicts build times

The **p-value** answers: "If there were no real relationship, what's the probability of seeing a correlation this strong by accident?"

For our data (r = 0.949, n = 109 crates), the p-value is essentially zero (< 0.0000...001 with many zeros). To put this in perspective: our correlation is **31 standard deviations** away from what we'd expect from random chance. This is not a fluke.

**The null hypothesis is decisively rejected.** The model captures something real about build costs.

### 3.3 Top Crate Accuracy

For identifying build time bottlenecks:

| Metric | Value |
|--------|-------|
| Top 10 correlation | **0.930** |
| Top 5 ranking accuracy | **4/5 correct** |
| Top 10 ranking accuracy | 6/10 correct |

The model correctly identifies the three most expensive crates:

| Rank | Crate | Actual Time |
|------|-------|-------------|
| 1 | `omicron_nexus` | 401s |
| 2 | `nexus_db_queries` | 141s |
| 3 | `omicron_sled_agent` | 85s |

### 3.4 Proportional Accuracy

Relative costs (normalized to most expensive crate):

| Crate | Actual | Model | Match |
|-------|--------|-------|-------|
| omicron_nexus | 1.00 | 1.00 | ✓ |
| nexus_db_queries | 0.35 | 0.61 | ✓ |
| omicron_sled_agent | 0.21 | 0.35 | ✓ |
| nexus_db_model | 0.15 | 0.08 | ✓ |
| oxide_client | 0.12 | 0.20 | ✓ |
| nexus_db_schema | 0.10 | 0.30 | ✗ |
| nexus_types | 0.10 | 0.09 | ✓ |

7/10 top crates have proportions within 2× of actual.

---

## 4. Known Limitations

### 4.1 Systematic Biases

**Overestimated crates** (model predicts too high):
- DB crates with massive frontend costs (e.g., `nexus_db_schema`: 79k symbols)
- Suggests frontend work may parallelize more than the model assumes

**Underestimated crates** (model predicts too low):
- "Thin wrapper" binaries with few symbols but large dependency trees
- Examples: `dns_server`, `wicketd`, `installinator`
- Missing some baseline/startup cost not captured by self-profile

### 4.2 Methodological Caveats

1. **Incremental vs clean builds**: Model uses `CARGO_INCREMENTAL=0` for complete profiling; actual builds used incremental compilation. The 0.90 R² suggests builds scale similarly despite this difference.

2. **Dev-dependency cycles**: When lib and test targets are merged, dev-dependencies create apparent cycles in the crate dependency graph. This prevents critical path computation for some workspaces.

---

## 5. Conclusions

The tarjanize cost model is **suitable for identifying build bottlenecks** in Rust workspaces:

- **90% of variance explained** by a simple linear model
- **Top bottlenecks correctly identified** (4/5 top crates, 0.93 correlation)
- **Relative costs accurate** for prioritizing optimization efforts

The model should be used for **relative comparisons** rather than absolute time predictions. The ~3× scaling factor varies by crate characteristics.

---

## Appendix A: Hypotheses Tested

During validation, we tested several hypotheses about model accuracy.

### A.1 Initial Data Mismatch (confirmed, fixed)

**Hypothesis**: Original comparison showed 11.9× scaling factor — too large.

**Finding**: We were comparing lib-only targets from --timings against lib+test from the model. After summing all targets (lib + test + bin), scaling factor dropped to 2.9×.

**Lesson**: Always compare equivalent build configurations.

### A.2 Test Code Ratio Affects Accuracy (disproved)

**Hypothesis**: Crates with more test code might have higher model error since tests have different compilation characteristics.

**Finding**: Correlation between test code ratio and error ratio was **r = -0.019** (essentially zero). Test code fraction does not explain model error.

### A.3 Frontend Parallelism (marginal improvement, not adopted)

**Hypothesis**: Frontend work may parallelize within rustc's query system, so `frontend^k` (k < 1) might fit better than `Σfrontend`.

**Tested**:

| Model | R² |
|-------|-----|
| Σfrontend (current) | 0.8998 |
| frontend^0.8 | 0.9102 |
| frontend^0.7 | 0.9096 |
| sqrt(frontend) | 0.9040 |

**Finding**: `frontend^0.8` gave best R² but only 1% improvement. It also produced negative predictions for low-frontend crates due to the linear fit intercept.

**Conclusion**: Not worth the added complexity.

### A.4 Residual Analysis

**Hypothesis**: Systematic patterns in residuals might reveal missing model factors.

**Correlations with residuals** (after linear correction):

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| Backend cost | r = -0.358 | Higher backend → overestimate |
| Symbol count | r = -0.264 | More symbols → overestimate |
| Actual build time | r = +0.408 | Bigger crates → underestimate |
| Number of dependencies | r = +0.066 | No significant effect |
| Linking time | r = -0.009 | No significant effect |
| Frontend/backend ratio | r = +0.034 | No significant effect |

**Findings**:
- Higher backend cost → model overestimates (backend parallelizes better than assumed?)
- More symbols → model overestimates
- Larger actual time → model underestimates (missing scaling factor for big crates)
- Dependency count and linking time show no significant correlation

### A.5 Outlier Characteristics

**Overestimated crates** (model too high):
- DB crates: `nexus_db_schema` (79k symbols), `nexus_db_queries`, `nexus_db_lookup`
- Characteristically high frontend costs (200-500s profiled)
- High symbol counts
- Average frontend/backend ratio: 37.5

**Underestimated crates** (model too low):
- Thin wrappers: `wicket`, `reconfigurator_cli`, `installinator`, `dns_server`, `wicketd`
- Very few symbols (1-100)
- High linking/metadata overhead relative to frontend
- Average frontend/backend ratio: 92.3

**Hypothesis**: Underestimated crates have cost from dependency integration (metadata loading, monomorphization) not captured by self-profile.

### A.6 Component Contribution Analysis

#### Initial Crate-Level Analysis

We first tested how much each cost component contributes to model accuracy at the crate level (merging all targets per crate):

| Model | r | R² |
|-------|-----|------|
| **Full model (current)** | 0.949 | **0.900** |
| Without linking | 0.948 | 0.899 |
| Without metadata | 0.907 | 0.823 |
| Without linking or metadata | 0.905 | 0.820 |
| Frontend only | 0.919 | 0.844 |
| Backend only | 0.187 | 0.035 |
| **Linking + metadata only** | 0.963 | **0.927** |

**Key findings:**

1. **Linking contributes almost nothing** — removing it drops R² by only 0.001
2. **Metadata matters significantly** — removing it drops R² from 0.90 to 0.82
3. **Backend alone is nearly useless** — R² = 0.035
4. **Linking + metadata alone beats the full model** — R² = 0.927 vs 0.900

The last finding is surprising: linking and metadata times alone predict build times better than our full model. This suggests they're excellent proxies for overall crate complexity.

**However, this is not actionable.** When tarjanize runs the condense phase, it creates synthetic crates (SCCs merged into single nodes). These synthetic crates don't have real linking/metadata times — we must estimate them from the constituent symbols. Therefore, we cannot rely on linking + metadata as the sole predictor; we need the frontend/backend costs to estimate synthetic crate costs.

#### Target-Level Analysis (LIB vs TEST vs BIN)

The crate-level analysis (R² = 0.90) masked important differences between target types. We re-ran the analysis at the **target level**, comparing each individual lib, test, and bin target against its actual build time:

| Target Type | Count | R² | Pearson r | Slope | Interpretation |
|-------------|-------|-----|-----------|-------|----------------|
| LIB only | 149 | **0.856** | 0.925 | 0.15 | Good fit |
| TEST only | 161 | 0.724 | 0.851 | 1.59 | Systematic underprediction |
| LIB + TEST merged | 160 | **0.917** | 0.958 | 0.31 | Best fit |
| BIN only | 6 | 0.677 | 0.823 | — | Small sample, moderate fit |

**Key findings:**

1. **LIB targets are well-predicted** (R² = 0.856) — the model captures library compilation accurately
2. **TEST targets have different scaling** — slope of 1.59 vs 0.15 for libs means test builds take ~10× more time per unit of modeled cost
3. **Merging LIB + TEST improves fit** — this is why the crate-level analysis showed R² = 0.90
4. **BIN targets have small sample size** — only 6 binaries, so R² = 0.677 is less reliable

The slope difference between LIB (0.15) and TEST (1.59) explains why merging them helps: the regression finds an average slope that works reasonably for both.

#### Component Contribution by Target Type

We isolated the contribution of linking and metadata to LIB-only predictions:

| Component | LIB R² | Δ R² from baseline |
|-----------|--------|-------------------|
| Frontend + Backend only | 0.752 | baseline |
| + Metadata | 0.856 | **+0.104** |
| + Linking | 0.756 | +0.004 |
| + Both (full model) | 0.858 | +0.106 |

**Conclusion:** Metadata contributes ~10% improvement to R²; linking contributes almost nothing (~0.4% from baseline, ~0.2% when metadata already included).

### A.7 Metadata Estimation for Synthetic Crates

Since synthetic crates don't have measured metadata times, we need to estimate them from available symbol data.

**Single-variable predictors tested:**

| Predictor | R² |
|-----------|-----|
| Frontend cost (all symbols) | **0.705** |
| Linking time | 0.386 |
| Frontend cost (public symbols only) | 0.154 |
| Symbol count (public only) | 0.036 |
| Symbol count (all) | 0.013 |

**Multivariate models tested:**

| Model | R² |
|-------|-----|
| Frontend + symbols + linking | 0.727 |
| Frontend + symbols | 0.719 |
| Frontend + linking | 0.719 |
| Frontend only | **0.705** |

Adding more predictors provides minimal improvement over frontend alone.

**Non-linear transformations tested:**

| Transformation | R² |
|----------------|-----|
| frontend^1.2 | 0.725 |
| frontend (linear) | **0.705** |
| frontend^0.9 | 0.689 |
| sqrt(frontend) | 0.535 |
| log(frontend) | 0.159 |

Non-linear transforms don't help; linear is optimal.

**Surprising finding:** Public symbols predict metadata *worse* than all symbols. This is counterintuitive since metadata supposedly encodes public APIs. However, rustc metadata actually includes everything needed for downstream compilation: type signatures, AST spans, symbol tables, and items affecting trait resolution and type inference — not just public APIs.

**Best model for synthetic crates:**

```
metadata_ms = 0.26 × frontend_ms + 1662
```

Where `frontend_ms` is the sum of `frontend_cost_ms` for all symbols in the synthetic crate.

**Error characteristics:**

| Metric | Value |
|--------|-------|
| R² | 0.705 |
| Mean absolute % error | 135% |
| Median absolute % error | 50% |
| 90th percentile error | 247% |

Individual crate errors can be large (up to +314% for `nexus_db_queries`), but for **ranking purposes** (identifying which synthetic crates are most expensive) this should be acceptable.

**Note on the intercept:** The 1662ms intercept represents a per-crate fixed overhead. When splitting one crate into N crates, the total estimated metadata cost will increase. This is realistic — creating more crates does increase total compilation overhead due to per-crate fixed costs (file I/O, metadata headers, etc.).

### A.8 Linking Time Analysis and Decision to Remove

Initially, we attempted to model linking time for synthetic crates. However, deeper analysis revealed that linking time:
1. Contributes negligibly to model accuracy
2. Has weak predictors (best R² = 0.42 for workspace deps)
3. Varies dramatically by target type

**Final decision: Linking removed from the cost model.**

#### Why Linking Was Originally Considered

Linking (loading rlibs, resolving symbols, etc.) is a real compilation cost. We hypothesized it might be significant for crates with many dependencies.

#### Predictor Analysis

**Single-variable predictors tested:**

| Predictor | R² |
|-----------|-----|
| workspace_deps | 0.42 |
| dep_count (total) | 0.27 |
| external_deps | 0.16 |
| symbols | ~0.00 |

The best predictor (`workspace_deps` — dependencies on other workspace crates) explains only 42% of variance — far weaker than metadata's 71% (from frontend cost). External deps actually *decrease* predictive power because they're pre-compiled and don't require much linking work. This suggests linking time is influenced by factors we don't capture (LTO settings, codegen units, binary size, etc.).

#### Linking Time Distribution by Target Type

| Target Type | Linking % of Build Time | Range |
|-------------|------------------------|-------|
| LIB | <1% | 0-2% |
| TEST | 10-36% | varies |
| BIN | 10-63% | varies |

**Key insight:** Linking is negligible for library targets (our primary modeling concern) but significant for test and binary targets. However, test/bin linking time is highly variable and poorly predicted by available features.

#### Impact on Model Accuracy

Comparing the full model with and without linking:

| Model | LIB R² |
|-------|--------|
| Without linking (frontend + backend + metadata) | 0.856 |
| With linking (full model) | 0.858 |
| **Δ R²** | **+0.002** |

Adding linking improves R² by only 0.2% — not worth the complexity of estimating it.

#### Comparison: Linking vs Metadata

| Aspect | Linking | Metadata |
|--------|---------|----------|
| Contribution to LIB R² | +0.002 | +0.104 |
| Best predictor R² | 0.42 | 0.71 |
| % of LIB build time | <1% | ~5-15% |
| Predictable? | Weakly | Yes |

**Conclusion:** Metadata contributes ~50× more to model accuracy than linking. Linking is too small for libs and too unpredictable for tests/bins to be worth modeling.

---

## Appendix B: TODOs

### B.1 ~~Improve Linking Cost Estimation~~ CLOSED

~~The current linking model (R² = 0.555) only counts workspace crate dependencies captured in the symbol graph. However, linking also involves external crates (from crates.io, git dependencies, etc.) which are not currently tracked.~~

~~**TODO:** Add a complete list of crate dependencies (including external crates) at the crate level in the symbol graph.~~

~~**DONE:** Added `dependencies` and `dev_dependencies` fields to the `Crate` struct, populated from `cargo metadata`. This captures all dependencies (workspace and external) at the crate level. The linking cost model can now count all dependencies, potentially improving R² for linking estimation.~~

~~**Next step:** Re-run the validation analysis to measure whether including external crates improves the linking cost model's R².~~

**CLOSED (2026-02):** After comprehensive target-level analysis, linking was **removed from the cost model entirely**. Key findings:
- Best predictor (workspace_deps) only achieves R² = 0.42 — too weak
- Linking contributes <1% of build time for LIB targets
- Adding linking to the model improves R² by only 0.002
- See Section A.8 for full analysis

### B.2 ~~Fix Critical Path Computation for Dev-Dependency Cycles~~ CLOSED

~~The critical path code currently fails when the crate dependency graph contains cycles. These cycles can occur when dev-dependencies create apparent circular dependencies (e.g., crate A's tests depend on crate B, which depends on crate A).~~

~~**Current behavior:** Prints a warning and skips critical path computation, only reporting per-crate costs.~~

~~**TODO:** Properly handle dev-dependency cycles by either:~~
1. ~~Separating lib and test targets in the dependency graph~~
2. ~~Detecting and breaking cycles at dev-dependency edges~~
3. ~~Computing critical path on the lib-only subgraph~~

**CLOSED (2026-02):** Fixed by separating lib/test/bin targets in the symbol graph schema (commit ab35032). Each target is now a separate node in the dependency graph, so dev-dependencies no longer create artificial cycles. Test targets depend on their package's lib target explicitly, making the graph acyclic.

### B.3 Validate Critical Path Cost Model

The per-crate cost model has been validated (R² = 0.856 for lib targets), but the **critical path cost** (sum of costs along the longest dependency chain) has not been validated against real build times with parallelism.

**INVESTIGATED (2026-02):** We simulated parallel execution of our dependency graph and compared to actual `cargo build --timings` data. Key findings:

| Metric | Actual | Model | Ratio |
|--------|--------|-------|-------|
| Critical path | 331s | 2045s | 6.2x |
| Peak parallelism | 64 | 31 | 0.5x |
| Avg parallelism | 4.8 | 2.8 | 0.6x |

The model correctly identifies the same three bottleneck targets (nexus-db-model → nexus-db-queries → omicron-nexus) and the same 71% critical path fraction. However, **the model dramatically overestimates build time and underestimates parallelism**.

**Root cause: rmeta pipelining** (see Section B.4 below).

### B.4 CRITICAL FINDING: Cargo Uses Rmeta Pipelining

**This finding fundamentally changes how we must model Rust builds.**

#### Discovery

When comparing simulated vs actual build parallelism, we found that test targets start almost immediately after their lib target starts — not when the lib finishes:

| Package | Lib Duration | Test Starts After Lib | Test Starts Before Lib Ends |
|---------|--------------|----------------------|----------------------------|
| oxide-client | 27.1s | 2.2s | 24.9s |
| omicron-nexus | 166.5s | 3.7s | 162.8s |

The `cargo --timings` data shows `oxide-client/lib` has:
- Frontend: 24.25s
- Codegen: 2.89s
- Test starts at 2.2s — **before frontend even finishes**

#### Explanation: Rmeta vs Rlib

Cargo uses **pipelined compilation**:

1. **Rmeta (metadata)**: Contains type signatures, trait impls, and everything needed for type checking downstream crates. Available after frontend compilation.

2. **Rlib (library)**: Contains actual compiled code. Required for linking but NOT for compiling downstream crates.

Downstream crates only need **rmeta** to start compiling. They don't wait for the full rlib.

#### Impact on Our Model

Our current model assumes:
```
downstream.start_time = max(dep.finish_time for dep in dependencies)
```

But Cargo actually uses:
```
downstream.start_time = max(dep.rmeta_ready_time for dep in dependencies)
```

Where `rmeta_ready_time ≈ start_time + frontend_cost` (much earlier than finish_time).

This explains the parallelism gap:
- **Model**: Treats all deps as "wait for full completion" → low parallelism
- **Actual**: Uses rmeta pipelining → high parallelism

#### Implications for Tarjanize

This finding has major implications for the tarjanize algorithm:

1. **Critical path calculation is wrong**: The critical path should be based on rmeta availability, not rlib completion. Frontend costs are on the critical path; codegen costs are not (they run in parallel).

2. **Cost model needs restructuring**: We need to track frontend and codegen costs separately, not just total cost.

3. **Optimization target changes**: Reducing frontend cost reduces critical path. Reducing codegen cost only helps if codegen is longer than downstream frontend work.

4. **Symbol-level analysis may be less valuable**: If critical path is dominated by frontend (type checking, borrow checking), and frontend is largely serial within a crate, then splitting crates may not help as much as expected.

#### Next Steps

**TODO:** Restructure the cost model to account for rmeta pipelining:

1. **Separate frontend vs codegen costs** in the symbol graph schema
2. **Model rmeta availability** as `start_time + frontend_cost`
3. **Recalculate critical path** using rmeta-based dependencies
4. **Re-validate** against actual build times
5. **Assess impact** on tarjanize's crate-splitting recommendations

### B.5 Implications of Rmeta Pipelining for Tarjanize Algorithm

#### Current Tarjanize Approach

1. Extract symbol dependency graph
2. Find strongly connected components (SCCs) that could be split
3. Estimate cost savings from splitting based on total compilation time
4. Recommend crate splits that reduce critical path

#### The Problem

We assumed that splitting a crate reduces the critical path because downstream crates can start sooner. But with rmeta pipelining, downstream crates already start as soon as the upstream crate's **frontend** is done — not when codegen finishes.

Example from omicron-nexus:
- Frontend: ~90% of compilation time (type checking the massive crate)
- Codegen: ~10% of compilation time
- Downstream test starts 3.7s after lib starts (basically immediate)

If we split omicron-nexus into smaller crates:
- Each smaller crate has less frontend work
- But downstream crates already weren't waiting for codegen
- The benefit is faster **rmeta availability**, not faster rlib availability

#### Key Insight

**The critical path is determined by frontend costs (serial within crate), not total costs.**

Codegen runs in parallel with downstream compilation, so it's largely "free" unless it's longer than the downstream frontend work.

#### What This Means for Tarjanize

1. **Cost model needs restructuring**
   - Must track `frontend_cost` vs `codegen_cost` separately
   - Critical path = sum of frontend costs along dependency chain
   - Codegen costs only matter if they exceed downstream frontend work

2. **Splitting still helps, but for different reasons**
   - Splitting reduces frontend work per crate (smaller type checking scope)
   - This makes rmeta available sooner
   - Benefit is proportional to **frontend cost reduction**, not total cost

3. **Symbol-level granularity may be overkill**
   - Frontend cost is dominated by type checking the whole crate
   - Individual symbol costs may not be meaningful for frontend
   - Might need crate-level frontend estimates instead

4. **CGU (codegen unit) analysis is less important**
   - We spent effort modeling CGU costs and backend parallelism
   - But codegen is mostly "free" (runs during downstream frontend)
   - Only matters for leaf crates with no downstream dependencies

#### Possible New Approach

1. **Focus on frontend costs for critical path**
   - `frontend_cost ≈ type_checking + borrow_checking + macro_expansion`
   - These are largely serial within a crate

2. **Model rmeta availability**
   - `rmeta_ready = start_time + frontend_cost`
   - `downstream.start = max(dep.rmeta_ready for dep in deps)`

3. **Estimate frontend cost for synthetic crates**
   - May need different predictors than total cost
   - Symbol count? Type complexity? Trait impl count?

4. **Re-evaluate whether tarjanize's SCC splitting helps**
   - If splitting reduces frontend cost proportionally, it helps
   - If frontend cost has high fixed overhead per crate, splitting may hurt

#### Frontend vs Codegen Breakdown (Answered)

Analysis of cargo timing data for 151 lib targets in Omicron:

| Package | Total | Frontend | Codegen | FE % |
|---------|-------|----------|---------|------|
| omicron-nexus | 166.5s | 114.7s | 51.8s | 69% |
| nexus-db-queries | 33.9s | 29.8s | 4.1s | 88% |
| nexus-db-model | 33.5s | 31.1s | 2.4s | 93% |
| nexus-db-schema | 20.2s | 19.8s | 0.4s | 98% |
| oxide-client | 27.1s | 24.2s | 2.9s | 89% |
| **TOTAL (151 libs)** | **596.0s** | **418.9s** | **177.0s** | **70%** |

**Key finding: Frontend is 70% of lib compilation time** (weighted by time).

The critical path bottlenecks are especially frontend-heavy:
- nexus-db-schema: 98% frontend
- nexus-db-model: 93% frontend
- nexus-db-queries: 88% frontend

This confirms the rmeta pipelining effect is massive. The 30% codegen time largely runs in parallel with downstream compilation and is effectively "free" for critical path purposes.

#### Remaining Open Questions

1. ~~**What fraction of lib compilation is frontend vs codegen?**~~ **ANSWERED: 70% frontend**

2. **Does splitting a crate reduce total frontend cost?**
   - Or does it increase due to cross-crate type checking overhead?

3. ~~**Can we estimate frontend cost from profile data?**~~ **ANSWERED: Yes**
   - We already capture `symbol.frontend_cost_ms` from self-profile
   - The model uses `Σ frontend_cost_ms` for cost estimation
   - Section A.7 shows frontend cost predicts metadata with R² = 0.705
   - This is the data we need for rmeta-based critical path calculation

4. **What's the overhead of adding a new crate?**
   - More crates = more rmeta files to read
   - May increase frontend cost for downstream crates
