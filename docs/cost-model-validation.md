# Cost Model Validation Report

This document validates tarjanize's cost model against actual build times from a large Rust workspace (Omicron, ~160 crates).

## Executive Summary

**Question**: Does our profiling-based cost model predict real build times?

**Answer**: Yes. The model explains **90% of variance** in actual build times (R² = 0.900) with a correlation of **0.949**. It correctly identifies the most expensive crates and captures their relative costs.

**Key findings**:
- Top 3 bottlenecks correctly identified
- 4/5 top crates ranked correctly
- Relative costs accurate within 2× for 7/10 top crates
- Model overestimates by ~3× (consistent, correctable)

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
  overhead      = linking_ms + metadata_ms
```

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

We tested how much each cost component contributes to model accuracy:

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

### A.8 Linking Time Estimation for Synthetic Crates

Since synthetic crates don't have measured linking times, we need to estimate them.

**Single-variable predictors tested:**

| Predictor | R² |
|-----------|-----|
| dep_crate_count | **0.555** |
| frontend | 0.399 |
| unique_external_deps | 0.358 |
| external_dep_edges | 0.341 |
| unique_deps | 0.246 |
| symbols | 0.000 |

**Key finding:** The number of crates depended on (`dep_crate_count`) is the best predictor of linking time, explaining 56% of variance. This makes sense because linking involves loading each dependency's rlib, resolving symbols across crate boundaries, and performing symbol table lookups — all of which scale with the number of crates, not the number of individual symbols.

**Best model for synthetic crates:**

```
linking_ms = 160 × dep_crate_count + 678
```

Where `dep_crate_count` is the number of unique external crates that symbols in the synthetic crate depend on (excluding self-references).

**Error characteristics:**

| Metric | Value |
|--------|-------|
| R² | 0.555 |
| Mean absolute % error | 93% |
| Median absolute % error | 58% |
| 90th percentile error | 219% |

**Why this works for synthetic crates:** When we create a synthetic crate by merging symbols, we know each symbol's dependencies. We can compute `dep_crate_count` by collecting all dependency paths, extracting their crate names (the first component of the path), and counting unique external crates.

**Note on outliers:** Some crates with 0 recorded dependencies still have significant linking times (e.g., `reconfigurator_cli`: 2760ms). These are typically binaries with external dependencies not captured in the symbol graph. The model's intercept (678ms) provides a baseline for such cases.

**Comparison with previous approach:** The old approach distributed linking time proportionally by symbol count, but symbol count has essentially zero correlation with linking time (R² ≈ 0). The new dependency-based model is a significant improvement.

---

## Appendix B: TODOs

### B.1 Improve Linking Cost Estimation

The current linking model (R² = 0.555) only counts workspace crate dependencies captured in the symbol graph. However, linking also involves external crates (from crates.io, git dependencies, etc.) which are not currently tracked.

**TODO:** Add a complete list of crate dependencies (including external crates) at the crate level in the symbol graph. This would allow the linking cost model to count all dependencies, not just workspace-internal ones. This may significantly improve the R² for linking estimation.

### B.2 Fix Critical Path Computation for Dev-Dependency Cycles

The critical path code currently fails when the crate dependency graph contains cycles. These cycles can occur when dev-dependencies create apparent circular dependencies (e.g., crate A's tests depend on crate B, which depends on crate A).

**Current behavior:** Prints a warning and skips critical path computation, only reporting per-crate costs.

**TODO:** Properly handle dev-dependency cycles by either:
1. Separating lib and test targets in the dependency graph
2. Detecting and breaking cycles at dev-dependency edges
3. Computing critical path on the lib-only subgraph

### B.3 Validate Critical Path Cost Model

The per-crate cost model has been validated (R² = 0.900), but the **critical path cost** (sum of costs along the longest dependency chain) has not been validated against real build times with parallelism.

**TODO:** Validate that our critical path cost correlates with actual wall-clock build times when using `cargo build -j N`. This requires:
1. Capturing actual parallel build times (not just per-crate times)
2. Computing critical path from our model
3. Comparing model predictions against actual parallel build duration
