# Design: Scaled Metadata Cost for Shattered Crates

## Problem
Currently, `shatter_target` predicts the cost of shattered groups by assuming each group inherits the *full* metadata decoding cost of the original crate. This is overly pessimistic for groups that only use a subset of dependencies.

## Solution
Scale the metadata cost component (`meta`) based on the number of dependencies each shattered group actually requires, relative to the original crate.

## Formula
For a shattered group $G$:

$$ Cost(G) = C_{attr} \cdot Attr(G) + C_{meta} \cdot Meta(G) + C_{other} \cdot Other(G) $$

Where:
*   **$Attr(G)$**: Sum of symbol costs in $G$ (unchanged).
*   **$Other(G)$**: $Other(Original)$ (Fixed/Conservative, as requested).
*   **$Meta(G)$**: $Meta(Original) 	imes \frac{Deps(G)}{Deps(Original)}$

## Dependency Counting
The dependency count $Deps(G)$ includes both:
1.  **External Dependencies**: Number of original upstream targets referenced by symbols in $G$.
2.  **Internal Dependencies**: Number of *other shattered groups* that $G$ depends on (induced by the SCC graph).

$$ Deps(G) = Count(External_{used}) + Count(Internal_{groups}) $$

## Edge Cases
*   **Zero Dependencies**: If $Deps(G) = 0$, the ratio is 0, and predicted metadata cost is 0.
*   **Zero Original Dependencies**: If the original crate had 0 deps, the ratio is undefined (0/0). We assume ratio = 0 (cost remains 0).

## Implementation Plan
1.  **Modify `shatter_target`** in `crates/tarjanize-schedule/src/recommend.rs`.
2.  **Compute Group Dependencies**:
    *   Reuse `group_ext` for external dependency counts.
    *   Analyze `intra` (SCC DAG) and `scc_to_group` mapping to count unique group-to-group edges.
3.  **Apply Scaling**: Calculate the ratio and scale the `meta` input to the cost model.
4.  **Verify**: Ensure `shatter_cost_repro` (or a new test) confirms the scaling behavior.
