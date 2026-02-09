//! MAGSAC++ robust estimator for N-variable, no-intercept linear models.
//!
//! This crate provides a focused implementation of the MAGSAC++ loop for
//! fitting `y = β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ` with strong robustness to
//! outliers. The implementation is tailored to scalar residuals (1 degree of
//! freedom), which allows closed-form expressions for the MAGSAC++ weights
//! and loss.
//!
//! Data is passed as `&[Vec<f64>]` where each inner vec is
//! `[x1, x2, ..., xn, y]` — predictors followed by the response. The number
//! of variables is inferred from `data[0].len() - 1`.

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2 * pi)
const K_99: f64 = 2.575_829_303_548_900_4; // 0.99-quantile of Chi(1)
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// N-variable no-intercept linear model: `y = Σ βᵢ·xᵢ`.
///
/// Coefficients are stored in a `Vec<f64>` where `coeffs[i]` corresponds
/// to the i-th predictor variable. The model always passes through the origin
/// (no intercept term) because zero code should produce zero compilation time.
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Coefficients for each predictor variable.
    pub coeffs: Vec<f64>,
}

impl LinearModel {
    /// Predicts the response for the given predictor values.
    ///
    /// Computes `Σ coeffs[i] * x[i]` (dot product). Panics if `x.len()`
    /// doesn't match `coeffs.len()`.
    #[inline]
    pub fn predict(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), self.coeffs.len());
        self.coeffs.iter().zip(x).map(|(c, xi)| c * xi).sum()
    }
}

/// Result from MAGSAC++ fitting.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Best model parameters.
    pub model: LinearModel,
    /// MAGSAC++ score (1 / total loss).
    pub score: f64,
    /// Count of points within the reference threshold.
    pub inlier_count: usize,
    /// Residual threshold used to decide inliers.
    pub inlier_threshold: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// The sigma-max used for marginalization.
    pub sigma_max: f64,
}

/// How to set the maximum noise scale (sigma-max).
#[derive(Debug, Clone, Copy)]
pub enum SigmaMax {
    /// Use a fixed sigma-max.
    Fixed(f64),
    /// Derive sigma-max from OLS residuals.
    ///
    /// This matches the intended bootstrap: fit a vanilla OLS model once,
    /// compute absolute residuals, then take a chosen percentile as σmax.
    /// Using a high percentile keeps most points eligible as inliers while
    /// still bounding the noise scale for MAGSAC++.
    FromOls {
        /// Percentile of absolute residuals to use (0..=1).
        percentile: f64,
        /// Minimum sigma value to avoid a zero scale.
        min_sigma: f64,
    },
}

/// Parameters controlling the MAGSAC++ loop.
#[derive(Debug, Clone, Copy)]
pub struct MagsacParams {
    /// Confidence for early termination (0 < confidence < 1).
    pub confidence: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Minimum number of iterations before early termination.
    pub min_iterations: usize,
    /// Iteratively reweighted least-squares iterations.
    pub irls_iters: usize,
    /// Sigma-max configuration.
    pub sigma_max: SigmaMax,
    /// RNG seed for random sampling.
    pub seed: u64,
    /// Residual threshold used to count inliers; defaults to `k * sigma_max`.
    pub reference_threshold: Option<f64>,
}

impl Default for MagsacParams {
    fn default() -> Self {
        Self {
            confidence: 0.99,
            max_iterations: 2_000,
            min_iterations: 50,
            irls_iters: 2,
            sigma_max: SigmaMax::FromOls {
                percentile: 0.90,
                min_sigma: 1e-6,
            },
            seed: 0x6d_61_67_73_61_63_2b_2b, // "magsac++"
            reference_threshold: None,
        }
    }
}

/// Fits `y = Σ βᵢ·xᵢ` (no intercept) using MAGSAC++ with default parameters.
///
/// Each element of `data` is a `Vec<f64>` of length `n_vars + 1`, where the
/// first `n_vars` values are predictors and the last is the response.
#[must_use]
pub fn fit_magsac(data: &[Vec<f64>]) -> Option<FitResult> {
    fit_magsac_with_params(data, &MagsacParams::default())
}

/// Fits `y = Σ βᵢ·xᵢ` (no intercept) using MAGSAC++ with custom parameters.
///
/// Each element of `data` is a `Vec<f64>` of length `n_vars + 1`. The number
/// of variables is inferred from `data[0].len() - 1`. All rows must have the
/// same length.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "single control loop is easier to follow here"
)]
pub fn fit_magsac_with_params(
    data: &[Vec<f64>],
    params: &MagsacParams,
) -> Option<FitResult> {
    if data.is_empty() {
        return None;
    }

    let n_vars = data[0].len().checked_sub(1)?;
    if n_vars == 0 {
        return None;
    }

    // Need at least n_vars + 1 data points to avoid underdetermined system.
    if data.len() < n_vars + 1 {
        return None;
    }

    let sigma_max = match params.sigma_max {
        SigmaMax::Fixed(value) => value,
        SigmaMax::FromOls {
            percentile,
            min_sigma,
        } => estimate_sigma_max_from_ols(data, n_vars, percentile, min_sigma)?,
    };

    if !sigma_max.is_finite() || sigma_max <= 0.0 {
        return None;
    }

    let reference_threshold =
        params.reference_threshold.unwrap_or(K_99 * sigma_max);

    // If any predictor column is entirely zero, fall back to OLS since the
    // minimal solver is under-determined in that degenerate case.
    if is_degenerate_predictor(data, n_vars) {
        let model = ols_n_var(data, n_vars)?;
        let score = magsac_score(data, &model, n_vars, sigma_max);
        let inlier_count =
            count_inliers(data, &model, n_vars, reference_threshold);
        return Some(FitResult {
            model,
            score,
            inlier_count,
            inlier_threshold: reference_threshold,
            iterations: 0,
            sigma_max,
        });
    }

    let mut best_model = None;
    let mut best_score = 0.0;
    let mut best_inliers = 0;

    // Seed the search with the OLS model as a strong candidate.
    if let Some(ols_model) = ols_n_var(data, n_vars) {
        let (model, score, inliers) = sigma_consensus_plus_plus(
            data,
            ols_model,
            n_vars,
            sigma_max,
            reference_threshold,
            params.irls_iters,
        );
        best_model = Some(model);
        best_score = score;
        best_inliers = inliers;
    }

    let mut rng = XorShift64::new(derive_seed(params.seed, data.len()));
    let mut max_iterations = params.max_iterations.max(params.min_iterations);
    let mut iterations = 0usize;

    // Minimal sample size equals the number of variables.
    let sample_size = n_vars;
    let n = data.len();

    // For small datasets, enumerate all combinations. Otherwise, random sample.
    let total_combinations = n_choose_k(n, sample_size);
    if total_combinations > 0 && total_combinations <= max_iterations {
        // Enumerate all combinations of `sample_size` points.
        enumerate_combinations(n, sample_size, &mut |indices: &[usize]| {
            iterations += 1;
            let sample: Vec<&Vec<f64>> =
                indices.iter().map(|&i| &data[i]).collect();
            if let Some(model) = solve_minimal(&sample, n_vars) {
                let (model, score, inliers) = sigma_consensus_plus_plus(
                    data,
                    model,
                    n_vars,
                    sigma_max,
                    reference_threshold,
                    params.irls_iters,
                );
                if score > best_score {
                    best_score = score;
                    best_model = Some(model);
                    best_inliers = inliers;
                }
            }
        });
    } else {
        while iterations < max_iterations {
            iterations += 1;
            let indices = sample_k_distinct(&mut rng, n, sample_size);
            let sample: Vec<&Vec<f64>> =
                indices.iter().map(|&i| &data[i]).collect();
            if let Some(model) = solve_minimal(&sample, n_vars) {
                let (model, score, inliers) = sigma_consensus_plus_plus(
                    data,
                    model,
                    n_vars,
                    sigma_max,
                    reference_threshold,
                    params.irls_iters,
                );
                if score > best_score {
                    best_score = score;
                    best_model = Some(model);
                    best_inliers = inliers;
                }
            }

            if iterations >= params.min_iterations
                && let Some(required) = required_iterations(
                    params.confidence,
                    best_inliers,
                    n,
                    sample_size,
                )
            {
                max_iterations = max_iterations.min(required);
            }
        }
    }

    let best_model = best_model?;
    Some(FitResult {
        model: best_model,
        score: best_score,
        inlier_count: best_inliers,
        inlier_threshold: reference_threshold,
        iterations,
        sigma_max,
    })
}

/// Computes R² for a no-intercept model using all data points.
///
/// For a no-intercept model, `SS_tot` = Σ yᵢ² (not centered), and
/// R² = 1 - `SS_res` / `SS_tot`. This is the appropriate formulation
/// because the model is constrained to pass through the origin.
#[must_use]
pub fn r_squared_no_intercept(data: &[Vec<f64>], model: &LinearModel) -> f64 {
    let n_vars = model.coeffs.len();
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for row in data {
        let y = row[n_vars];
        let predicted = model.predict(&row[..n_vars]);
        ss_res += (y - predicted).powi(2);
        ss_tot += y * y;
    }
    if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Computes R² for a no-intercept model using only inliers.
///
/// Points whose absolute residual exceeds `threshold` are excluded from
/// both `SS_res` and `SS_tot`. This gives an unbiased quality metric for
/// the model on its consensus set.
#[must_use]
pub fn r_squared_no_intercept_inliers(
    data: &[Vec<f64>],
    model: &LinearModel,
    threshold: f64,
) -> f64 {
    let n_vars = model.coeffs.len();
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let mut count = 0usize;

    for row in data {
        let y = row[n_vars];
        let predicted = model.predict(&row[..n_vars]);
        let residual = (y - predicted).abs();
        if residual > threshold {
            continue;
        }
        ss_res += (y - predicted).powi(2);
        ss_tot += y * y;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// MAGSAC++ sigma-consensus refinement with IRLS.
///
/// Given an initial model, computes MAGSAC weights for each data point, fits
/// a weighted least squares model, and iterates. Returns the best model found
/// along with its score and inlier count.
fn sigma_consensus_plus_plus(
    data: &[Vec<f64>],
    initial_model: LinearModel,
    n_vars: usize,
    sigma_max: f64,
    reference_threshold: f64,
    irls_iters: usize,
) -> (LinearModel, f64, usize) {
    let mut best_model = initial_model.clone();
    let mut best_score = magsac_score(data, &initial_model, n_vars, sigma_max);
    let mut best_inliers =
        count_inliers(data, &initial_model, n_vars, reference_threshold);

    let max_threshold = K_99 * sigma_max;
    let e1_k = expint_e1(K_99 * K_99 / 2.0);

    let mut current_model = initial_model;

    for _ in 0..irls_iters {
        let mut indices = Vec::new();
        let mut weights = Vec::new();

        for (idx, row) in data.iter().enumerate() {
            let y = row[n_vars];
            let residual = (y - current_model.predict(&row[..n_vars])).abs();
            if residual >= max_threshold {
                continue;
            }

            let weight = magsac_weight(residual, sigma_max, e1_k);
            if weight > 0.0 {
                indices.push(idx);
                weights.push(weight);
            }
        }

        if indices.len() < n_vars {
            break;
        }

        let Some(model) =
            weighted_least_squares(data, n_vars, &indices, &weights)
        else {
            break;
        };

        let score = magsac_score(data, &model, n_vars, sigma_max);
        if score > best_score {
            best_score = score;
            best_model = model.clone();
            best_inliers =
                count_inliers(data, &model, n_vars, reference_threshold);
        }

        current_model = model;
    }

    (best_model, best_score, best_inliers)
}

/// Computes the MAGSAC++ score for a model across all data points.
///
/// The score is `1 / total_loss` where each point's loss is the MAGSAC++
/// marginalized loss. Higher scores indicate better models. Outliers
/// (residual >= k·σ) receive a fixed penalty.
fn magsac_score(
    data: &[Vec<f64>],
    model: &LinearModel,
    n_vars: usize,
    sigma_max: f64,
) -> f64 {
    let max_threshold = K_99 * sigma_max;
    let e1_k = expint_e1(K_99 * K_99 / 2.0);
    let outlier_loss =
        sigma_max * INV_SQRT_2PI * (1.0 - (-K_99 * K_99 / 2.0).exp());

    let mut total_loss = 0.0;
    for row in data {
        let y = row[n_vars];
        let residual = (y - model.predict(&row[..n_vars])).abs();
        let loss = if residual >= max_threshold {
            outlier_loss
        } else {
            let x = (residual * residual) / (2.0 * sigma_max * sigma_max);
            let e1_x = expint_e1(x.max(1e-12));
            let term_a = sigma_max * sigma_max * (1.0 - (-x).exp());
            let diff = (e1_x - e1_k).max(0.0);
            let term_b = (residual * residual / 2.0) * diff;
            (term_a + term_b) * INV_SQRT_2PI / sigma_max
        };
        total_loss += loss;
    }

    if total_loss > 0.0 {
        1.0 / total_loss
    } else {
        0.0
    }
}

/// Computes the MAGSAC++ weight for a single data point.
///
/// The weight is derived from the marginalized loss function. Points with
/// small residuals get high weights; points near the threshold get low
/// weights; outliers get zero weight.
fn magsac_weight(residual: f64, sigma_max: f64, e1_k: f64) -> f64 {
    if residual <= 0.0 || sigma_max <= 0.0 {
        return 0.0;
    }

    let max_threshold = K_99 * sigma_max;
    if residual >= max_threshold {
        return 0.0;
    }

    let x = (residual * residual).max(1e-12) / (2.0 * sigma_max * sigma_max);
    let e1_x = expint_e1(x);
    let diff = (e1_x - e1_k).max(0.0);
    let weight = diff * INV_SQRT_2PI / sigma_max;
    if weight.is_finite() && weight > 0.0 {
        weight
    } else {
        0.0
    }
}

/// Counts data points whose absolute residual is below the threshold.
fn count_inliers(
    data: &[Vec<f64>],
    model: &LinearModel,
    n_vars: usize,
    threshold: f64,
) -> usize {
    data.iter()
        .filter(|row| {
            let y = row[n_vars];
            (y - model.predict(&row[..n_vars])).abs() < threshold
        })
        .count()
}

/// Solves the minimal system: N data points → N×N linear system.
///
/// Given exactly `n_vars` data points, builds an N×N system and solves it
/// via Gaussian elimination. This is the minimal sample needed to determine
/// a unique no-intercept linear model.
fn solve_minimal(sample: &[&Vec<f64>], n_vars: usize) -> Option<LinearModel> {
    // Build augmented matrix [X | y] where X is n_vars × n_vars.
    let mut augmented = vec![vec![0.0; n_vars + 1]; n_vars];
    for (row_idx, point) in sample.iter().enumerate() {
        augmented[row_idx][..n_vars].copy_from_slice(&point[..n_vars]);
        augmented[row_idx][n_vars] = point[n_vars];
    }

    solve_gaussian(&mut augmented, n_vars)
}

/// Fits the N-variable no-intercept model using weighted least squares.
///
/// Builds the normal equations `(X'WX) β = X'Wy` for the selected subset
/// of data points with the given weights, then solves via Gaussian
/// elimination.
fn weighted_least_squares(
    data: &[Vec<f64>],
    n_vars: usize,
    indices: &[usize],
    weights: &[f64],
) -> Option<LinearModel> {
    if indices.is_empty() || indices.len() != weights.len() {
        return None;
    }

    // Build X'WX (n_vars × n_vars) and X'Wy (n_vars × 1).
    let mut xtx = vec![vec![0.0; n_vars]; n_vars];
    let mut xty = vec![0.0; n_vars];

    for (&idx, &w) in indices.iter().zip(weights) {
        let row = &data[idx];
        let y = row[n_vars];
        for i in 0..n_vars {
            let wxi = w * row[i];
            for j in 0..n_vars {
                xtx[i][j] += wxi * row[j];
            }
            xty[i] += wxi * y;
        }
    }

    solve_normal_equations(&xtx, &xty, n_vars)
}

/// Fits the N-variable no-intercept model using ordinary least squares.
///
/// Builds the normal equations `(X'X) β = X'y` from all data points and
/// solves via Gaussian elimination.
fn ols_n_var(data: &[Vec<f64>], n_vars: usize) -> Option<LinearModel> {
    let mut xtx = vec![vec![0.0; n_vars]; n_vars];
    let mut xty = vec![0.0; n_vars];

    for row in data {
        let y = row[n_vars];
        for i in 0..n_vars {
            for j in 0..n_vars {
                xtx[i][j] += row[i] * row[j];
            }
            xty[i] += row[i] * y;
        }
    }

    solve_normal_equations(&xtx, &xty, n_vars)
}

/// Solves the normal equations `A β = b` via Gaussian elimination.
///
/// For degenerate cases where one predictor column is all-zero, the
/// corresponding row/column in A will be zero. The solver detects this
/// and returns zero for that coefficient rather than failing.
fn solve_normal_equations(
    xtx: &[Vec<f64>],
    xty: &[f64],
    n_vars: usize,
) -> Option<LinearModel> {
    // Build augmented matrix [A | b].
    let mut augmented = Vec::with_capacity(n_vars);
    for i in 0..n_vars {
        let mut row = Vec::with_capacity(n_vars + 1);
        row.extend_from_slice(&xtx[i]);
        row.push(xty[i]);
        augmented.push(row);
    }

    solve_gaussian(&mut augmented, n_vars)
}

/// Gaussian elimination with partial pivoting for an augmented matrix.
///
/// Solves an N×(N+1) augmented matrix `[A | b]` in place. Uses partial
/// pivoting (row swaps to maximize pivot magnitude) for numerical stability.
/// Returns `None` if a pivot is near-zero (singular/degenerate system).
///
/// For small N (≤ ~5 in our use case), performance is irrelevant —
/// correctness and numerical stability are what matter.
fn solve_gaussian(augmented: &mut [Vec<f64>], n: usize) -> Option<LinearModel> {
    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot: row with maximum absolute value in current column.
        let mut max_val = augmented[col][col].abs();
        let mut max_row = col;
        for (row, aug_row) in augmented.iter().enumerate().take(n).skip(col + 1)
        {
            let val = aug_row[col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // If pivot is near-zero, this predictor has no signal. Set its
        // coefficient to zero and skip this column.
        if max_val < 1e-15 {
            // Zero out the column to prevent numerical noise.
            for aug_row in augmented.iter_mut().take(n).skip(col) {
                aug_row[col] = 0.0;
            }
            continue;
        }

        // Swap pivot row into position.
        if max_row != col {
            augmented.swap(max_row, col);
        }

        // Eliminate below the pivot. We split the slice to get mutable
        // access to the pivot row and the rows below simultaneously.
        let (top, bottom) = augmented.split_at_mut(col + 1);
        let pivot_row = &top[col];
        let pivot = pivot_row[col];
        for target_row in bottom.iter_mut().take(n - col - 1) {
            let factor = target_row[col] / pivot;
            target_row[col] = 0.0;
            for j in (col + 1)..=n {
                target_row[j] -= factor * pivot_row[j];
            }
        }
    }

    // Back-substitution.
    let mut coeffs = vec![0.0; n];
    for col in (0..n).rev() {
        if augmented[col][col].abs() < 1e-15 {
            // Degenerate column — coefficient stays zero.
            coeffs[col] = 0.0;
            continue;
        }

        let mut sum = augmented[col][n];
        for j in (col + 1)..n {
            sum -= augmented[col][j] * coeffs[j];
        }
        coeffs[col] = sum / augmented[col][col];

        if !coeffs[col].is_finite() {
            return None;
        }
    }

    Some(LinearModel { coeffs })
}

/// Estimates sigma-max from OLS residuals.
///
/// Fits a vanilla OLS model, computes absolute residuals, sorts them, and
/// returns the residual at the given percentile (clamped to `min_sigma`).
fn estimate_sigma_max_from_ols(
    data: &[Vec<f64>],
    n_vars: usize,
    percentile: f64,
    min_sigma: f64,
) -> Option<f64> {
    let model = ols_n_var(data, n_vars)?;
    let mut residuals: Vec<f64> = data
        .iter()
        .map(|row| {
            let y = row[n_vars];
            (y - model.predict(&row[..n_vars])).abs()
        })
        .collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = percentile.clamp(0.0, 1.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "percentile index is clamped to valid bounds"
    )]
    let idx = ((residuals.len() - 1) as f64 * p).round() as usize;
    let sigma = residuals[idx].max(min_sigma);
    Some(sigma)
}

/// Estimates the number of RANSAC iterations needed for the given confidence.
fn required_iterations(
    confidence: f64,
    inlier_count: usize,
    total_points: usize,
    sample_size: usize,
) -> Option<usize> {
    if !(0.0..1.0).contains(&confidence) || total_points == 0 {
        return None;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "ratio only needs approximate precision"
    )]
    let inlier_ratio = inlier_count as f64 / total_points as f64;
    if inlier_ratio <= 0.0 || inlier_ratio >= 1.0 {
        return None;
    }

    let log_confidence = (1.0 - confidence).ln();
    let sample_size_i32 = i32::try_from(sample_size).ok()?;
    let log_prob_good = (1.0 - inlier_ratio.powi(sample_size_i32)).ln();
    if log_prob_good >= 0.0 {
        return None;
    }

    let required = (log_confidence / log_prob_good).ceil();
    if !(required.is_finite() && required > 0.0) {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "required iterations is a positive finite integer"
    )]
    let required_u64 = required as u64;
    usize::try_from(required_u64).ok()
}

/// Checks if any predictor column is entirely zero (degenerate).
///
/// When a predictor is all-zero, the minimal solver can't determine its
/// coefficient uniquely, so we fall back to OLS which handles degeneracy
/// via the normal equations.
fn is_degenerate_predictor(data: &[Vec<f64>], n_vars: usize) -> bool {
    for col in 0..n_vars {
        if data.iter().all(|row| row[col].abs() < 1e-15) {
            return true;
        }
    }
    false
}

/// Samples `k` distinct indices from `0..len` using the given RNG.
fn sample_k_distinct(rng: &mut XorShift64, len: usize, k: usize) -> Vec<usize> {
    // Fisher-Yates partial shuffle approach.
    let mut result = Vec::with_capacity(k);
    let mut available: Vec<usize> = (0..len).collect();
    for i in 0..k {
        let j = i + rng.next_usize(len - i);
        available.swap(i, j);
        result.push(available[i]);
    }
    result
}

/// Enumerates all combinations of `k` indices from `0..n` and calls `f`
/// for each combination.
///
/// Used when the total number of combinations is small enough to enumerate
/// exhaustively rather than random sampling.
fn enumerate_combinations(n: usize, k: usize, f: &mut impl FnMut(&[usize])) {
    let mut indices: Vec<usize> = (0..k).collect();
    loop {
        f(&indices);

        // Find the rightmost index that can be incremented.
        let mut i = k;
        while i > 0 {
            i -= 1;
            if indices[i] != i + n - k {
                break;
            }
            if i == 0 && indices[0] == n - k {
                return;
            }
        }

        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

/// Computes n choose k, returning 0 on overflow.
fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    // Use the smaller of k and n-k to minimize intermediate values.
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

fn derive_seed(seed: u64, len: usize) -> u64 {
    seed ^ (len as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Exponential integral E1(x) for x > 0.
fn expint_e1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    if x <= 1.0 {
        let mut sum = 0.0;
        let mut term = 1.0;
        for k in 1..=60 {
            let kf = f64::from(k);
            term *= -x / kf;
            let add = term / kf;
            sum += add;
            if add.abs() < 1e-16 {
                break;
            }
        }
        -x.ln() - EULER_GAMMA - sum
    } else {
        let mut sum = 1.0;
        let mut term = 1.0;
        let mut prev_abs = f64::INFINITY;
        for k in 1..=60 {
            let kf = f64::from(k);
            term *= -kf / x;
            let abs_term = term.abs();
            if abs_term > prev_abs {
                break;
            }
            sum += term;
            prev_abs = abs_term;
            if abs_term < 1e-16 {
                break;
            }
        }
        (-x).exp() * sum / x
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xdead_beef_cafe_f00d
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        let upper_u64 = u64::try_from(upper).unwrap_or(u64::MAX);
        let value = self.next_u64() % upper_u64;
        usize::try_from(value).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expint_e1_known_value() {
        let value = expint_e1(1.0);
        assert!((value - 0.219_383_934_395_520_3).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_least_squares_exact_2var() {
        // 2-variable test: y = 3·x1 + 2·x2
        let data = vec![
            vec![1.0, 2.0, 7.0],  // 3*1 + 2*2
            vec![2.0, 1.0, 8.0],  // 3*2 + 2*1
            vec![3.0, 4.0, 17.0], // 3*3 + 2*4
        ];

        let indices = vec![0, 1, 2];
        let weights = vec![1.0, 1.0, 1.0];
        let model =
            weighted_least_squares(&data, 2, &indices, &weights).unwrap();
        assert!((model.coeffs[0] - 3.0).abs() < 1e-10);
        assert!((model.coeffs[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_least_squares_exact_3var() {
        // 3-variable test: y = 2·x1 + 3·x2 + 4·x3
        let data = vec![
            vec![1.0, 0.0, 0.0, 2.0], // 2*1
            vec![0.0, 1.0, 0.0, 3.0], // 3*1
            vec![0.0, 0.0, 1.0, 4.0], // 4*1
            vec![1.0, 1.0, 1.0, 9.0], // 2+3+4
        ];

        let indices = vec![0, 1, 2, 3];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let model =
            weighted_least_squares(&data, 3, &indices, &weights).unwrap();
        assert!(
            (model.coeffs[0] - 2.0).abs() < 1e-10,
            "a = {}",
            model.coeffs[0]
        );
        assert!(
            (model.coeffs[1] - 3.0).abs() < 1e-10,
            "b = {}",
            model.coeffs[1]
        );
        assert!(
            (model.coeffs[2] - 4.0).abs() < 1e-10,
            "c = {}",
            model.coeffs[2]
        );
    }

    #[test]
    fn test_magsac_recovers_with_outliers_2var() {
        // 2-variable test with outliers.
        let mut data = Vec::new();
        let mut rng = XorShift64::new(12345);

        for _ in 0..60 {
            let x1 =
                f64::from(u32::try_from(rng.next_u64() % 1000).unwrap()) / 10.0;
            let x2 =
                f64::from(u32::try_from(rng.next_u64() % 500).unwrap()) / 10.0;
            let noise = f64::from(u32::try_from(rng.next_u64() % 100).unwrap())
                / 100.0
                - 0.5;
            let y = 3.0 * x1 + 2.0 * x2 + noise;
            data.push(vec![x1, x2, y]);
        }

        for _ in 0..15 {
            let x1 =
                f64::from(u32::try_from(rng.next_u64() % 1000).unwrap()) / 10.0;
            let x2 =
                f64::from(u32::try_from(rng.next_u64() % 500).unwrap()) / 10.0;
            let y =
                200.0 + f64::from(u32::try_from(rng.next_u64() % 500).unwrap());
            data.push(vec![x1, x2, y]);
        }

        let params = MagsacParams {
            max_iterations: 3_000,
            ..MagsacParams::default()
        };

        let result = fit_magsac_with_params(&data, &params).unwrap();
        assert!(
            (result.model.coeffs[0] - 3.0).abs() < 0.2,
            "a = {}",
            result.model.coeffs[0]
        );
        assert!(
            (result.model.coeffs[1] - 2.0).abs() < 0.2,
            "b = {}",
            result.model.coeffs[1]
        );
    }

    #[test]
    fn test_magsac_recovers_with_outliers_3var() {
        // 3-variable test with outliers: y = 2·x1 + 3·x2 + 4·x3
        let mut data = Vec::new();
        let mut rng = XorShift64::new(54321);

        for _ in 0..80 {
            let x1 =
                f64::from(u32::try_from(rng.next_u64() % 1000).unwrap()) / 10.0;
            let x2 =
                f64::from(u32::try_from(rng.next_u64() % 500).unwrap()) / 10.0;
            let x3 =
                f64::from(u32::try_from(rng.next_u64() % 300).unwrap()) / 10.0;
            let noise = f64::from(u32::try_from(rng.next_u64() % 100).unwrap())
                / 100.0
                - 0.5;
            let y = 2.0 * x1 + 3.0 * x2 + 4.0 * x3 + noise;
            data.push(vec![x1, x2, x3, y]);
        }

        for _ in 0..20 {
            let x1 =
                f64::from(u32::try_from(rng.next_u64() % 1000).unwrap()) / 10.0;
            let x2 =
                f64::from(u32::try_from(rng.next_u64() % 500).unwrap()) / 10.0;
            let x3 =
                f64::from(u32::try_from(rng.next_u64() % 300).unwrap()) / 10.0;
            let y =
                500.0 + f64::from(u32::try_from(rng.next_u64() % 500).unwrap());
            data.push(vec![x1, x2, x3, y]);
        }

        let params = MagsacParams {
            max_iterations: 5_000,
            ..MagsacParams::default()
        };

        let result = fit_magsac_with_params(&data, &params).unwrap();
        assert!(
            (result.model.coeffs[0] - 2.0).abs() < 0.3,
            "a = {}",
            result.model.coeffs[0]
        );
        assert!(
            (result.model.coeffs[1] - 3.0).abs() < 0.3,
            "b = {}",
            result.model.coeffs[1]
        );
        assert!(
            (result.model.coeffs[2] - 4.0).abs() < 0.3,
            "c = {}",
            result.model.coeffs[2]
        );
    }

    #[test]
    fn test_r_squared_no_intercept_perfect() {
        let data = vec![
            vec![1.0, 1.0, 5.0],
            vec![2.0, 3.0, 12.0],
            vec![4.0, 1.0, 14.0],
        ];
        let model = LinearModel {
            coeffs: vec![3.0, 2.0],
        };
        let r2 = r_squared_no_intercept(&data, &model);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_elimination_3x3() {
        // Solve: 2a + 1b + 1c = 10
        //        1a + 3b + 2c = 13
        //        1a + 0b + 4c = 11
        // Solution: a=3, b=2, c=2
        // Verify: 2*3+1*2+1*2=10, 1*3+3*2+2*2=13, 1*3+0*2+4*2=11 ✓
        let mut augmented = vec![
            vec![2.0, 1.0, 1.0, 10.0],
            vec![1.0, 3.0, 2.0, 13.0],
            vec![1.0, 0.0, 4.0, 11.0],
        ];
        let model = solve_gaussian(&mut augmented, 3).unwrap();
        assert!(
            (model.coeffs[0] - 3.0).abs() < 1e-10,
            "a = {}",
            model.coeffs[0]
        );
        assert!(
            (model.coeffs[1] - 2.0).abs() < 1e-10,
            "b = {}",
            model.coeffs[1]
        );
        assert!(
            (model.coeffs[2] - 2.0).abs() < 1e-10,
            "c = {}",
            model.coeffs[2]
        );
    }

    #[test]
    fn test_gaussian_elimination_degenerate_column() {
        // Second column all zero: effectively y = a·x1 + c·x3.
        // 2a + 0b + 1c = 7  → a=3, c=1
        // 1a + 0b + 2c = 5
        // 3a + 0b + 1c = 10
        let mut augmented = vec![
            vec![2.0, 0.0, 1.0, 7.0],
            vec![1.0, 0.0, 2.0, 5.0],
            vec![3.0, 0.0, 1.0, 10.0],
        ];
        let model = solve_gaussian(&mut augmented, 3).unwrap();
        assert!(
            (model.coeffs[0] - 3.0).abs() < 1e-10,
            "a = {}",
            model.coeffs[0]
        );
        assert!(
            model.coeffs[1].abs() < 1e-10,
            "b should be 0, got {}",
            model.coeffs[1]
        );
        assert!(
            (model.coeffs[2] - 1.0).abs() < 1e-10,
            "c = {}",
            model.coeffs[2]
        );
    }

    #[test]
    fn test_n_choose_k() {
        assert_eq!(n_choose_k(5, 2), 10);
        assert_eq!(n_choose_k(3, 3), 1);
        assert_eq!(n_choose_k(3, 0), 1);
        assert_eq!(n_choose_k(10, 3), 120);
        assert_eq!(n_choose_k(2, 3), 0);
    }

    #[test]
    fn test_enumerate_combinations() {
        let mut combos: Vec<Vec<usize>> = Vec::new();
        enumerate_combinations(4, 2, &mut |indices| {
            combos.push(indices.to_vec());
        });
        assert_eq!(
            combos,
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![1, 2],
                vec![1, 3],
                vec![2, 3],
            ]
        );
    }

    #[test]
    fn test_fit_magsac_insufficient_data() {
        // Fewer than n_vars + 1 points → None.
        assert!(fit_magsac(&[]).is_none());
        assert!(fit_magsac(&[vec![1.0, 2.0]]).is_none());
        // 2 vars, need 3 points minimum.
        assert!(
            fit_magsac(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).is_none()
        );
    }
}
