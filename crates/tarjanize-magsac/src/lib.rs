//! MAGSAC++ robust estimator for a two-variable, no-intercept linear model.
//!
//! This crate provides a focused implementation of the MAGSAC++ loop for
//! fitting `y = a * x1 + b * x2` with strong robustness to outliers. The
//! implementation is tailored to scalar residuals (1 degree of freedom),
//! which allows closed-form expressions for the MAGSAC++ weights and loss.

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2 * pi)
const K_99: f64 = 2.575_829_303_548_900_4; // 0.99-quantile of Chi(1)
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
/// Two-variable no-intercept linear model: `y = a * x1 + b * x2`.
#[derive(Debug, Clone, Copy)]
pub struct TwoVarModel {
    /// Coefficient for the first predictor.
    pub a: f64,
    /// Coefficient for the second predictor.
    pub b: f64,
}

impl TwoVarModel {
    #[inline]
    pub fn predict(&self, x1: f64, x2: f64) -> f64 {
        self.a * x1 + self.b * x2
    }
}

/// Result from MAGSAC++ fitting.
#[derive(Debug, Clone, Copy)]
pub struct FitResult {
    /// Best model parameters.
    pub model: TwoVarModel,
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

/// Fits `y = a * x1 + b * x2` using MAGSAC++ with default parameters.
#[must_use]
pub fn fit_two_var_magsac(data: &[(f64, f64, f64)]) -> Option<FitResult> {
    fit_two_var_magsac_with_params(data, &MagsacParams::default())
}

/// Fits `y = a * x1 + b * x2` using MAGSAC++ with custom parameters.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "single control loop is easier to follow here"
)]
pub fn fit_two_var_magsac_with_params(
    data: &[(f64, f64, f64)],
    params: &MagsacParams,
) -> Option<FitResult> {
    if data.len() < 3 {
        return None;
    }

    let sigma_max = match params.sigma_max {
        SigmaMax::Fixed(value) => value,
        SigmaMax::FromOls {
            percentile,
            min_sigma,
        } => estimate_sigma_max_from_ols(data, percentile, min_sigma)?,
    };

    if !sigma_max.is_finite() || sigma_max <= 0.0 {
        return None;
    }

    let reference_threshold =
        params.reference_threshold.unwrap_or(K_99 * sigma_max);

    // If a predictor is entirely zero, fall back to OLS since the minimal
    // solver is under-determined in that degenerate case.
    if is_single_predictor(data) {
        let model = ols_two_var(data)?;
        let score = magsac_score(data, model, sigma_max);
        let inlier_count = count_inliers(data, model, reference_threshold);
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
    if let Some(ols_model) = ols_two_var(data) {
        let (model, score, inliers) = sigma_consensus_plus_plus(
            data,
            ols_model,
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

    let sample_size = 2usize;
    let n = data.len();
    let total_pairs = n.saturating_mul(n.saturating_sub(1)) / 2;

    if total_pairs > 0 && total_pairs <= max_iterations {
        for i in 0..n.saturating_sub(1) {
            for j in (i + 1)..n {
                iterations += 1;
                if let Some(model) = solve_minimal(&data[i], &data[j]) {
                    let (model, score, inliers) = sigma_consensus_plus_plus(
                        data,
                        model,
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
            }
        }
    } else {
        while iterations < max_iterations {
            iterations += 1;
            let (i, j) = sample_two_distinct(&mut rng, n);
            if let Some(model) = solve_minimal(&data[i], &data[j]) {
                let (model, score, inliers) = sigma_consensus_plus_plus(
                    data,
                    model,
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

/// Computes R² for a no-intercept model.
#[must_use]
pub fn r_squared_no_intercept(
    data: &[(f64, f64, f64)],
    model: TwoVarModel,
) -> f64 {
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for &(x1, x2, y) in data {
        let predicted = model.predict(x1, x2);
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
#[must_use]
pub fn r_squared_no_intercept_inliers(
    data: &[(f64, f64, f64)],
    model: TwoVarModel,
    threshold: f64,
) -> f64 {
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let mut count = 0usize;

    for &(x1, x2, y) in data {
        let residual = (y - model.predict(x1, x2)).abs();
        if residual > threshold {
            continue;
        }
        let predicted = model.predict(x1, x2);
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

fn sigma_consensus_plus_plus(
    data: &[(f64, f64, f64)],
    initial_model: TwoVarModel,
    sigma_max: f64,
    reference_threshold: f64,
    irls_iters: usize,
) -> (TwoVarModel, f64, usize) {
    let mut best_model = initial_model;
    let mut best_score = magsac_score(data, initial_model, sigma_max);
    let mut best_inliers =
        count_inliers(data, initial_model, reference_threshold);

    let max_threshold = K_99 * sigma_max;
    let e1_k = expint_e1(K_99 * K_99 / 2.0);

    let mut current_model = initial_model;

    for _ in 0..irls_iters {
        let mut indices = Vec::new();
        let mut weights = Vec::new();

        for (idx, &(x1, x2, y)) in data.iter().enumerate() {
            let residual = (y - current_model.predict(x1, x2)).abs();
            if residual >= max_threshold {
                continue;
            }

            let weight = magsac_weight(residual, sigma_max, e1_k);
            if weight > 0.0 {
                indices.push(idx);
                weights.push(weight);
            }
        }

        if indices.len() < 2 {
            break;
        }

        let Some(model) = weighted_least_squares(data, &indices, &weights)
        else {
            break;
        };

        let score = magsac_score(data, model, sigma_max);
        if score > best_score {
            best_score = score;
            best_model = model;
            best_inliers = count_inliers(data, model, reference_threshold);
        }

        current_model = model;
    }

    (best_model, best_score, best_inliers)
}

fn magsac_score(
    data: &[(f64, f64, f64)],
    model: TwoVarModel,
    sigma_max: f64,
) -> f64 {
    let max_threshold = K_99 * sigma_max;
    let e1_k = expint_e1(K_99 * K_99 / 2.0);
    let outlier_loss =
        sigma_max * INV_SQRT_2PI * (1.0 - (-K_99 * K_99 / 2.0).exp());

    let mut total_loss = 0.0;
    for &(x1, x2, y) in data {
        let residual = (y - model.predict(x1, x2)).abs();
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

fn count_inliers(
    data: &[(f64, f64, f64)],
    model: TwoVarModel,
    threshold: f64,
) -> usize {
    data.iter()
        .filter(|&&(x1, x2, y)| (y - model.predict(x1, x2)).abs() < threshold)
        .count()
}

fn solve_minimal(
    a: &(f64, f64, f64),
    b: &(f64, f64, f64),
) -> Option<TwoVarModel> {
    let (x1a, x2a, ya) = *a;
    let (x1b, x2b, yb) = *b;
    let det = x1a * x2b - x1b * x2a;

    if det.abs() < 1e-12 {
        return None;
    }

    let a_coeff = (ya * x2b - yb * x2a) / det;
    let b_coeff = (x1a * yb - x1b * ya) / det;
    Some(TwoVarModel {
        a: a_coeff,
        b: b_coeff,
    })
}

fn weighted_least_squares(
    data: &[(f64, f64, f64)],
    indices: &[usize],
    weights: &[f64],
) -> Option<TwoVarModel> {
    if indices.is_empty() || indices.len() != weights.len() {
        return None;
    }

    let mut s11 = 0.0;
    let mut s12 = 0.0;
    let mut s22 = 0.0;
    let mut sy1 = 0.0;
    let mut sy2 = 0.0;

    for (&idx, &w) in indices.iter().zip(weights) {
        let (x1, x2, y) = data[idx];
        let wx1 = w * x1;
        let wx2 = w * x2;
        s11 += wx1 * x1;
        s12 += wx1 * x2;
        s22 += wx2 * x2;
        sy1 += wx1 * y;
        sy2 += wx2 * y;
    }

    solve_normal_equations(s11, s12, s22, sy1, sy2)
}

fn ols_two_var(data: &[(f64, f64, f64)]) -> Option<TwoVarModel> {
    let mut s11 = 0.0;
    let mut s12 = 0.0;
    let mut s22 = 0.0;
    let mut sy1 = 0.0;
    let mut sy2 = 0.0;

    for &(x1, x2, y) in data {
        s11 += x1 * x1;
        s12 += x1 * x2;
        s22 += x2 * x2;
        sy1 += x1 * y;
        sy2 += x2 * y;
    }

    solve_normal_equations(s11, s12, s22, sy1, sy2)
}

fn solve_normal_equations(
    s11: f64,
    s12: f64,
    s22: f64,
    sy1: f64,
    sy2: f64,
) -> Option<TwoVarModel> {
    let det = s11 * s22 - s12 * s12;
    if det.abs() < 1e-15 {
        if s11 > 1e-15 && s22.abs() < 1e-15 {
            return Some(TwoVarModel {
                a: sy1 / s11,
                b: 0.0,
            });
        }
        if s22 > 1e-15 && s11.abs() < 1e-15 {
            return Some(TwoVarModel {
                a: 0.0,
                b: sy2 / s22,
            });
        }
        return None;
    }

    let a = (s22 * sy1 - s12 * sy2) / det;
    let b = (s11 * sy2 - s12 * sy1) / det;
    Some(TwoVarModel { a, b })
}

fn estimate_sigma_max_from_ols(
    data: &[(f64, f64, f64)],
    percentile: f64,
    min_sigma: f64,
) -> Option<f64> {
    let model = ols_two_var(data)?;
    let mut residuals: Vec<f64> = data
        .iter()
        .map(|&(x1, x2, y)| (y - model.predict(x1, x2)).abs())
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
    let log_prob_good =
        (1.0 - inlier_ratio.powi(sample_size_i32)).ln();
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

fn is_single_predictor(data: &[(f64, f64, f64)]) -> bool {
    let all_x1_zero = data.iter().all(|&(x1, _, _)| x1.abs() < 1e-15);
    let all_x2_zero = data.iter().all(|&(_, x2, _)| x2.abs() < 1e-15);
    all_x1_zero || all_x2_zero
}

fn sample_two_distinct(rng: &mut XorShift64, len: usize) -> (usize, usize) {
    let i = rng.next_usize(len);
    let mut j = rng.next_usize(len - 1);
    if j >= i {
        j += 1;
    }
    (i, j)
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
    fn test_weighted_least_squares_exact() {
        let data = vec![
            (1.0, 2.0, 7.0),  // 3*1 + 2*2
            (2.0, 1.0, 8.0),  // 3*2 + 2*1
            (3.0, 4.0, 17.0), // 3*3 + 2*4
        ];

        let indices = vec![0, 1, 2];
        let weights = vec![1.0, 1.0, 1.0];
        let model = weighted_least_squares(&data, &indices, &weights).unwrap();
        assert!((model.a - 3.0).abs() < 1e-10);
        assert!((model.b - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_magsac_recovers_with_outliers() {
        let mut data = Vec::new();
        let mut rng = XorShift64::new(12345);

        for _ in 0..60 {
            let x1 =
                f64::from(u32::try_from(rng.next_u64() % 1000).unwrap())
                    / 10.0;
            let x2 =
                f64::from(u32::try_from(rng.next_u64() % 500).unwrap())
                    / 10.0;
            let noise =
                f64::from(u32::try_from(rng.next_u64() % 100).unwrap())
                    / 100.0
                    - 0.5;
            let y = 3.0 * x1 + 2.0 * x2 + noise;
            data.push((x1, x2, y));
        }

        for _ in 0..15 {
            let x1 =
                f64::from(u32::try_from(rng.next_u64() % 1000).unwrap())
                    / 10.0;
            let x2 =
                f64::from(u32::try_from(rng.next_u64() % 500).unwrap())
                    / 10.0;
            let y =
                200.0 + f64::from(u32::try_from(rng.next_u64() % 500).unwrap());
            data.push((x1, x2, y));
        }

        let params = MagsacParams {
            max_iterations: 3_000,
            ..MagsacParams::default()
        };

        let result = fit_two_var_magsac_with_params(&data, &params).unwrap();
        assert!((result.model.a - 3.0).abs() < 0.2, "a = {}", result.model.a);
        assert!((result.model.b - 2.0).abs() < 0.2, "b = {}", result.model.b);
    }

    #[test]
    fn test_r_squared_no_intercept_perfect() {
        let data = vec![(1.0, 1.0, 5.0), (2.0, 3.0, 12.0), (4.0, 1.0, 14.0)];
        let model = TwoVarModel { a: 3.0, b: 2.0 };
        let r2 = r_squared_no_intercept(&data, model);
        assert!((r2 - 1.0).abs() < 1e-10);
    }
}
