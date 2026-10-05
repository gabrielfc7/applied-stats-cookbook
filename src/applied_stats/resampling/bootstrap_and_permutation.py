"""Vectorized bootstrapping and permutation test routines with BCa confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
from scipy import stats

from applied_stats.base import check_sample


@dataclass
class BootstrapResult:
    """Container for bootstrap point estimate, standard error, and confidence intervals."""

    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    method: str
    n_resamples: int
    bootstrap_distribution: np.ndarray = field(repr=False)

    def summary(self) -> str:
        ci_pct = int(self.ci_level * 100)
        return (
            f"Bootstrap Estimate: {self.estimate:.4f} (SE = {self.standard_error:.4f})\n"
            f"{ci_pct}% {self.method.capitalize()} CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


@dataclass
class PermutationResult:
    """Container for two-sample permutation test results."""

    observed_statistic: float
    p_value: float
    alternative: str
    n_permutations: int
    sample_sizes: tuple[int, int]
    null_distribution: np.ndarray = field(repr=False)

    def summary(self) -> str:
        p_str = "< .0001" if self.p_value < 0.0001 else f"= {self.p_value:.4f}"
        return (
            f"Observed Statistic: {self.observed_statistic:.4f}\n"
            f"Permutation p-value ({self.alternative}): {p_str} (B = {self.n_permutations})"
        )


def bootstrap_ci(
    data: Sequence[float] | np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 4999,
    ci_level: float = 0.95,
    method: Literal["percentile", "bca", "basic"] = "percentile",
    random_state: Optional[int] = None,
) -> BootstrapResult:
    """Compute vectorized bootstrap confidence intervals for arbitrary scalar statistics.

    Parameters
    ----------
    data : array-like
        1D array of numeric observations.
    statistic_fn : callable, default=np.mean
        Function taking a 1D NumPy array and returning a scalar estimate.
    n_resamples : int, default=4999
        Number of bootstrap replications.
    ci_level : float, default=0.95
        Confidence level (e.g. 0.95 for 95% CI).
    method : {'percentile', 'bca', 'basic'}, default='percentile'
        Bootstrap CI computation algorithm.
    random_state : int, optional
        Seed for reproducibility.
    """
    arr = check_sample(data, "data")
    n = len(arr)
    rng = np.random.RandomState(random_state)

    point_est = float(statistic_fn(arr))

    # Vectorized bootstrap resample index generation
    boot_indices = rng.randint(0, n, size=(n_resamples, n))
    boot_samples = arr[boot_indices]
    boot_dist = np.array([statistic_fn(sample) for sample in boot_samples], dtype=np.float64)

    se_boot = float(np.std(boot_dist, ddof=1))
    alpha = 1.0 - ci_level

    if method == "percentile":
        q_low = 100.0 * (alpha / 2.0)
        q_high = 100.0 * (1.0 - alpha / 2.0)
        ci_lower = float(np.percentile(boot_dist, q_low))
        ci_upper = float(np.percentile(boot_dist, q_high))

    elif method == "basic":
        q_low = 100.0 * (alpha / 2.0)
        q_high = 100.0 * (1.0 - alpha / 2.0)
        ci_lower = float(2.0 * point_est - np.percentile(boot_dist, q_high))
        ci_upper = float(2.0 * point_est - np.percentile(boot_dist, q_low))

    elif method == "bca":
        # Bias-correction parameter z0
        prop_less = np.mean(boot_dist < point_est)
        prop_less = np.clip(prop_less, 1e-6, 1.0 - 1e-6)
        z0 = float(stats.norm.ppf(prop_less))

        # Acceleration parameter 'a' via jackknife influence values
        jack_vals = np.empty(n, dtype=np.float64)
        for i in range(n):
            jack_sample = np.delete(arr, i)
            jack_vals[i] = statistic_fn(jack_sample)
        jack_mean = np.mean(jack_vals)
        jack_diff = jack_mean - jack_vals

        num = np.sum(jack_diff ** 3)
        denom = 6.0 * (np.sum(jack_diff ** 2) ** 1.5)
        a = float(num / denom) if denom != 0 else 0.0

        # Percentile boundaries
        z_alpha1 = stats.norm.ppf(alpha / 2.0)
        z_alpha2 = stats.norm.ppf(1.0 - alpha / 2.0)

        a1_adj = stats.norm.cdf(z0 + (z0 + z_alpha1) / (1.0 - a * (z0 + z_alpha1)))
        a2_adj = stats.norm.cdf(z0 + (z0 + z_alpha2) / (1.0 - a * (z0 + z_alpha2)))

        a1_adj = np.clip(a1_adj, 0.0, 1.0)
        a2_adj = np.clip(a2_adj, 0.0, 1.0)

        ci_lower = float(np.percentile(boot_dist, 100.0 * a1_adj))
        ci_upper = float(np.percentile(boot_dist, 100.0 * a2_adj))

    else:
        raise ValueError(f"Unknown bootstrap method '{method}'. Choose 'percentile', 'bca', or 'basic'.")

    return BootstrapResult(
        estimate=point_est,
        standard_error=se_boot,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        method=method,
        n_resamples=n_resamples,
        bootstrap_distribution=boot_dist,
    )


def permutation_test(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    statistic_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_permutations: int = 9999,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    random_state: Optional[int] = None,
) -> PermutationResult:
    """Exact Monte Carlo permutation test for two independent samples.

    Parameters
    ----------
    x, y : array-like
        Numeric sample vectors.
    statistic_fn : callable, optional
        Function taking (x, y) arrays and returning a test statistic.
        Defaults to difference in means: mean(x) - mean(y).
    n_permutations : int, default=9999
        Number of Monte Carlo permutations B.
    alternative : {'two-sided', 'greater', 'less'}, default='two-sided'
        Alternative hypothesis.
    random_state : int, optional
        Seed for reproducibility.
    """
    s1 = check_sample(x, "x")
    s2 = check_sample(y, "y")
    n1, n2 = len(s1), len(s2)
    n_total = n1 + n2

    if statistic_fn is None:
        def diff_means(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.mean(a) - np.mean(b))
        statistic_fn = diff_means

    t_obs = float(statistic_fn(s1, s2))

    pooled = np.concatenate([s1, s2])
    rng = np.random.RandomState(random_state)

    null_dist = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        p1 = shuffled[:n1]
        p2 = shuffled[n1:]
        null_dist[i] = statistic_fn(p1, p2)

    # Davison & Hinkley (1997) continuity adjustment: (1 + count) / (1 + B)
    if alternative == "two-sided":
        extreme_count = np.sum(np.abs(null_dist) >= np.abs(t_obs))
    elif alternative == "greater":
        extreme_count = np.sum(null_dist >= t_obs)
    elif alternative == "less":
        extreme_count = np.sum(null_dist <= t_obs)
    else:
        raise ValueError(f"Unknown alternative '{alternative}'. Choose 'two-sided', 'greater', or 'less'.")

    p_val = float((1.0 + extreme_count) / (1.0 + n_permutations))

    return PermutationResult(
        observed_statistic=t_obs,
        p_value=p_val,
        alternative=alternative,
        n_permutations=n_permutations,
        sample_sizes=(n1, n2),
        null_distribution=null_dist,
    )
