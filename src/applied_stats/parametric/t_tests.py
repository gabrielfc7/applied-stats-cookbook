"""Parametric hypothesis testing: One-sample, Two-sample, and Paired Student/Welch t-tests."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import stats

from applied_stats.base import TestResult, check_sample


def one_sample_t_test(
    x: Any,
    popmean: float = 0.0,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """One-sample Student's t-test with Cohen's d effect size and confidence interval."""
    sample = check_sample(x, "x")
    n = len(sample)
    mean_val = float(np.mean(sample))
    std_val = float(np.std(sample, ddof=1))

    if std_val == 0.0:
        raise ValueError("Sample standard deviation is zero; cannot compute t-statistic.")

    se = std_val / np.sqrt(n)
    t_stat = (mean_val - popmean) / se
    df = n - 1

    # P-value calculation
    if alternative == "two-sided":
        p_val = 2.0 * float(stats.t.sf(np.abs(t_stat), df=df))
    elif alternative == "greater":
        p_val = float(stats.t.sf(t_stat, df=df))
    elif alternative == "less":
        p_val = float(stats.t.cdf(t_stat, df=df))
    else:
        raise ValueError(f"Unknown alternative '{alternative}'. Choose 'two-sided', 'less', or 'greater'.")

    # Cohen's d effect size
    cohen_d = (mean_val - popmean) / std_val

    # Confidence interval for the mean
    if alternative == "two-sided":
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        ci_lower = mean_val - t_crit * se
        ci_upper = mean_val + t_crit * se
    elif alternative == "greater":
        t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
        ci_lower = mean_val - t_crit * se
        ci_upper = float("inf")
    else:
        t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
        ci_lower = float("-inf")
        ci_upper = mean_val + t_crit * se

    return TestResult(
        test_name="One-Sample t-test",
        statistic_name="t",
        statistic=t_stat,
        p_value=p_val,
        df=df,
        effect_size=cohen_d,
        effect_size_name="Cohen's d",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=1.0 - alpha,
        alternative=alternative,
        null_value=popmean,
        sample_sizes=(n,),
        extra_info={"mean": mean_val, "std": std_val, "se": se},
    )


def two_sample_t_test(
    x: Any,
    y: Any,
    equal_var: bool = False,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """Independent two-sample Student's t-test or Welch's t-test."""
    s1 = check_sample(x, "x")
    s2 = check_sample(y, "y")
    n1, n2 = len(s1), len(s2)
    m1, m2 = float(np.mean(s1)), float(np.mean(s2))
    v1, v2 = float(np.var(s1, ddof=1)), float(np.var(s2, ddof=1))

    diff = m1 - m2

    if equal_var:
        # Pooled variance
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        sp = np.sqrt(sp2)
        se = sp * np.sqrt(1.0 / n1 + 1.0 / n2)
        df: float = float(n1 + n2 - 2)
        test_name = "Two-Sample Student's t-test"

        # Cohen's d using pooled std
        cohen_d = diff / sp if sp > 0 else 0.0
        # Hedges' g correction factor J
        j_factor = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
        hedges_g = cohen_d * j_factor
    else:
        # Welch's unpooled variance
        se = np.sqrt(v1 / n1 + v2 / n2)
        # Welch-Satterthwaite degrees of freedom
        num = (v1 / n1 + v2 / n2) ** 2
        denom = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
        df = num / denom
        test_name = "Welch's Two-Sample t-test"

        # Cohen's d using root mean square of standard deviations
        s_avg = np.sqrt((v1 + v2) / 2.0)
        cohen_d = diff / s_avg if s_avg > 0 else 0.0
        hedges_g = None

    if se == 0.0:
        raise ValueError("Standard error of difference is zero; cannot compute t-statistic.")

    t_stat = diff / se

    if alternative == "two-sided":
        p_val = 2.0 * float(stats.t.sf(np.abs(t_stat), df=df))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        ci_lower = diff - t_crit * se
        ci_upper = diff + t_crit * se
    elif alternative == "greater":
        p_val = float(stats.t.sf(t_stat, df=df))
        t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
        ci_lower = diff - t_crit * se
        ci_upper = float("inf")
    else:
        p_val = float(stats.t.cdf(t_stat, df=df))
        t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
        ci_lower = float("-inf")
        ci_upper = diff + t_crit * se

    extra: dict[str, Any] = {
        "mean_x": m1,
        "mean_y": m2,
        "mean_diff": diff,
        "se_diff": se,
        "equal_var": equal_var,
    }
    if hedges_g is not None:
        extra["hedges_g"] = hedges_g

    return TestResult(
        test_name=test_name,
        statistic_name="t",
        statistic=t_stat,
        p_value=p_val,
        df=round(df, 3) if not equal_var else int(df),
        effect_size=cohen_d,
        effect_size_name="Cohen's d",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=1.0 - alpha,
        alternative=alternative,
        null_value=0.0,
        sample_sizes=(n1, n2),
        extra_info=extra,
    )


def paired_t_test(
    x: Any,
    y: Any,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """Paired samples Student's t-test on differences."""
    s1 = np.asarray(x, dtype=np.float64).ravel()
    s2 = np.asarray(y, dtype=np.float64).ravel()

    if len(s1) != len(s2):
        raise ValueError(f"Paired samples must have equal lengths (got len(x)={len(s1)}, len(y)={len(s2)}).")

    valid_mask = ~(np.isnan(s1) | np.isnan(s2) | np.isinf(s1) | np.isinf(s2))
    diffs = s1[valid_mask] - s2[valid_mask]

    res = one_sample_t_test(diffs, popmean=0.0, alternative=alternative, alpha=alpha)

    return TestResult(
        test_name="Paired Samples t-test",
        statistic_name="t",
        statistic=res.statistic,
        p_value=res.p_value,
        df=res.df,
        effect_size=res.effect_size,
        effect_size_name="Cohen's d_z",
        ci_lower=res.ci_lower,
        ci_upper=res.ci_upper,
        ci_level=res.ci_level,
        alternative=alternative,
        null_value=0.0,
        sample_sizes=(len(diffs),),
        extra_info={"mean_diff": float(np.mean(diffs)), "std_diff": float(np.std(diffs, ddof=1))},
    )
