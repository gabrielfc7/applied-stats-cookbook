"""Non-parametric rank tests: Mann-Whitney U, Wilcoxon signed-rank, and Kruskal-Wallis H."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import stats

from applied_stats.base import TestResult, check_sample


def mann_whitney_u_test(
    x: Any,
    y: Any,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    use_continuity: bool = True,
) -> TestResult:
    """Mann-Whitney U rank-sum test with rank-biserial correlation effect size."""
    s1 = check_sample(x, "x")
    s2 = check_sample(y, "y")
    n1, n2 = len(s1), len(s2)

    # Compute using scipy.stats.mannwhitneyu for exact tie and continuity handling
    res = stats.mannwhitneyu(s1, s2, alternative=alternative, use_continuity=use_continuity)
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Rank-biserial correlation: r = 1 - (2 * U) / (n1 * n2)
    # Range is [-1, 1], identical to Cliff's delta
    u_total = n1 * n2
    rank_biserial = 1.0 - (2.0 * u_stat) / u_total

    # Common language effect size (probability of superiority)
    p_superiority = u_stat / u_total if alternative != "less" else 1.0 - (u_stat / u_total)

    return TestResult(
        test_name="Mann-Whitney U Test",
        statistic_name="U",
        statistic=u_stat,
        p_value=p_val,
        df=None,
        effect_size=rank_biserial,
        effect_size_name="Rank-biserial r",
        ci_lower=None,
        ci_upper=None,
        ci_level=0.95,
        alternative=alternative,
        null_value=0.0,
        sample_sizes=(n1, n2),
        extra_info={
            "common_language_effect_size": p_superiority,
            "median_x": float(np.median(s1)),
            "median_y": float(np.median(s2)),
        },
    )


def wilcoxon_signed_rank_test(
    x: Any,
    y: Any = None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    correction: bool = True,
) -> TestResult:
    """Wilcoxon signed-rank test for paired differences or single-sample median."""
    s1 = check_sample(x, "x")
    if y is not None:
        s2 = check_sample(y, "y")
        if len(s1) != len(s2):
            raise ValueError(f"Paired samples must have equal lengths (got {len(s1)} and {len(s2)}).")
        diffs = s1 - s2
    else:
        diffs = s1

    # Verify not all differences are zero
    non_zeros = diffs[diffs != 0]
    if len(non_zeros) == 0:
        raise ValueError("All paired differences are zero; test is undefined.")

    res = stats.wilcoxon(
        diffs,
        alternative=alternative,
        zero_method=zero_method,
        correction=correction,
    )
    w_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Compute matched-pairs rank biserial correlation
    # W+ is sum of ranks of positive diffs, W- is sum of ranks of negative diffs
    abs_d = np.abs(non_zeros)
    ranks = stats.rankdata(abs_d)
    w_plus = float(np.sum(ranks[non_zeros > 0]))
    w_minus = float(np.sum(ranks[non_zeros < 0]))
    total_w = w_plus + w_minus
    rank_biserial = (w_plus - w_minus) / total_w if total_w > 0 else 0.0

    return TestResult(
        test_name="Wilcoxon Signed-Rank Test",
        statistic_name="W",
        statistic=w_stat,
        p_value=p_val,
        df=None,
        effect_size=rank_biserial,
        effect_size_name="Matched-pairs rank-biserial r",
        ci_lower=None,
        ci_upper=None,
        ci_level=0.95,
        alternative=alternative,
        null_value=0.0,
        sample_sizes=(len(diffs),),
        extra_info={"W_plus": w_plus, "W_minus": w_minus, "median_diff": float(np.median(diffs))},
    )


def kruskal_wallis_test(*groups: Any) -> TestResult:
    """Kruskal-Wallis non-parametric one-way ANOVA across k independent groups."""
    cleaned_groups = [check_sample(g, f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned_groups)
    if k < 2:
        raise ValueError("Kruskal-Wallis test requires at least 2 groups.")

    res = stats.kruskal(*cleaned_groups)
    h_stat = float(res.statistic)
    p_val = float(res.pvalue)

    n_total = sum(len(g) for g in cleaned_groups)
    df = k - 1

    # Epsilon-squared effect size: epsilon^2 = (H - k + 1) / (N - k)
    epsilon_squared = (h_stat - k + 1.0) / (n_total - k)
    epsilon_squared = max(0.0, min(1.0, epsilon_squared))

    return TestResult(
        test_name="Kruskal-Wallis H Test",
        statistic_name="H",
        statistic=h_stat,
        p_value=p_val,
        df=df,
        effect_size=epsilon_squared,
        effect_size_name="Epsilon-squared",
        ci_lower=None,
        ci_upper=None,
        ci_level=0.95,
        alternative="two-sided",
        null_value=0.0,
        sample_sizes=tuple(len(g) for g in cleaned_groups),
        extra_info={"group_medians": [float(np.median(g)) for g in cleaned_groups]},
    )
