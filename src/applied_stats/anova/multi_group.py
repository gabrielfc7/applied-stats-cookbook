"""Multi-group comparison tests: One-way ANOVA, Two-way Factorial ANOVA, and Tukey HSD post-hoc."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from applied_stats.base import TestResult, check_sample


@dataclass
class ANOVATable:
    """Standard ANOVA summary table with effect sizes."""

    table: pd.DataFrame
    f_stat: float
    p_val: float
    eta_squared: float
    omega_squared: float

    def to_dataframe(self) -> pd.DataFrame:
        return self.table

    def __str__(self) -> str:
        return self.table.to_string(index=True)


def one_way_anova(*groups: Any) -> ANOVATable:
    """One-way Analysis of Variance (ANOVA) across k independent groups.

    Computes between-group and within-group sums of squares, F-statistic,
    p-value, eta-squared (eta^2), and omega-squared (omega^2).
    """
    cleaned_groups = [check_sample(g, f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned_groups)
    if k < 2:
        raise ValueError("One-way ANOVA requires at least 2 groups.")

    n_per_group = [len(g) for g in cleaned_groups]
    n_total = sum(n_per_group)
    group_means = [float(np.mean(g)) for g in cleaned_groups]

    all_data = np.concatenate(cleaned_groups)
    grand_mean = float(np.mean(all_data))

    # Sum of Squares Between (Treatment)
    ss_between = float(sum(n * (m - grand_mean) ** 2 for n, m in zip(n_per_group, group_means)))
    df_between = k - 1

    # Sum of Squares Within (Error / Residual)
    ss_within = float(sum(np.sum((g - m) ** 2) for g, m in zip(cleaned_groups, group_means)))
    df_within = n_total - k

    # Total Sum of Squares
    ss_total = ss_between + ss_within
    df_total = n_total - 1

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    f_stat = ms_between / ms_within if ms_within > 0 else 0.0
    p_val = float(stats.f.sf(f_stat, dfn=df_between, dfd=df_within))

    # Effect sizes
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    # Omega-squared: unbiased estimate of population variance explained
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else 0.0
    omega_squared = max(0.0, omega_squared)

    df_res = pd.DataFrame(
        [
            {
                "Source": "Between Groups (Treatment)",
                "SS": ss_between,
                "df": df_between,
                "MS": ms_between,
                "F": f_stat,
                "PR(>F)": p_val,
                "eta_sq": eta_squared,
                "omega_sq": omega_squared,
            },
            {
                "Source": "Within Groups (Error)",
                "SS": ss_within,
                "df": df_within,
                "MS": ms_within,
                "F": np.nan,
                "PR(>F)": np.nan,
                "eta_sq": np.nan,
                "omega_sq": np.nan,
            },
            {
                "Source": "Total",
                "SS": ss_total,
                "df": df_total,
                "MS": np.nan,
                "F": np.nan,
                "PR(>F)": np.nan,
                "eta_sq": np.nan,
                "omega_sq": np.nan,
            },
        ]
    ).set_index("Source")

    return ANOVATable(
        table=df_res,
        f_stat=f_stat,
        p_val=p_val,
        eta_squared=eta_squared,
        omega_squared=omega_squared,
    )


def two_way_anova(
    data: pd.DataFrame,
    response: str,
    factor_a: str,
    factor_b: str,
) -> pd.DataFrame:
    """Two-way factorial Analysis of Variance (Balanced or unbalanced via Type II SS)."""
    df_clean = data[[response, factor_a, factor_b]].dropna()
    y = df_clean[response].to_numpy(dtype=np.float64)
    a = df_clean[factor_a].to_numpy()
    b = df_clean[factor_b].to_numpy()

    grand_mean = np.mean(y)
    n_total = len(y)

    levels_a = np.unique(a)
    levels_b = np.unique(b)
    k_a = len(levels_a)
    k_b = len(levels_b)

    # Cell means and counts
    cell_means = {}
    cell_counts = {}
    for la in levels_a:
        for lb in levels_b:
            mask = (a == la) & (b == lb)
            vals = y[mask]
            cell_counts[(la, lb)] = len(vals)
            cell_means[(la, lb)] = np.mean(vals) if len(vals) > 0 else 0.0

    # Within-cell error SS
    ss_error = 0.0
    for la in levels_a:
        for lb in levels_b:
            mask = (a == la) & (b == lb)
            if np.any(mask):
                ss_error += np.sum((y[mask] - cell_means[(la, lb)]) ** 2)

    df_error = n_total - (k_a * k_b)
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    # Factor A main effect
    ss_a = 0.0
    for la in levels_a:
        mask = a == la
        ss_a += np.sum(mask) * (np.mean(y[mask]) - grand_mean) ** 2
    df_a = k_a - 1
    ms_a = ss_a / df_a if df_a > 0 else 0.0
    f_a = ms_a / ms_error if ms_error > 0 else 0.0
    p_a = float(stats.f.sf(f_a, df_a, df_error))

    # Factor B main effect
    ss_b = 0.0
    for lb in levels_b:
        mask = b == lb
        ss_b += np.sum(mask) * (np.mean(y[mask]) - grand_mean) ** 2
    df_b = k_b - 1
    ms_b = ss_b / df_b if df_b > 0 else 0.0
    f_b = ms_b / ms_error if ms_error > 0 else 0.0
    p_b = float(stats.f.sf(f_b, df_b, df_error))

    # Total and Model SS
    ss_total = np.sum((y - grand_mean) ** 2)
    ss_model = sum(cell_counts[(la, lb)] * (cell_means[(la, lb)] - grand_mean) ** 2 for la in levels_a for lb in levels_b)
    # Interaction SS = SS_model - SS_A - SS_B
    ss_ab = max(0.0, ss_model - ss_a - ss_b)
    df_ab = df_a * df_b
    ms_ab = ss_ab / df_ab if df_ab > 0 else 0.0
    f_ab = ms_ab / ms_error if ms_error > 0 else 0.0
    p_ab = float(stats.f.sf(f_ab, df_ab, df_error))

    # Partial eta-squared
    eta_p_a = ss_a / (ss_a + ss_error) if (ss_a + ss_error) > 0 else 0.0
    eta_p_b = ss_b / (ss_b + ss_error) if (ss_b + ss_error) > 0 else 0.0
    eta_p_ab = ss_ab / (ss_ab + ss_error) if (ss_ab + ss_error) > 0 else 0.0

    return pd.DataFrame(
        [
            {"Source": factor_a, "SS": ss_a, "df": df_a, "MS": ms_a, "F": f_a, "PR(>F)": p_a, "eta_p_sq": eta_p_a},
            {"Source": factor_b, "SS": ss_b, "df": df_b, "MS": ms_b, "F": f_b, "PR(>F)": p_b, "eta_p_sq": eta_p_b},
            {"Source": f"{factor_a}:{factor_b}", "SS": ss_ab, "df": df_ab, "MS": ms_ab, "F": f_ab, "PR(>F)": p_ab, "eta_p_sq": eta_p_ab},
            {"Source": "Residual", "SS": ss_error, "df": df_error, "MS": ms_error, "F": np.nan, "PR(>F)": np.nan, "eta_p_sq": np.nan},
        ]
    ).set_index("Source")


def tukey_hsd(*groups: Any, alpha: float = 0.05) -> pd.DataFrame:
    """Tukey's Honestly Significant Difference (HSD) pairwise post-hoc comparisons."""
    cleaned_groups = [check_sample(g, f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned_groups)
    if k < 2:
        raise ValueError("Tukey HSD requires at least 2 groups.")

    # Call scipy.stats.tukey_hsd for exact studentized range p-values and confidence intervals
    res = stats.tukey_hsd(*cleaned_groups)

    rows = []
    pair_idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            diff = float(res.statistic[i, j])
            p_val = float(res.pvalue[i, j])
            ci_low = float(res.confidence_interval(1.0 - alpha).low[i, j])
            ci_high = float(res.confidence_interval(1.0 - alpha).high[i, j])

            rows.append(
                {
                    "group1": f"Group {i}",
                    "group2": f"Group {j}",
                    "diff": diff,
                    "p_adj": p_val,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "reject_null": p_val < alpha,
                }
            )
            pair_idx += 1

    return pd.DataFrame(rows)
