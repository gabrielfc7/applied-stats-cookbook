"""Tests for ANOVA and post-hoc pairwise comparisons."""

import pytest
from scipy import stats

from applied_stats.anova import one_way_anova, tukey_hsd, two_way_anova


def test_one_way_anova_parity(multi_group_data):
    g1, g2, g3 = multi_group_data
    anova_res = one_way_anova(g1, g2, g3)
    scipy_res = stats.f_oneway(g1, g2, g3)

    assert pytest.approx(anova_res.f_stat, abs=1e-7) == scipy_res.statistic
    assert pytest.approx(anova_res.p_val, abs=1e-7) == scipy_res.pvalue
    assert anova_res.eta_squared > 0.50
    assert anova_res.omega_squared > 0.45

    df_tab = anova_res.to_dataframe()
    assert "Between Groups (Treatment)" in df_tab.index
    assert "Within Groups (Error)" in df_tab.index


def test_two_way_anova(factorial_data):
    table = two_way_anova(
        factorial_data,
        response="Yield",
        factor_a="FactorA",
        factor_b="FactorB",
    )
    assert "FactorA" in table.index
    assert "FactorB" in table.index
    assert "FactorA:FactorB" in table.index
    assert "Residual" in table.index

    # Both main effects should be statistically significant
    assert table.loc["FactorA", "PR(>F)"] < 0.001
    assert table.loc["FactorB", "PR(>F)"] < 0.001


def test_tukey_hsd_post_hoc(multi_group_data):
    g1, g2, g3 = multi_group_data
    res_df = tukey_hsd(g1, g2, g3, alpha=0.05)

    assert len(res_df) == 3  # 3 pairwise comparisons: (0,1), (0,2), (1,2)
    assert "p_adj" in res_df.columns
    assert "ci_lower" in res_df.columns
    assert "ci_upper" in res_df.columns
    # Groups 1 vs 3 should reject null
    row_0_2 = res_df[(res_df["group1"] == "Group 0") & (res_df["group2"] == "Group 2")].iloc[0]
    assert row_0_2["reject_null"] is True or row_0_2["p_adj"] < 0.05
