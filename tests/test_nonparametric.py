"""Tests for non-parametric tests asserting exact parity with scipy.stats."""

import pytest
from scipy import stats

from applied_stats.nonparametric import (
    kruskal_wallis_test,
    mann_whitney_u_test,
    wilcoxon_signed_rank_test,
)


def test_mann_whitney_u_test_parity(two_sample_data):
    s1, s2 = two_sample_data
    res = mann_whitney_u_test(s1, s2, alternative="two-sided")
    scipy_res = stats.mannwhitneyu(s1, s2, alternative="two-sided")

    assert pytest.approx(res.statistic, abs=1e-7) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-7) == scipy_res.pvalue
    assert -1.0 <= res.effect_size <= 1.0
    assert "Rank-biserial r" in res.effect_size_name


def test_wilcoxon_signed_rank_test_parity(paired_data):
    before, after = paired_data
    res = wilcoxon_signed_rank_test(after, before)
    scipy_res = stats.wilcoxon(after - before)

    assert pytest.approx(res.statistic, abs=1e-7) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-7) == scipy_res.pvalue
    assert res.is_significant


def test_kruskal_wallis_test_parity(multi_group_data):
    g1, g2, g3 = multi_group_data
    res = kruskal_wallis_test(g1, g2, g3)
    scipy_res = stats.kruskal(g1, g2, g3)

    assert pytest.approx(res.statistic, abs=1e-7) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-7) == scipy_res.pvalue
    assert res.df == 2
    assert 0.0 <= res.effect_size <= 1.0
