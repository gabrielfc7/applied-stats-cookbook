"""Tests for parametric hypothesis tests asserting exact parity with scipy.stats."""

import numpy as np
import pytest
from scipy import stats

from applied_stats.parametric import (
    one_sample_t_test,
    paired_t_test,
    two_sample_t_test,
)


def test_one_sample_t_test_parity_with_scipy(one_sample_data):
    popmean = 5.0
    res = one_sample_t_test(one_sample_data, popmean=popmean)
    scipy_res = stats.ttest_1samp(one_sample_data, popmean=popmean)

    # Assert t-statistic and p-value parity with scipy
    assert pytest.approx(res.statistic, abs=1e-7) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-7) == scipy_res.pvalue
    assert res.df == len(one_sample_data) - 1
    assert res.ci_lower < float(np.mean(one_sample_data)) < res.ci_upper
    assert "Cohen's d" in res.apa_format()


def test_two_sample_welch_t_test_parity_with_scipy(two_sample_data):
    s1, s2 = two_sample_data
    # Welch's t-test (equal_var=False)
    res = two_sample_t_test(s1, s2, equal_var=False)
    scipy_res = stats.ttest_ind(s1, s2, equal_var=False)

    assert pytest.approx(res.statistic, abs=1e-7) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-7) == scipy_res.pvalue
    assert res.effect_size is not None
    assert res.p_value < 0.05
    assert res.is_significant


def test_two_sample_student_t_test_parity_with_scipy(two_sample_data):
    s1, s2 = two_sample_data
    # Student's t-test (equal_var=True)
    res = two_sample_t_test(s1, s2, equal_var=True)
    scipy_res = stats.ttest_ind(s1, s2, equal_var=True)

    assert pytest.approx(res.statistic, abs=1e-7) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-7) == scipy_res.pvalue
    assert res.df == len(s1) + len(s2) - 2
    assert "hedges_g" in res.extra_info


def test_paired_t_test_parity_with_scipy(paired_data):
    before, after = paired_data
    res = paired_t_test(after, before)
    scipy_res = stats.ttest_rel(after, before)

    assert pytest.approx(res.statistic, abs=1e-7) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-7) == scipy_res.pvalue
    assert res.df == len(before) - 1
    assert res.is_significant
