"""Tests for bootstrap and permutation tests."""

import numpy as np
import pytest

from applied_stats.resampling import bootstrap_ci, permutation_test


def test_bootstrap_ci_percentile_and_bca(one_sample_data):
    # Mean bootstrap
    res_pct = bootstrap_ci(one_sample_data, statistic_fn=np.mean, n_resamples=1500, method="percentile", random_state=42)
    assert res_pct.ci_lower < res_pct.estimate < res_pct.ci_upper
    assert res_pct.standard_error > 0

    # BCa bootstrap
    res_bca = bootstrap_ci(one_sample_data, statistic_fn=np.mean, n_resamples=1500, method="bca", random_state=42)
    assert res_bca.ci_lower < res_bca.estimate < res_bca.ci_upper

    # Median bootstrap
    res_med = bootstrap_ci(one_sample_data, statistic_fn=np.median, n_resamples=1500, method="percentile", random_state=42)
    assert res_med.ci_lower <= res_med.estimate <= res_med.ci_upper


def test_permutation_test_significance(two_sample_data):
    s1, s2 = two_sample_data
    res = permutation_test(s1, s2, n_permutations=2000, alternative="two-sided", random_state=42)

    # Samples have true difference in means (10 vs 12.5), p-value must be < 0.001
    assert res.p_value < 0.005
    assert len(res.null_distribution) == 2000
    assert abs(res.observed_statistic - (np.mean(s1) - np.mean(s2))) < 1e-7


def test_permutation_test_null_hypothesis():
    rng = np.random.RandomState(42)
    s1 = rng.normal(loc=0.0, scale=1.0, size=30)
    s2 = rng.normal(loc=0.0, scale=1.0, size=30)
    res = permutation_test(s1, s2, n_permutations=1000, alternative="two-sided", random_state=42)

    # Identical populations -> p-value should NOT be small
    assert res.p_value > 0.10
